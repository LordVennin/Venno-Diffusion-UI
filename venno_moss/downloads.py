"""Downloading MOSS-Image models from Hugging Face into the models folder (models/moss/MOSS-Image-Base, …).

The text encoder (17 GB), tokenizer and VAE are the same files for Base and SFT, so they are only downloaded
with the first model; later models bring their transformer (10 GB).

Plain HTTPS (urllib), like Venno's model downloader, rather than the huggingface_hub client: on some systems the
client's requests hang where urllib's work. One request lists the files with their sizes, then each file is
fetched from its resolve URL into a .part file that resumes (HTTP Range) after a dropped connection or a
restart; files already complete are skipped. HF_TOKEN is sent when set.
"""

from __future__ import annotations

import fnmatch
import http.client
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid

from . import config, models

HF = "https://huggingface.co"
REPOS = {"base": "OpenMOSS-Team/MOSS-Image-Base", "sft": "OpenMOSS-Team/MOSS-Image-SFT", "kromoss": "lodestones/Kromoss"}
DESCRIPTIONS = {  # from OpenMOSS's user guide
    "base": "Left without post-training so LoRAs train well on it: OpenMOSS's recommended model to train LoRAs on. "
            "Needs good settings: 50+ steps, CFG 3–6 (higher for short prompts).",
    "sft": "Base with OpenMOSS's own supervised fine-tune: more polished pictures and text layout out of the box.",
    "kromoss": "Community full fine-tune of Base by lodestones (Apache-2.0): 704k images at 1024 px; scores above Base and SFT "
               "on PartiPrompts. Update 11000. Same text encoder and VAE as Base, so 10 GB once those are here.",
}
# the diffusers VAE only: Kromoss's vae/ also holds a ComfyUI copy, and two files in one folder would load as one
LORAS = {  # single files into the LoRA folder
    "turbo": {
        "repo": "lodestones/Kromoss", "file": config.TURBO_FILES[0], "name": "8-step turbo LoRA (DMD2)", "size_gb": 1.2,
        "description": "For Fast preview: 8 steps at CFG 1, about 12× faster than 50 steps with CFG. By lodestones, distilled from "
                       "MOSS-Image Base (Apache-2.0); works on Base, SFT and Kromoss, together with your own LoRAs.",
    },
}
SHARED = ["text_encoder/*", "tokenizer/*", "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
OWN = ["model_index.json", "transformer/*"]
TIMEOUT = 60  # seconds without an answer (file list) or without data (a file) before giving up / retrying
RETRIES = 5  # per file, resuming where it stopped
CHUNK = 1 << 20

JOBS: dict[str, "Download"] = {}
_lock = threading.Lock()


CONNECT_TIMEOUT = 10  # per address: a dead IPv6 route must not eat the whole TIMEOUT


def _connect(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, **_):
    """socket.create_connection, but IPv4 addresses first and a short timeout per address. Python tries the
    addresses one by one in the system's order, usually IPv6 first; where IPv6 is configured but doesn't route
    (common on home Linux setups) every request waited for the full timeout there. Browsers and curl try both at
    once, so they don't notice."""
    host, port = address
    infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    infos.sort(key=lambda info: info[0] != socket.AF_INET)
    error = None
    for family, kind, proto, _, addr in infos:
        sock = socket.socket(family, kind, proto)
        try:
            limit = timeout if isinstance(timeout, (int, float)) else None
            sock.settimeout(min(CONNECT_TIMEOUT, limit) if limit else CONNECT_TIMEOUT)
            if source_address:
                sock.bind(source_address)
            sock.connect(addr)
            sock.settimeout(limit)
            return sock
        except OSError as e:
            error = e
            sock.close()
    raise error or OSError(f"no address for {host}")


class _Connection(http.client.HTTPSConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = _connect


class _Handler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_Connection, req, context=self._context)


_opener = urllib.request.build_opener(_Handler())  # keeps the proxy settings from the environment


def _open(req: urllib.request.Request, timeout: float):
    return _opener.open(req, timeout=timeout)


def _request(url: str, start: int = 0) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"User-Agent": "VennoMOSS/1.0"})
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if start:
        req.add_header("Range", f"bytes={start}-")
    return req


def file_list(repo: str) -> list[tuple[str, int]]:
    """(path, size) of every file in the repository's main branch."""
    try:
        with _open(_request(f"{HF}/api/models/{repo}/tree/main?recursive=true"), TIMEOUT) as r:
            items = json.load(r)
    except (TimeoutError, OSError) as e:
        if "timed out" in str(e).lower() or isinstance(e, TimeoutError):
            raise RuntimeError(f"Hugging Face didn't answer within {TIMEOUT} s. Check the internet connection (or a proxy / firewall) and try again.") from None
        raise RuntimeError(f"Could not reach Hugging Face: {e}") from None
    return [(i["path"], int(i.get("lfs", {}).get("size") or i.get("size") or 0)) for i in items if i.get("type") == "file"]


class Cancelled(Exception):
    pass


class Download:
    def __init__(self, variant: str):
        self.id = uuid.uuid4().hex[:12]
        self.variant = variant
        if variant in LORAS:
            spec = LORAS[variant]
            self.repo, self.name, self.target, self.patterns = spec["repo"], spec["name"], config.default_lora_path(), [spec["file"]]
        else:
            self.repo = REPOS[variant]
            self.name = models.KNOWN[variant]
            self.target = os.path.join(config.models_path(), self.name)
            found = models.scan()
            self.patterns = OWN + ([] if (found.text_encoder and found.tokenizer and found.vae) else SHARED)
        self.total = 0
        self.finished = 0  # bytes of the files already complete
        self.current = 0  # bytes of the file being fetched
        self.status = "queued"  # queued | downloading | done | error
        self.phase = "Waiting…"
        self.error = ""
        self.started = time.time()
        self.cancel = threading.Event()

    def to_dict(self) -> dict:
        done = self.total if self.status == "done" else min(self.finished + self.current, self.total)
        return {"id": self.id, "variant": self.variant, "name": self.name, "status": self.status, "phase": self.phase, "error": self.error,
                "total": self.total, "done": done, "seconds": round(time.time() - self.started), "shared": "text_encoder/*" in self.patterns}

    def _fetch(self, path: str, size: int):
        dest = os.path.join(self.target, *path.split("/"))
        if os.path.isfile(dest) and (not size or os.path.getsize(dest) == size):
            self.finished += size
            return
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        part = dest + ".part"
        url = f"{HF}/{self.repo}/resolve/main/{path}"
        attempt = 0
        while True:
            have = os.path.getsize(part) if os.path.isfile(part) else 0
            if size and have > size:  # a stale part from another file version
                os.remove(part)
                have = 0
            self.current = have
            if size and have == size:
                break
            try:
                with _open(_request(url, have), TIMEOUT) as r:
                    if have and r.status != 206:  # the server ignored the range: start over
                        have = 0
                    with open(part, "ab" if have else "wb") as f:
                        while True:
                            if self.cancel.is_set():
                                raise Cancelled
                            chunk = r.read(CHUNK)
                            if not chunk:
                                break
                            f.write(chunk)
                            self.current += len(chunk)
                if not size or os.path.getsize(part) >= size:
                    break
            except Cancelled:
                raise
            except urllib.error.HTTPError as e:
                if e.code == 416 and size and os.path.getsize(part) == size:  # already complete
                    break
                raise RuntimeError(f"{path}: HTTP {e.code} {e.reason}") from None
            except (OSError, http.client.HTTPException) as e:
                attempt += 1
                if attempt > RETRIES:
                    raise RuntimeError(f"{path}: the connection kept dropping ({e})") from None
                print(f"[Venno MOSS] {path}: connection dropped ({e}); resuming ({attempt}/{RETRIES})", flush=True)
                time.sleep(2 * attempt)
        os.replace(part, dest)
        self.finished += size or os.path.getsize(dest)
        self.current = 0

    def run(self):
        try:
            self.status, self.phase = "downloading", "Asking Hugging Face for the file list…"
            print(f"[Venno MOSS] {self.name}: asking Hugging Face ({self.repo}) for the file list", flush=True)
            files = [(p, s) for p, s in file_list(self.repo) if any(fnmatch.fnmatch(p, pat) for pat in self.patterns)]
            if not files:
                raise RuntimeError(f"{self.repo} lists none of the expected files")
            self.total = sum(s for _, s in files)
            print(f"[Venno MOSS] downloading {len(files)} files ({self.total / 1e9:.1f} GB) into {self.target}; resumes where it stopped", flush=True)
            for k, (path, size) in enumerate(sorted(files, key=lambda f: f[1])):  # small files (configs) first
                self.phase = f"File {k + 1} of {len(files)}: {path}"
                self._fetch(path, size)
            self.status, self.phase = "done", "Done"
            print(f"[Venno MOSS] {self.name} downloaded", flush=True)
        except Cancelled:
            self.status, self.phase, self.error = "error", "Stopped", "Stopped; start again to resume"
        except Exception as e:
            self.status, self.error = "error", str(e) or type(e).__name__
            print(f"[Venno MOSS] download of {self.repo} failed: {self.error}", flush=True)


def catalog() -> list[dict]:
    found = models.scan()
    have = {m.name for m in found.models}
    shared = bool(found.text_encoder and found.tokenizer and found.vae)
    from . import lora

    out = [{"variant": v, "kind": "model", "name": models.KNOWN[v], "repo": r, "description": DESCRIPTIONS[v], "installed": models.KNOWN[v] in have,
            "size_gb": 10.2 if shared else 27.2} for v, r in REPOS.items()]
    turbo = lora.find_turbo()
    out += [{"variant": v, "kind": "lora", "name": spec["name"], "repo": spec["repo"], "description": spec["description"],
             "installed": turbo is not None, "size_gb": spec["size_gb"]} for v, spec in LORAS.items()]
    return out


def start(variant: str) -> Download:
    if variant not in REPOS and variant not in LORAS:
        raise ValueError(f"Unknown download: {variant}")
    with _lock:
        for job in JOBS.values():
            if job.variant == variant and job.status in ("queued", "downloading"):
                return job
        job = Download(variant)
        JOBS[job.id] = job
    threading.Thread(target=job.run, daemon=True, name=f"venno-moss-download-{variant}").start()
    return job


def jobs() -> list[dict]:
    return [j.to_dict() for j in sorted(JOBS.values(), key=lambda j: j.started, reverse=True)]
