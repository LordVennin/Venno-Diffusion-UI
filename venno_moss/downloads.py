"""Downloading MOSS-Image models from Hugging Face into the models folder (models/moss/MOSS-Image-Base, …).

The text encoder (17 GB), tokenizer and VAE are the same files for Base and SFT, so they are only downloaded
with the first model; later models bring their transformer (10 GB).

Each model also comes in two small sizes, as GGUF files (venno_moss/gguf.py): Q8 (5.5 GB, for 12 GB cards)
and Q4 (3.2 GB, for 8 GB cards), from unsloth's conversions of Base and SFT and molbal's of Kromoss. They go
into the models folder as single files; what every model needs goes into models/moss/shared: a GGUF of the
Qwen3.5 9B text encoder (unsloth's: Q6, 7.5 GB, with a Q8 model; Q4, 5.7 GB, with a Q4 model) with its
config, the tokenizer and the VAE.

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
from dataclasses import dataclass, field

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

# ---- the small sizes (GGUF)
SIZES = {  # size → the transformer's quantization, and the text encoder download that comes with it when there is none yet
    "q8": {"label": "12 GB cards", "transformer": "Q8_0", "text_encoder": "te-q6"},
    "q4": {"label": "8 GB cards", "transformer": "Q4_K_M", "text_encoder": "te-q4"},
}
GGUF = {  # model → repository, file in it ({q} = quantization), name here, size in GB per quantization
    "base": ("unsloth/MOSS-Image-Base-GGUF", "MOSS-Image-Base-{q}.gguf", "MOSS-Image-Base-{q}", {"Q8_0": 5.5, "Q4_K_M": 3.2}),
    "sft": ("unsloth/MOSS-Image-SFT-GGUF", "MOSS-Image-SFT-{q}.gguf", "MOSS-Image-SFT-{q}", {"Q8_0": 5.5, "Q4_K_M": 3.2}),
    "kromoss": ("molbal/Kromoss-GGUF", "kromoss-{q}.gguf", "Kromoss-{q}", {"Q8_0": 5.7, "Q4_K_M": 3.6}),
}
TEXT_ENCODER_REPO = "unsloth/Qwen3.5-9B-GGUF"  # the stock Qwen3.5 9B: MOSS-Image's text encoder is the same weights
TEXT_ENCODER_FILE = "Qwen3.5-9B-{q}.gguf"
TEXT_ENCODERS = {"te-full": ("", 16.8), "te-q6": ("Q6_K", 7.5), "te-q4": ("Q4_K_M", 5.7)}  # download → quantization, GB
PARTS_REPO = REPOS["base"]  # the text encoder's config, the tokenizer and the VAE: the same in every MOSS-Image repository
SHARED_FOLDER = "shared"


@dataclass
class Source:
    """Files of one repository: those matching `patterns` go into `folder`, keeping their path (or renamed)."""

    repo: str
    patterns: list[str]
    folder: str
    rename: dict[str, str] = field(default_factory=dict)  # path in the repository → path in the folder


def shared_folder() -> str:
    return os.path.join(config.models_path(), SHARED_FOLDER)


def text_encoder_sources(quantization: str) -> list[Source]:
    """A text encoder into the shared folder: the full one (safetensors) or a GGUF, with the model's config beside it."""
    if not quantization:
        return [Source(PARTS_REPO, ["text_encoder/*"], shared_folder())]
    file = TEXT_ENCODER_FILE.format(q=quantization)
    return [Source(PARTS_REPO, ["text_encoder/config.json"], shared_folder()),
            Source(TEXT_ENCODER_REPO, [file], shared_folder(), {file: f"text_encoder/{file}"})]


def small_sources(model: str, size: str, found: models.Found) -> list[Source]:
    """A model's GGUF transformer, and whichever of text encoder, tokenizer and VAE isn't here yet."""
    repo, file, name, _ = GGUF[model]
    q = SIZES[size]["transformer"]
    out = [Source(repo, [file.format(q=q)], config.models_path(), {file.format(q=q): f"{name.format(q=q)}.gguf"})]
    if not found.text_encoder:
        out += text_encoder_sources(TEXT_ENCODERS[SIZES[size]["text_encoder"]][0])
    parts = ([] if found.tokenizer else ["tokenizer/*"]) + ([] if found.vae else ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"])
    if parts:
        out.append(Source(PARTS_REPO, parts, shared_folder()))
    return out


def split_variant(variant: str) -> tuple[str, str]:
    """"sft-q4" → ("sft", "q4"); "sft" → ("sft", "")."""
    model, _, size = variant.partition("-")
    return (model, size) if model in REPOS and (not size or size in SIZES) else ("", "")
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
        found = models.scan()
        model, size = split_variant(variant)
        if variant in LORAS:
            spec = LORAS[variant]
            self.name, self.sources = spec["name"], [Source(spec["repo"], [spec["file"]], config.default_lora_path())]
        elif variant in TEXT_ENCODERS:
            q = TEXT_ENCODERS[variant][0]
            self.name, self.sources = f"Text encoder ({q or 'full'})", text_encoder_sources(q)
        elif size:
            self.name, self.sources = GGUF[model][2].format(q=SIZES[size]["transformer"]), small_sources(model, size, found)
        else:
            self.name = models.KNOWN[variant]
            patterns = OWN + ([] if (found.text_encoder and found.tokenizer and found.vae) else SHARED)
            self.sources = [Source(REPOS[variant], patterns, os.path.join(config.models_path(), self.name))]
        self.shared = any(p.startswith("text_encoder/") or p.startswith("Qwen") for src in self.sources for p in src.patterns) and variant not in TEXT_ENCODERS
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
                "total": self.total, "done": done, "seconds": round(time.time() - self.started), "shared": self.shared}

    def _fetch(self, repo: str, path: str, dest: str, size: int):
        if os.path.isfile(dest) and (not size or os.path.getsize(dest) == size):
            self.finished += size
            return
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        part = dest + ".part"
        url = f"{HF}/{repo}/resolve/main/{path}"
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
            files = []  # (repo, path, destination, size)
            listed: dict[str, list] = {}
            for src in self.sources:
                if src.repo not in listed:
                    print(f"[Venno MOSS] {self.name}: asking Hugging Face ({src.repo}) for the file list", flush=True)
                    listed[src.repo] = file_list(src.repo)
                wanted = [(p, s) for p, s in listed[src.repo] if any(fnmatch.fnmatch(p, pat) for pat in src.patterns)]
                if not wanted:
                    raise RuntimeError(f"{src.repo} lists none of the expected files ({', '.join(src.patterns)})")
                files += [(src.repo, p, os.path.join(src.folder, *src.rename.get(p, p).split("/")), s) for p, s in wanted]
            self.total = sum(f[3] for f in files)
            folders = sorted({src.folder for src in self.sources})
            print(f"[Venno MOSS] downloading {len(files)} files ({self.total / 1e9:.1f} GB) into {', '.join(folders)}; resumes where it stopped", flush=True)
            for k, (repo, path, dest, size) in enumerate(sorted(files, key=lambda f: f[3])):  # small files (configs) first
                self.phase = f"File {k + 1} of {len(files)}: {path}"
                self._fetch(repo, path, dest, size)
            self.status, self.phase = "done", "Done"
            print(f"[Venno MOSS] {self.name} downloaded", flush=True)
        except Cancelled:
            self.status, self.phase, self.error = "error", "Stopped", "Stopped; start again to resume"
        except Exception as e:
            self.status, self.error = "error", str(e) or type(e).__name__
            print(f"[Venno MOSS] download of {self.name} failed: {self.error}", flush=True)


def catalog() -> list[dict]:
    """What Settings → Models offers: each model in three sizes, the text encoders, the turbo LoRA."""
    found = models.scan()
    have = {m.name for m in found.models}
    encoders = {t.name for t in found.text_encoders}
    parts = 0.0 if found.vae else 0.3  # VAE and tokenizer
    from . import lora

    def size(variant, label, hint, gb, installed):
        return {"variant": variant, "label": label, "hint": hint, "size_gb": round(gb, 1), "installed": installed}

    out = []
    for v, repo in REPOS.items():
        gguf_repo, _, name, gb = GGUF[v]
        sizes = [size(v, "Full", "24 GB cards", 10.2 + (0 if found.text_encoder and found.tokenizer and found.vae else 17.0), models.KNOWN[v] in have)]
        for key, spec in SIZES.items():
            q = spec["transformer"]
            te = 0.0 if found.text_encoder else TEXT_ENCODERS[spec["text_encoder"]][1]
            sizes.append(size(f"{v}-{key}", q.split("_")[0], spec["label"], gb[q] + te + parts, name.format(q=q) in have))
        out.append({"kind": "model", "name": models.KNOWN[v], "repo": repo, "small_repo": gguf_repo, "description": DESCRIPTIONS[v], "sizes": sizes})
    names = {"te-full": models.FULL_TEXT_ENCODER, **{k: TEXT_ENCODER_FILE.format(q=q)[: -len(".gguf")] for k, (q, _) in TEXT_ENCODERS.items() if q}}
    out.append({"kind": "text_encoder", "name": "Text encoder (Qwen3.5 9B)", "repo": TEXT_ENCODER_REPO,
                "description": "Comes with the first model. Get another size here: it stays in system RAM, so its size is the RAM it needs; "
                               "pick the one to use below. The small ones are GGUFs of the same encoder (by unsloth).",
                "sizes": [size(k, "Full" if not q else q.split("_")[0], {"te-full": "32 GB of RAM", "te-q6": "16 GB of RAM", "te-q4": "smallest"}[k], gb, names[k] in encoders)
                          for k, (q, gb) in TEXT_ENCODERS.items()]})
    turbo = lora.find_turbo()
    out += [{"kind": "lora", "name": spec["name"], "repo": spec["repo"], "description": spec["description"],
             "sizes": [size(v, "Download", "", spec["size_gb"], turbo is not None)]} for v, spec in LORAS.items()]
    for entry in out:
        entry["installed"] = all(x["installed"] for x in entry["sizes"])
    return out


def start(variant: str) -> Download:
    if variant not in LORAS and variant not in TEXT_ENCODERS and not split_variant(variant)[0]:
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
