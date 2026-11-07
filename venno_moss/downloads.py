"""Downloading MOSS-Image models from Hugging Face into the models folder (models/moss/MOSS-Image-SFT, …).

The text encoder (17 GB), tokenizer and VAE are the same files for Base and SFT, so they are only downloaded
with the first model; later models bring their transformer (10 GB). Downloads resume where they stopped.
"""

from __future__ import annotations

import fnmatch
import os
import threading
import time
import uuid

from . import config, models

REPOS = {"base": "OpenMOSS-Team/MOSS-Image-Base", "sft": "OpenMOSS-Team/MOSS-Image-SFT"}
DESCRIPTIONS = {  # from OpenMOSS's user guide
    "base": "Left without post-training so LoRAs train well on it: OpenMOSS's recommended model to train LoRAs on. "
            "Needs good settings: 50+ steps, CFG 3–6 (higher for short prompts).",
    "sft": "Base with OpenMOSS's own supervised fine-tune: more polished pictures and text layout out of the box.",
}
SHARED = ["text_encoder/*", "tokenizer/*", "vae/*"]
OWN = ["model_index.json", "transformer/*"]

JOBS: dict[str, "Download"] = {}
_lock = threading.Lock()


LIST_TIMEOUT = 60  # seconds to wait for Hugging Face's file list


def _arrived(root: str) -> int:
    """Bytes downloaded so far: finished files plus the hub's partial (*.incomplete) files in .cache, which are
    renamed into place when done (anything else there is bookkeeping, and counting it would count twice)."""
    total = 0
    for dirpath, _, names in os.walk(root):
        in_cache = f"{os.sep}.cache" in dirpath[len(root):]
        for n in names:
            if in_cache and not n.endswith(".incomplete"):
                continue
            try:
                total += os.path.getsize(os.path.join(dirpath, n))
            except OSError:
                pass
    return total


class Download:
    def __init__(self, variant: str):
        self.id = uuid.uuid4().hex[:12]
        self.variant = variant
        self.repo = REPOS[variant]
        self.name = models.KNOWN[variant]
        self.target = os.path.join(config.models_path(), self.name)
        found = models.scan()
        self.patterns = OWN + ([] if (found.text_encoder and found.tokenizer and found.vae) else SHARED)
        self.total = 0
        self.done = 0
        self.status = "queued"  # queued | downloading | done | error
        self.phase = "Waiting…"
        self.error = ""
        self.started = time.time()

    def to_dict(self) -> dict:
        if self.status == "downloading" and self.total:
            self.done = max(self.done, min(_arrived(self.target), self.total))  # never backwards
        elif self.status == "done":
            self.done = self.total
        return {"id": self.id, "variant": self.variant, "name": self.name, "status": self.status, "phase": self.phase, "error": self.error,
                "total": self.total, "done": self.done, "seconds": round(time.time() - self.started), "shared": "text_encoder/*" in self.patterns}

    def _file_list(self):
        """Hugging Face's file list (sizes), or an error when it doesn't answer: a hung request would otherwise
        leave the download waiting forever with nothing to show."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as Timeout

        from huggingface_hub import HfApi

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            return pool.submit(HfApi().model_info, self.repo, files_metadata=True).result(timeout=LIST_TIMEOUT)
        except Timeout:
            raise RuntimeError(f"Hugging Face didn't answer within {LIST_TIMEOUT} s. Check the internet connection (or a proxy / firewall) and try again.") from None
        finally:
            pool.shutdown(wait=False)

    def run(self):
        try:
            from huggingface_hub import snapshot_download

            self.status, self.phase = "downloading", "Asking Hugging Face for the file list…"
            print(f"[Venno MOSS] {self.name}: asking Hugging Face ({self.repo}) for the file list", flush=True)
            info = self._file_list()
            self.total = sum(s.size or 0 for s in info.siblings if any(fnmatch.fnmatch(s.rfilename, p) for p in self.patterns))
            os.makedirs(self.target, exist_ok=True)
            self.phase = "Downloading"
            print(f"[Venno MOSS] downloading {self.repo} into {self.target} ({self.total / 1e9:.1f} GB; resumes where it stopped)", flush=True)
            snapshot_download(self.repo, allow_patterns=self.patterns, local_dir=self.target)
            self.status = "done"
            print(f"[Venno MOSS] {self.name} downloaded", flush=True)
        except Exception as e:
            self.status, self.error = "error", str(e) or type(e).__name__
            print(f"[Venno MOSS] download of {self.repo} failed: {self.error}", flush=True)


def catalog() -> list[dict]:
    found = models.scan()
    have = {m.name for m in found.models}
    shared = bool(found.text_encoder and found.tokenizer and found.vae)
    return [{"variant": v, "name": models.KNOWN[v], "repo": r, "description": DESCRIPTIONS[v], "installed": models.KNOWN[v] in have,
             "size_gb": 10.2 if shared else 27.2} for v, r in REPOS.items()]


def start(variant: str) -> Download:
    if variant not in REPOS:
        raise ValueError(f"Unknown model: {variant}")
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
