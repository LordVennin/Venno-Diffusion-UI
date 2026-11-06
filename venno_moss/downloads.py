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

REPOS = {"sft": "OpenMOSS-Team/MOSS-Image-SFT", "base": "OpenMOSS-Team/MOSS-Image-Base"}
DESCRIPTIONS = {
    "sft": "The supervised fine-tune: cleaner composition and text; what most LoRAs are trained on.",
    "base": "The base model: more varied, less polished; better for some style training.",
}
SHARED = ["text_encoder/*", "tokenizer/*", "vae/*"]
OWN = ["model_index.json", "transformer/*"]

JOBS: dict[str, "Download"] = {}
_lock = threading.Lock()


def _folder_bytes(root: str) -> int:
    total = 0
    for dirpath, _, names in os.walk(root):
        for n in names:
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
        self.start_bytes = 0
        self.status = "queued"  # queued | downloading | done | error
        self.error = ""
        self.started = time.time()

    def to_dict(self) -> dict:
        done = max(0, _folder_bytes(self.target) - self.start_bytes) if self.status == "downloading" else (self.total if self.status == "done" else 0)
        return {"id": self.id, "variant": self.variant, "name": self.name, "status": self.status, "error": self.error,
                "total": self.total, "done": min(done, self.total) if self.total else done, "shared": "text_encoder/*" in self.patterns}

    def run(self):
        try:
            from huggingface_hub import HfApi, snapshot_download

            self.status = "downloading"
            info = HfApi().model_info(self.repo, files_metadata=True)
            self.total = sum(s.size or 0 for s in info.siblings if any(fnmatch.fnmatch(s.rfilename, p) for p in self.patterns))
            os.makedirs(self.target, exist_ok=True)
            self.start_bytes = _folder_bytes(self.target)
            self.total = max(0, self.total - self.start_bytes)  # a resumed download only needs the rest
            print(f"[Venno MOSS] downloading {self.repo} into {self.target} ({self.total / 1e9:.1f} GB)", flush=True)
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
