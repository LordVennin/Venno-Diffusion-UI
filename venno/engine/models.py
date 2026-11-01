"""Loads and caches the Anima engine (diffusion model + text encoder + VAE) for Venno jobs."""

from __future__ import annotations

import gc
import os
import threading
import time

import torch

from backend import memory_management
from backend.loader import forge_loader


MODEL_EXTENSIONS = (".safetensors", ".sft", ".gguf", ".ckpt", ".pt", ".pth", ".bin")


def _scan(kind: str) -> dict[str, str]:
    """title (path relative to its folder, "/"-separated) → file, for every model file of a kind."""
    from venno import config

    found: dict[str, str] = {}
    for root in config.model_folders(kind):
        for dirpath, _, files in os.walk(root, followlinks=True):
            for f in sorted(files):
                if f.lower().endswith(MODEL_EXTENSIONS):
                    path = os.path.join(dirpath, f)
                    found.setdefault(os.path.relpath(path, root).replace(os.sep, "/"), path)
    return dict(sorted(found.items(), key=lambda kv: kv[0].lower()))


NEEDS_TEXT_ENCODER = ("Anima needs the Qwen3 0.6B text encoder (qwen_3_06b_base.safetensors): pick it as the text encoder in the "
                      "top bar, or download it in ⚙ Settings → Models. (Anima 3.8B adds its Qwen3.5 4B encoder by itself.)")


def checkpoints() -> dict[str, str]:
    return _scan("checkpoint")


def modules() -> dict[str, str]:
    """Text encoders and VAEs by file name (what Studio's module pickers list)."""
    out: dict[str, str] = {}
    for kind in ("text_encoder", "vae"):
        for title, path in _scan(kind).items():
            out.setdefault(os.path.basename(title), path)
    return out


def checkpoint_name(title: str) -> str:
    return os.path.splitext(title)[0]


class ModelManager:
    def __init__(self):
        self.engine = None  # backend diffusion engine (backend/diffusion_engine/anima.py)
        self.key: tuple | None = None
        self.info: dict = {}
        self.lock = threading.RLock()

    @staticmethod
    def resolve_checkpoint(name: str) -> tuple[str, str]:
        """A checkpoint title, file name or name (an "A1111 title" with a "[hash]" suffix works too) → (path, name)."""
        if os.path.isfile(name):
            return name, os.path.splitext(os.path.basename(name))[0]
        want = name.split(" [")[0].strip().replace(os.sep, "/").lower()
        found = checkpoints()
        for title, path in found.items():
            t = title.lower()
            if want in (t, checkpoint_name(t), os.path.basename(t), checkpoint_name(os.path.basename(t))):
                return path, checkpoint_name(os.path.basename(title))
        raise ValueError(f'Checkpoint "{name}" not found')

    @staticmethod
    def resolve_module(name: str) -> str:
        if os.path.isfile(name):
            return name
        if os.path.basename(name).lower().startswith("qwen35"):  # Anima 3.8B's second encoder is found on its own
            raise ValueError("Pick the Qwen3 0.6B text encoder here; Anima 3.8B finds qwen35_4b by itself")
        path = modules().get(os.path.basename(name))
        if path is None:
            raise ValueError(f'Text encoder / VAE "{name}" not found')
        return path

    # -- loading --
    @torch.inference_mode()
    def get(self, checkpoint: str, modules: list[str]):
        ckpt_path, ckpt_name = self.resolve_checkpoint(checkpoint)
        module_paths = sorted(self.resolve_module(m) for m in modules if m)
        encoders = {os.path.realpath(p) for p in _scan("text_encoder").values()}
        if not any(os.path.realpath(p) in encoders for p in module_paths):
            raise ValueError(NEEDS_TEXT_ENCODER)
        key = (ckpt_path, tuple(module_paths))
        with self.lock:
            if self.engine is not None and self.key == key:
                return _clean(self.engine)
            self.unload()
            t0 = time.time()
            print(f"[Venno] loading {ckpt_name} + {[os.path.basename(m) for m in module_paths]}")
            try:
                self.engine = forge_loader(ckpt_path, additional_state_dicts=module_paths)
            except (ValueError, AssertionError) as e:
                if "Qwen3" not in str(e):  # "This file doesn't fit Qwen3Model…" / "You do not have Qwen3 state dict!"
                    raise
                chosen = [os.path.basename(p) for p in module_paths if os.path.realpath(p) in encoders]
                raise ValueError(f'{", ".join(chosen)} is not the text encoder Anima needs. {NEEDS_TEXT_ENCODER}') from e
            self.key = key
            self.info = {"name": ckpt_name, "path": ckpt_path, "modules": [os.path.basename(m) for m in module_paths]}
            from venno.engine import anima38

            try:
                if anima38.attach(self.engine, ckpt_path) is not None:
                    self.info["variant"] = "anima38"
            except Exception:
                self.unload()
                raise
            print(f"[Venno] model ready in {time.time() - t0:.1f}s")
            return self.engine

    @torch.inference_mode()
    def unload(self):
        with self.lock:
            if self.engine is None:
                return
            self.engine = None
            self.key = None
            self.info = {}
            memory_management.unload_all_models()
            from venno.engine import anima38, guides

            anima38.release()  # Qwen3.5 4B: ~8 GB of RAM
            guides.release()  # keyed by id() of the old diffusion model, which a new model can reuse
            memory_management.soft_empty_cache()
            gc.collect()


def _clean(engine):
    """Start every job from the unpatched objects (a previous job's LoRA / guide clones are dropped)."""
    engine.forge_objects = engine.forge_objects_original.shallow_copy()
    return engine


manager = ModelManager()
