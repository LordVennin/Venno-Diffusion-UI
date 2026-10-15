"""Loads and caches the Anima engine (diffusion model + text encoder + VAE) for Venno jobs."""

from __future__ import annotations

import gc
import os
import threading
import time
import types
import weakref

from backend import memory_management
from backend.loader import forge_loader


class ModelManager:
    def __init__(self):
        self.engine = None  # backend ForgeDiffusionEngine (backend/diffusion_engine/anima.py)
        self.key: tuple | None = None
        self.info: dict = {}
        self.lock = threading.RLock()

    # -- name resolution (Phase 1: reuse Forge's folder scan; replaced by Venno's own in Phase 4) --
    @staticmethod
    def resolve_checkpoint(name: str) -> tuple[str, str]:
        from modules import sd_models

        if os.path.isfile(name):
            return name, os.path.splitext(os.path.basename(name))[0]
        info = sd_models.get_closet_checkpoint_match(name)
        if info is None:
            raise ValueError(f'Checkpoint "{name}" not found')
        return info.filename, info.model_name

    @staticmethod
    def resolve_module(name: str) -> str:
        from modules_forge.main_entry import module_list

        if os.path.isfile(name):
            return name
        path = module_list.get(os.path.basename(name))
        if path is None:
            raise ValueError(f'Text encoder / VAE "{name}" not found')
        return path

    # -- loading --
    def get(self, checkpoint: str, modules: list[str]):
        ckpt_path, ckpt_name = self.resolve_checkpoint(checkpoint)
        module_paths = sorted(self.resolve_module(m) for m in modules if m)
        key = (ckpt_path, tuple(module_paths))
        with self.lock:
            if self.engine is not None and self.key == key:
                return _clean(self.engine)
            self.unload()
            adopted = _forge_model_for(ckpt_path, module_paths)
            if adopted is not None:  # Forge already holds exactly this model: share it
                self.engine, self.key = adopted, key
                self.info = {"name": ckpt_name, "path": ckpt_path, "modules": [os.path.basename(m) for m in module_paths]}
                return _clean(self.engine)
            _release_forge_model()
            t0 = time.time()
            print(f"[Venno] loading {ckpt_name} + {[os.path.basename(m) for m in module_paths]}")
            self.engine = forge_loader(ckpt_path, additional_state_dicts=module_paths)
            self.key = key
            self.info = {"name": ckpt_name, "path": ckpt_path, "modules": [os.path.basename(m) for m in module_paths]}
            print(f"[Venno] model ready in {time.time() - t0:.1f}s")
            return self.engine

    def release_for_forge(self):
        """Before a Forge-path job: drop a Venno-only copy; keep one shared with Forge."""
        with self.lock:
            if self.engine is None:
                return
            from modules import sd_models

            if self.engine is not sd_models.model_data.sd_model:
                self.unload()

    def unload(self):
        with self.lock:
            if self.engine is None:
                return
            self.engine = None
            self.key = None
            self.info = {}
            memory_management.unload_all_models()
            memory_management.soft_empty_cache()
            gc.collect()


def _clean(engine):
    """Start from the unpatched objects: a Forge job may have left LoRA / script clones in forge_objects
    (Forge re-applies its own from forge_objects_after_applying_lora on its next job)."""
    engine.forge_objects = engine.forge_objects_original.shallow_copy()
    return engine


def _forge_model_for(ckpt_path: str, module_paths: list[str]):
    """The model Forge currently holds, if it was loaded from the same files."""
    try:
        from modules import sd_models

        data = sd_models.model_data
        model = data.sd_model
        if not data.forge_hash or getattr(model, "filename", None) is None:
            return None
        same = lambda a, b: sorted(map(os.path.realpath, a)) == sorted(map(os.path.realpath, b))
        if os.path.realpath(model.filename) == os.path.realpath(ckpt_path) and same(data.forge_loading_parameters.get("additional_modules") or [], module_paths):
            return model
    except Exception as e:
        print(f"[Venno] could not inspect the Forge model: {e}")
    return None


def _release_forge_model():
    """While both pipelines exist (Phases 1-3), never hold two copies of the model in memory."""
    try:
        from modules import sd_models

        if sd_models.model_data.sd_model is not None and sd_models.model_data.forge_hash:
            ref = weakref.ref(sd_models.model_data.sd_model)
            sd_models.unload_model_weights()
            if ref() is not None:
                holders = sorted({type(r).__name__ for r in gc.get_referrers(ref()) if not isinstance(r, types.FrameType)})
                print(f"[Venno] warning: the Forge model is still referenced after unloading ({', '.join(holders)})")
    except Exception as e:
        print(f"[Venno] could not release the Forge model: {e}")


manager = ModelManager()
