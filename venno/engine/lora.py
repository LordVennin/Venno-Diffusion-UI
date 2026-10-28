"""LoRA files: discovery, loading and patching the engine's diffusion model / text encoder.

Patches are non-destructive (cloned patchers); memory_management merges them into the weights when the
patched model is loaded and restores the originals when another set is loaded.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

import torch

from backend.nn.anima import BLOCK_MAPPINGS
from backend.patcher.lora import load_lora, model_lora_keys_clip, model_lora_keys_unet
from backend.state_dict import state_dict_prefix_replace
from backend.utils import load_torch_file

EXTENSIONS = (".safetensors", ".pt", ".ckpt")
_lock = threading.Lock()
_files: dict[str, str] = {}  # name → path
_cache: "OrderedDict[tuple, dict]" = OrderedDict()  # (path, mtime) → state dict
CACHE_SIZE = 4


# ---------------------------------------------------------------- discovery


def folders() -> list[str]:
    from venno import config

    return config.model_folders("lora")


def scan() -> dict[str, str]:
    found: dict[str, str] = {}
    for root in folders():
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            for f in sorted(filenames):
                if f.lower().endswith(EXTENSIONS) and not f.lower().startswith("anima_ref_adapter"):  # used by the Reference card, not as a LoRA
                    found.setdefault(os.path.splitext(f)[0], os.path.join(dirpath, f))
    with _lock:
        _files.clear()
        _files.update(found)
    return found


def available() -> list[dict]:
    files = scan()
    return [{"name": n, "folder": os.path.basename(os.path.dirname(p))} for n, p in sorted(files.items(), key=lambda x: x[0].lower())]


def resolve(name: str) -> str:
    path = _files.get(name)
    if path is None or not os.path.isfile(path):
        path = scan().get(name)
    if path is None:
        raise ValueError(f'LoRA "{name}" not found')
    return path


# ---------------------------------------------------------------- loading


def _state_dict(path: str) -> dict[str, torch.Tensor]:
    key = (path, os.path.getmtime(path))
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    sd = load_torch_file(path, safe_load=True)
    if any(k.startswith("lora_unet__") for k in sd):
        sd = state_dict_prefix_replace(sd, {"lora_unet__": "lora_unet_"})
    with _lock:
        _cache[key] = sd
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return sd


def _remap_blocks(lora: dict[str, torch.Tensor], blocks: int) -> dict[str, torch.Tensor] | None:
    """Anima-specific key fixes. LoRAs trained on one Anima size can be mapped onto another (e.g. 2B → 2.9B)."""
    lora = dict(lora)  # the cached original stays untouched
    for k in list(lora):  # the LLM adapter lives in the text encoder here
        if k.startswith("diffusion_model.llm_adapter"):
            lora[k.replace("diffusion_model", "text_encoders.qwen3_06b")] = lora.pop(k)
        elif k.startswith("lora_unet_llm_adapter"):
            lora[k.replace("lora_unet_llm_adapter", "lora_te_llm_adapter")] = lora.pop(k)

    keys = list(lora)
    prefix, sep = ("lora_unet_blocks_", "_") if any(k.startswith("lora_unet_blocks_") for k in keys) else ("diffusion_model.blocks.", ".")
    parsed: dict[str, tuple[int, str]] = {}
    for k in keys:
        if k.startswith(prefix):
            num, s, tail = k[len(prefix) :].partition(sep)
            if s and num.isdigit():
                parsed[k] = (int(num), tail)

    used = (max(n for n, _ in parsed.values()) + 1) if parsed else 0
    lora_blocks = next((size for size in (28, 40, 52) if used <= size), used)
    if lora_blocks == blocks:
        return lora
    if lora_blocks > blocks or (mapping := BLOCK_MAPPINGS.get((lora_blocks, blocks))) is None:
        return None

    reverse: dict[int, list[int]] = {}
    for target, source in enumerate(mapping):
        reverse.setdefault(source, []).append(target)
    original = {k: lora[k] for k in parsed}
    for k, (source, tail) in parsed.items():
        for target in reverse.get(source, []):
            lora[f"{prefix}{target}{sep}{tail}"] = original[k]
    return lora


def apply(engine, loras: list[tuple[str, float]]):
    """Patch engine.forge_objects (already reset to the originals) with the given (name, weight) list.

    Returns warnings for the job (e.g. a LoRA made for a different model)."""
    loras = [(name, float(weight)) for name, weight in loras if weight != 0]
    paths = [resolve(name) for name, _ in loras]
    key = tuple((p, os.path.getmtime(p), w) for p, (_, w) in zip(paths, loras))
    cached = getattr(engine, "_venno_lora", None)
    if cached and cached[0] == key:  # same set as last job: reuse the patched objects (no re-merge)
        _, engine.forge_objects.unet, engine.forge_objects.clip, warnings = cached
        return list(warnings)

    warnings: list[str] = []
    unet, clip = engine.forge_objects.unet, engine.forge_objects.clip
    blocks = len(unet.model.diffusion_model.blocks)
    unet_keys = model_lora_keys_unet(unet.model)
    clip_keys = model_lora_keys_clip(clip.cond_stage_model)

    for (name, weight), path in zip(loras, paths):
        lora = _remap_blocks(_state_dict(path), blocks)
        if lora is None:
            warnings.append(f"{name}: made for a different Anima size, skipped")
            continue
        patches_unet, rest = load_lora(lora, unet_keys)
        patches_clip, rest = load_lora(rest, clip_keys)
        if not patches_unet and not patches_clip or len(rest) / max(len(lora), 1) > 0.5:
            warnings.append(f"{name}: not an Anima LoRA, skipped")
            continue
        if patches_unet:
            unet = unet.clone()
            unet.add_patches(filename=path, patches=patches_unet, strength_patch=weight)
        if patches_clip:
            clip = clip.clone()
            clip.add_patches(filename=path, patches=patches_clip, strength_patch=weight)
        print(f"[Venno] LoRA {name} @ {weight:g}: {len(patches_unet)} diffusion + {len(patches_clip)} text encoder keys")

    engine.forge_objects.unet, engine.forge_objects.clip = unet, clip
    engine._venno_lora = (key, unet, clip, tuple(warnings))
    return warnings
