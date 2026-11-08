"""Finding MOSS-Image models on disk.

A model is a diffusers-layout folder (``transformer/``, usually with ``text_encoder/``, ``tokenizer/`` and
``vae/``; what the Hub, venno-trainer and our downloads produce) or a single ``.safetensors`` transformer
(a fine-tune). MOSS-Image Base and SFT share the text encoder and the VAE (identical files), so those come
from whichever folder has them and only the transformer differs between models.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from safetensors import safe_open

from . import config

KNOWN = {"base": "MOSS-Image-Base", "sft": "MOSS-Image-SFT"}  # Hub names (OpenMOSS-Team/…); Base first: LoRAs train on it


@dataclass
class Model:
    name: str
    transformer: str  # folder of (sharded) weights, or one .safetensors file
    root: Optional[str] = None  # the diffusers folder it came from

    @property
    def size(self) -> int:
        if os.path.isfile(self.transformer):
            return os.path.getsize(self.transformer)
        return sum(os.path.getsize(os.path.join(self.transformer, n)) for n in os.listdir(self.transformer) if n.endswith(".safetensors"))


@dataclass
class Found:
    models: list[Model] = field(default_factory=list)
    text_encoder: Optional[str] = None
    tokenizer: Optional[str] = None
    vae: Optional[str] = None

    def model(self, name: str) -> Model:
        for m in self.models:
            if m.name == name:
                return m
        raise ValueError(f'MOSS-Image model "{name}" not found (Settings → Models)')

    @property
    def complete(self) -> bool:
        return bool(self.models and self.text_encoder and self.tokenizer and self.vae)


def _has_weights(folder: str) -> bool:
    """Weights are there; for sharded weights, every shard the index names (not while a download is under way)."""
    if not os.path.isdir(folder):
        return False
    names = os.listdir(folder)
    index = [n for n in names if n.endswith(".safetensors.index.json")]
    if index:
        try:
            with open(os.path.join(folder, index[0]), encoding="utf-8") as f:
                shards = set(json.load(f)["weight_map"].values())
        except (OSError, ValueError, KeyError):
            return False
        return all(os.path.isfile(os.path.join(folder, s)) for s in shards)
    return any(n.endswith(".safetensors") for n in names)


def _display_name(folder: str) -> str:
    name = os.path.basename(os.path.normpath(folder))
    return name.split("__", 1)[-1]  # "OpenMOSS-Team__MOSS-Image-SFT" (venno-trainer's downloads) → "MOSS-Image-SFT"


QUANTIZED = ("weight_scale", "comfy_quant", "input_scale")  # ComfyUI's int8 / fp8 formats (e.g. *_int8_convrot)


def is_transformer_file(path: str) -> bool:
    """A single-file MOSS-Image transformer in bf16 / fp16 / fp32 (not a LoRA, not another model, and not
    ComfyUI's int8 or fp8 files: those need ComfyUI's quantized kernels; use the bf16 file and Memory: Low)."""
    try:
        with safe_open(path, "pt") as f:
            keys = list(f.keys())
    except Exception:  # not a safetensors file we can read
        return False
    if not any(k.endswith("video_embedder.weight") for k in keys) or any("lora" in k for k in keys):
        return False
    if any(k.endswith(QUANTIZED) for k in keys):
        print(f"[Venno MOSS] skipping {os.path.basename(path)}: a ComfyUI int8 / fp8 file (use the bf16 one; Memory: Low stores it in fp8)", flush=True)
        return False
    return True


def scan() -> Found:
    found = Found()
    names: set[str] = set()

    def add(model: Model):
        base, k = model.name, 2
        while model.name in names:  # the same model in two folders: keep both, told apart
            model.name = f"{base} ({k})"
            k += 1
        names.add(model.name)
        found.models.append(model)

    for top in config.model_folders():
        candidates = [top] + sorted(os.path.join(top, d) for d in os.listdir(top) if os.path.isdir(os.path.join(top, d)))
        for folder in candidates:
            if _has_weights(os.path.join(folder, "transformer")):
                add(Model(_display_name(folder), os.path.join(folder, "transformer"), folder))
            for part in ("text_encoder", "vae"):
                if getattr(found, part) is None and _has_weights(os.path.join(folder, part)):
                    setattr(found, part, os.path.join(folder, part))
            if found.tokenizer is None and os.path.isfile(os.path.join(folder, "tokenizer", "tokenizer.json")):
                found.tokenizer = os.path.join(folder, "tokenizer")
            if folder != top and _has_weights(os.path.join(folder, "transformer")):
                continue  # a diffusers folder: its files are its parts, not fine-tunes
            for n in sorted(os.listdir(folder)):
                path = os.path.join(folder, n)
                if n.endswith(".safetensors") and os.path.isfile(path) and is_transformer_file(path):
                    add(Model(os.path.splitext(n)[0], path))
    order = {v: i for i, v in enumerate(KNOWN.values())}  # Base, SFT first; then fine-tunes by name
    found.models.sort(key=lambda m: (order.get(m.name, len(order)), m.name.lower()))
    return found
