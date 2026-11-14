"""Finding MOSS-Image models on disk.

A model is a diffusers-layout folder (``transformer/``, usually with ``text_encoder/``, ``tokenizer/`` and
``vae/``; what the Hub, venno-trainer and our downloads produce), a single ``.safetensors`` transformer
(a fine-tune) or a single ``.gguf`` transformer (a quantized one: the small downloads). MOSS-Image Base and
SFT share the text encoder and the VAE (identical files), so those come from whichever folder has them and
only the transformer differs between models.

The text encoder is the Qwen3.5 9B of a model folder (bf16 safetensors, 17 GB) or a GGUF of Qwen3.5 9B in a
``text_encoder/`` folder next to its ``config.json`` (5 to 10 GB). With several, Settings picks; by default
the full one is used.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from safetensors import safe_open

from . import config, gguf

KNOWN = {"base": "MOSS-Image-Base", "sft": "MOSS-Image-SFT", "kromoss": "Kromoss"}  # folder names of the downloads; Base first: LoRAs train on it


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
class TextEncoder:
    name: str  # "Qwen3.5 9B (full)" or the GGUF's file name without .gguf
    path: str  # a folder of safetensors weights, or a .gguf file with config.json beside it

    @property
    def quantized(self) -> bool:
        return self.path.lower().endswith(".gguf")

    @property
    def size(self) -> int:
        if os.path.isfile(self.path):
            return os.path.getsize(self.path)
        return sum(os.path.getsize(os.path.join(self.path, n)) for n in os.listdir(self.path) if n.endswith(".safetensors"))


FULL_TEXT_ENCODER = "Qwen3.5 9B (full)"


@dataclass
class Found:
    models: list[Model] = field(default_factory=list)
    text_encoders: list[TextEncoder] = field(default_factory=list)
    text_encoder: Optional[str] = None  # the one in use: Settings' choice, else the full one, else the largest GGUF
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

    @property
    def text_encoder_name(self) -> str:
        return next((t.name for t in self.text_encoders if t.path == self.text_encoder), "")


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
        print(f"[Venno MOSS] skipping {os.path.basename(path)}: a ComfyUI int8 / fp8 file (use the bf16 one, or a GGUF: Settings → Models has Q8 and Q4)", flush=True)
        return False
    return True


_gguf_kinds: dict[tuple, Optional[str]] = {}  # (path, mtime, size) → "transformer" | "text_encoder" | None


def gguf_kind(path: str) -> Optional[str]:
    """What a .gguf file holds: a MOSS-Image transformer, a Qwen3.5 text encoder, or nothing Venno MOSS reads."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_mtime, st.st_size)
    if key not in _gguf_kinds:
        kind = None
        try:
            f = gguf.File(path)
            unread = sorted({t.kind for t in f.tensors.values() if t.kind not in gguf.PLAIN and t.kind not in gguf.BLOCKS})
            if "video_embedder.weight" in f.tensors:
                kind = "transformer"
            elif f.architecture == "qwen35" and "token_embd.weight" in f.tensors:
                kind = "text_encoder"
            if kind and unread:
                print(f"[Venno MOSS] skipping {os.path.basename(path)}: stored as {', '.join(unread)}, which isn't read (use a Q4_K, Q5_K, Q6_K or Q8_0 file)", flush=True)
                kind = None
        except Exception as e:  # not a GGUF file, or cut short
            print(f"[Venno MOSS] skipping {os.path.basename(path)}: {e}", flush=True)
        _gguf_kinds[key] = kind
    return _gguf_kinds[key]


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
            if found.vae is None and _has_weights(os.path.join(folder, "vae")):
                found.vae = os.path.join(folder, "vae")
            te = os.path.join(folder, "text_encoder")
            if _has_weights(te) and not any(not t.quantized for t in found.text_encoders):  # the full ones are all the same files
                found.text_encoders.append(TextEncoder(FULL_TEXT_ENCODER, te))
            if os.path.isfile(os.path.join(te, "config.json")):  # a GGUF needs the model's config beside it
                for n in sorted(os.listdir(te)):
                    path = os.path.join(te, n)
                    name = os.path.splitext(n)[0]
                    if n.lower().endswith(".gguf") and name not in {t.name for t in found.text_encoders} and gguf_kind(path) == "text_encoder":
                        found.text_encoders.append(TextEncoder(name, path))
            if found.tokenizer is None and os.path.isfile(os.path.join(folder, "tokenizer", "tokenizer.json")):
                found.tokenizer = os.path.join(folder, "tokenizer")
            if folder != top and _has_weights(os.path.join(folder, "transformer")):
                continue  # a diffusers folder: its files are its parts, not fine-tunes
            for n in sorted(os.listdir(folder)):
                path = os.path.join(folder, n)
                if n.endswith(".safetensors") and os.path.isfile(path) and is_transformer_file(path):
                    add(Model(os.path.splitext(n)[0], path))
                elif n.lower().endswith(".gguf") and os.path.isfile(path) and gguf_kind(path) == "transformer":
                    add(Model(os.path.splitext(n)[0], path))
    order = {v: i for i, v in enumerate(KNOWN.values())}  # Base, SFT, Kromoss first; then other models by name
    found.models.sort(key=lambda m: (order.get(m.name, len(order)), m.name.lower()))
    found.text_encoders.sort(key=lambda t: (t.quantized, -t.size))  # the full one first, then GGUFs from large to small
    wanted = config.load_settings().get("text_encoder", "")
    chosen = next((t for t in found.text_encoders if t.name == wanted), found.text_encoders[0] if found.text_encoders else None)
    found.text_encoder = chosen.path if chosen else None
    return found
