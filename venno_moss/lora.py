"""LoRAs on the MOSS-Image transformer, applied as extra terms in the Linear layers' forward (never merged), so
changing a LoRA's weight costs nothing and works on fp8 weights too.

Files from venno-trainer use ComfyUI's MOSS layout: ``diffusion_model.layers.N.proj.{q,k,v,o,ffn_up_gate,
ffn_down}.{lora_A,lora_B}.weight`` and ``.alpha``. The model fuses q/k/v into ``proj.qkv`` (rows q, k, v), so
q/k/v LoRAs add to their third of its output. Also read: ``transformer.`` / ``model.diffusion_model.``
prefixes, ``lora_down`` / ``lora_up`` names and kohya-style ``lora_unet_layers_0_proj_qkv`` keys. Math as
kohya / venno-trainer: out = W x + up(down(x)) * weight * alpha / rank.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from . import config

EXTENSIONS = (".safetensors",)
PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.", "base_model.model.", "lora_unet_")
SUFFIXES = {".lora_A.weight": "down", ".lora_down.weight": "down", ".lora_B.weight": "up", ".lora_up.weight": "up", ".alpha": "alpha"}
QKV = {"q": 0, "k": 1, "v": 2}
_cache: "OrderedDict[tuple, dict]" = OrderedDict()  # (path, mtime) → state dict on the CPU
_lock = threading.Lock()
CACHE_SIZE = 6


# ---------------------------------------------------------------- files


def scan() -> list[dict]:
    """Every LoRA in the LoRA folders: {name, path, folder, size, mtime}; name is the path inside its folder."""
    out, seen = [], set()
    for root in config.lora_folders():
        for dirpath, _, files in os.walk(root, followlinks=True):
            for n in sorted(files):
                if not n.lower().endswith(EXTENSIONS):
                    continue
                path = os.path.join(dirpath, n)
                name = os.path.relpath(path, root).replace(os.sep, "/")
                if name in seen or not is_lora_file(path):
                    continue
                seen.add(name)
                st = os.stat(path)
                out.append({"name": name, "path": path, "folder": root, "size": st.st_size, "mtime": st.st_mtime, **summary(path)})
    out.sort(key=lambda x: x["name"].lower())
    return out


def is_lora_file(path: str) -> bool:
    try:
        with safe_open(path, "pt") as f:
            return any(k.endswith(tuple(SUFFIXES)) and "lora" in k for k in f.keys())
    except Exception:
        return False


def resolve(name: str) -> str:
    if os.path.isfile(name):
        return name
    for item in scan():
        if item["name"] == name or os.path.splitext(item["name"])[0] == os.path.splitext(name)[0]:
            return item["path"]
    raise ValueError(f'LoRA "{name}" not found in the LoRA folders (Settings)')


TRAINED_ON = ("ss_sd_model_name", "venno_base_model", "base_model", "base_model_name", "modelspec.base_model", "ss_base_model")


def summary(path: str) -> dict:
    """What a LoRA file says about itself: the model it was trained on (when recorded), rank and alpha."""
    out = {"trained_on": "", "rank": None, "alpha": None}
    try:
        with safe_open(path, "pt") as f:
            meta = f.metadata() or {}
            down = next((k for k in f.keys() if k.endswith((".lora_A.weight", ".lora_down.weight"))), None)
            if down is not None:
                out["rank"] = f.get_slice(down).get_shape()[0]
    except Exception:
        return out
    name = next((meta[k].strip() for k in TRAINED_ON if meta.get(k, "").strip()), "")
    out["trained_on"] = os.path.splitext(os.path.basename(name.replace("\\", "/")))[0] if name else ""
    for key in ("ss_network_alpha", "alpha"):
        try:
            out["alpha"] = float(meta[key])
            break
        except (KeyError, ValueError):
            pass
    for key in ("ss_network_dim", "rank"):
        if out["rank"] is None and meta.get(key, "").isdigit():
            out["rank"] = int(meta[key])
    return out


def metadata(path: str) -> dict:
    try:
        with safe_open(path, "pt") as f:
            return f.metadata() or {}
    except Exception:
        return {}


def _state_dict(path: str) -> dict:
    key = (path, os.path.getmtime(path))
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    sd = load_file(path)
    with _lock:
        _cache[key] = sd
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return sd


# ---------------------------------------------------------------- keys → model layers


@dataclass
class Part:
    down: torch.Tensor  # (rank, in)
    up: torch.Tensor  # (out or out/3, rank)
    scale: float  # alpha / rank
    rows: Optional[tuple[int, int]] = None  # slice of the layer's output (q, k or v of a fused qkv)


def parse(sd: dict[str, torch.Tensor], modules: dict[str, nn.Linear]) -> tuple[dict[str, list[Part]], list[str]]:
    """{Linear name: [Part]} for a LoRA state dict, and the keys that match no layer."""
    flat = {name.replace(".", "_"): name for name in modules}  # kohya-style keys use "_" for "."
    groups: dict[str, dict] = {}
    unknown = []
    for key, value in sd.items():
        suffix = next((s for s in SUFFIXES if key.endswith(s)), None)
        if suffix is None:
            unknown.append(key)
            continue
        stem = key[: -len(suffix)]
        for p in PREFIXES:
            if stem.startswith(p):
                stem = stem[len(p) :]
                break
        groups.setdefault(stem, {})[SUFFIXES[suffix]] = value

    patches: dict[str, list[Part]] = {}
    for stem, g in groups.items():
        if "down" not in g or "up" not in g:
            unknown.append(stem)
            continue
        name, rows = stem if stem in modules else flat.get(stem.replace(".", "_")), None
        if name is None and stem[-2:-1] in "._" and stem[-1] in QKV:  # q / k / v of a fused projection
            part = stem[-1]
            fused = flat.get(f"{stem[:-2]}_qkv".replace(".", "_"))
            if fused is not None:
                name = fused
                third = modules[fused].out_features // 3
                rows = (QKV[part] * third, (QKV[part] + 1) * third)
        if name is None:
            unknown.append(stem)
            continue
        down, up = g["down"], g["up"]
        rank = down.shape[0]
        layer = modules[name]
        want_out = (rows[1] - rows[0]) if rows else layer.out_features
        if down.shape != (rank, layer.in_features) or up.shape != (want_out, rank):
            raise ValueError(f"LoRA layer {stem} doesn't fit this model ({tuple(down.shape)} / {tuple(up.shape)}); is it a MOSS-Image LoRA?")
        alpha = float(g["alpha"]) if "alpha" in g else float(rank)
        patches.setdefault(name, []).append(Part(down, up, alpha / rank, rows))
    return patches, unknown


def _merge_shared_down(parts: list[Part]) -> list[Part]:
    """q, k and v with one shared down projection (venno-trainer, ai-toolkit) → one part over the whole qkv."""
    sliced = [p for p in parts if p.rows]
    if len(sliced) == 3 and len({p.scale for p in sliced}) == 1 and all(torch.equal(sliced[0].down, p.down) for p in sliced[1:]):
        ordered = sorted(sliced, key=lambda p: p.rows[0])
        if ordered[0].rows[0] == 0 and all(a.rows[1] == b.rows[0] for a, b in zip(ordered, ordered[1:])):
            whole = Part(ordered[0].down, torch.cat([p.up for p in ordered]), ordered[0].scale)
            return [p for p in parts if not p.rows] + [whole]
    return parts


# ---------------------------------------------------------------- applying


class LoraStack:
    """The LoRAs on one transformer. set([(path, weight), …]) loads what changed and only re-weights the rest."""

    def __init__(self, model: nn.Module, device, dtype):
        self.device, self.dtype = device, dtype
        self.modules = {name: m for name, m in model.named_modules() if isinstance(m, nn.Linear)}
        self.loaded: list[tuple[str, float]] = []  # (path@mtime, ...) of the loaded files
        self.active: dict[str, list[tuple[Part, int]]] = {}  # Linear name → [(part on device, lora index)]
        self.weights: list[float] = []
        self.originals: dict[str, object] = {}

    def set(self, loras: list[tuple[str, float]]) -> list[str]:
        """Returns warnings (e.g. keys that matched nothing)."""
        loras = [(p, w) for p, w in loras if w != 0]
        files = [(p, os.path.getmtime(p)) for p, _ in loras]
        warnings: list[str] = []
        if files != self.loaded:
            self.clear()
            for index, (path, _) in enumerate(loras):
                patches, unknown = parse(_state_dict(path), self.modules)
                if not patches:
                    raise ValueError(f"{os.path.basename(path)}: none of its weights fit the MOSS-Image transformer (is it a LoRA for another model?)")
                if unknown:
                    warnings.append(f"{os.path.basename(path)}: {len(unknown)} weights matched no layer (e.g. {unknown[0]})")
                for name, parts in patches.items():
                    for part in _merge_shared_down(parts):
                        moved = Part(part.down.to(self.device, self.dtype), part.up.to(self.device, self.dtype), part.scale, part.rows)
                        self.active.setdefault(name, []).append((moved, index))
            for name in self.active:
                self._hook(name)
            self.loaded = files
        self.weights = [w for _, w in loras]
        return warnings

    def _hook(self, name: str):
        module = self.modules[name]
        original = module.forward
        self.originals[name] = original
        parts = self.active[name]

        def forward(x):
            out = original(x)
            for part, index in parts:
                weight = self.weights[index] if index < len(self.weights) else 0.0
                if weight == 0:
                    continue
                h = F.linear(F.linear(x.to(part.down.dtype), part.down), part.up) * (weight * part.scale)
                if part.rows:
                    out[..., part.rows[0] : part.rows[1]] += h.to(out.dtype)
                else:
                    out = out + h.to(out.dtype)
            return out

        module.forward = forward

    def clear(self):
        for name, original in self.originals.items():
            self.modules[name].forward = original
        self.originals.clear()
        self.active.clear()
        self.loaded = []
        self.weights = []
