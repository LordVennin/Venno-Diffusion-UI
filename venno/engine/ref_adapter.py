"""The Anima reference adapter: a LoRA that teaches Anima to take a character from one reference picture.

Method and weights by ElectronGirl ("Anima Reference Adapter: Single-Image Character and Style Transfer with an
Offset-RoPE Reference Frame", github.com/ElectronGirl/ComfyUI-Anima-Ref-Adapter, Apache-2.0; weights
huggingface.co/ElectronGirl/anima-ref-adapter under the CircleStone Non-Commercial License, like Anima). This is
Venno's own implementation of the published method:

- the reference is letterboxed to the picture's size (white padding) and added as an extra frame on Anima's
  time axis, clean (timestep 0), like Venno's reference pictures (venno/engine/reference.py);
- its rotary positions are shifted one picture height down (`OFFSET`), so the model can't read it as the
  previous frame of the same shot and copy its pose: it takes who the character is, the prompt sets the scene;
- this LoRA (rank 32 on every block's attention and MLP layers of the 28-block Anima) reads that frame. It is
  applied through forward hooks only during model calls that carry the reference, so the same model also gives
  the reference-free prediction the reference guidance needs (venno/engine/sampling.py).

Trained on Anima base v1.0 with one reference: other 28-block Anima checkpoints load it, the 2.9B / 3.8B ones
don't fit, and only the first reference picture is used.
"""

from __future__ import annotations

import contextlib
import os
import threading
import weakref
from typing import Optional

import torch
import torch.nn.functional as F

PREFIX = "anima_ref_adapter"  # weight files are found by this name (anima_ref_adapter_pilot_r32.safetensors)
OFFSET = 1.0  # reference frame's RoPE shift in picture heights (training value)
BLOCKS = 28
_lock = threading.Lock()
_loaded: dict[tuple, "Adapter"] = {}


def find() -> Optional[str]:
    """The adapter weights: models/reference_adapter first, then the LoRA folders (where ComfyUI keeps them)."""
    from venno import config

    for kind in ("reference_adapter", "lora"):
        for root in config.model_folders(kind):
            for dirpath, _, files in os.walk(root, followlinks=True):
                for f in sorted(files):
                    if f.lower().startswith(PREFIX) and f.lower().endswith(".safetensors"):
                        return os.path.join(dirpath, f)
    return None


def get() -> "Adapter":
    path = find()
    if path is None:
        raise FileNotFoundError("The reference adapter isn't installed: get it in ⚙ Settings → Models (Anima reference adapter)")
    key = (os.path.abspath(path), os.path.getmtime(path))
    with _lock:
        if key not in _loaded:
            _loaded.clear()
            _loaded[key] = Adapter(path)
        return _loaded[key]


class Adapter:
    def __init__(self, path: str):
        from safetensors.torch import load_file

        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        tensors = load_file(path, device="cpu")
        self.weights: dict[str, list] = {}  # module name in the diffusion model → [down, up, scale]
        for key, down in tensors.items():
            if not key.endswith(".lora_down.weight"):
                continue
            name = key.removeprefix("diffusion_model.").removesuffix(".lora_down.weight")
            up = tensors[key.replace("lora_down", "lora_up")]
            alpha = float(tensors.get(key.replace("lora_down.weight", "alpha"), torch.tensor(down.shape[0])))
            self.weights[name] = [down, up, alpha / down.shape[0]]
        if not self.weights:
            raise ValueError(f"{os.path.basename(path)} has no LoRA weights")
        self.on = False
        self._hooked: "weakref.WeakKeyDictionary[torch.nn.Module, list]" = weakref.WeakKeyDictionary()

    def attach(self, dit: torch.nn.Module) -> None:
        """Hook the adapter's layers of a diffusion model (once per model); they act only inside `active()`."""
        if dit in self._hooked:
            return
        blocks = len(getattr(dit, "blocks", []))
        if blocks != BLOCKS:
            raise ValueError(f"The reference adapter is made for the 28-block Anima (base, aesthetic, turbo); this checkpoint has {blocks} blocks")
        handles = []
        for name, entry in self.weights.items():
            handles.append(dit.get_submodule(name).register_forward_hook(self._hook(entry)))
        self._hooked[dit] = handles

    def _hook(self, entry):
        def hook(module, inputs, output):
            if not self.on:
                return output
            x = inputs[0]
            down, up, scale = entry
            if down.device != x.device or down.dtype != x.dtype:
                entry[0], entry[1] = down, up = down.to(device=x.device, dtype=x.dtype), up.to(device=x.device, dtype=x.dtype)
            return output + (F.linear(F.linear(x, down), up) * scale).to(output.dtype)

        return hook

    @contextlib.contextmanager
    def active(self):
        """The LoRA applies to the model calls made inside (those that carry the reference)."""
        previous, self.on = self.on, True
        try:
            yield
        finally:
            self.on = previous


def guidance(pos: torch.Tensor, neg: torch.Tensor, base: torch.Tensor, cfg: float, ref_scale: float, correction: str = "apg", phi: float = 0.5) -> torch.Tensor:
    """Separate text and reference guidance (in the spirit of InstructPix2Pix), on denoised predictions:
        base + ref_scale · (neg − base) + cfg · (pos − neg)
    pos / neg: prompt / negative with the reference; base: negative without it (nor the adapter). ref_scale 1 is
    plain CFG with the reference in both branches. Higher reference guidance over-saturates; `apg` drops the part
    of the update parallel to pos (adaptive projected guidance), `rescale` matches pos's spread (blend `phi`)."""
    guided = base + ref_scale * (neg - base) + cfg * (pos - neg)
    dims = tuple(range(1, pos.ndim))
    if correction == "apg":
        update = (guided - pos).double()
        unit = F.normalize(pos.double().flatten(1), dim=1).view_as(pos)
        parallel = (update * unit).sum(dim=dims, keepdim=True) * unit
        guided = pos + (update - parallel).to(pos.dtype)
    elif correction == "rescale":
        ratio = pos.std(dim=dims, keepdim=True) / guided.std(dim=dims, keepdim=True).clamp_min(1e-8)
        guided = phi * guided * ratio + (1 - phi) * guided
    return guided
