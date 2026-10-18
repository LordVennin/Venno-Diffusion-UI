"""Sketch guides: Anima ControlLLLite models (kohya) driven by the Venno denoiser.

Differences from Forge's port:
- the active step range comes from the denoiser's real step (Forge counted model calls, which drifts with
  samplers that evaluate twice per step and with image to image)
- the guide image is fitted to every pass's exact size, so guides also steer the hires pass
- patches are always removed after sampling (the model is shared with the Forge path)

Control modes (as in Forge's ControlNet):
- balanced: the guide acts on every block, for the prompt and the negative pass alike.
- prompt ("My prompt is more important"): the guide fades over the blocks (later blocks ~0.1×, like
  Forge's soft injection), so it sets the layout and the prompt decides the details.
- guide ("ControlNet is more important"): the guide acts on the prompt rows only, so CFG amplifies it.
  Needs CFG > 1 (with CFG 1 there is no negative pass and it equals balanced).
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image

EXTENSIONS = (".safetensors", ".pth", ".pt", ".bin")
_lock = threading.Lock()
_nets: dict = {}  # (path, id(diffusion model)) → ControlNetLLLiteDiT


def folders() -> list[str]:
    from venno import config

    return config.model_folders("controlnet")


def models() -> dict[str, str]:
    found: dict[str, str] = {}
    for root in folders():
        for dirpath, _, files in os.walk(root):
            for f in sorted(files):
                if f.lower().endswith(EXTENSIONS):
                    found.setdefault(os.path.splitext(f)[0], os.path.join(dirpath, f))
    return found


def _net(engine, path: str):
    from backend.utils import load_torch_file

    from .lllite_anima import ControlNetLLLiteDiT, infer_anima_config, load_lllite_weights_from_dict, map_blocks

    unet = engine.forge_objects.unet
    dit = unet.model.diffusion_model
    key = (path, id(dit))
    with _lock:
        if key not in _nets:
            for k in [k for k in _nets if k[1] != id(dit)]:  # a different model was loaded: drop stale nets
                del _nets[k]
            sd = load_torch_file(path, safe_load=True)
            if not any(k.startswith("lllite_dit") for k in sd):
                raise ValueError(f"{os.path.basename(path)} is not an Anima ControlLLLite model")
            net = ControlNetLLLiteDiT(map_blocks(dit, sd), **infer_anima_config(sd))
            load_lllite_weights_from_dict(net, sd)
            _nets[key] = net.eval()
        net = _nets[key]
    return net.to(device=unet.load_device, dtype=unet.model.computation_dtype)


@dataclass
class Guide:
    path: str
    image: Image.Image  # RGB, polarity as the model expects (Studio inverts scribble-type guides)
    weight: float = 0.8
    start: float = 0.0
    end: float = 0.8
    mode: str = "balanced"  # balanced | prompt | guide


SOFT_DECAY = 0.825  # Forge's soft injection: 0.825 ** 12 ≈ 0.1 for the last blocks


def layer_scales(count: int, mode: str) -> list[float]:
    if mode != "prompt" or count < 2:
        return [1.0] * count
    return [SOFT_DECAY ** (12 * i / (count - 1)) for i in range(count)]


class GuideSet:
    """All guides of a job. `prepare` per pass, then `attach`/`detach` around sampling; `at_step` per model call."""

    def __init__(self, engine, guides: list[Guide]):
        self.engine = engine
        self.guides = guides
        self.nets = [_net(engine, g.path) for g in guides]
        self.scales = [layer_scales(len(net.lllite_modules), g.mode) for g, net in zip(guides, self.nets)]
        self.ranges: list[tuple[int, int]] = []
        self._last_step = None

    def prepare(self, width: int, height: int, steps_run: int, mask: Optional[Image.Image] = None):
        unet = self.engine.forge_objects.unet
        device, dtype = unet.load_device, unet.model.computation_dtype
        self.ranges = []
        for g, net in zip(self.guides, self.nets):
            img = g.image.convert("RGB").resize((width, height), Image.LANCZOS)
            x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
            if net.conditioning1.conv1.weight.shape[1] == 4:  # inpaint LLLite: masked image + mask channel
                m = mask.resize((width, height), Image.BILINEAR) if mask is not None else Image.new("L", (width, height), 0)
                mt = torch.from_numpy((np.asarray(m, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32))[None, None]
                x = torch.cat([x * (mt < 0.5), mt * 2.0 - 1.0], dim=1)
            with torch.inference_mode():
                net.set_cond_image(x.to(device=device, dtype=dtype))
            for m in net.lllite_modules:
                m.num_steps = 0  # Venno sets the multiplier per step instead of the module's call counter
                m.cond_only = g.mode == "guide"
            start = math.floor(steps_run * g.start) if g.start > 0 else 0
            end = math.floor(steps_run * g.end) if g.end > 0 else steps_run
            self.ranges.append((start, max(end, start + 1)))
        self._last_step = None

    def at_step(self, step: int):
        if step == self._last_step:
            return
        self._last_step = step
        for g, net, scales, (start, end) in zip(self.guides, self.nets, self.scales, self.ranges):
            weight = g.weight if start <= step - 1 < end else 0.0
            net.multiplier = weight
            for m, scale in zip(net.lllite_modules, scales):
                m.multiplier = weight * scale

    def attach(self):
        for net in self.nets:
            net.apply_to()

    def detach(self):
        for net in reversed(self.nets):
            for m in net.lllite_modules:
                m.cond_only = False  # the nets are cached and shared between jobs
            net.restore()  # also clears the conditioning image
