"""Reference images on Cosmos's time axis.

Anima is Cosmos-Predict2 Text2Image: a video DiT whose 3D RoPE and per-frame timesteps it still has, trained
on single frames. A clean, VAE-encoded picture appended as an extra frame is attended to by every image
token through self-attention, so the generated frame takes the reference's character, outfit and style
(the closer the reference's composition, the closer the copy). The model's output for reference frames is
dropped (backend/nn/anima.py).

Each reference frame carries its own timestep:
- fixed (`noise`): the model is told the frame is that noisy, although it is clean. Near 0 the model copies
  the reference too literally and shows a dotted pattern; about 0.5 works best for plain Anima.
- `follow`: the frame's timestep follows the generated frames' at every step. This is what the Anima Edit
  LoRAs were trained with (references concatenated on the time axis, same timestep).

On its own a clean reference is a strong pull: the picture tends to copy it (pose and framing included) and
the prompt loses. `strength` < 1 damps the generated frames' self-attention to the reference frames (by up
to 1000× at 0), so the prompt keeps the composition while the reference still lends the character.

With the reference adapter (venno/engine/ref_adapter.py) the reference is letterboxed instead of cropped, fed
clean (timestep 0), placed `offset` picture heights down in the rotary positions so it is not read as the same
shot, and a LoRA trained for exactly that reads it; `guidance` sets how strongly it is followed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL import Image

from . import images, vae


@dataclass
class Reference:
    pictures: list[Image.Image]
    noise: float = 0.5  # fixed timestep of the reference frames, 0..1
    follow: bool = False  # timestep follows the steps instead (Anima Edit LoRAs)
    strength: float = 1.0  # 0..1: how much the generated frames attend to the references
    adapter: object = None  # ref_adapter.Adapter: the reference adapter's LoRA (one reference, letterboxed, t = 0)
    guidance: float = 1.0  # reference guidance with the adapter (1 = plain CFG with the reference in both branches)
    correction: str = "apg"  # how high reference guidance is kept from over-saturating: apg | rescale | none
    offset: float = 0.0  # reference frames' rotary positions shifted this many picture heights down
    _latents: dict = field(default_factory=dict, repr=False)
    _current: torch.Tensor = field(default=None, repr=False)  # (1, C, R, h, w) for the pass being sampled
    _timestep: float = field(default=0.0, repr=False)

    def prepare(self, engine, width: int, height: int) -> None:
        """Encode the pictures for a pass of this size (cropped to fill), once per size."""
        if (width, height) not in self._latents:
            fit = letterbox if self.adapter is not None else (lambda p, w, h: images.resize(p.convert("RGB"), w, h, 1))
            frames = [vae.encode(engine, fit(p, width, height)) for p in self.pictures]
            self._latents[(width, height)] = torch.cat(frames, dim=2)  # (1, C, R, h, w)
        self._current = self._latents[(width, height)]
        predictor = engine.forge_objects.unet.model.predictor
        self._timestep = float(predictor.timestep(torch.tensor([float(self.noise)])))

    def latents_like(self, x: torch.Tensor) -> torch.Tensor:
        """The reference frames for a model call on x (B, C, T, h, w): (B, C, R, h, w)."""
        ref = self._current
        if ref is None:
            raise RuntimeError("Reference.prepare was not called for this pass")
        if ref.shape[-2:] != x.shape[-2:]:
            raise ValueError(f"reference latents are {tuple(ref.shape[-2:])}, the pass is {tuple(x.shape[-2:])}")
        return ref.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1, -1, -1)

    def timesteps(self, t: torch.Tensor, count: int) -> torch.Tensor:
        """Timesteps of the reference frames, (B, count), given the generated frames' (B, 1)."""
        if self.follow:
            return t.expand(-1, count)
        return torch.full((t.shape[0], count), self._timestep, device=t.device, dtype=t.dtype)

    MAX_DAMPING = -6.9  # log(1/1000)

    def key_bias(self, thw: tuple[int, int, int], like: torch.Tensor):
        """Self-attention key bias per query frame, (t, N): generated frames' queries see the reference frames
        damped by `strength`; or None at full strength. The references are the last frames."""
        if self.strength >= 1.0 or self._current is None:
            return None
        t, h, w = thw
        refs = self._current.shape[2]
        bias = torch.zeros(t, t)
        bias[: t - refs, t - refs :] = self.MAX_DAMPING * (1.0 - max(0.0, self.strength))
        return bias.repeat_interleave(h * w, dim=1).to(device=like.device, dtype=like.dtype)

    def h_offsets(self, frames: int, height: int) -> Optional[list[int]]:
        """Rotary h offset per frame (generated frames first, then the references), in patches; None: none."""
        if not self.offset or self._current is None:
            return None
        return [0] * (frames - self._current.shape[2]) + [round(self.offset * height)] * self._current.shape[2]

    def info(self) -> dict:
        if self.adapter is not None:
            out = {"Reference images": len(self.pictures), "Reference adapter": self.adapter.name, "Reference guidance": self.guidance}
            if self.guidance != 1.0 and self.correction != "apg":
                out["Reference correction"] = self.correction
            return out
        out = {"Reference images": len(self.pictures), "Reference noise": "follow" if self.follow else self.noise}
        if self.strength < 1.0:
            out["Reference strength"] = self.strength
        return out


def letterbox(picture: Image.Image, width: int, height: int, fill: int = 255) -> Image.Image:
    """The whole picture fitted inside width × height, centred on a plain background (white, as the adapter was trained)."""
    picture = picture.convert("RGB")
    scale = min(width / picture.width, height / picture.height)
    size = (max(1, round(picture.width * scale)), max(1, round(picture.height * scale)))
    out = Image.new("RGB", (width, height), (fill, fill, fill))
    out.paste(picture.resize(size, Image.LANCZOS), ((width - size[0]) // 2, (height - size[1]) // 2))
    return out
