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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from PIL import Image

from . import images, vae


@dataclass
class Reference:
    pictures: list[Image.Image]
    noise: float = 0.5  # fixed timestep of the reference frames, 0..1
    follow: bool = False  # timestep follows the steps instead (Anima Edit LoRAs)
    _latents: dict = field(default_factory=dict, repr=False)
    _current: torch.Tensor = field(default=None, repr=False)  # (1, C, R, h, w) for the pass being sampled
    _timestep: float = field(default=0.0, repr=False)

    def prepare(self, engine, width: int, height: int) -> None:
        """Encode the pictures for a pass of this size (cropped to fill), once per size."""
        if (width, height) not in self._latents:
            frames = [vae.encode(engine, images.resize(p.convert("RGB"), width, height, 1)) for p in self.pictures]
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

    def info(self) -> dict:
        return {"Reference images": len(self.pictures), "Reference noise": "follow" if self.follow else self.noise}
