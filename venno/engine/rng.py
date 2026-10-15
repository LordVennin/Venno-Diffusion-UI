"""Seeded noise.

Every image in a batch has its own CPU generator, so results don't depend on batch size,
and the initial noise for a seed matches Forge's default "CPU" RNG.
"""

from __future__ import annotations

import random

import torch


def resolve_seed(seed: int | None) -> int:
    return random.randrange(4294967294) if seed is None or seed < 0 else int(seed)


class ImageNoise:
    def __init__(self, seeds: list[int], shape: tuple[int, ...]):
        self.shape = tuple(shape)  # per-image latent shape, e.g. (C, 1, H, W)
        self.generators = [torch.Generator("cpu").manual_seed(int(s)) for s in seeds]

    def first(self) -> torch.Tensor:
        return torch.stack([torch.randn(self.shape, generator=g) for g in self.generators])

    def like(self, x: torch.Tensor) -> torch.Tensor:
        """Fresh noise for ancestral / SDE sampler steps, continuing each image's generator."""
        per_image = x.shape[1:]
        return torch.stack([torch.randn(per_image, generator=g) for g in self.generators[: x.shape[0]]]).to(device=x.device, dtype=x.dtype)


class TorchProxy:
    """Stands in for `torch` inside k_diffusion.sampling so `torch.randn_like` uses the job's seeded generators."""

    def __init__(self, noise: ImageNoise):
        self._noise = noise

    def __getattr__(self, item):
        return getattr(torch, item)

    def randn_like(self, x, *args, **kwargs):
        return self._noise.like(x)
