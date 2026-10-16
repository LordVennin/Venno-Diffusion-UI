"""Latent ↔ image conversion and cheap live previews."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def latent_shape(engine, width: int, height: int) -> tuple[int, ...]:
    """Per-image latent shape. Anima uses the (Wan) Qwen-Image VAE: (C, 1, H/8, W/8)."""
    vae = engine.forge_objects.vae
    f = vae.upscale_ratio if isinstance(vae.upscale_ratio, int) else 8
    c = vae.latent_channels
    return (c, 1, height // f, width // f) if engine.is_wan else (c, height // f, width // f)


def to_pil(x: torch.Tensor) -> list[Image.Image]:
    """[-1, 1] image tensor (B, C, H, W) or (B, C, T, H, W) → PIL images."""
    x = x.float()
    if x.ndim == 5:
        x = x.reshape(-1, *x.shape[-3:])
    x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    arr = (x.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


@torch.inference_mode()
def decode(engine, latents: torch.Tensor) -> list[Image.Image]:
    images = []
    for i in range(latents.shape[0]):  # one at a time keeps VRAM flat
        out = engine.decode_first_stage(latents[i : i + 1])
        if torch.isnan(out).any():
            raise RuntimeError("The VAE produced NaNs (try a different VAE precision)")
        images += to_pil(out)
    return images


@torch.inference_mode()
def encode(engine, image: Image.Image) -> torch.Tensor:
    """PIL image → one latent, (1, C, 1, H/8, W/8) for Anima's Wan VAE."""
    vae = engine.forge_objects.vae
    x = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0)[None]  # (1, H, W, 3) in [0, 1]
    latent = vae.first_stage_model.process_in(vae.encode(x))
    if torch.isnan(latent).any():
        raise RuntimeError("The VAE produced NaNs while encoding the input image")
    return latent.float().cpu()


@torch.inference_mode()
def preview(engine, latent: torch.Tensor) -> Image.Image | None:
    """Fast latent → RGB approximation of the first image (no VAE)."""
    fmt = engine.model_config.latent_format
    factors = getattr(fmt, "latent_rgb_factors", None)
    if factors is None:
        return None
    lat = latent[0].float().cpu()
    if lat.ndim == 4:  # (C, T, H, W) → first frame
        lat = lat[:, 0]
    w = torch.tensor(factors, dtype=torch.float32)  # (C, 3)
    bias = getattr(fmt, "latent_rgb_factors_bias", None)
    rgb = torch.einsum("chw,cr->rhw", lat, w)
    if bias is not None:
        rgb = rgb + torch.tensor(bias, dtype=torch.float32)[:, None, None]
    return to_pil(rgb.unsqueeze(0))[0]
