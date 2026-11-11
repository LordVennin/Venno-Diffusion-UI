"""Upscaling for hires fix, as in Venno (venno/engine/upscale.py): Spandrel models (ESRGAN, DAT, SwinIR, HAT, …)
run in feathered tiles, plus plain resampling. The models are the ones in Venno's upscaler folders
(models/ESRGAN, …), so an upscaler downloaded in Venno shows up here too.

Upscaler names:
    "Latent…"       latent resize, done by the engine (see `is_latent`)
    "Lanczos"       plain resampling
    "Nearest"
    <file stem>     a model in an upscaler folder
"""

from __future__ import annotations

import os
import threading

import numpy as np
import torch
from PIL import Image

from . import config

EXTENSIONS = (".safetensors", ".pth", ".pt", ".ckpt", ".bin")
LATENT = ("Latent", "Latent (bicubic)", "Latent (nearest-exact)")
RESAMPLE = {"Lanczos": Image.LANCZOS, "Nearest": Image.NEAREST}

_lock = threading.Lock()
_loaded: dict = {}  # path → spandrel ImageModelDescriptor (one at a time)


def is_latent(name: str) -> bool:
    return (name or "Latent").startswith("Latent")


def latent_mode(name: str) -> str:
    return {"Latent (bicubic)": "bicubic", "Latent (nearest-exact)": "nearest-exact"}.get(name, "bilinear")


def models() -> dict[str, str]:
    found: dict[str, str] = {}
    for root in config.upscaler_folders():
        for f in sorted(os.listdir(root)):
            if f.lower().endswith(EXTENSIONS):
                found.setdefault(os.path.splitext(f)[0], os.path.join(root, f))
    return found


def available() -> list[str]:
    return [*RESAMPLE, *LATENT, *sorted(models(), key=str.lower)]


def feather(h: int, w: int, overlap: int, device) -> torch.Tensor:
    """Weight ramp at the tile borders so overlapping tiles blend without seams."""
    if overlap <= 0:
        return torch.ones(1, 1, h, w, device=device)
    ramp = lambda n: torch.clamp(torch.minimum(torch.arange(n, device=device) + 1, torch.arange(n, 0, -1, device=device)) / overlap, max=1.0)
    return (ramp(h)[:, None] * ramp(w)[None, :])[None, None]


def tiled(fn, x: torch.Tensor, tile: int, overlap: int, factor: float) -> torch.Tensor:
    """fn over overlapping tiles of x (B, C, H, W), blended with feathered borders. `factor` is fn's output size
    over its input size (16 for a VAE decode of packed latents, 1/16 for the encode, an upscaler's scale)."""
    _, _, H, W = x.shape
    step = max(1, tile - overlap)
    starts = lambda n: [0] if n <= tile else sorted({*range(0, n - tile, step), n - tile})  # last tile flush with the edge
    out = weight = None
    for ya in starts(H):
        for xa in starts(W):
            part = fn(x[:, :, ya : ya + tile, xa : xa + tile]).float()
            if out is None:
                out = torch.zeros(part.shape[0], part.shape[1], round(H * factor), round(W * factor), device=part.device)
                weight = torch.zeros(1, 1, *out.shape[2:], device=part.device)
            oy, ox = round(ya * factor), round(xa * factor)
            w = feather(part.shape[2], part.shape[3], round(overlap * factor), part.device)
            out[:, :, oy : oy + part.shape[2], ox : ox + part.shape[3]] += part * w
            weight[:, :, oy : oy + part.shape[2], ox : ox + part.shape[3]] += w
    return out / weight.clamp_min(1e-6)


def _model(path: str, device):
    try:
        import spandrel
    except ImportError:
        raise RuntimeError("Upscaler models need the spandrel package: start Venno MOSS with moss.sh / moss.bat (without --skip-install) to install it") from None

    with _lock:
        if path not in _loaded:
            _loaded.clear()
            desc = spandrel.ModelLoader().load_from_file(path)
            if not isinstance(desc, spandrel.ImageModelDescriptor):
                raise ValueError(f"{os.path.basename(path)} is not an image upscaler")
            desc.model.eval()
            _loaded[path] = desc
        desc = _loaded[path]
    dtype = torch.float16 if device.type == "cuda" and desc.supports_half else torch.float32
    return desc.to(device).to(dtype), dtype


@torch.inference_mode()
def run_model(image: Image.Image, name: str, device, *, tile: int = 512, overlap: int = 32) -> Image.Image:
    """Upscale by the model's own factor; the model goes back to system RAM afterwards (the transformer needs the VRAM)."""
    path = models().get(name)
    if path is None:
        raise ValueError(f'Upscaler "{name}" not found')
    desc, dtype = _model(path, device)
    try:
        x = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        out = tiled(lambda t: desc(t.to(device=device, dtype=dtype)), x, tile, overlap, desc.scale).clamp(0, 1)
        return Image.fromarray((out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8))
    finally:
        desc.to("cpu")


def upscale(image: Image.Image, name: str, width: int, height: int, device) -> Image.Image:
    """Image → exactly width × height with the named upscaler (models run until big enough, then resample)."""
    if name in RESAMPLE or name not in models():
        return image.resize((width, height), RESAMPLE.get(name, Image.LANCZOS))
    out = image
    for _ in range(3):  # e.g. a 2× model for a 3× target runs twice
        if out.width >= width and out.height >= height:
            break
        out = run_model(out, name, device)
    return out if out.size == (width, height) else out.resize((width, height), Image.LANCZOS)
