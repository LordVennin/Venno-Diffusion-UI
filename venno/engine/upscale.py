"""Image upscaling: Spandrel models (ESRGAN, DAT, SwinIR, HAT, …) with feathered tiles, plus plain resampling.

Upscaler names (as shown in Studio):
    "Latent…"       latent resize, handled by the hires pipeline (see `is_latent`)
    "Lanczos"       plain resampling
    "Nearest"
    <file stem>     a model in the upscaler folder
"""

from __future__ import annotations

import os
import threading

import numpy as np
import torch
from PIL import Image

EXTENSIONS = (".safetensors", ".pth", ".pt", ".ckpt", ".bin")
LATENT = ("Latent", "Latent (bicubic)", "Latent (nearest-exact)")
RESAMPLE = {"Lanczos": Image.LANCZOS, "Nearest": Image.NEAREST}

_lock = threading.Lock()
_loaded: dict = {}  # path → spandrel ImageModelDescriptor (one at a time)


def is_latent(name: str) -> bool:
    return (name or "Latent").startswith("Latent")


def latent_mode(name: str) -> str:
    return {"Latent (bicubic)": "bicubic", "Latent (nearest-exact)": "nearest-exact"}.get(name, "bilinear")


def folders() -> list[str]:
    from venno import config

    return config.model_folders("upscaler")


def models() -> dict[str, str]:
    found: dict[str, str] = {}
    for root in folders():
        for f in sorted(os.listdir(root)):
            if f.lower().endswith(EXTENSIONS):
                found.setdefault(os.path.splitext(f)[0], os.path.join(root, f))
    return found


def available() -> list[str]:
    return [*LATENT, *RESAMPLE, *sorted(models(), key=str.lower)]


def _model(path: str):
    import spandrel

    from backend import memory_management

    with _lock:
        if path not in _loaded:
            _loaded.clear()
            desc = spandrel.ModelLoader().load_from_file(path)
            if not isinstance(desc, spandrel.ImageModelDescriptor):
                raise ValueError(f"{os.path.basename(path)} is not an image upscaler")
            desc.model.eval()
            _loaded[path] = desc
        desc = _loaded[path]
    device = memory_management.get_torch_device()
    dtype = torch.float16 if device.type == "cuda" and desc.supports_half else torch.float32
    return desc.to(device).to(dtype), device, dtype


def _feather(h: int, w: int, overlap: int, device) -> torch.Tensor:
    """Weight ramp at the tile borders so overlapping tiles blend without seams."""
    if overlap <= 0:
        return torch.ones(1, 1, h, w, device=device)
    ramp = lambda n: torch.clamp(torch.minimum(torch.arange(n, device=device) + 1, torch.arange(n, 0, -1, device=device)) / overlap, max=1.0)
    return (ramp(h)[:, None] * ramp(w)[None, :])[None, None]


@torch.inference_mode()
def run_model(image: Image.Image, name: str, *, tile: int = 512, overlap: int = 32) -> Image.Image:
    """Upscale by the model's own factor."""
    path = models().get(name)
    if path is None:
        raise ValueError(f'Upscaler "{name}" not found')
    desc, device, dtype = _model(path)
    s = desc.scale
    x = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    _, _, H, W = x.shape
    out = torch.zeros(1, 3, H * s, W * s, device=device)
    weight = torch.zeros(1, 1, H * s, W * s, device=device)
    step = max(1, tile - overlap)
    starts = lambda n: [0] if n <= tile else sorted({*range(0, n - tile, step), n - tile})  # last tile flush with the edge
    for ya in starts(H):
        for xa in starts(W):
            y1, x1 = min(ya + tile, H), min(xa + tile, W)
            part = desc(x[:, :, ya:y1, xa:x1].to(device=device, dtype=dtype)).float()
            w = _feather(part.shape[2], part.shape[3], overlap * s, device)
            out[:, :, ya * s : y1 * s, xa * s : x1 * s] += part * w
            weight[:, :, ya * s : y1 * s, xa * s : x1 * s] += w
    out = (out / weight.clamp_min(1e-6)).clamp(0, 1)
    return Image.fromarray((out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8))


def upscale(image: Image.Image, name: str, width: int, height: int) -> Image.Image:
    """Image → exactly width × height with the named upscaler (models run until big enough, then resample)."""
    if name in RESAMPLE or name not in models():
        return image.resize((width, height), RESAMPLE.get(name, Image.LANCZOS))
    out = image
    for _ in range(3):  # e.g. a 2× model for a 3× target runs twice
        if out.width >= width and out.height >= height:
            break
        out = run_model(out, name)
    return out if out.size == (width, height) else out.resize((width, height), Image.LANCZOS)
