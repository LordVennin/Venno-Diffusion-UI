"""Image and mask handling for image to image / inpainting (PIL in, PIL out; float tensors inside)."""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

# ---------------------------------------------------------------- decoding


def load(data: str) -> Image.Image:
    """Base64 or data: URL → PIL image."""
    if data.lstrip().startswith("data:"):
        data = data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data)))


def load_rgb(data: str, background=(255, 255, 255)) -> Image.Image:
    img = load(data)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, background + (255,))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def load_mask(data: str) -> Image.Image:
    """White = regenerate. Transparent pixels count as black."""
    img = load(data)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("L")


# ---------------------------------------------------------------- tensors


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL → (C, H, W) float in [0, 1]."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr[..., 0] if arr.shape[2] == 1 else arr)


# ---------------------------------------------------------------- sizing


def snap(v: int, multiple: int = 16) -> int:
    """Anima works on 8× VAE latents with 2×2 patches: sizes must be multiples of 16."""
    return max(multiple, int(round(v / multiple)) * multiple)


def resize(img: Image.Image, width: int, height: int, mode: int = 0) -> Image.Image:
    """0 = stretch, 1 = crop to fill, 2 = fit and extend the edges."""
    if img.size == (width, height):
        return img
    resample = Image.LANCZOS if img.mode != "L" else Image.BILINEAR
    if mode == 1:
        scale = max(width / img.width, height / img.height)
        w, h = max(width, round(img.width * scale)), max(height, round(img.height * scale))
        img = img.resize((w, h), resample)
        left, top = (w - width) // 2, (h - height) // 2
        return img.crop((left, top, left + width, top + height))
    if mode == 2:
        scale = min(width / img.width, height / img.height)
        w, h = max(1, round(img.width * scale)), max(1, round(img.height * scale))
        inner = img.resize((w, h), resample)
        if img.mode == "L":  # masks: new borders stay unmasked
            out = Image.new("L", (width, height), 0)
            out.paste(inner, ((width - w) // 2, (height - h) // 2))
            return out
        # stretch the outermost pixels into the border, softened, so the model sees plausible colours
        out = inner.resize((width, height), resample).filter(ImageFilter.GaussianBlur(max(width, height) / 32))
        out.paste(inner, ((width - w) // 2, (height - h) // 2))
        return out
    return img.resize((width, height), resample)


# ---------------------------------------------------------------- masks


def blur_mask(mask: Image.Image, radius: float) -> Image.Image:
    return mask.filter(ImageFilter.GaussianBlur(radius)) if radius > 0 else mask


def has_content(mask: Image.Image) -> bool:
    return mask.getbbox() is not None


def crop_region(mask: Image.Image, padding: int, aspect: float) -> tuple[int, int, int, int]:
    """Box around the mask (+padding), grown to `aspect` (w/h) and kept inside the image."""
    W, H = mask.size
    x1, y1, x2, y2 = mask.point(lambda v: 255 if v > 0 else 0).getbbox() or (0, 0, W, H)
    x1, y1, x2, y2 = x1 - padding, y1 - padding, x2 + padding, y2 + padding
    w, h = x2 - x1, y2 - y1
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    w, h = min(W, round(w)), min(H, round(h))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = int(min(max(0, round(cx - w / 2)), W - w))
    top = int(min(max(0, round(cy - h / 2)), H - h))
    return left, top, left + w, top + h


def latent_mask(mask: Image.Image, width: int, height: int) -> torch.Tensor:
    """Pixel mask → (1, 1, 1, H/8, W/8) soft latent mask (area average keeps the feathering)."""
    m = to_tensor(mask)[None]  # (1, 1, H, W)
    m = F.interpolate(m, size=(height, width), mode="area")
    return m[:, :, None]  # time axis for the Wan VAE latent layout


def fill_masked(img: Image.Image, mask: Image.Image) -> Image.Image:
    """Fill the masked area with colours flowing in from its surroundings (push-pull / normalised convolution)."""
    x = to_tensor(img)[None]
    keep = (to_tensor(mask)[None] < 0.5).float()
    if keep.sum() == 0:
        return img
    levels = [(x * keep, keep)]
    while min(levels[-1][1].shape[-2:]) > 2:
        cx, ck = levels[-1]
        levels.append((F.avg_pool2d(cx, 2, ceil_mode=True), F.avg_pool2d(ck, 2, ceil_mode=True)))
    fill = levels[-1][0] / levels[-1][1].clamp_min(1e-6)
    for cx, ck in reversed(levels[:-1]):
        fill = F.interpolate(fill, size=cx.shape[-2:], mode="bilinear", align_corners=False)
        a = (ck * 4).clamp(0, 1)  # how much this level knows about each pixel
        fill = (cx / ck.clamp_min(1e-6)) * a + fill * (1 - a)
    out = x * keep + fill * (1 - keep)
    return to_pil(out[0])


def composite(original: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
    """Generated pixels where the (feathered) mask is white, untouched original pixels elsewhere."""
    if generated.size != original.size:
        generated = generated.resize(original.size, Image.LANCZOS)
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.BILINEAR)
    return Image.composite(generated.convert("RGB"), original.convert("RGB"), mask)
