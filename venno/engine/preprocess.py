"""Guide preprocessors: turn any picture into the map a guide (ControlLLLite) model expects.

- canny: OpenCV edges.
- lineart: anime-style line art by colour dodge (pencil-sketch effect), no model needed.
- softedge: thick, rough lines from the line art, for scribble-type models.
- depth: Depth Anything V2 Small (Apache-2.0, ~100 MB, downloaded on first use), white = near.

Line maps come out in the polarity of the chosen model: the Anima lineart / any-test models were trained on
dark lines on white, the scribble model on white lines on black.
"""

from __future__ import annotations

import os
import threading

import numpy as np
from PIL import Image, ImageFilter, ImageOps

PREPROCESSORS = {
    "none": "None (the image is already a map)",
    "canny": "Canny edges",
    "lineart": "Line art",
    "softedge": "Soft edge / scribble",
    "depth": "Depth (Depth Anything V2)",
}
# the guide models each preprocessor's maps suit, by file-name keyword, best first
SUITS = {
    "canny": ("lineart", "any-test", "anytest"),
    "lineart": ("lineart", "any-test", "anytest"),
    "softedge": ("scribble", "any-test", "anytest"),
    "depth": ("depth",),
}
DEPTH_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
_depth = None
_lock = threading.Lock()


def folder() -> str:
    from venno import config

    return os.path.join(config.models_path(), "preprocessors")


def white_lines(model_name: str) -> bool:
    """Whether a guide model expects white lines on black (scribble-type) rather than dark lines on white."""
    return "scribble" in (model_name or "").lower()


def suggested_model(preprocessor: str, models: list[str]) -> str:
    for key in SUITS.get(preprocessor, ()):
        for name in models:
            if key in name.lower():
                return name
    return ""


def run(name: str, image: Image.Image, model_name: str = "", canny_low: int = 100, canny_high: int = 200) -> Image.Image:
    """The map for `image` (RGB, same size)."""
    image = image.convert("RGB")
    if name in ("", "none"):
        return image
    if name == "depth":
        return depth(image)
    if name == "canny":
        lines = canny(image, canny_low, canny_high)  # white on black
    elif name == "lineart":
        lines = ImageOps.invert(lineart(image))
    elif name == "softedge":
        lines = softedge(image)
    else:
        raise ValueError(f'Unknown preprocessor "{name}"')
    if not white_lines(model_name):
        lines = ImageOps.invert(lines)
    return lines.convert("RGB")


def canny(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    import cv2

    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    return Image.fromarray(cv2.Canny(gray, int(low), int(high)))


def lineart(image: Image.Image) -> Image.Image:
    """Dark lines on white: colour dodge of the picture by its blurred negative, then cleaned up."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    radius = max(1.0, max(image.size) / 512)  # line width follows the picture size
    blurred = np.asarray(Image.fromarray(255 - gray.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    dodge = np.clip(gray * 255.0 / np.maximum(255.0 - blurred, 1.0), 0, 255)
    # flat areas → white, lines → dark; a gamma curve removes paper-like grey noise
    lines = 255.0 * (dodge / 255.0) ** 6
    return Image.fromarray(lines.astype(np.uint8))


def softedge(image: Image.Image) -> Image.Image:
    """White, thick, slightly rough lines on black (like a scribble or HED map)."""
    lines = 255 - np.asarray(lineart(image), dtype=np.uint8)
    mask = Image.fromarray(np.where(lines > 110, 255, 0).astype(np.uint8))
    return mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(1))


def _depth_model():
    global _depth
    with _lock:
        if _depth is None:
            from huggingface_hub import snapshot_download
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            local = os.path.join(folder(), "depth-anything-v2-small")
            if not os.path.isfile(os.path.join(local, "config.json")):
                print(f"[Venno] downloading the depth preprocessor from {DEPTH_REPO}")
                snapshot_download(DEPTH_REPO, local_dir=local, allow_patterns=["*.json", "*.safetensors"])
            _depth = (AutoImageProcessor.from_pretrained(local), AutoModelForDepthEstimation.from_pretrained(local).eval())
        return _depth


def depth(image: Image.Image) -> Image.Image:
    """Relative depth, white = near (as the Anima depth guide was trained)."""
    import torch

    processor, model = _depth_model()
    with torch.inference_mode():
        out = model(**processor(images=image, return_tensors="pt")).predicted_depth  # (1, h, w), larger = nearer
        d = torch.nn.functional.interpolate(out[None].float(), size=image.size[::-1], mode="bicubic", align_corners=False)[0, 0]
    d = (d - d.min()) / (d.max() - d.min()).clamp_min(1e-6)
    return Image.fromarray((d.clamp(0, 1).numpy() * 255).astype(np.uint8)).convert("RGB")
