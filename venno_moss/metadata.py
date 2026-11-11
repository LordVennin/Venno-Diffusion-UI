"""Settings written into result PNGs: an A1111-style "parameters" text (readable anywhere) and the exact job as
JSON under "venno_moss" (what "Use settings" reads back)."""

from __future__ import annotations

import io
import json
import os
import time

from PIL import Image, PngImagePlugin

from . import VERSION, config


def infotext(job, seed: int) -> str:
    params = {
        "Steps": job.steps, "CFG scale": f"{job.cfg:g}", "Shift": f"{job.shift:g}", "Seed": seed,
        "Size": f"{job.width}x{job.height}", "Model": job.model,
        "Skip layers": ",".join(str(k) for k in job.skip_layers) if job.skip_layers else "none",
    }
    if job.hires is not None:
        from .engine import hires_size

        h = job.hires
        tw, th = hires_size(job.width, job.height, h.scale, h.resize_x, h.resize_y)
        params.update({"Hires resize": f"{tw}x{th}", "Hires upscaler": h.upscaler, "Denoising strength": f"{h.denoise:g}"})
        if h.steps:
            params["Hires steps"] = h.steps
        if h.cfg:
            params["Hires CFG scale"] = f"{h.cfg:g}"
        if h.shift:
            params["Hires shift"] = f"{h.shift:g}"
    if job.fast:
        params["Fast preview"] = "turbo LoRA"
    for k, lr in enumerate([lr for lr in job.loras if lr.weight != 0], start=1):
        params[f"LoRA {k}"] = f"{lr.name}:{lr.weight:g}"
    params["Version"] = f"venno-moss {VERSION}"
    tail = ", ".join(f"{k}: {json.dumps(v) if isinstance(v, str) and ',' in v else v}" for k, v in params.items())
    lines = [job.prompt]
    if job.negative:
        lines.append(f"Negative prompt: {job.negative}")
    lines.append(tail)
    return "\n".join(lines)


def settings_json(job, seed: int) -> str:
    keep = ("prompt", "negative", "model", "width", "height", "steps", "cfg", "shift", "skip_layers")
    data = {k: getattr(job, k) for k in keep}
    loras = [lr for lr in job.loras if not (job.fast and config.is_turbo(lr.name))]  # Fast preview adds its LoRA itself
    data.update(seed=seed, loras=[{"name": lr.name, "weight": lr.weight} for lr in loras], version=VERSION,
                fast=job.fast, hires=job.hires.model_dump() if job.hires is not None else None)
    return json.dumps(data, ensure_ascii=False)


def png_bytes(image: Image.Image, info: str, settings: str) -> bytes:
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", info)
    meta.add_text("venno_moss", settings)
    buf = io.BytesIO()
    image.save(buf, format="PNG", pnginfo=meta)
    return buf.getvalue()


def save(png: bytes, seed: int, sub: str = "") -> str:
    folder = os.path.join(config.outputs_path(), time.strftime("%Y-%m-%d"), sub)
    os.makedirs(folder, exist_ok=True)
    n = len([f for f in os.listdir(folder) if f.endswith(".png")])
    path = os.path.join(folder, f"{n:05d}-{seed}.png")
    with open(path, "wb") as f:
        f.write(png)
    return path


def read(data: bytes) -> dict:
    """{"parameters": text or "", "settings": dict or None} of a PNG."""
    img = Image.open(io.BytesIO(data))
    text = img.info.get("parameters", "") or ""
    raw = img.info.get("venno_moss")
    try:
        settings = json.loads(raw) if raw else None
    except ValueError:
        settings = None
    return {"parameters": text, "settings": settings, "width": img.width, "height": img.height}
