"""Settings written into result PNGs: an A1111-style "parameters" text (readable anywhere) and the exact job as
JSON under "venno_moss" (what "Use settings" reads back)."""

from __future__ import annotations

import io
import json
import os
import re
import time

from PIL import Image, PngImagePlugin

from . import VERSION, config


def infotext(job, seed: int, masked: bool = False, size: tuple[int, int] | None = None) -> str:
    """masked: an inpaint / outpaint job whose mask had content; size: the result's size when it differs from the
    job's (an "only masked" result has the input's size)."""
    width, height = size or (job.width, job.height)
    params = {
        "Steps": job.steps, "CFG scale": f"{job.cfg:g}", "Shift": f"{job.shift:g}", "Seed": seed,
        "Size": f"{job.width}x{job.height}" if job.mode == "txt2img" else f"{width}x{height}", "Model": job.model,
        "Skip layers": ",".join(str(k) for k in job.skip_layers) if job.skip_layers else "none",
    }
    if job.mode == "img2img":
        params["Denoising strength"] = "1" if masked and job.fill == "latent_noise" else f"{job.strength:g}"
        if masked:
            params.update({"Mask blur": job.mask_blur, "Masked content": job.fill, "Inpaint area": "Only masked" if job.only_masked else "Whole picture",
                           "Soft inpainting": job.soft_inpaint})
    elif job.hires is not None:
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
    keep = ("mode", "prompt", "negative", "model", "width", "height", "steps", "cfg", "shift", "skip_layers",
            "strength", "resize_mode", "mask_blur", "fill", "only_masked", "only_masked_padding", "soft_inpaint")  # not the pictures
    data = {k: getattr(job, k) for k in keep}
    loras = [lr for lr in job.loras if not (job.fast and config.is_turbo(lr.name))]  # Fast preview adds its LoRA itself
    data.update(seed=seed, loras=[{"name": lr.name, "weight": lr.weight} for lr in loras], version=VERSION,
                fast=job.fast, hires=job.hires.model_dump() if job.hires is not None and job.mode == "txt2img" else None)
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


_FIELD = re.compile(r'\s*(\w[\w \-/]*):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')


def parse_infotext(text: str) -> dict:
    """An A1111-style parameters text → {"prompt", "negative", "fields": [[name, value], …]} (fields in order)."""
    lines = text.strip().split("\n")
    fields = []
    if lines and len(_FIELD.findall(lines[-1])) >= 3:
        for name, value in _FIELD.findall(lines.pop()):
            if value.startswith('"'):
                try:
                    value = json.loads(value)
                except ValueError:
                    pass
            fields.append([name, value])
    prompt, negative, target = [], [], None
    for line in lines:
        if line.startswith("Negative prompt:"):
            target = negative
            line = line[len("Negative prompt:"):].strip()
        (prompt if target is None else target).append(line)
    return {"prompt": "\n".join(prompt).strip(), "negative": "\n".join(negative).strip(), "fields": fields}


def _exif_parameters(img: Image.Image) -> str:
    """A1111 / Forge JPEG and WebP files keep the parameters in the EXIF user comment."""
    try:
        raw = img.getexif().get_ifd(0x8769).get(0x9286)
    except Exception:
        return ""
    if isinstance(raw, bytes):
        if raw.startswith(b"UNICODE\0"):
            return raw[8:].decode("utf-16-be", "replace")
        return raw[8:].decode("utf-8", "replace") if raw.startswith(b"ASCII\0\0\0") else raw.decode("utf-8", "replace")
    return raw or ""


def read(data: bytes) -> dict:
    """What a picture says about how it was made: {"parameters": its text, "fields": [[name, value], …],
    "settings": the exact Venno MOSS settings, or what a picture from another UI (Venno, Forge, A1111) can give
    (prompt, negative, seed, size; marked "partial": its steps and CFG belong to another model), or None}."""
    img = Image.open(io.BytesIO(data))
    text = img.info.get("parameters", "") or _exif_parameters(img)
    parsed = parse_infotext(text) if text else {"prompt": "", "negative": "", "fields": []}
    raw = img.info.get("venno_moss")
    try:
        settings = json.loads(raw) if raw else None
    except ValueError:
        settings = None
    if settings is None and text:
        fields = dict(parsed["fields"])
        settings = {"partial": True, "prompt": parsed["prompt"], "negative": parsed["negative"]}
        if str(fields.get("Seed", "")).isdigit():
            settings["seed"] = int(fields["Seed"])
        size = re.fullmatch(r"(\d+)x(\d+)", str(fields.get("Size", "")))
        if size:
            fit = lambda v: min(2048, max(256, round(int(v) / 16) * 16))
            settings["width"], settings["height"] = fit(size.group(1)), fit(size.group(2))
    return {"parameters": text, "fields": parsed["fields"], "settings": settings, "width": img.width, "height": img.height}
