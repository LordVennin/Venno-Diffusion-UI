"""Infotext (A1111-compatible "parameters" PNG chunk) and saving results."""

from __future__ import annotations

import datetime
import io
import os
import re
import threading

from PIL import Image, PngImagePlugin

VERSION = "venno"
_save_lock = threading.Lock()


def infotext(prompt: str, negative: str, params: dict) -> str:
    def fmt(v):
        v = str(v)
        return f'"{v}"' if ("," in v or ":" in v) and not v.startswith('"') else v

    lines = [prompt]
    if negative:
        lines.append(f"Negative prompt: {negative}")
    lines.append(", ".join(f"{k}: {fmt(v)}" for k, v in params.items() if v is not None and v != ""))
    return "\n".join(lines)


def png_bytes(image: Image.Image, info: str) -> bytes:
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", info)
    buf = io.BytesIO()
    image.save(buf, format="PNG", pnginfo=pnginfo)
    return buf.getvalue()


def outputs_root() -> str:
    from modules import paths  # Phase 4: Venno settings

    return os.path.join(paths.data_path, "outputs")


def save(image: Image.Image, info: str, mode: str, seed: int) -> str:
    folder = os.path.join(outputs_root(), f"{mode}-images", datetime.date.today().isoformat())
    os.makedirs(folder, exist_ok=True)
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", info)
    with _save_lock:
        numbers = [int(m.group(1)) for f in os.listdir(folder) if (m := re.match(r"(\d+)-", f))]
        path = os.path.join(folder, f"{(max(numbers) + 1) if numbers else 0:05d}-{seed}.png")
        image.save(path, pnginfo=pnginfo)
    return path
