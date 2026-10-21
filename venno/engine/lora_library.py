"""The LoRA library behind the Studio's LoRA browser: folders, previews, tags and notes per LoRA.

Everything is stored next to the LoRA file in the A1111 / Forge format, so previews and notes made in Forge
show up here and the other way round:
- preview: `<name>.preview.png` (also read: `<name>.png/.jpg/.jpeg/.webp`, `.preview.jpg/.webp`), or the
  thumbnail some trainers embed in the safetensors header (`modelspec.thumbnail`)
- notes: `<name>.json` with "activation text", "preferred weight", "description", "notes", "sd version";
  Venno adds "tags". Civitai Helper's `<name>.civitai.info` "trainedWords" fill in missing trigger words.
"""

from __future__ import annotations

import base64
import io
import json
import os
import struct
import threading
from typing import Optional

from PIL import Image

from . import lora

PREVIEW_NAMES = (".preview.png", ".preview.jpg", ".preview.jpeg", ".preview.webp", ".png", ".jpg", ".jpeg", ".webp")
PREVIEW_SIZE = 768  # longest side of saved previews
HEADER_LIMIT = 8 * 1024 * 1024  # don't parse safetensors headers bigger than this (some embed huge tag lists)
_lock = threading.Lock()
_headers: dict[tuple, dict] = {}  # (path, mtime) → {"base": str, "thumbnail": data URL or ""}


def _base(path: str) -> str:
    return os.path.splitext(path)[0]


def _root(path: str) -> str:
    """The LoRA folder a file lives under (the longest matching one)."""
    path = os.path.abspath(path)
    roots = sorted((os.path.abspath(r) for r in lora.folders()), key=len, reverse=True)
    return next((r for r in roots if path.startswith(r + os.sep)), os.path.dirname(path))


def folder_of(path: str) -> str:
    """Sub-folder of a LoRA inside its LoRA folder, '/'-separated ('' for the top level)."""
    rel = os.path.relpath(os.path.dirname(os.path.abspath(path)), _root(path))
    return "" if rel == "." else rel.replace(os.sep, "/")


def preview_path(path: str) -> Optional[str]:
    base = _base(path)
    return next((base + ext for ext in PREVIEW_NAMES if os.path.isfile(base + ext)), None)


def _header(path: str) -> dict:
    """Base model and embedded thumbnail from a safetensors header (cached by file and mtime)."""
    if not path.lower().endswith(".safetensors"):
        return {"base": "", "thumbnail": ""}
    key = (path, os.path.getmtime(path))
    with _lock:
        if key in _headers:
            return _headers[key]
    info = {"base": "", "thumbnail": ""}
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if 0 < n <= HEADER_LIMIT:
                meta = json.loads(f.read(n)).get("__metadata__", {}) or {}
                info["base"] = meta.get("modelspec.architecture") or meta.get("ss_base_model_version") or ""
                thumb = meta.get("modelspec.thumbnail") or ""
                info["thumbnail"] = thumb if thumb.startswith("data:image/") else ""
    except (OSError, ValueError, struct.error):
        pass
    with _lock:
        _headers[key] = info
    return info


def _read_json(file: str) -> dict:
    try:
        with open(file, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def meta(path: str) -> dict:
    data = _read_json(_base(path) + ".json")
    activation = data.get("activation text", "")
    if not activation:
        words = _read_json(_base(path) + ".civitai.info").get("trainedWords") or []
        activation = ", ".join(w.strip() for w in words if isinstance(w, str) and w.strip())
    try:
        weight = float(data.get("preferred weight") or 0) or None
    except (TypeError, ValueError):
        weight = None
    tags = data.get("tags") or []
    return {
        "tags": sorted({str(t).strip() for t in tags if str(t).strip()}, key=str.lower) if isinstance(tags, list) else [],
        "activation": activation,
        "weight": weight,
        "description": data.get("description", ""),
        "notes": data.get("notes", ""),
    }


def entries() -> list[dict]:
    """Every LoRA with its folder, file info, preview and notes."""
    out = []
    for name, path in sorted(lora.scan().items(), key=lambda x: x[0].lower()):
        try:
            stat = os.stat(path)
        except OSError:
            continue
        header = _header(path)
        preview = preview_path(path)
        version = int(os.path.getmtime(preview)) if preview else int(stat.st_mtime)
        out.append({
            "name": name,
            "folder": folder_of(path),
            "file": os.path.basename(path),
            "size": stat.st_size,
            "modified": int(stat.st_mtime),
            "base": header["base"],
            "preview": f"/venno/v1/loras/preview?name={_quote(name)}&v={version}" if preview or header["thumbnail"] else None,
            **meta(path),
        })
    return out


def _quote(name: str) -> str:
    from urllib.parse import quote

    return quote(name, safe="")


def preview_bytes(name: str) -> tuple[bytes, str]:
    """The preview image of a LoRA: (data, media type)."""
    path = lora.resolve(name)
    file = preview_path(path)
    if file:
        ext = os.path.splitext(file)[1].lower()
        with open(file, "rb") as f:
            return f.read(), {".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
    thumb = _header(path)["thumbnail"]
    if thumb:
        head, data = thumb.split(",", 1)
        return base64.b64decode(data), head[5:].split(";")[0]
    raise FileNotFoundError(name)


def set_preview(name: str, image: Optional[Image.Image]) -> None:
    """Save (or with None, remove) a LoRA's preview as `<name>.preview.png`."""
    base = _base(lora.resolve(name))
    target = base + ".preview.png"
    if image is None:
        for ext in PREVIEW_NAMES:  # every preview file Venno or Forge would show
            if os.path.isfile(base + ext):
                os.remove(base + ext)
        return
    image = image.convert("RGB")
    image.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    with open(target, "wb") as f:
        f.write(buf.getvalue())


def set_meta(name: str, *, tags=None, activation=None, weight=None, description=None, notes=None) -> dict:
    """Update a LoRA's `<name>.json` (keys Venno doesn't know are kept)."""
    path = lora.resolve(name)
    file = _base(path) + ".json"
    data = _read_json(file)
    if tags is not None:
        data["tags"] = sorted({t.strip() for t in tags if t and t.strip()}, key=str.lower)
    if activation is not None:
        data["activation text"] = activation.strip()
    if weight is not None:
        data["preferred weight"] = weight
    if description is not None:
        data["description"] = description
    if notes is not None:
        data["notes"] = notes
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return meta(path)
