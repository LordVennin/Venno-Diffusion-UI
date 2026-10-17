"""Reading generation settings back from images (A1111 / Forge / Venno "parameters" text).

PNG: the "parameters" text chunk. JPEG / WebP: EXIF UserComment (how A1111 stores it).
`settings()` turns the text into Venno job-style fields for Studio to apply.
"""

from __future__ import annotations

import io
import re
from typing import Optional

from PIL import Image

_PARAM = re.compile(r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')
_SIZE = re.compile(r"^(\d+)x(\d+)$")
_SCHEDULER_SUFFIXES = ("Karras", "Exponential", "SGM Uniform", "Simple", "Beta", "Normal", "Uniform", "Polyexponential", "Align Your Steps", "KL Optimal")


def read_text(data: bytes) -> tuple[Optional[str], str]:
    """(parameters text or None, note) from image bytes."""
    img = Image.open(io.BytesIO(data))
    info = img.info or {}
    if isinstance(info.get("parameters"), str):
        return info["parameters"], ""
    exif = img.getexif()
    comment = exif.get_ifd(0x8769).get(0x9286) if exif else None  # Exif → UserComment
    if isinstance(comment, bytes):
        comment = _decode_user_comment(comment)
    if comment and comment.strip("\0").strip():
        return comment.strip("\0").strip(), ""
    if "prompt" in info or "workflow" in info:
        return None, "This image was made with ComfyUI; its workflow can't be read as settings."
    return None, "No generation settings found in this image."


def _decode_user_comment(raw: bytes) -> str:
    """EXIF UserComment: 8-byte charset prefix, then the text (A1111 writes UTF-16; byte order varies by writer)."""
    prefix, body = raw[:8], raw[8:]
    if prefix.startswith(b"UNICODE"):
        candidates = [body.decode(enc, errors="ignore") for enc in ("utf-16-be", "utf-16-le")]
        return max(candidates, key=lambda t: sum(ch.isascii() and ch.isprintable() for ch in t))
    if prefix.startswith(b"ASCII"):
        return body.decode("ascii", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def parse(text: str) -> dict:
    """A1111 infotext → {prompt, negative, params}."""
    lines = text.strip().split("\n")
    last = lines[-1] if lines else ""
    params = {}
    if len(_PARAM.findall(last)) >= 3:
        for k, v in _PARAM.findall(last):
            v = v.strip()
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1].replace('\\"', '"').replace("\\\\", "\\")
            params[k.strip()] = v
        lines = lines[:-1]
    prompt, negative, in_neg = [], [], False
    for line in lines:
        if line.startswith("Negative prompt:"):
            in_neg = True
            line = line[len("Negative prompt:") :].strip()
        (negative if in_neg else prompt).append(line)
    return {"prompt": "\n".join(prompt).strip(), "negative": "\n".join(negative).strip(), "params": params}


def _num(v, kind=float):
    try:
        return kind(float(v)) if kind is int else kind(v)
    except (TypeError, ValueError):
        return None


def settings(text: str) -> dict:
    from venno.engine.conditioning import extract_lora_tags

    p = parse(text)
    prm = p["params"]
    prompt, loras = extract_lora_tags(p["prompt"], keep_random=True)
    negative, _ = extract_lora_tags(p["negative"], keep_random=True)
    out: dict = {"prompt": prompt, "negative": negative, "loras": [{"name": n, "weight": w} for n, w in loras], "raw": prm}

    sampler, scheduler = prm.get("Sampler"), prm.get("Schedule type")
    if sampler and not scheduler:  # old A1111 names: "DPM++ 2M Karras"
        for suffix in _SCHEDULER_SUFFIXES:
            if sampler.endswith(" " + suffix):
                sampler, scheduler = sampler[: -len(suffix) - 1], suffix
                break
    out |= {
        "steps": _num(prm.get("Steps"), int),
        "sampler": sampler,
        "scheduler": scheduler,
        "cfg": _num(prm.get("CFG scale")),
        "shift": _num(prm.get("Shift", prm.get("Distilled CFG Scale"))),
        "seed": _num(prm.get("Seed"), int),
        "checkpoint": prm.get("Model"),
        "template": prm.get("Template"),
    }
    if m := _SIZE.match(prm.get("Size", "")):
        out["width"], out["height"] = int(m.group(1)), int(m.group(2))

    hires = None
    if any(k in prm for k in ("Hires upscale", "Hires resize", "Hires upscaler")):
        hires = {"upscaler": prm.get("Hires upscaler", "Latent"), "scale": _num(prm.get("Hires upscale")) or 0, "resize_x": 0, "resize_y": 0,
                 "steps": _num(prm.get("Hires steps"), int) or 0, "denoise": _num(prm.get("Denoising strength")) or 0.5,
                 "cfg": _num(prm.get("Hires CFG Scale")) or 0, "shift": _num(prm.get("Hires shift", prm.get("Hires Distilled CFG Scale"))) or 0}
        if m := _SIZE.match(prm.get("Hires resize", "")):
            hires["resize_x"], hires["resize_y"] = int(m.group(1)), int(m.group(2))
            if not hires["scale"] and out.get("width"):
                hires["scale"] = round(hires["resize_x"] / out["width"], 3)
    else:
        out["denoise"] = _num(prm.get("Denoising strength"))
    out["hires"] = hires

    if "Prompt switch" in prm:
        out["switch_prompt"] = prm["Prompt switch"]
        out["switch_at"] = _num(prm.get("Prompt switch at")) or 0.4

    detailer = []
    for suffix in ("", " 2nd", " 3rd", " 4th"):
        if f"ADetailer model{suffix}" in prm:
            detailer.append({
                "model": prm[f"ADetailer model{suffix}"],
                "confidence": _num(prm.get(f"ADetailer confidence{suffix}")) or 0.3,
                "denoise": _num(prm.get(f"ADetailer denoising strength{suffix}")) or 0.4,
                "prompt": prm.get(f"ADetailer prompt{suffix}", ""),
            })
    out["detailer"] = detailer
    out["regions"] = [prm[k] for k in sorted((k for k in prm if re.fullmatch(r"Region \d+", k)), key=lambda k: int(k.split()[1]))]
    out["region_negatives"] = [prm.get(f"Region {k + 1} negative", "") for k in range(len(out["regions"]))]
    out["region_base"] = _num(prm.get("Region base"))
    return {k: v for k, v in out.items() if v is not None}
