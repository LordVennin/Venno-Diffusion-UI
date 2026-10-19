"""Detailer (ADetailer replacement): find faces / hands / people and re-render each at a proper resolution.

Per unit:
  1. detect (YOLO boxes or segmentation masks, or mediapipe), grow the masks by `dilate`
  2. keep only detections that touch the job's inpaint mask, if the job has one
  3. merge detections whose crops would overlap (one crop each, no fighting pastes)
  4. crop around each group, sized so the object is rendered at ~4× up to the job's pixel count
  5. refine all crops that share a size in one batched pass: differential soft inpainting inside the
     feathered mask, everything else held to the image
  6. paste back with the feathered mask (untouched pixels stay exact)

`[PROMPT]` in a detailer prompt is replaced by the main prompt; an empty prompt means the main prompt.
`<lora:…>` tags in a detailer prompt apply to that unit's pass only.

Like Forge's ADetailer, every run also returns a detection preview: the picture before detailing with each
detection boxed and labelled with its score (grey when it lies outside the inpaint mask and is skipped), and a
note for units that found nothing. It is also sent as the live preview while the detailer works.
"""

from __future__ import annotations

import io
import math
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from venno.engine import conditioning, detect, images, lora, sampling, vae

from .passes import make_conditioning, sample_pass

UNIT_COLORS = ((255, 120, 60), (60, 200, 255), (140, 230, 90), (230, 110, 230))  # per detector, as in the preview
SKIPPED = (150, 150, 150)
MAX_UPSCALE = 4.0
ASPECTS = (1.0, 3 / 4, 4 / 3, 2 / 3, 3 / 2, 1 / 2, 2 / 1)


def _grow(mask: Image.Image, px: int) -> Image.Image:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return Image.fromarray(cv2.dilate(np.asarray(mask), k))


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _groups(masks: list[Image.Image], pad: int, size: tuple[int, int]) -> list[Image.Image]:
    """Union masks whose padded boxes overlap, so each region of the image is refined by one crop."""
    W, H = size
    boxes = []
    for m in masks:
        x1, y1, x2, y2 = m.getbbox()
        boxes.append((max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)))
    parent = list(range(len(masks)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if _overlaps(boxes[i], boxes[j]):
                parent[root(i)] = root(j)
    merged: dict[int, np.ndarray] = {}
    for i, m in enumerate(masks):
        r = root(i)
        merged[r] = np.maximum(merged[r], np.asarray(m)) if r in merged else np.asarray(m).copy()
    return [Image.fromarray(v) for v in merged.values()]


def resolve_prompt(text: str, main: str) -> str:
    text = (text or "").strip()
    if not text:
        return main
    return text.replace("[PROMPT]", main)


def draw_detections(image: Image.Image, report: list[dict]) -> Image.Image:
    """The picture with every unit's detections boxed and scored; `report` items: model, confidence, found, kept."""
    out = image.convert("RGB").copy()
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    size = max(12, round(max(out.size) / 48))
    font = ImageFont.load_default(size=size)
    width = max(2, round(max(out.size) / 300))
    for k, r in enumerate(report):
        color = UNIT_COLORS[k % len(UNIT_COLORS)]
        for d, kept in zip(r["found"], r["kept"]):
            c = color if kept else SKIPPED
            tint = Image.new("RGBA", out.size, (*c, 70 if kept else 40))
            overlay.paste(tint, (0, 0), d.mask)  # the detected shape, tinted
            draw.rectangle(d.box, outline=(*c, 255), width=width)
            label = f"{r['model'].rsplit('.', 1)[0].split('_')[0]} {d.score:.2f}" + ("" if kept else " (outside mask)")
            tx, ty = d.box[0], max(0, d.box[1] - size - 6)
            tb = draw.textbbox((tx + 4, ty + 2), label, font=font)
            draw.rectangle((tb[0] - 4, tb[1] - 2, tb[2] + 4, tb[3] + 2), fill=(*c, 220))
            draw.text((tx + 4, ty + 2), label, font=font, fill=(0, 0, 0, 255))
    lines = [f"{r['model']}: {'nothing found' if not r['found'] else ', '.join(f'{d.score:.2f}' for d in r['found'])} (confidence ≥ {r['confidence']:g})" for r in report]
    y = out.size[1] - (size + 8) * len(lines) - 4
    for k, line in enumerate(lines):
        tb = draw.textbbox((8, y), line, font=font)
        draw.rectangle((tb[0] - 4, tb[1] - 3, tb[2] + 4, tb[3] + 3), fill=(0, 0, 0, 170))
        draw.text((8, y), line, font=font, fill=(*UNIT_COLORS[k % len(UNIT_COLORS)], 255))
        y += size + 8
    return Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")


def summary(report: list[dict]) -> str:
    """One line for the Studio, e.g. "face_yolov8n.pt: 2 found (0.91, 0.54)"."""
    parts = []
    for r in report:
        kept = [d for d, k in zip(r["found"], r["kept"]) if k]
        skipped = len(r["found"]) - len(kept)
        text = f"{r['model']}: " + (f"{len(kept)} found ({', '.join(f'{d.score:.2f}' for d in kept)})" if kept else "nothing found")
        if skipped:
            text += f", {skipped} outside the mask"
        parts.append(text)
    return "; ".join(parts)


def run(
    engine,
    job,
    progress,
    image: Image.Image,
    seed: int,
    units: list,
    *,
    job_loras: list[tuple[str, float]],
    cache: dict,
    limit_mask: Optional[Image.Image] = None,
    on_unit: Callable[[int, int], None] = lambda done, total: None,
) -> tuple[Image.Image, list[dict]]:
    """Refine `image` with every detailer unit in turn. Returns the image and the detection report
    (per unit: model, confidence, found, kept) for `draw_detections` / `summary`."""
    report = []
    original = image
    target_area = min(job.width * job.height, 1024 * 1024)
    for u_index, unit in enumerate(units):
        found = detect.detect(unit.model, image, unit.confidence)
        masks = [_grow(d.mask, unit.dilate) for d in found]
        if limit_mask is not None:  # inpainting: only what touches the area being regenerated
            lm = np.asarray(limit_mask.resize(image.size, Image.BILINEAR)).astype(np.float32) / 255.0
            masks = [Image.fromarray((np.asarray(m) * lm).astype(np.uint8)) for m in masks]
        kept = [m.getbbox() is not None for m in masks]
        report.append(dict(model=unit.model, confidence=unit.confidence, found=found, kept=kept))
        _show(progress, draw_detections(original, report))  # live preview: what this unit found
        masks = [m for m, k in zip(masks, kept) if k]
        if not masks:
            continue

        # crops at a sensible resolution, bucketed so similar crops share a size (and a batch)
        crops = []
        for group in _groups(masks, unit.padding, image.size):
            x1, y1, x2, y2 = group.getbbox()
            aspect = min(ASPECTS, key=lambda a: abs(math.log(a) - math.log(max(1, x2 - x1) / max(1, y2 - y1))))
            box = images.crop_region(group, unit.padding + 2 * unit.mask_blur, aspect)
            bw, bh = box[2] - box[0], box[3] - box[1]
            scale = min(MAX_UPSCALE, math.sqrt(target_area / max(1, bw * bh)))
            scale = max(scale, 1.0)
            pw, ph = images.snap(bw * scale), images.snap(bh * scale)
            soft = images.blur_mask(group, unit.mask_blur)
            crops.append(dict(box=box, size=(pw, ph), image=image.crop(box).resize((pw, ph), Image.LANCZOS), mask=soft.crop(box).resize((pw, ph), Image.BILINEAR), soft=soft))

        # prompts and LoRAs for this unit
        prompt_text, unit_loras = conditioning.extract_lora_tags(resolve_prompt(unit.prompt, job.prompt))
        negative_text, neg_loras = conditioning.extract_lora_tags(resolve_prompt(unit.negative, job.negative))
        extra = [lr for lr in unit_loras + neg_loras if lr[0] not in {n for n, _ in job_loras}]
        if extra:
            engine.forge_objects = engine.forge_objects_original.shallow_copy()
            lora.apply(engine, job_loras + extra)
        custom_prompt = bool((unit.prompt or "").strip())

        steps_run = sampling.run_steps(job.steps, unit.denoise)
        by_size: dict[tuple[int, int], list[dict]] = {}
        for c in crops:
            by_size.setdefault(c["size"], []).append(c)
        try:
            for (pw, ph), batch in by_size.items():
                engine.set_shift(shift=job.shift, width=pw, height=ph)
                cond = make_conditioning(engine, job, steps_run, pw, ph, cache, prompt=prompt_text, negative=negative_text, switch=None if custom_prompt else (job.switch_prompt or None))
                latent = torch.cat([vae.encode(engine, c["image"]) for c in batch])
                lmask = torch.cat([images.latent_mask(c["mask"], pw // 8, ph // 8) for c in batch])
                out = sample_pass(
                    engine, job, progress, [seed + 1000 * (u_index + 1) + k for k in range(len(batch))], vae.latent_shape(engine, pw, ph), cond, None,
                    steps=job.steps, cfg=job.cfg, init_latent=latent, denoise=unit.denoise,
                    inpaint=sampling.Inpaint(lmask, differential=True, keep=latent), on_step=lambda i, n, d: None,
                )
                for c, generated in zip(batch, vae.decode(engine, out)):
                    box = c["box"]
                    w, h = box[2] - box[0], box[3] - box[1]
                    region = images.composite(image.crop(box), generated.resize((w, h), Image.LANCZOS), c["soft"].crop(box))
                    image = image.copy()
                    image.paste(region, box[:2])
        finally:
            if extra:  # back to the job's own LoRAs
                engine.forge_objects = engine.forge_objects_original.shallow_copy()
                lora.apply(engine, job_loras)
        on_unit(u_index + 1, len(units))
    return image, report


def preview_png(image: Image.Image, report: list[dict]) -> bytes:
    """The detection preview as a PNG (no generation parameters: it is not a result)."""
    buf = io.BytesIO()
    draw_detections(image, report).save(buf, format="PNG")
    return buf.getvalue()


def _show(progress, image: Image.Image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    progress.on_preview(buf.getvalue())
