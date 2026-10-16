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
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from venno.engine import conditioning, detect, images, lora, sampling, vae

from .passes import make_conditioning, sample_pass

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
) -> tuple[Image.Image, list[str]]:
    """Refine `image` with every detailer unit in turn. Returns the image and notes for the infotext."""
    notes = []
    target_area = min(job.width * job.height, 1024 * 1024)
    for u_index, unit in enumerate(units):
        found = detect.detect(unit.model, image, unit.confidence)
        masks = [_grow(d.mask, unit.dilate) for d in found]
        if limit_mask is not None:  # inpainting: only what touches the area being regenerated
            lm = np.asarray(limit_mask.resize(image.size, Image.BILINEAR)).astype(np.float32) / 255.0
            masks = [Image.fromarray((np.asarray(m) * lm).astype(np.uint8)) for m in masks]
        masks = [m for m in masks if m.getbbox() is not None]
        notes.append(f"{unit.model}: {len(masks)}")
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
    return image, notes
