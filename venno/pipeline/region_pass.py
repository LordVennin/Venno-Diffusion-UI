"""Region pass: after the image is generated, re-render each region on its own, like inpainting it again.

With several characters in one picture each gets only a slice of the pixels, and whatever bled between
them is baked in. The region pass crops each region (grown a little, with context around it), renders
the crop at up to ~1 MP with only that region's prompt and negative, at a moderate denoise, and blends it
back with a soft edge (differential soft inpainting, like the detailer). Regions run one at a time from
the bottom layer up, each on the result of the previous one, so neighbours stay consistent; regions that
span most of the picture (backgrounds) are skipped. Sketch guides are cropped along with the region.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PIL import Image

from venno.engine import guides, images, regions, sampling, vae

from .detail import ASPECTS, MAX_UPSCALE, _grow
from .passes import make_conditioning, sample_pass

FEATHER = 0.012  # soft edge, as a fraction of the picture's long side
GROW = 0.02  # how far the refined area reaches past the region (hair and hands that stick out)
CONTEXT = 0.05  # context around the region in the crop


def _to_output(job, src, img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """An image given in the coordinates of the job's input (or output) → the finished image's coordinates."""
    resample = Image.BILINEAR if img.mode == "L" else Image.LANCZOS
    if src is not None and src.box is None:
        return images.resize(img, size[0], size[1], job.resize_mode)
    return img.resize(size, resample)


def is_background(mask: Image.Image) -> bool:
    box = mask.point(lambda v: 255 if v > 127 else 0).getbbox()
    if box is None:
        return True
    return (box[2] - box[0]) * (box[3] - box[1]) / (mask.width * mask.height) > regions.RegionalAttention.BACKGROUND_BOX


def run(engine, job, progress, image: Image.Image, seed: int, *, src, cache: dict, limit_mask: Optional[Image.Image] = None) -> tuple[Image.Image, int]:
    """Refine every (non-background) region of `image`. Returns the image and how many regions were refined."""
    W, H = image.size
    long_side = max(W, H)
    grow, feather, context = round(long_side * GROW), max(2, round(long_side * FEATHER)), round(long_side * CONTEXT)
    target_area = min(job.width * job.height, 1024 * 1024)
    steps_run = sampling.run_steps(job.steps, job.region_refine)
    from .generate import guide_map

    sw, sh = images.snap(job.width), images.snap(job.height)
    guide_images = [(g, _to_output(job, src, guide_map(job, g, src, sw, sh), (W, H))) for g in job.guides]
    known = guides.models() if guide_images else {}
    lm = np.asarray(limit_mask.resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0 if limit_mask is not None else None

    done = 0
    for k, r in enumerate(job.regions):  # bottom layer first
        mask = _to_output(job, src, images.load_mask(r.mask), (W, H))
        if is_background(mask):
            continue
        area = _grow(mask.point(lambda v: 255 if v > 127 else 0), grow)
        if lm is not None:  # inpainting: only inside the area being regenerated
            area = Image.fromarray((np.asarray(area) * lm).astype(np.uint8))
        if area.getbbox() is None:
            continue

        x1, y1, x2, y2 = area.getbbox()
        aspect = min(ASPECTS, key=lambda a: abs(math.log(a) - math.log(max(1, x2 - x1) / max(1, y2 - y1))))
        box = images.crop_region(area, context + 2 * feather, aspect)
        bw, bh = box[2] - box[0], box[3] - box[1]
        scale = max(1.0, min(MAX_UPSCALE, math.sqrt(target_area / max(1, bw * bh))))
        pw, ph = images.snap(bw * scale), images.snap(bh * scale)
        soft = images.blur_mask(area, feather)
        crop = image.crop(box).resize((pw, ph), Image.LANCZOS)
        crop_mask = soft.crop(box).resize((pw, ph), Image.BILINEAR)

        guide_set = None
        if guide_images:
            items = [guides.Guide(known[g.model], img.crop(box), g.weight, g.start, g.end, g.mode) for g, img in guide_images if g.model in known]
            guide_set = guides.GuideSet(engine, items) if items else None

        engine.set_shift(shift=job.shift, width=pw, height=ph)
        cond = make_conditioning(engine, job, steps_run, pw, ph, cache, prompt=r.prompt, negative=r.negative.strip() or job.negative, switch=None)
        latent = vae.encode(engine, crop)
        out = sample_pass(
            engine, job, progress, [seed + 5000 + k], vae.latent_shape(engine, pw, ph), cond, None,
            steps=job.steps, cfg=job.cfg, init_latent=latent, denoise=job.region_refine,
            inpaint=sampling.Inpaint(images.latent_mask(crop_mask, pw // 8, ph // 8), differential=True, keep=latent),
            on_step=lambda i, n, d: None, guides=guide_set, guide_mask=crop_mask if guide_set is not None else None,
        )
        generated = vae.decode(engine, out)[0].resize((bw, bh), Image.LANCZOS)
        image = image.copy()
        image.paste(images.composite(image.crop(box), generated, soft.crop(box)), box[:2])
        done += 1
    return image, done
