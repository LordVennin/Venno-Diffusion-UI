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

from venno.engine import guides, images, prompts, regions, sampling, vae

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


def areas(job, src, size: tuple[int, int], *, backgrounds: bool = False, limit_mask: Optional[Image.Image] = None) -> list[tuple[int, Image.Image]]:
    """(region index, area to re-render) for each region the pass works on, bottom layer first."""
    W, H = size
    grow = round(max(W, H) * GROW)
    lm = np.asarray(limit_mask.resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0 if limit_mask is not None else None
    out = []
    for k, r in enumerate(job.regions):
        mask = _to_output(job, src, images.load_mask(r.mask), (W, H))
        if not backgrounds and is_background(mask):
            continue
        area = _grow(mask.point(lambda v: 255 if v > 127 else 0), grow)
        if lm is not None:  # inpainting: only inside the area being regenerated
            area = Image.fromarray((np.asarray(area) * lm).astype(np.uint8))
        if area.getbbox() is not None:
            out.append((k, area))
    return out


def coverage(found: list[tuple[int, Image.Image]], size: tuple[int, int]) -> Optional[Image.Image]:
    """Everything the pass re-renders, as one mask (None when nothing)."""
    if not found:
        return None
    out = np.zeros((size[1], size[0]), np.uint8)
    for _, area in found:
        out = np.maximum(out, np.asarray(area))
    return Image.fromarray(out)


def _steering(engine, r, cond, steps_run: int, mask, width: int, height: int, base: float) -> regions.RegionalAttention:
    """Regional attention for one region inside its crop (mask: (H, W) tensor at the crop's processing size)."""
    plan = prompts.PromptPlan(steps_run, r.prompt, allow_and=False)
    texts = {t for step in range(1, steps_run + 1) for t, _ in plan.parts(step)}
    region = regions.Region(plan=plan, contexts={t: cond.tensor(engine, t, False, width, height) for t in texts}, mask=mask, strength=r.strength)
    if r.negative.strip():
        region.negative_plan = prompts.PromptPlan(steps_run, r.negative, allow_and=False)
        texts = {t for step in range(1, steps_run + 1) for t, _ in region.negative_plan.parts(step)}
        region.negative_contexts = {t: cond.tensor(engine, t, True, width, height) for t in texts}
    return regions.RegionalAttention([region], base_strength=base)


def run(
    engine, job, progress, image: Image.Image, seed: int, *, src, cache: dict, limit_mask: Optional[Image.Image] = None,
    denoise: Optional[float] = None, backgrounds: bool = False, context_share: Optional[float] = None, steer: Optional[float] = None, on_step=None,
) -> tuple[Image.Image, int]:
    """Refine every (non-background) region of `image`. Returns the image and how many regions were refined.

    denoise: instead of job.region_refine; backgrounds: also regions spanning most of the picture;
    context_share: surrounding picture in each crop as a share of the region's size (like the Canvas's
    Context) instead of the small fixed margin. Drawing a region anew needs it: with little of the picture
    around it, the model composes a whole new little picture from the prompt inside the region;
    steer: the crop is prompted with the main prompt and the region's prompt steers the region's area through
    regional attention (as in a normal generation), with this much of the main prompt inside the region,
    instead of the region's prompt for the whole crop. Drawing regions anew follows their prompts far better
    this way, best with 0 (the main prompt then only describes the surroundings);
    on_step(step, steps): progress over all the regions."""
    W, H = image.size
    long_side = max(W, H)
    feather, context = max(2, round(long_side * FEATHER)), round(long_side * CONTEXT)
    target_area = min(job.width * job.height, 1024 * 1024)
    denoise = job.region_refine if denoise is None else denoise
    steps_run = sampling.run_steps(job.steps, denoise)
    from .generate import guide_map

    sw, sh = images.snap(job.width), images.snap(job.height)
    guide_images = [(g, _to_output(job, src, guide_map(job, g, src, sw, sh), (W, H))) for g in job.guides]
    known = guides.models() if guide_images else {}

    found = areas(job, src, (W, H), backgrounds=backgrounds, limit_mask=limit_mask)
    done = 0
    for n, (k, area) in enumerate(found):
        r = job.regions[k]
        x1, y1, x2, y2 = area.getbbox()
        aspect = min(ASPECTS, key=lambda a: abs(math.log(a) - math.log(max(1, x2 - x1) / max(1, y2 - y1))))
        pad = context
        if context_share is not None:
            pad = max(pad, round(max(x2 - x1, y2 - y1) * context_share), round(long_side * context_share * 0.25))
        box = images.crop_region(area, pad + 2 * feather, aspect)
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
        regional = None
        if steer is not None:
            cond = make_conditioning(engine, job, steps_run, pw, ph, cache, switch=None)
            regional = _steering(engine, r, cond, steps_run, images.to_tensor(crop_mask)[0], pw, ph, steer)
        else:
            cond = make_conditioning(engine, job, steps_run, pw, ph, cache, prompt=r.prompt, negative=r.negative.strip() or job.negative, switch=None)
        latent = vae.encode(engine, crop)
        out = sample_pass(
            engine, job, progress, [seed + 5000 + k], vae.latent_shape(engine, pw, ph), cond, regional,
            steps=job.steps, cfg=job.cfg, init_latent=latent, denoise=denoise,
            inpaint=sampling.Inpaint(images.latent_mask(crop_mask, pw // 8, ph // 8), differential=True, keep=latent),
            on_step=lambda i, _, d, n=n: on_step and on_step(n * steps_run + i, len(found) * steps_run),
            guides=guide_set, guide_mask=crop_mask if guide_set is not None else None,
        )
        generated = vae.decode(engine, out)[0].resize((bw, bh), Image.LANCZOS)
        image = image.copy()
        image.paste(images.composite(image.crop(box), generated, soft.crop(box)), box[:2])
        done += 1
    return image, done
