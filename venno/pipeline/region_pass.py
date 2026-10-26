"""Passes over single regions of a finished picture: the region pass (refine each subject region) and the
redraw of regions that came out empty.

With several characters in one picture each gets only a slice of the pixels, and whatever bled between
them is baked in. The **region pass** crops each subject region (grown a little, with context around it),
renders the crop at up to ~1 MP with only that region's prompt and negative, at a moderate denoise, and
blends it back with a soft edge (differential soft inpainting, like the detailer). Regions run one at a
time from the bottom layer up, each on the result of the previous one, so neighbours stay consistent;
scenery regions are skipped. Sketch guides are cropped along with the region.

The **redraw** is the fallback for a subject that the generation left out: the detailer's face and person
detectors say which character regions hold no character (`missing`), and each of those is inpainted anew
on a crop with plenty of the picture around it, prompted like the generation itself (the main prompt, with
every region's prompt steering its area and the region's own pass inside the crop), at the crop's upscaled
size. A region that fails at the picture's resolution usually comes out right at that scale.
"""

from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np
from PIL import Image

from venno.engine import detect, guides, images, regions, sampling, vae

from .detail import ASPECTS, MAX_UPSCALE, _grow
from .passes import make_conditioning, sample_pass

FEATHER = 0.012  # soft edge, as a fraction of the picture's long side
GROW = 0.02  # how far the refined area reaches past the region (hair and hands that stick out)
CONTEXT = 0.05  # context around the region in the crop (region pass)
REDRAW_CONTEXT = 0.5  # context around the region as a share of its size (redraw: the subject must fit the scene)
FACE_MODEL, PERSON_MODEL = "face_yolov8n.pt", "person_yolov8n-seg.pt"
CHARACTER = re.compile(r"\b[12]?(girl|boy|other|woman|man|person|character|witch|kobold|anthro|furry)s?\b", re.I)


def _to_output(job, src, img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """An image given in the coordinates of the job's input (or output) → the finished image's coordinates."""
    resample = Image.BILINEAR if img.mode == "L" else Image.LANCZOS
    if src is not None and src.box is None:
        return images.resize(img, size[0], size[1], job.resize_mode)
    return img.resize(size, resample)


def is_scenery(mask: Image.Image, kind: str = "auto") -> bool:
    """A scenery region: declared so, or (auto) its box spans most of the picture."""
    if kind != "auto":
        return kind == "scenery"
    box = mask.point(lambda v: 255 if v > 127 else 0).getbbox()
    if box is None:
        return True
    return (box[2] - box[0]) * (box[3] - box[1]) / (mask.width * mask.height) > regions.RegionalAttention.BACKGROUND_BOX


def _masks(job, src, size: tuple[int, int]) -> list[tuple[int, Image.Image]]:
    """(region index, hard mask in the finished image's coordinates) for every subject region."""
    out = []
    for k, r in enumerate(job.regions):
        mask = _to_output(job, src, images.load_mask(r.mask), size)
        if not is_scenery(mask, r.kind):
            out.append((k, mask.point(lambda v: 255 if v > 127 else 0)))
    return out


def areas(job, src, size: tuple[int, int], *, limit_mask: Optional[Image.Image] = None) -> list[tuple[int, Image.Image]]:
    """(region index, area to re-render) for each subject region, bottom layer first."""
    W, H = size
    grow = round(max(W, H) * GROW)
    lm = np.asarray(limit_mask.resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0 if limit_mask is not None else None
    out = []
    for k, mask in _masks(job, src, size):
        area = _grow(mask, grow)
        if lm is not None:  # inpainting: only inside the area being regenerated
            area = Image.fromarray((np.asarray(area) * lm).astype(np.uint8))
        if area.getbbox() is not None:
            out.append((k, area))
    return out


# ---------------------------------------------------------------- which character regions hold a character


def characters(image: Image.Image, boxes: dict[int, tuple[int, int, int, int]], faces=None, persons=None) -> tuple[set[int], int]:
    """Which of the `boxes` (region → x1, y1, x2, y2) hold a character, and how many characters lie outside all of
    them. A box counts when a face lies in it (grown by 5% of the picture: regions are rough placements) or at
    least 30% of a detected person's pixels do, so one big figure sprawling over two boxes counts for one at most.
    faces / persons: detections to use instead of running the detectors (tests)."""
    W, H = image.size
    grow = round(0.05 * max(W, H))
    faces = detect.detect(FACE_MODEL, image, 0.3) if faces is None else faces
    persons = detect.detect(PERSON_MODEL, image, 0.2) if persons is None else persons
    hit, extra = set(), 0
    for d in faces:
        cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
        inside = [k for k, b in boxes.items() if b[0] - grow <= cx <= b[2] + grow and b[1] - grow <= cy <= b[3] + grow]
        if inside:
            hit.add(min(inside, key=lambda k: k in hit))  # a box already hit gives way to another one containing the face
        else:
            extra += 1
    for d in persons:
        m = np.asarray(d.mask.resize((W, H))) > 127
        total = m.sum()
        if total < 0.005 * W * H or not boxes:
            continue
        best = max(boxes, key=lambda k: m[boxes[k][1]:boxes[k][3] + 1, boxes[k][0]:boxes[k][2] + 1].sum())
        b = boxes[best]
        if m[b[1]:b[3] + 1, b[0]:b[2] + 1].sum() >= 0.3 * total:
            hit.add(best)
    return hit, extra


def missing(job, src, image: Image.Image) -> list[int]:
    """Subject regions whose prompt names a character but in which the detectors find none."""
    boxes = {k: mask.getbbox() for k, mask in _masks(job, src, image.size) if CHARACTER.search(job.regions[k].prompt)}
    boxes = {k: b for k, b in boxes.items() if b is not None}
    if not boxes:
        return []
    hit, _ = characters(image, boxes)
    return [k for k in boxes if k not in hit]


# ---------------------------------------------------------------- the passes


def _crop_pass(engine, job, progress, image, seed, *, src, cache, k, area, pad, cond_for, denoise, regional_for=None, on_step=None):
    """Re-render `area` of `image` on a crop (padded by `pad` px, snapped to a trained aspect, up to ~1 MP) and
    blend it back with a soft edge. cond_for(pw, ph) gives the crop's conditioning, regional_for(pw, ph, box)
    the regional attention inside it (None: the crop's prompt alone)."""
    W, H = image.size
    long_side = max(W, H)
    feather = max(2, round(long_side * FEATHER))
    target_area = min(job.width * job.height, 1024 * 1024)
    x1, y1, x2, y2 = area.getbbox()
    aspect = min(ASPECTS, key=lambda a: abs(math.log(a) - math.log(max(1, x2 - x1) / max(1, y2 - y1))))
    box = images.crop_region(area, pad + 2 * feather, aspect)
    bw, bh = box[2] - box[0], box[3] - box[1]
    scale = max(1.0, min(MAX_UPSCALE, math.sqrt(target_area / max(1, bw * bh))))
    pw, ph = images.snap(bw * scale), images.snap(bh * scale)
    soft = images.blur_mask(area, feather)
    crop = image.crop(box).resize((pw, ph), Image.LANCZOS)
    crop_mask = soft.crop(box).resize((pw, ph), Image.BILINEAR)

    guide_set = None
    if job.guides:
        from .generate import guide_map

        sw, sh = images.snap(job.width), images.snap(job.height)
        known = guides.models()
        items = [guides.Guide(known[g.model], _to_output(job, src, guide_map(job, g, src, sw, sh), (W, H)).crop(box), g.weight, g.start, g.end, g.mode) for g in job.guides if g.model in known]
        guide_set = guides.GuideSet(engine, items) if items else None

    engine.set_shift(shift=job.shift, width=pw, height=ph)
    cond = cond_for(pw, ph)
    regional = regional_for(pw, ph, box) if regional_for is not None else None
    latent = vae.encode(engine, crop)
    out = sample_pass(
        engine, job, progress, [seed + 5000 + k], vae.latent_shape(engine, pw, ph), cond, regional,
        steps=job.steps, cfg=job.cfg, init_latent=latent, denoise=denoise,
        inpaint=sampling.Inpaint(images.latent_mask(crop_mask, pw // 8, ph // 8), differential=True, keep=latent),
        on_step=on_step, guides=guide_set, guide_mask=crop_mask if guide_set is not None else None,
    )
    generated = vae.decode(engine, out)[0].resize((bw, bh), Image.LANCZOS)
    image = image.copy()
    image.paste(images.composite(image.crop(box), generated, soft.crop(box)), box[:2])
    return image


def run(engine, job, progress, image: Image.Image, seed: int, *, src, cache: dict, limit_mask: Optional[Image.Image] = None, on_step=None) -> tuple[Image.Image, int]:
    """Refine every subject region of `image` at job.region_refine. Returns the image and how many regions were refined.
    on_step(step, steps): progress over all the regions."""
    denoise = job.region_refine
    steps_run = sampling.run_steps(job.steps, denoise)
    context = round(max(image.size) * CONTEXT)
    found = areas(job, src, image.size, limit_mask=limit_mask)
    for n, (k, area) in enumerate(found):
        r = job.regions[k]
        image = _crop_pass(
            engine, job, progress, image, seed, src=src, cache=cache, k=k, area=area, pad=context, denoise=denoise,
            cond_for=lambda pw, ph: make_conditioning(engine, job, steps_run, pw, ph, cache, prompt=r.prompt, negative=r.negative.strip() or job.negative, switch=None),
            on_step=lambda i, _, d, n=n: on_step and on_step(n * steps_run + i, len(found) * steps_run),
        )
    return image, len(found)


def redraw(engine, job, progress, image: Image.Image, seed: int, *, src, cache: dict, which: list[int], on_step=None) -> Image.Image:
    """Draw the regions `which` anew: each on a crop with REDRAW_CONTEXT of the picture around it, from noise inside
    the region (the rest of the crop is kept), prompted as the generation was: the main prompt, every region's
    prompt steering its area and the region's own pass, at the crop's upscaled size."""
    from .generate import make_regional

    W, H = image.size
    grow = round(max(W, H) * GROW)
    masks = dict(_masks(job, src, (W, H)))
    for n, k in enumerate(which):
        area = _grow(masks[k], grow)
        x1, y1, x2, y2 = area.getbbox()
        pad = max(round(max(x2 - x1, y2 - y1) * REDRAW_CONTEXT), round(max(W, H) * REDRAW_CONTEXT * 0.25))

        crop_cond = {}

        def cond_for(pw, ph):
            crop_cond["cond"] = make_conditioning(engine, job, job.steps, pw, ph, cache)
            return crop_cond["cond"]

        def regional_for(pw, ph, box):  # every region's mask, cropped like the picture
            cropped = [images.to_tensor(_to_output(job, src, images.load_mask(r.mask), (W, H)).crop(box).resize((pw, ph), Image.BILINEAR))[0] for r in job.regions]
            return make_regional(engine, job, None, job.steps, pw, ph, crop_cond["cond"], masks=cropped)

        image = _crop_pass(
            engine, job, progress, image, seed + 2000, src=src, cache=cache, k=k, area=area, pad=pad, denoise=1.0,
            cond_for=cond_for, regional_for=regional_for,
            on_step=lambda i, _, d, n=n: on_step and on_step(n * job.steps + i, len(which) * job.steps),
        )
    return image
