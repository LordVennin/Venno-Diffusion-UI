"""The input of an image to image / inpaint / outpaint job, as in Venno (venno/pipeline/generate.py): the picture
(and mask) at processing size, the latents the sampler starts from and holds the unmasked area to, and the
result pasted back over the input. The image and mask helpers are Venno's own (venno/engine/images.py), so both
Studios treat masks, "only masked" crops and the masked-content fills the same way.

MOSS-Image's latents are 16 px a cell (Anima's are 8), so the latent mask is coarser; the pixel-space paste at
the end (`finish`) is what makes mask edges exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image

from venno.engine import images


@dataclass
class Source:
    image: Image.Image  # width × height, what the model sees
    mask: Optional[Image.Image] = None  # feathered, white = regenerate (None = whole image)
    hard_mask: Optional[Image.Image] = None  # un-feathered, for the colour fill
    original: Optional[Image.Image] = None  # "only masked": the full-size input the result is pasted back into
    box: Optional[tuple[int, int, int, int]] = None  # "only masked": where `image` came from in `original`


def prepare(job, width: int, height: int) -> Source:
    init = images.load_rgb(job.init_image)
    mask = images.load_mask(job.mask) if job.mask else None
    if mask is not None:
        if mask.size != init.size:
            mask = mask.resize(init.size, Image.BILINEAR)
        if not images.has_content(mask):
            mask = None
    if mask is not None and job.only_masked:
        # work on a crop around the mask at full model resolution, paste the result back at native size
        box = images.crop_region(mask, job.only_masked_padding + 2 * job.mask_blur, width / height)
        soft = images.blur_mask(mask, job.mask_blur)
        return Source(
            image=init.crop(box).resize((width, height), Image.LANCZOS),
            mask=soft.crop(box).resize((width, height), Image.BILINEAR),
            hard_mask=mask.crop(box).resize((width, height), Image.BILINEAR),
            original=init,
            box=box,
        )
    image = images.resize(init, width, height, job.resize_mode)
    if mask is None:
        return Source(image=image)
    hard = images.resize(mask, width, height, job.resize_mode)
    return Source(image=image, mask=images.blur_mask(hard, job.mask_blur), hard_mask=hard)


def denoise(job, src: Source) -> float:
    """"Latent noise" discards what is under the mask, and the unmasked area is held to the input at every step
    anyway, so the masked area is regenerated over the full schedule (a partial one can't resolve pure noise)."""
    return 1.0 if src.mask is not None and job.fill == "latent_noise" else job.strength


def latents(engine, job, src: Source) -> dict:
    """What Engine.generate starts from: {latent, keep, mask, soft, denoise}; mask is (1, 1, h, w), 1 = regenerate."""
    keep = engine.encode_image(src.image)
    init, mask = keep, None
    if src.mask is not None:
        mask = images.latent_mask(src.mask, keep.shape[-1], keep.shape[-2])[:, :, 0].to(keep)  # Venno's has a time axis
        if not job.soft_inpaint:
            mask = (mask >= 0.5).to(keep)
        if job.fill == "fill":  # start the masked area from colours of its surroundings
            init = engine.encode_image(images.fill_masked(src.image, src.hard_mask))
        elif job.fill == "latent_nothing":
            init = keep * (1 - mask)
    return {"latent": init, "keep": keep, "mask": mask, "soft": job.soft_inpaint, "denoise": denoise(job, src)}


def out_size(job, src: Optional[Source]) -> tuple[int, int]:
    return src.original.size if src is not None and src.box is not None else (job.width, job.height)


def finish(src: Optional[Source], generated: Image.Image) -> Image.Image:
    """Put a generated picture back together with its input: untouched pixels stay exactly as they were."""
    if src is None:
        return generated
    if src.box is not None:
        w, h = src.box[2] - src.box[0], src.box[3] - src.box[1]
        region = images.composite(src.original.crop(src.box), generated.resize((w, h), Image.LANCZOS), src.mask.resize((w, h), Image.BILINEAR))
        full = src.original.copy()
        full.paste(region, src.box[:2])
        return full
    if src.mask is not None:
        return images.composite(src.image, generated, src.mask)
    return generated
