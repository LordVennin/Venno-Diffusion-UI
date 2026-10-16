"""Text to image, image to image and inpainting on the Venno engine."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from PIL import Image

from venno.engine import conditioning, images, lora, prompts, regions, rng, sampling, vae
from venno.engine.models import manager

from . import metadata


@dataclass
class Progress:
    """Callbacks the job runner hands to a pipeline."""

    on_step: Callable[[int, int, int, int], None] = lambda step, steps, run, runs: None  # sampling step within run
    on_preview: Callable[[bytes], None] = lambda jpeg: None
    is_cancelled: Callable[[], bool] = lambda: False
    preview_every: int = 1


@dataclass
class Result:
    images: list[bytes] = field(default_factory=list)  # PNG
    seeds: list[int] = field(default_factory=list)
    infotexts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cancelled: bool = False


@dataclass
class Source:
    """The input of an image to image job, prepared at processing size."""

    image: Image.Image  # width × height, what the model sees
    mask: Optional[Image.Image] = None  # feathered, white = regenerate (None = whole image)
    hard_mask: Optional[Image.Image] = None  # un-feathered, for the colour fill
    original: Optional[Image.Image] = None  # "only masked": the full-size input the result is pasted back into
    box: Optional[tuple[int, int, int, int]] = None  # "only masked": where `image` came from in `original`


def prepare_source(job, width: int, height: int) -> Source:
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


def finish(src: Optional[Source], generated: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Put a generated image back together with its input."""
    if src is None:
        return generated if generated.size == size else generated.resize(size, Image.LANCZOS)
    out = generated
    if src.mask is not None and src.box is None:
        out = images.composite(src.image, generated, src.mask)  # untouched pixels stay exactly as they were
    if src.box is not None:
        w, h = src.box[2] - src.box[0], src.box[3] - src.box[1]
        region = images.composite(src.original.crop(src.box), generated.resize((w, h), Image.LANCZOS), src.mask.resize((w, h), Image.BILINEAR))
        full = src.original.copy()
        full.paste(region, src.box[:2])
        return full
    return out if out.size == size else out.resize(size, Image.LANCZOS)


def region_mask(job, src: Optional[Source], data: str, width: int, height: int):
    """A region mask in the coordinates of the job's input (or output, for text to image) → (H, W) tensor."""
    mask = images.load_mask(data)
    if src is not None and src.box is not None:
        mask = mask.resize(src.original.size, Image.BILINEAR).crop(src.box)
        mask = mask.resize((width, height), Image.BILINEAR)
    elif src is not None:
        mask = images.resize(mask, width, height, job.resize_mode)
    else:
        mask = mask.resize((width, height), Image.BILINEAR)
    return images.to_tensor(mask)[0]


def run(job, progress: Progress) -> Result:
    W, H = images.snap(job.width), images.snap(job.height)
    engine = manager.get(job.model.checkpoint, [job.model.text_encoder, job.model.vae])
    engine.set_shift(shift=job.shift, width=W, height=H)
    loras = [(lr.name, lr.weight) for lr in job.loras]
    result = Result(warnings=lora.apply(engine, loras))  # before encoding: LoRAs can patch the text encoder too

    base_seed = rng.resolve_seed(job.seed)
    shape = vae.latent_shape(engine, W, H)

    # everything below is the same for every run of the job: prepare once
    src = prepare_source(job, W, H) if job.mode == "img2img" and job.init_image else None
    init_latent = keep_latent = latent_mask = None
    if src is not None:
        keep_latent = vae.encode(engine, src.image)
        init_latent = keep_latent
        if src.mask is not None:
            latent_mask = images.latent_mask(src.mask, shape[-1], shape[-2])
            if not job.soft_inpaint:
                latent_mask = (latent_mask >= 0.5).float()
            if job.fill == "fill":  # start the masked area from colours of its surroundings
                init_latent = vae.encode(engine, images.fill_masked(src.image, src.hard_mask))
            elif job.fill == "latent_nothing":
                init_latent = keep_latent * (1 - latent_mask)
    denoise = 1.0
    if src is not None:
        # "latent noise" discards what is under the mask, and the unmasked area is held to the input at every
        # step anyway, so the masked area is regenerated over the full schedule (a partial one can't resolve noise)
        denoise = 1.0 if latent_mask is not None and job.fill == "latent_noise" else job.strength
    out_size = src.original.size if src is not None and src.box is not None else (job.width, job.height)

    # prompts per step: scheduling / AND / the "then switch to" prompt, over the steps actually sampled
    steps_run = sampling.run_steps(job.steps, denoise if src is not None else 1.0)
    positive = prompts.PromptPlan(steps_run, job.prompt, switch=job.switch_prompt or None, switch_at=job.switch_at)
    negative = prompts.PromptPlan(steps_run, job.negative, allow_and=False)
    cond = conditioning.PromptConditioning(engine, positive, negative, width=W, height=H)

    regional = None
    if job.regions:
        region_list = []
        for r in job.regions:
            plan = prompts.PromptPlan(steps_run, r.prompt, allow_and=False)
            texts = {t for step in range(1, steps_run + 1) for t, _ in plan.parts(step)}
            contexts = {t: cond.tensor(engine, t, False, W, H) for t in texts}
            region_list.append(regions.Region(plan=plan, contexts=contexts, mask=region_mask(job, src, r.mask, W, H), strength=r.strength))
        regional = regions.RegionalAttention(region_list, base_strength=job.region_base)

    params = {
        "Steps": job.steps,
        "Sampler": job.sampler,
        "Schedule type": job.scheduler,
        "CFG scale": job.cfg,
        "Shift": job.shift,
        "Seed": None,  # filled per image
        "Size": f"{out_size[0]}x{out_size[1]}",
        "Model": manager.info.get("name"),
    }
    if src is not None:
        params["Denoising strength"] = denoise
        if src.mask is not None:
            params["Mask blur"] = job.mask_blur
            params["Masked content"] = job.fill
            params["Inpaint area"] = "Only masked" if src.box else "Whole picture"
            params["Soft inpainting"] = job.soft_inpaint
    if job.switch_prompt:
        params["Prompt switch"] = job.switch_prompt
        params["Prompt switch at"] = job.switch_at
    if regional is not None:
        params["Regions"] = len(job.regions)
        params["Region base"] = job.region_base
        if job.region_end < 1.0:
            params["Region end"] = job.region_end
        for k, r in enumerate(job.regions, start=1):
            params[f"Region {k}"] = r.prompt
    for k, m in enumerate(manager.info.get("modules", []), start=1):
        params[f"Module {k}"] = m.rsplit(".", 1)[0]
    params["Version"] = metadata.VERSION

    for run_index in range(job.count):
        seeds = [base_seed + run_index * job.batch + i for i in range(job.batch)]
        last_preview = [0.0]

        def on_step(i, n, denoised, run_index=run_index):
            progress.on_step(i, n, run_index, job.count)
            now = time.time()
            if progress.preview_every and i % progress.preview_every == 0 and i < n and now - last_preview[0] > 0.25:
                last_preview[0] = now
                img = vae.preview(engine, denoised)
                if img is not None:
                    buf = io.BytesIO()
                    img.convert("RGB").save(buf, format="JPEG", quality=80)
                    progress.on_preview(buf.getvalue())

        try:
            latents = sampling.sample(
                sampling.SampleRequest(
                    engine=engine,
                    noise=rng.ImageNoise(seeds, shape),
                    conditioning=cond,
                    cfg=job.cfg,
                    sampler=job.sampler,
                    scheduler=job.scheduler,
                    steps=job.steps,
                    seeds=seeds,
                    init_latent=init_latent,
                    denoise=denoise,
                    inpaint=sampling.Inpaint(latent_mask, differential=job.soft_inpaint, keep=keep_latent) if latent_mask is not None else None,
                    regional=regional,
                    regional_end=job.region_end,
                    on_step=on_step,
                    is_cancelled=progress.is_cancelled,
                )
            )
        except sampling.Cancelled:
            result.cancelled = True
            break

        for seed, generated in zip(seeds, vae.decode(engine, latents)):
            image = finish(src, generated, out_size)
            info = metadata.infotext(metadata.with_lora_tags(job.prompt, loras), job.negative, {**params, "Seed": seed})
            if job.save:
                metadata.save(image, info, job.mode, seed)
            result.images.append(metadata.png_bytes(image, info))
            result.seeds.append(seed)
            result.infotexts.append(info)
    return result
