"""Text to image, image to image and inpainting on the Venno engine."""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from venno.engine import conditioning, guides, images, lora, preprocess, prompts, reference, regions, rng, sampling, upscale, vae, wildcards
from venno.engine.models import manager

from . import detail, metadata, region_pass
from .passes import make_conditioning, sample_pass


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
    detections: list = field(default_factory=list)  # per image: ADetailer detection preview (PNG) or None
    detection_notes: list[str] = field(default_factory=list)  # per image: what the detailer found ("" without one)


@dataclass
class Source:
    """The input of an image to image job, prepared at processing size."""

    image: Image.Image  # width × height, what the model sees
    mask: Optional[Image.Image] = None  # feathered, white = regenerate (None = whole image)
    hard_mask: Optional[Image.Image] = None  # un-feathered, for the colour fill
    original: Optional[Image.Image] = None  # "only masked": the full-size input the result is pasted back into
    box: Optional[tuple[int, int, int, int]] = None  # "only masked": where `image` came from in `original`
    full_mask: Optional[Image.Image] = None  # the mask in output coordinates (limits the detailer)


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
            full_mask=mask,
        )

    image = images.resize(init, width, height, job.resize_mode)
    if mask is None:
        return Source(image=image)
    hard = images.resize(mask, width, height, job.resize_mode)
    return Source(image=image, mask=images.blur_mask(hard, job.mask_blur), hard_mask=hard, full_mask=hard)


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


def fit_to_source(job, src: Optional[Source], img: Image.Image, width: int, height: int) -> Image.Image:
    """An image given in the coordinates of the job's input (or output, for text to image) → processing size."""
    resample = Image.BILINEAR if img.mode == "L" else Image.LANCZOS
    if src is not None and src.box is not None:
        return img.resize(src.original.size, resample).crop(src.box).resize((width, height), resample)
    if src is not None:
        return images.resize(img, width, height, job.resize_mode)
    return img.resize((width, height), resample)


def region_mask(job, src: Optional[Source], data: str, width: int, height: int):
    """A region mask → (H, W) tensor at processing size."""
    return images.to_tensor(fit_to_source(job, src, images.load_mask(data), width, height))[0]


def guide_map(job, g, src: Optional[Source], width: int, height: int) -> Image.Image:
    """A guide's map in the coordinates of the job's input (or output, for text to image): an uploaded picture
    is first fitted to that frame with its own resize mode, then run through its preprocessor."""
    img = images.load_rgb(g.image)
    if g.resize_mode is not None:
        frame = src.original.size if src is not None and src.box is not None else (width, height)
        img = images.resize(img, frame[0], frame[1], g.resize_mode)
    if g.preprocessor != "none":
        img = preprocess.run(g.preprocessor, img, g.model, g.canny_low, g.canny_high)
    return img


def make_guides(engine, job, src: Optional[Source], width: int, height: int):
    if not job.guides:
        return None
    known = guides.models()
    items = []
    for g in job.guides:
        if g.model not in known:
            raise ValueError(f'Guide model "{g.model}" not found')
        image = fit_to_source(job, src, guide_map(job, g, src, width, height), width, height)
        items.append(guides.Guide(known[g.model], image, g.weight, g.start, g.end, g.mode))
    return guides.GuideSet(engine, items)


def uses_wildcards(job) -> bool:
    texts = [job.prompt, job.negative, job.switch_prompt, *job.frame_prompts, *(r.prompt for r in job.regions), *(r.negative for r in job.regions), *(d.prompt for d in job.detailer), *(d.negative for d in job.detailer)]
    return any(wildcards.has_syntax(t) for t in texts)


def run_each(job, progress: Progress) -> Result:
    """Wildcard jobs: every image gets its own random choices (from its own seed), so images run one by one."""
    base_seed = rng.resolve_seed(job.seed)
    total = job.count * job.batch
    result = Result()
    for k in range(total):
        seed = base_seed + k
        pick = lambda text, salt: wildcards.for_seed(text, seed, salt) if text else text
        prompt, picked_loras = conditioning.extract_lora_tags(pick(job.prompt, "prompt"))
        negative, picked_neg = conditioning.extract_lora_tags(pick(job.negative, "negative"))
        have = {lr.name for lr in job.loras}
        lora_ref = type(job).model_fields["loras"].annotation.__args__[0]  # the job schema's LoRA entry type
        extra_loras = [lora_ref(name=n, weight=w) for n, w in picked_loras + picked_neg if n not in have]
        sub = job.model_copy(
            update=dict(
                prompt=prompt,
                negative=negative,
                loras=list(job.loras) + extra_loras,
                switch_prompt=pick(job.switch_prompt, "switch"),
                frame_prompts=[pick(fp, f"frame{i}") for i, fp in enumerate(job.frame_prompts)],
                regions=[r.model_copy(update={"prompt": pick(r.prompt, f"region{i}"), "negative": pick(r.negative, f"region{i}n")}) for i, r in enumerate(job.regions)],
                detailer=[d.model_copy(update={"prompt": pick(d.prompt, f"detail{i}"), "negative": pick(d.negative, f"detail{i}n")}) for i, d in enumerate(job.detailer)],
                seed=seed,
                batch=1,
                count=1,
            )
        )
        sub_progress = Progress(
            on_step=lambda step, steps, run, runs, k=k: progress.on_step(step, steps, k, total),
            on_preview=progress.on_preview,
            is_cancelled=progress.is_cancelled,
            preview_every=progress.preview_every,
        )
        extra = {"Template": job.prompt} if wildcards.has_syntax(job.prompt) else {}
        if wildcards.has_syntax(job.negative):
            extra["Negative Template"] = job.negative
        part = run(sub, sub_progress, extra_params=extra)
        result.images += part.images
        result.seeds += part.seeds
        result.infotexts += part.infotexts
        result.detections += part.detections
        result.detection_notes += part.detection_notes
        result.warnings += [w for w in part.warnings if w not in result.warnings]
        if part.cancelled:
            result.cancelled = True
            break
    return result


def run(job, progress: Progress, extra_params: Optional[dict] = None) -> Result:
    if getattr(job, "grid", None) is not None:
        from . import grid

        return grid.run(job, progress, run)
    if uses_wildcards(job):
        return run_each(job, progress)
    W, H = images.snap(job.width), images.snap(job.height)
    engine = manager.get(job.model.checkpoint, [job.model.text_encoder, job.model.vae])
    if getattr(engine, "venno_anima38", None) is not None:
        engine.venno_anima38.reset()  # prompts of earlier jobs are no longer needed
    engine.set_shift(shift=job.shift, width=W, height=H)
    loras = [(lr.name, lr.weight) for lr in job.loras]
    result = Result(warnings=lora.apply(engine, loras))  # before encoding: LoRAs can patch the text encoder too

    base_seed = rng.resolve_seed(job.seed)
    frames = job.frames if job.mode == "txt2img" else 1  # consistent set: frames sampled together on the time axis
    shape = with_frames(vae.latent_shape(engine, W, H), frames)
    refs = make_reference(job)

    # everything below is the same for every run of the job: prepare once
    src = prepare_source(job, W, H) if job.mode == "img2img" and job.init_image else None
    # regions only: the input is kept and only the regions are regenerated, one at a time (no whole-image pass)
    regions_only = job.regions_only and src is not None and bool(job.regions)
    init_latent = keep_latent = latent_mask = None
    if src is not None and not regions_only:
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

    tensors: dict = {}  # encoded prompt texts, shared by every pass of the job
    steps_run = sampling.run_steps(job.steps, denoise if src is not None else 1.0)
    cond = regional = guide_set = None
    if not regions_only:
        cond = make_conditioning(engine, job, steps_run, W, H, tensors)
        regional = make_regional(engine, job, src, steps_run, W, H, cond, frames)
        guide_set = make_guides(engine, job, src, W, H)
    guide_mask = src.hard_mask if src is not None and src.mask is not None else None
    region_areas = region_pass.areas(job, src, out_size, backgrounds=True, limit_mask=src.full_mask) if regions_only else []
    if regions_only:  # progress: every region's steps, for every image of the run
        steps_run = len(region_areas) * sampling.run_steps(job.steps, job.strength) * job.batch

    hires = job.hires if job.mode == "txt2img" else None
    if hires is not None:
        tw, th = hires_size(job)
        TW, TH = images.snap(tw), images.snap(th)
        hires_steps = hires.steps or job.steps
        hires_run = sampling.run_steps(hires_steps, hires.denoise)
        cond2 = make_conditioning(engine, job, hires_run, TW, TH, tensors)
        regional2 = make_regional(engine, job, None, hires_run, TW, TH, cond2, frames)
        out_size = (tw, th)
    total_steps = steps_run + (hires_run if hires is not None else 0)

    params = {
        "Steps": job.steps,
        "Sampler": job.sampler,
        "Schedule type": job.scheduler,
        "CFG scale": job.cfg,
        "Shift": job.shift,
        "Seed": None,  # filled per image
        "Size": f"{job.width}x{job.height}" if hires is not None else f"{out_size[0]}x{out_size[1]}",
        "Model": manager.info.get("name"),
    }
    if src is not None:
        params["Denoising strength"] = denoise
        if regions_only:
            params["Regions only"] = True
        if src.mask is not None:
            params["Mask blur"] = job.mask_blur
            params["Masked content"] = job.fill
            params["Inpaint area"] = "Only masked" if src.box else "Whole picture"
            params["Soft inpainting"] = job.soft_inpaint
    if hires is not None:
        params["Denoising strength"] = hires.denoise
        params["Hires resize"] = f"{tw}x{th}"
        params["Hires upscaler"] = hires.upscaler
        params["Hires steps"] = hires_steps
        if hires.cfg:
            params["Hires CFG Scale"] = hires.cfg
        if hires.shift:
            params["Hires shift"] = hires.shift
    if job.switch_prompt:
        params["Prompt switch"] = job.switch_prompt
        params["Prompt switch at"] = job.switch_at
    if frames > 1:
        params["Frames"] = frames
        for k, fp in enumerate(job.frame_prompts[:frames], start=1):
            if fp.strip():
                params[f"Frame {k} prompt"] = fp
        if any(fp.strip() for fp in job.frame_prompts[:frames]):
            params["Frame separation"] = job.frame_separation
    if refs is not None:
        params.update(refs.info())
    if job.regions:
        params["Regions"] = len(job.regions)
        params["Region base"] = job.region_base
        if job.region_separation > 0:
            params["Region separation"] = job.region_separation
        if job.region_refine > 0:
            params["Region pass"] = job.region_refine
        if job.region_end < 1.0:
            params["Region end"] = job.region_end
        for k, r in enumerate(job.regions, start=1):
            params[f"Region {k}"] = r.prompt
            if r.negative.strip():
                params[f"Region {k} negative"] = r.negative
    for k, g in enumerate(job.guides, start=1):
        extra = (f", {preprocess.PREPROCESSORS[g.preprocessor]}" if g.preprocessor != "none" else "") + (f", {MODE_NAMES[g.mode]}" if g.mode != "balanced" else "")
        params[f"Guide {k}"] = f"{g.model}, weight {g.weight:g}, steps {g.start:g}-{g.end:g}{extra}"
    units = [d for d in job.detailer if d.model and d.model != "None"]
    for k, unit in enumerate(units):
        suffix = "" if k == 0 else f" {k + 1}{['nd', 'rd'][k - 1] if k < 3 else 'th'}"
        params[f"ADetailer model{suffix}"] = unit.model
        params[f"ADetailer confidence{suffix}"] = unit.confidence
        params[f"ADetailer denoising strength{suffix}"] = unit.denoise
        if unit.prompt.strip():
            params[f"ADetailer prompt{suffix}"] = unit.prompt
    for k, m in enumerate(manager.info.get("modules", []), start=1):
        params[f"Module {k}"] = m.rsplit(".", 1)[0]
    if getattr(engine, "venno_anima38", None) is not None:
        params.update(engine.venno_anima38.info)
    params.update(extra_params or {})
    params["Version"] = metadata.VERSION

    for run_index in range(job.count):
        seeds = [base_seed + run_index * job.batch + i for i in range(job.batch)]
        last_preview = [0.0]

        def stepper(offset: int, run_index=run_index):
            def on_step(i, n, denoised):
                progress.on_step(offset + i, total_steps, run_index, job.count)
                now = time.time()
                if progress.preview_every and i % progress.preview_every == 0 and i < n and now - last_preview[0] > 0.25:
                    last_preview[0] = now
                    img = vae.preview(engine, denoised)
                    if img is not None:
                        buf = io.BytesIO()
                        img.convert("RGB").save(buf, format="JPEG", quality=80)
                        progress.on_preview(buf.getvalue())

            return on_step

        latents = None
        if not regions_only:
            try:
                engine.set_shift(shift=job.shift, width=W, height=H)
                latents = sample_pass(
                    engine, job, progress, seeds, shape, cond, regional,
                    steps=job.steps, cfg=job.cfg, init_latent=init_latent, denoise=denoise,
                    inpaint=sampling.Inpaint(latent_mask, differential=job.soft_inpaint, keep=keep_latent) if latent_mask is not None else None,
                    on_step=stepper(0), guides=guide_set, guide_mask=guide_mask, reference=refs,
                )
                if hires is not None:
                    latents = upscale_latents(engine, latents, hires.upscaler, TW, TH)
                    engine.set_shift(shift=hires.shift or job.shift, width=TW, height=TH)  # shift follows the new resolution
                    latents = sample_pass(
                        engine, job, progress, seeds, with_frames(vae.latent_shape(engine, TW, TH), frames), cond2, regional2,
                        steps=hires_steps, cfg=hires.cfg or job.cfg, init_latent=latents, denoise=hires.denoise,
                        on_step=stepper(steps_run), guides=guide_set, reference=refs,
                    )
            except sampling.Cancelled:
                result.cancelled = True
                break

        outputs = [None] * len(seeds) if regions_only else vae.decode(engine, latents)
        for k, generated in enumerate(outputs):  # frames of an image are consecutive
            seed, frame = seeds[k // frames], k % frames
            if regions_only:
                per_image = steps_run // job.batch
                try:
                    image, _ = region_pass.run(
                        engine, job, progress, finish(src, src.image, out_size), seed, src=src, cache=tensors, limit_mask=src.full_mask,
                        denoise=job.strength, backgrounds=True,
                        on_step=lambda i, n, k=k, run_index=run_index: progress.on_step(k * per_image + i, total_steps, run_index, job.count),
                    )
                except sampling.Cancelled:
                    result.cancelled = True
                    return result
            else:
                image = finish(src, generated, out_size)
            if job.region_refine > 0 and job.regions and not regions_only:
                try:
                    image, _ = region_pass.run(engine, job, progress, image, seed, src=src, cache=tensors, limit_mask=src.full_mask if src is not None else None)
                except sampling.Cancelled:
                    result.cancelled = True
                    return result
            found, note = None, ""
            if units:
                try:
                    before = image
                    limit = region_pass.coverage(region_areas, out_size) if regions_only else src.full_mask if src is not None else None
                    image, report = detail.run(engine, job, progress, image, seed, units, job_loras=loras, cache=tensors, limit_mask=limit)
                    found, note = detail.preview_png(before, report), detail.summary(report)
                except sampling.Cancelled:
                    result.cancelled = True
                    return result
            info = metadata.infotext(metadata.with_lora_tags(job.prompt, loras), job.negative, {**params, "Seed": seed, **({"Frame": frame + 1} if frames > 1 else {})})
            if job.save:
                metadata.save(image, info, job.mode, seed)
            result.images.append(metadata.png_bytes(image, info))
            result.seeds.append(seed)
            result.infotexts.append(info)
            result.detections.append(found)
            result.detection_notes.append(note)
    return result


# ---------------------------------------------------------------- passes


MODE_NAMES = {"balanced": "Balanced", "prompt": "My prompt is more important", "guide": "ControlNet is more important"}


def with_frames(shape: tuple[int, ...], frames: int) -> tuple[int, ...]:
    """A per-image latent shape (C, 1, h, w) with `frames` frames on the time axis."""
    return shape if frames == 1 else (shape[0], frames, *shape[2:])


def make_reference(job) -> Optional[reference.Reference]:
    if not job.references:
        return None
    pictures = [images.load_rgb(r) for r in job.references]
    return reference.Reference(pictures, noise=job.reference_noise, follow=job.reference_follow, strength=job.reference_strength)


def make_regional(engine, job, src: Optional[Source], steps_run: int, width: int, height: int, cond, frames: int = 1):
    frame_prompts = [(k, fp) for k, fp in enumerate(job.frame_prompts[:frames]) if fp.strip()] if frames > 1 else []
    if not job.regions and not frame_prompts:
        return None
    region_list = []
    for k, fp in frame_prompts:  # frame prompts: the prompt with the frame's own text in front, on that frame only
        plan = prompts.PromptPlan(steps_run, f"{fp}, {job.prompt}", switch=f"{fp}, {job.switch_prompt}" if job.switch_prompt else None, switch_at=job.switch_at, allow_and=False)
        texts = {t for step in range(1, steps_run + 1) for t, _ in plan.parts(step)}
        region_list.append(regions.Region(plan=plan, contexts={t: cond.tensor(engine, t, False, width, height) for t in texts}, mask=None, frame=k))
    for r in job.regions:
        plan = prompts.PromptPlan(steps_run, r.prompt, allow_and=False)
        texts = {t for step in range(1, steps_run + 1) for t, _ in plan.parts(step)}
        contexts = {t: cond.tensor(engine, t, False, width, height) for t in texts}
        region = regions.Region(plan=plan, contexts=contexts, mask=region_mask(job, src, r.mask, width, height), strength=r.strength)
        if r.negative.strip():
            region.negative_plan = prompts.PromptPlan(steps_run, r.negative, allow_and=False)
            texts = {t for step in range(1, steps_run + 1) for t, _ in region.negative_plan.parts(step)}
            region.negative_contexts = {t: cond.tensor(engine, t, True, width, height) for t in texts}
        region_list.append(region)
    return regions.RegionalAttention(
        region_list, base_strength=job.region_base, frames=frames, frame_separation=job.frame_separation if frame_prompts else 0.0,
        separation=job.region_separation,
        separate_from=max(2, math.floor(steps_run * job.region_end * 0.25) + 1),  # layout first, as one picture
        separate_until=math.ceil(steps_run * job.region_end * 0.6),
    )


def hires_size(job) -> tuple[int, int]:
    """Final size of a hires job: explicit width/height (one may be 0 = keep aspect), else the scale."""
    h = job.hires
    if h.resize_x and h.resize_y:
        return h.resize_x, h.resize_y
    if h.resize_x:
        return h.resize_x, round(job.height * h.resize_x / job.width / 8) * 8
    if h.resize_y:
        return round(job.width * h.resize_y / job.height / 8) * 8, h.resize_y
    return round(job.width * h.scale / 8) * 8, round(job.height * h.scale / 8) * 8


def upscale_latents(engine, latents, upscaler_name: str, width: int, height: int):
    """Latent resize, or decode → upscale (model / resampling) → encode. Latents are (B, C, T, h, w)."""
    b, c, t = latents.shape[:3]
    if upscale.is_latent(upscaler_name):
        lat = latents.float().transpose(1, 2).reshape(b * t, c, *latents.shape[-2:])  # every frame on its own
        lat = F.interpolate(lat, size=(height // 8, width // 8), mode=upscale.latent_mode(upscaler_name))
        return lat.reshape(b, t, c, *lat.shape[-2:]).transpose(1, 2)
    big = [upscale.upscale(im, upscaler_name, width, height) for im in vae.decode(engine, latents)]
    lat = torch.cat([vae.encode(engine, im) for im in big])  # (B·T, C, 1, h, w)
    return lat.reshape(b, t, c, *lat.shape[-2:]).transpose(1, 2)
