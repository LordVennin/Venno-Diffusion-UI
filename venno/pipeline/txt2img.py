"""Text to image on the Venno engine."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Callable

from venno.engine import conditioning, lora, rng, sampling, vae
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


def run(job, progress: Progress) -> Result:
    engine = manager.get(job.model.checkpoint, [job.model.text_encoder, job.model.vae])
    engine.set_shift(shift=job.shift, width=job.width, height=job.height)
    loras = [(lr.name, lr.weight) for lr in job.loras]
    result_warnings = lora.apply(engine, loras)  # before encoding: LoRAs can patch the text encoder too

    base_seed = rng.resolve_seed(job.seed)
    result = Result(warnings=result_warnings)
    shape = vae.latent_shape(engine, job.width, job.height)

    # conditioning is the same for every run of the job: encode once
    cond = conditioning.encode(engine, job.prompt, job.batch, negative=False, width=job.width, height=job.height)
    uncond = conditioning.encode(engine, job.negative, job.batch, negative=True, width=job.width, height=job.height)

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
                    cond=cond,
                    uncond=uncond,
                    cfg=job.cfg,
                    sampler=job.sampler,
                    scheduler=job.scheduler,
                    steps=job.steps,
                    seeds=seeds,
                    on_step=on_step,
                    is_cancelled=progress.is_cancelled,
                )
            )
        except sampling.Cancelled:
            result.cancelled = True
            break

        params = {
            "Steps": job.steps,
            "Sampler": job.sampler,
            "Schedule type": job.scheduler,
            "CFG scale": job.cfg,
            "Shift": job.shift,
            "Seed": None,  # filled per image
            "Size": f"{job.width}x{job.height}",
            "Model": manager.info.get("name"),
        }
        for k, m in enumerate(manager.info.get("modules", []), start=1):
            params[f"Module {k}"] = m.rsplit(".", 1)[0]
        params["Version"] = metadata.VERSION

        for seed, image in zip(seeds, vae.decode(engine, latents)):
            info = metadata.infotext(metadata.with_lora_tags(job.prompt, loras), job.negative, {**params, "Seed": seed})
            if job.save:
                metadata.save(image, info, "txt2img", seed)
            result.images.append(metadata.png_bytes(image, info))
            result.seeds.append(seed)
            result.infotexts.append(info)
    return result
