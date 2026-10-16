"""One sampling pass of a job (first pass, hires pass, detailer pass) with its prompts."""

from __future__ import annotations

from venno.engine import conditioning, prompts, rng, sampling

_KEEP = object()


def make_conditioning(engine, job, steps_run: int, width: int, height: int, cache: dict, *, prompt=None, negative=None, switch=_KEEP):
    """Prompt plans (scheduling / AND / switch prompt) over the steps a pass samples, encoded via the shared cache."""
    switch = (job.switch_prompt or None) if switch is _KEEP else switch
    positive = prompts.PromptPlan(steps_run, job.prompt if prompt is None else prompt, switch=switch, switch_at=job.switch_at)
    neg = prompts.PromptPlan(steps_run, job.negative if negative is None else negative, allow_and=False)
    return conditioning.PromptConditioning(engine, positive, neg, width=width, height=height, cache=cache)


def sample_pass(engine, job, progress, seeds, shape, cond, regional, *, steps, cfg, init_latent=None, denoise=1.0, inpaint=None, on_step, guides=None, guide_mask=None):
    """guides: engine.guides.GuideSet, fitted to this pass (size from `shape`, steps from denoise); guide_mask: PIL, for inpaint guides."""
    hooks = []
    if guides is not None:
        guides.prepare(shape[-1] * 8, shape[-2] * 8, sampling.run_steps(steps, denoise if init_latent is not None else 1.0), guide_mask)
        guides.attach()
        hooks.append(guides.at_step)
    try:
        return _sample(engine, job, progress, seeds, shape, cond, regional, steps=steps, cfg=cfg, init_latent=init_latent, denoise=denoise, inpaint=inpaint, on_step=on_step, hooks=hooks)
    finally:
        if guides is not None:
            guides.detach()


def _sample(engine, job, progress, seeds, shape, cond, regional, *, steps, cfg, init_latent, denoise, inpaint, on_step, hooks):
    return sampling.sample(
        sampling.SampleRequest(
            engine=engine,
            noise=rng.ImageNoise(seeds, shape),
            conditioning=cond,
            cfg=cfg,
            sampler=job.sampler,
            scheduler=job.scheduler,
            steps=steps,
            seeds=seeds,
            init_latent=init_latent,
            denoise=denoise,
            inpaint=inpaint,
            regional=regional,
            regional_end=job.region_end,
            on_step=on_step,
            step_hooks=hooks,
            is_cancelled=progress.is_cancelled,
        )
    )
