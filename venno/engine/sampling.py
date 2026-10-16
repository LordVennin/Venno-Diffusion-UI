"""k-diffusion sampling over the backend engine (replaces Forge's sd_samplers_* glue)."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import k_diffusion.external
import k_diffusion.sampling as kd
import torch

from backend import utils
from backend.sampling.sampling_function import sampling_cleanup, sampling_function_inner, sampling_prepare

from . import rng, schedulers


@dataclass(frozen=True)
class SamplerSpec:
    func: str  # k_diffusion.sampling function name
    scheduler: str = "Normal"  # used when the job asks for "Automatic"
    discard_penultimate: bool = False
    brownian: bool = False  # SDE samplers: Brownian-tree noise (batch-size independent)


SAMPLERS: dict[str, SamplerSpec] = {
    "Euler": SamplerSpec("sample_euler"),
    "Euler a": SamplerSpec("sample_euler_ancestral"),
    "ER SDE": SamplerSpec("sample_er_sde"),
    "DPM++ 2M": SamplerSpec("sample_dpmpp_2m", scheduler="Karras"),
    "DPM++ SDE": SamplerSpec("sample_dpmpp_sde", scheduler="Karras", brownian=True),
    "DPM++ 2M SDE": SamplerSpec("sample_dpmpp_2m_sde", scheduler="Exponential", brownian=True),
    "DPM++ 3M SDE": SamplerSpec("sample_dpmpp_3m_sde", scheduler="Exponential", discard_penultimate=True, brownian=True),
    "DPM++ 2s a RF": SamplerSpec("sample_dpmpp_2s_ancestral_RF"),
    "Res Multistep": SamplerSpec("sample_res_multistep"),
    "Heun": SamplerSpec("sample_heun"),
    "DPM2": SamplerSpec("sample_dpm_2", scheduler="Karras", discard_penultimate=True),
    "LMS": SamplerSpec("sample_lms"),
    "LCM": SamplerSpec("sample_lcm"),
    "Kohaku LoNyu Yog": SamplerSpec("sample_Kohaku_LoNyu_Yog"),
}


class Cancelled(Exception):
    pass


@dataclass
class Inpaint:
    """Keep the unmasked area of the init latent; regenerate the rest.

    mask: (1, 1, 1, h, w) in [0, 1], 1 = regenerate. With `differential`, the mask acts as a per-pixel
    denoising strength (differential diffusion): a pixel joins the generation once the remaining noise
    level drops below its mask value, so feathered edges blend in gradually instead of as a seam.
    """

    mask: torch.Tensor
    differential: bool = True
    keep: Optional[torch.Tensor] = None  # latent the unmasked area is held to (default: init_latent)


@dataclass
class SampleRequest:
    engine: object  # backend ForgeDiffusionEngine
    noise: rng.ImageNoise
    conditioning: object  # conditioning.PromptConditioning (per-step prompts)
    cfg: float
    sampler: str
    scheduler: str
    steps: int
    seeds: list[int]
    init_latent: Optional[torch.Tensor] = None  # img2img: (1, C, 1, h, w) encoded input
    denoise: float = 1.0
    inpaint: Optional[Inpaint] = None  # needs init_latent
    regional: Optional[object] = None  # regions.RegionalAttention
    regional_end: float = 1.0  # stop regional prompts after this fraction of the full schedule
    on_step: Callable[[int, int, torch.Tensor], None] = lambda i, n, denoised: None
    is_cancelled: Callable[[], bool] = lambda: False
    extra_model_options: dict = field(default_factory=dict)


def run_steps(steps: int, denoise: float = 1.0) -> int:
    """Steps actually sampled: the last `denoise` part of the schedule (all of it at 1.0)."""
    return steps if denoise >= 1.0 else max(1, min(steps, math.floor(steps * denoise + 0.5)))  # half up, like Studio


class Denoiser:
    """k-diffusion model wrapper: CFG over the engine's conditioning pipeline."""

    def __init__(self, req: SampleRequest, linker, *, sigmas=None, noise=None, latent=None):
        self.req = req
        self.inner_model = linker  # samplers read inner_model.predictor (flow / const checks)
        self.predictor = linker.predictor
        self.noise, self.latent = noise, latent  # inpainting: initial noise and the input latent (on the device)
        self.sigmas = sigmas.float().cpu()
        self.regional_sigma_end = float(self.predictor.percent_to_sigma(req.regional_end)) if req.regional is not None and req.regional_end < 1.0 else -1.0
        if req.inpaint is not None:
            self.mask = req.inpaint.mask.to(latent)
            positive = sigmas[sigmas > 0]
            self.sigma_from, self.sigma_to = float(sigmas[0]), float(positive.min()) if len(positive) else 0.0
        self.unet = req.engine.forge_objects.unet

    def step_at(self, sigma) -> int:
        """1-based step for a model call (also right for samplers that evaluate between steps, e.g. Heun)."""
        return max(1, int((self.sigmas[:-1] >= float(sigma.reshape(-1)[0]) - 1e-6).sum()))

    def __call__(self, x, sigma, **_):
        if self.req.is_cancelled():
            raise Cancelled
        step = self.step_at(sigma)
        model_options = utils.join_dicts(self.unet.model_options, self.req.extra_model_options)
        if self.req.conditioning.needs_uncond:
            model_options = {**model_options, "disable_cfg1_optimization": True}
        if self.req.regional is not None and float(sigma.max()) >= self.regional_sigma_end:
            self.req.regional.step = step
            transformer_options = dict(model_options.get("transformer_options", {}))
            transformer_options["venno_regional"] = self.req.regional  # read by the Anima blocks' cross-attention
            model_options = {**model_options, "transformer_options": transformer_options}
        cond, uncond = self.req.conditioning.at(step)
        mask = None
        if self.req.inpaint is not None:
            mask = self.mask_at(sigma)
            known = self.predictor.noise_scaling(sigma, self.noise, self.latent, max_denoise=False)  # input, noised to this step
            x = x * mask + known * (1 - mask)
        for modifier in model_options.get("conditioning_modifiers", []):  # e.g. regional prompting / guides
            _, x, sigma, uncond, cond, _, model_options, _ = modifier(self.unet.model, x, sigma, uncond, cond, self.req.cfg, model_options, self.req.seeds[0])
        denoised = sampling_function_inner(self.unet.model, x, sigma, uncond, cond, self.req.cfg, model_options, self.req.seeds[0])
        if mask is not None:
            denoised = denoised * mask + self.latent * (1 - mask)
        return denoised

    def mask_at(self, sigma) -> torch.Tensor:
        if not self.req.inpaint.differential:
            return self.mask
        span = max(self.sigma_from - self.sigma_to, 1e-6)
        threshold = (float(sigma.reshape(-1)[0]) - self.sigma_to) / span  # 1 at the first step → 0 at the last
        return (self.mask >= threshold).to(self.mask)


def sample(req: SampleRequest) -> torch.Tensor:
    spec = SAMPLERS.get(req.sampler)
    if spec is None:
        raise ValueError(f'Sampler "{req.sampler}" is not supported by the Venno engine')

    unet = req.engine.forge_objects.unet
    predictor = unet.model.predictor
    linker = k_diffusion.external.ForgeScheduleLinker(predictor)
    sigmas = schedulers.get_sigmas(req.scheduler, req.steps, linker, default=spec.scheduler, discard_penultimate=spec.discard_penultimate)

    noise = req.noise.first()
    if req.init_latent is None:
        x = predictor.noise_scaling(sigmas[0], noise, torch.zeros_like(noise), max_denoise=False)
    else:
        # the last `denoise` part of the full schedule: strength 0.5 at 20 steps runs the final 10 steps,
        # strength 1.0 runs all of them (from pure noise)
        run = run_steps(req.steps, req.denoise)
        sigmas = sigmas[len(sigmas) - 1 - run :]
        x = predictor.noise_scaling(sigmas[0], noise, req.init_latent.to(noise), max_denoise=False)

    sampling_prepare(unet, x=x)
    device = unet.load_device
    x = x.to(device)
    sigmas = sigmas.to(device)
    extra = {"sigmas": sigmas}
    if req.inpaint is not None:
        if req.init_latent is None:
            raise ValueError("inpainting needs an init latent")
        extra.update(noise=noise.to(device=device, dtype=x.dtype), latent=(req.inpaint.keep if req.inpaint.keep is not None else req.init_latent).to(device=device, dtype=x.dtype))
    unet.model_options.setdefault("transformer_options", {})["sampling_sigmas"] = sigmas

    func = getattr(kd, spec.func)
    params = inspect.signature(func).parameters
    kwargs = {}
    if spec.brownian and "noise_sampler" in params:
        positive = sigmas[sigmas > 0]
        kwargs["noise_sampler"] = kd.BrownianTreeNoiseSampler(x, positive.min(), sigmas.max(), seed=req.seeds)

    total = len(sigmas) - 1

    def callback(d):
        req.on_step(d["i"] + 1, total, d["denoised"])

    previous_torch = kd.torch
    kd.torch = rng.TorchProxy(req.noise)  # ancestral / SDE noise from the job's seeded generators
    try:
        with torch.inference_mode():
            return func(Denoiser(req, linker, **extra), x, sigmas, extra_args={}, callback=callback, disable=True, **kwargs)
    finally:
        kd.torch = previous_torch
        sampling_cleanup(unet)
