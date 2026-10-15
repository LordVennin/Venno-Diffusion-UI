"""k-diffusion sampling over the backend engine (replaces Forge's sd_samplers_* glue)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Optional

import k_diffusion.external
import k_diffusion.sampling as kd
import torch

from backend import utils
from backend.sampling.condition import compile_conditions
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
class SampleRequest:
    engine: object  # backend ForgeDiffusionEngine
    noise: rng.ImageNoise
    cond: torch.Tensor  # (B, L, D) positive conditioning
    uncond: torch.Tensor  # (B, L, D) negative conditioning
    cfg: float
    sampler: str
    scheduler: str
    steps: int
    seeds: list[int]
    init_latent: Optional[torch.Tensor] = None  # img2img (Phase 2)
    denoise: float = 1.0
    on_step: Callable[[int, int, torch.Tensor], None] = lambda i, n, denoised: None
    is_cancelled: Callable[[], bool] = lambda: False
    extra_model_options: dict = field(default_factory=dict)


class Denoiser:
    """k-diffusion model wrapper: CFG over the engine's conditioning pipeline."""

    def __init__(self, req: SampleRequest, linker):
        self.req = req
        self.inner_model = linker  # samplers read inner_model.predictor (flow / const checks)
        self.unet = req.engine.forge_objects.unet
        self.cond = compile_conditions(req.cond)
        self.uncond = compile_conditions(req.uncond)

    def __call__(self, x, sigma, **_):
        if self.req.is_cancelled():
            raise Cancelled
        model_options = utils.join_dicts(self.unet.model_options, self.req.extra_model_options)
        cond, uncond = self.cond, self.uncond
        for modifier in model_options.get("conditioning_modifiers", []):  # e.g. regional prompting / guides
            _, x, sigma, uncond, cond, _, model_options, _ = modifier(self.unet.model, x, sigma, uncond, cond, self.req.cfg, model_options, self.req.seeds[0])
        denoised = sampling_function_inner(self.unet.model, x, sigma, uncond, cond, self.req.cfg, model_options, self.req.seeds[0])
        return denoised


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
        t_enc = int(min(req.denoise, 0.999) * req.steps)  # same as Forge's setup_img2img_steps
        sigmas = sigmas[req.steps - t_enc - 1 :]
        x = predictor.noise_scaling(sigmas[0], noise, req.init_latent.to(noise), max_denoise=False)

    sampling_prepare(unet, x=x)
    device = unet.load_device
    x = x.to(device)
    sigmas = sigmas.to(device)
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
            return func(Denoiser(req, linker), x, sigmas, extra_args={}, callback=callback, disable=True, **kwargs)
    finally:
        kd.torch = previous_torch
        sampling_cleanup(unet)
