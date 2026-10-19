"""Sigma schedules for Anima (rectified-flow / discrete-flow predictor).

Ported from Forge's sd_schedulers.py, minus the SD-only and other-architecture entries
(Align Your Steps, Flux2, FlowMatchEulerDiscrete).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, pi
from typing import Callable

import k_diffusion.sampling as kd
import numpy as np
import torch
from scipy import stats

BETA_ALPHA = 0.6
BETA_BETA = 0.6


def normal(n, sigma_min, sigma_max, linker, sgm=False):
    start = linker.sigma_to_t(torch.tensor(sigma_max))
    end = linker.sigma_to_t(torch.tensor(sigma_min))
    timesteps = torch.linspace(start, end, n + 1)[:-1] if sgm else torch.linspace(start, end, n)
    return torch.FloatTensor([float(linker.t_to_sigma(t)) for t in timesteps] + [0.0])


def sgm_uniform(n, sigma_min, sigma_max, linker):
    return normal(n, sigma_min, sigma_max, linker, sgm=True)


def simple(n, sigma_min, sigma_max, linker):
    ss = len(linker.sigmas) / n
    return torch.FloatTensor([float(linker.sigmas[-(1 + int(x * ss))]) for x in range(n)] + [0.0])


def uniform(n, sigma_min, sigma_max, linker):
    """Evenly spaced over the model's own timestep table (A1111's "Uniform"). Read from the table directly:
    `linker.get_sigmas` passes table indices (0..999) to `t_to_sigma`, which Anima's flow schedule reads as
    fractions, giving sigmas above 1 (and NaN images)."""
    log = linker.sigmas.float().log()
    t = torch.linspace(len(log) - 1, 0, n)
    lo, w = t.floor().long(), t.frac()
    hi = (lo + 1).clamp(max=len(log) - 1)
    return torch.FloatTensor(((1 - w) * log[lo] + w * log[hi]).exp().tolist() + [0.0])


def ddim(n, sigma_min, sigma_max, linker):
    sigs, ss, x = [], max(len(linker.sigmas) // n, 1), 1
    while x < len(linker.sigmas):
        sigs.append(float(linker.sigmas[x]))
        x += ss
    return torch.FloatTensor(sigs[::-1] + [0.0])


def beta(n, sigma_min, sigma_max, linker, alpha=BETA_ALPHA, beta_=BETA_BETA):
    """"Beta Sampling is All You Need" (Lee et al., 2024)"""
    total = len(linker.sigmas) - 1
    ts = np.rint(stats.beta.ppf(1 - np.linspace(0, 1, n, endpoint=False), alpha, beta_) * total)
    sigs, last = [], -1
    for t in ts:
        if t != last:
            sigs.append(float(linker.sigmas[int(t)]))
        last = t
    return torch.FloatTensor(sigs + [0.0])


def turbo(n, sigma_min, sigma_max, linker):
    timesteps = torch.flip(torch.arange(1, n + 1) * float(1000.0 / n) - 1, (0,)).round().long().clip(0, 999)
    sigmas = linker.predictor.sigma(timesteps)
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def linear_quadratic(n, sigma_min, sigma_max, linker, threshold_noise=0.025):
    if n == 1:
        schedule = [1.0, 0.0]
    else:
        linear_steps = n // 2
        linear = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        diff = linear_steps - threshold_noise * n
        quadratic_steps = n - linear_steps
        q_coef = diff / (linear_steps * quadratic_steps**2)
        l_coef = threshold_noise / linear_steps - 2 * diff / (quadratic_steps**2)
        const = q_coef * (linear_steps**2)
        quadratic = [q_coef * (i**2) + l_coef * i + const for i in range(linear_steps, n)]
        schedule = [1.0 - x for x in linear + quadratic + [1.0]]
    return torch.FloatTensor(schedule) * sigma_max


def kl_optimal(n, sigma_min, sigma_max, linker):
    a_min = torch.arctan(torch.tensor(sigma_min))
    a_max = torch.arctan(torch.tensor(sigma_max))
    idx = torch.arange(n + 1)
    return torch.tan(idx / n * a_min + (1.0 - idx / n) * a_max)


def _bong(steps, slope, pivot, start, end):
    smax = ((2 / pi) * atan(-slope * (0 - pivot)) + 1) / 2
    smin = ((2 / pi) * atan(-slope * ((steps - 1) - pivot)) + 1) / 2
    srange, sscale = smax - smin, start - end
    return [((((2 / pi) * atan(-slope * (x - pivot)) + 1) / 2) - smin) * (1 / srange) * sscale + end for x in range(steps)]


def bong_tangent(n, sigma_min, sigma_max, linker, start=1.0, middle=0.5, end=0.0, pivot_1=0.6, pivot_2=0.6, slope_1=0.2, slope_2=0.2):
    """https://github.com/ClownsharkBatwing/RES4LYF"""
    n += 2
    midpoint = int((n * pivot_1 + n * pivot_2) / 2)
    p1, p2 = int(n * pivot_1), int(n * pivot_2)
    s1, s2 = slope_1 / (n / 40), slope_2 / (n / 40)
    len2 = n - midpoint
    len1 = n - len2
    return torch.tensor(_bong(len1, s1, p1, start, middle)[:-1] + _bong(len2, s2, p2 - len1, middle, end))


def karras(n, sigma_min, sigma_max, linker):
    return kd.get_sigmas_karras(n=n, sigma_min=sigma_min, sigma_max=sigma_max, rho=7.0)


def exponential(n, sigma_min, sigma_max, linker):
    return kd.get_sigmas_exponential(n=n, sigma_min=sigma_min, sigma_max=sigma_max)


def polyexponential(n, sigma_min, sigma_max, linker):
    return kd.get_sigmas_polyexponential(n=n, sigma_min=sigma_min, sigma_max=sigma_max, rho=1.0)


@dataclass(frozen=True)
class Scheduler:
    label: str
    fn: Callable


SCHEDULERS: dict[str, Scheduler] = {
    s.label: s
    for s in (
        Scheduler("Normal", normal),
        Scheduler("Simple", simple),
        Scheduler("Beta", beta),
        Scheduler("SGM Uniform", sgm_uniform),
        Scheduler("Uniform", uniform),
        Scheduler("Karras", karras),
        Scheduler("Exponential", exponential),
        Scheduler("Polyexponential", polyexponential),
        Scheduler("Linear Quadratic", linear_quadratic),
        Scheduler("KL Optimal", kl_optimal),
        Scheduler("DDIM", ddim),
        Scheduler("Turbo", turbo),
        Scheduler("Bong Tangent", bong_tangent),
    )
}


def get_sigmas(label: str, steps: int, linker, *, default: str = "Normal", discard_penultimate: bool = False) -> torch.Tensor:
    """Sigmas (descending, ending in 0) for `steps` steps. `linker` is a k_diffusion ForgeScheduleLinker."""
    if label in (None, "", "Automatic"):
        label = default
    sched = SCHEDULERS.get(label)
    if sched is None:
        raise ValueError(f'Unknown scheduler "{label}"')
    n = steps + (1 if discard_penultimate else 0)
    sigma_min, sigma_max = float(linker.sigmas[0]), float(linker.sigmas[-1])
    sigmas = sched.fn(n, sigma_min, sigma_max, linker).float().cpu()
    if discard_penultimate:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas
