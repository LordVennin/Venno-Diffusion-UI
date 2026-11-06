"""SDE samplers' Brownian-tree noise: no torchsde warning on every step when a sigma sits on the interval's edge."""

import warnings

import torch

from backend.vendor.k_diffusion.sampling import BrownianTreeNoiseSampler


def test_edge_sigma_does_not_warn():
    # torchsde rounds the interval's ends to 1e-6: the float32 sigma itself (0.74758845…) lies past 0.747588
    smax, smin = torch.tensor(0.7475884556770325, dtype=torch.float32), torch.tensor(0.0291675, dtype=torch.float32)
    sampler = BrownianTreeNoiseSampler(torch.zeros(1, 4, 8, 8), smin, smax, seed=1, cpu=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        noise = sampler(smax, torch.tensor(0.5))
    assert noise.shape == (1, 4, 8, 8) and 0.5 < float(noise.std()) < 2
