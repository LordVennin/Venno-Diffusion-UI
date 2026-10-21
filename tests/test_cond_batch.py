"""Combining weighted conds (AND) in backend/sampling/sampling_function.calc_cond_uncond_batch."""

import pytest
import torch

from backend.sampling.condition import ConditionCrossAttn
from backend.sampling.sampling_function import calc_cond_uncond_batch


class Model:
    """Predicts the prompt's value everywhere, so the combination of the outputs is easy to check."""

    def memory_required(self, shape):
        return 0

    def apply_model(self, x, t, c_crossattn=None, transformer_options=None, **_):
        return c_crossattn[:, :1, :1, None].expand_as(x).clone()


def cond(value, weight=None):
    c = {"model_conds": {"c_crossattn": ConditionCrossAttn(torch.full((1, 1, 1), float(value)))}}
    if weight is not None:
        c["strength"] = weight
    return c


X = torch.zeros(1, 4, 2, 2)
T = torch.ones(1)


def run(conds, uncond_value=0.0):
    out_cond, out_uncond = calc_cond_uncond_batch(Model(), conds, [cond(uncond_value)], X, T, {})
    assert torch.isfinite(out_cond).all()
    return out_cond.flatten()[0].item(), out_uncond.flatten()[0].item()


def test_single_prompt():
    assert run([cond(3.0)], 1.0) == pytest.approx((3.0, 1.0))


def test_positive_weights_average():
    assert run([cond(2.0, 1.0), cond(5.0, 0.5)], 1.0)[0] == pytest.approx((2.0 + 2.5) / 1.5)


def test_weights_summing_to_zero():
    # uncond + 1*(4-1) - 1*(2-1)
    assert run([cond(4.0, 1.0), cond(2.0, -1.0)], 1.0)[0] == pytest.approx(3.0)


def test_negative_weight_sum_keeps_direction():
    # uncond + (1*(4-1) - 2*(2-1)) / 1: the first prompt still pulls towards itself
    assert run([cond(4.0, 1.0), cond(2.0, -2.0)], 1.0)[0] == pytest.approx(2.0)
