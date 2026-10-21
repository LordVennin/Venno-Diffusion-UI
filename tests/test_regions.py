"""Region / frame-prompt blend weights and the self-attention biases (venno/engine/regions.py, reference.py)."""

import pytest
import torch
import torch.nn.functional as F

from venno.engine import regions
from venno.engine.reference import Reference


class Plan:
    def __init__(self, text):
        self.text = text

    def parts(self, step):
        return [(self.text, 1.0)]


H, W, D, L = 4, 6, 8, 5
LIKE = torch.zeros(1, 1, D)


def region(left: bool, **kw):
    mask = torch.zeros(H * 8, W * 8)
    mask[:, : W * 4] = 1 if left else 0
    mask[:, W * 4 :] = 0 if left else 1
    return regions.Region(plan=Plan("l" if left else "r"), contexts={("l" if left else "r"): torch.randn(1, L, D)}, mask=mask, **kw)


def frame_prompt(k):
    return regions.Region(plan=Plan(f"f{k}"), contexts={f"f{k}": torch.randn(1, L, D)}, mask=None, frame=k)


@pytest.mark.parametrize("t", [1, 3])
def test_weights_sum_to_one(t):
    base, reg = regions.RegionalAttention([region(True), region(False)], frames=t).weights((t, H, W), LIKE)
    assert base.shape == (t * H * W, 1) and reg.shape == (2, t * H * W, 1)
    assert torch.allclose(base + reg.sum(0), torch.ones_like(base))


def test_region_area_is_mostly_its_prompt():
    base, reg = regions.RegionalAttention([region(True), region(False)], base_strength=0.25).weights((1, H, W), LIKE)
    left, right = 0, W - 1  # first row: a left and a right token
    assert reg[0, left, 0] == pytest.approx(0.8) and reg[1, right, 0] == pytest.approx(0.8)
    assert base[left, 0] == pytest.approx(0.2)


def test_reference_frames_see_the_main_prompt_only():
    base, reg = regions.RegionalAttention([region(True)], frames=1).weights((3, H, W), LIKE)  # 1 generated + 2 references
    n = H * W
    assert (base[n:] == 1).all() and (reg[:, n:] == 0).all()


def test_frame_prompt_replaces_the_main_prompt_on_its_frame():
    base, reg = regions.RegionalAttention([region(True), frame_prompt(1)], frames=2).weights((2, H, W), LIKE)
    n = H * W
    assert (reg[1, :n] == 0).all(), "absent on other frames"
    assert (base[n:] == 0).all(), "main prompt replaced on its frame"
    assert torch.allclose(base + reg.sum(0), torch.ones_like(base))


def test_separation_bias_only_in_its_window_and_single_frame():
    ra = regions.RegionalAttention([region(True), region(False)], separation=1.0, separate_from=2, separate_until=3)
    ra.step = 1
    assert ra.self_bias((1, H, W), LIKE) is None
    ra.step = 2
    bias = ra.self_bias((1, H, W), LIKE)
    assert bias.shape == (1, 1, H * W, H * W) and bias.min() < -6
    assert ra.self_bias((2, H, W), LIKE) is None, "no spatial separation across frames"


def test_frame_bias_matches_a_dense_mask():
    ra = regions.RegionalAttention([frame_prompt(1)], frames=3, frame_separation=0.7)
    T = 4  # 3 frames + 1 reference
    fb = ra.frame_bias((T, H, W), torch.zeros(1))
    n = H * W
    q, k, v = (torch.randn(2, 4, T * n, 8) for _ in range(3))
    dense = F.scaled_dot_product_attention(q, k, v, attn_mask=fb.repeat_interleave(n, dim=0)[None, None])
    per_frame = torch.cat([F.scaled_dot_product_attention(q[:, :, f * n : (f + 1) * n], k, v, attn_mask=fb[f][None, None, None]) for f in range(T)], dim=2)
    assert torch.allclose(dense, per_frame, atol=1e-5)
    assert fb[0, 0] == 0 and fb[0, n] < 0, "own frame free, other frames damped"
    assert (fb[:, 3 * n :] == 0).all(), "the reference stays visible"


def test_frame_bias_off_without_frame_prompts():
    assert regions.RegionalAttention([region(True)], frames=3, frame_separation=0.7).frame_bias((3, H, W), torch.zeros(1)) is None


def test_reference_key_bias_and_timesteps():
    r = Reference([], strength=0.5)
    r._current = torch.zeros(1, 4, 2, H, W)  # two reference frames
    bias = r.key_bias((3, H, W), torch.zeros(1))
    n = H * W
    assert bias.shape == (3, 3 * n)
    assert (bias[0, :n] == 0).all() and (bias[0, n:] < 0).all(), "the generated frame sees references damped"
    assert (bias[1:] == 0).all(), "reference frames are not damped"
    assert Reference([], strength=1.0).key_bias((3, H, W), torch.zeros(1)) is None
    t = torch.tensor([[0.7], [0.7]])
    r._timestep = 0.5
    assert torch.equal(r.timesteps(t, 2), torch.full((2, 2), 0.5))
    r.follow = True
    assert torch.equal(r.timesteps(t, 2), t.expand(-1, 2))
