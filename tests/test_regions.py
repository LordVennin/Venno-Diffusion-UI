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


def test_contexts_follow_fresh_tensors():
    """Anima 3.8B resolves region prompts to new tensors on every model call: a freed tensor's id gets
    reused, which must not return the previous call's cached context."""
    r = regions.Region(plan=Plan("a"), contexts={"a": {"placeholder": 0}}, mask=torch.ones(H * 8, W * 8))
    att = regions.RegionalAttention([r])
    like = torch.zeros(1, L, D)
    for value in range(20):
        att.resolve_context = lambda c, v=value: torch.full((1, L, D), float(v))
        assert att.contexts(like)[0, 0, 0].item() == value


def test_contexts_padded_to_the_main_context():
    att = regions.RegionalAttention([region(True)])
    assert att.contexts(torch.zeros(1, L, D)).shape == (1, L, D)
    assert att.contexts(torch.zeros(1, L + 3, D)).shape == (1, L + 3, D)


def test_couple_box():
    """Region own passes run on a crop around the region: grown, at least 32 latents a side, even, inside the picture."""
    from venno.engine.sampling import _couple_box

    w = torch.zeros(64, 64)
    w[41:51, 29:55] = 1
    y0, y1, x0, x1 = _couple_box(w, 64, 64)
    assert y0 <= 41 and y1 >= 51 and x0 <= 29 and x1 >= 55  # covers the region
    assert y1 - y0 >= 32 and x1 - x0 >= 32 and (y1 - y0) % 2 == 0 and (x1 - x0) % 2 == 0
    assert 0 <= y0 and y1 <= 64 and 0 <= x0 and x1 <= 64
    assert _couple_box(torch.ones(64, 64), 64, 64) is None  # a background region: the whole picture
    assert _couple_box(torch.zeros(64, 64), 64, 64) is None


def test_background_gives_way_to_subjects():
    """A region over the whole picture (scenery) never shares a subject's area, even with Blend prompts."""
    full = regions.Region(plan=Plan("room"), contexts={"room": torch.randn(1, L, D)}, mask=torch.ones(H * 8, W * 8))
    att = regions.RegionalAttention([full, region(True)], base_strength=0.1)
    assert att.backgrounds() == [0]
    base, reg = att.weights((1, H, W), LIKE)
    inside, outside = 0, W - 1  # token columns inside / outside the left-half subject
    grid = reg.reshape(2, H, W)
    assert grid[1, :, inside].min() > 0.85 and grid[0, :, inside].max() < 0.05  # the subject's prompt rules its area
    assert grid[0, :, outside].min() > 0.85  # the scenery everywhere else


# ---------------------------------------------------------------- region layout guide


def test_layout_map_and_guide_for_regions(monkeypatch):
    """The colour mask a region layout guide (LLLite regional) reads, drawn from a job's regions."""
    import numpy as np

    from venno.engine import guides
    from venno.pipeline import generate, infotext
    from venno.pipeline import grid as grid_module
    from venno.server.jobs import Job

    scenery = torch.ones(32, 48)  # spans the picture: stays white
    left, right, small = torch.zeros(32, 48), torch.zeros(32, 48), torch.zeros(32, 48)
    left[4:28, 2:22] = 1
    right[4:28, 26:46] = 0.8
    small[10:16, 16:30] = 1  # overlaps both: the smaller shape stays whole
    image = guides.layout_map([scenery, left, None, right, small, torch.zeros(32, 48)], background=[0])
    px = np.asarray(image)
    assert image.size == (48, 32) and image.mode == "RGB"
    assert tuple(px[0, 0]) == (255, 255, 255) and tuple(px[20, 5]) == (255, 0, 0) and tuple(px[20, 40]) == (0, 0, 255)
    assert tuple(px[12, 20]) == (0, 255, 0) and tuple(px[12, 28]) == (0, 255, 0)
    assert len({tuple(c) for c in px.reshape(-1, 3)}) == 4  # flat colours only
    with pytest.raises(ValueError):
        guides.layout_map([None])

    known = {"anima-lllite-any-test-like-v2": "/a", "anima-lllite-regional-exp-v3": "/r", "other": "/o"}
    assert guides.regional_model(known) == "anima-lllite-regional-exp-v3" and guides.regional_model(known, "other") == "other"
    assert guides.regional_model(known, "gone") is None and guides.regional_model({"lineart": "/l"}) is None

    # the pipeline adds it as a guide over the first steps, on the first pass only
    made = []
    monkeypatch.setattr(guides, "models", lambda: known)
    monkeypatch.setattr(guides, "GuideSet", lambda engine, items: made.append(items) or items)
    regional = regions.RegionalAttention([regions.Region(plan=Plan("bg"), contexts={}, mask=scenery), region(True), region(False)])
    mask = "data:image/png;base64,"  # not read here: the masks come from `regional`
    job = Job(prompt="x", regions=[{"prompt": p, "mask": mask} for p in ("bg", "l", "r")], region_guide=0.9, region_guide_end=0.4)
    items = generate.make_guides(None, job, None, 48, 32, regional)
    assert len(items) == 1 and (items[0].path, items[0].weight, items[0].start, items[0].end, items[0].flat) == ("/r", 0.9, 0.0, 0.4, True)
    assert {tuple(c) for c in np.asarray(items[0].image).reshape(-1, 3)} == {(255, 0, 0), (0, 0, 255)}  # two subjects fill the picture
    assert generate.make_guides(None, job, None, 48, 32, layout=False) is None  # the hires pass: the job's own guides only
    assert generate.make_guides(None, job.model_copy(update={"region_guide": 0.0}), None, 48, 32, regional) is None
    assert generate.make_guides(None, job.model_copy(update={"regions": []}), None, 48, 32, None) is None
    monkeypatch.setattr(guides, "models", lambda: {"lineart": "/l"})
    with pytest.raises(ValueError, match="LLLite regional"):
        generate.make_guides(None, job, None, 48, 32, regional)

    parsed = infotext.settings("x\nSteps: 8, Region layout guide: \"anima-lllite-regional-exp-v3, weight 0.9, steps 0-0.4\", Regions: 2")
    assert (parsed["region_guide_model"], parsed["region_guide"], parsed["region_guide_end"]) == ("anima-lllite-regional-exp-v3", 0.9, 0.4)
    assert grid_module.AXES["region_guide"].apply(job, 0.5, "") == {"region_guide": 0.5}
