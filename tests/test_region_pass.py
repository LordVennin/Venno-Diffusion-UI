"""Region pass: which areas it re-renders, and the region kind (venno/pipeline/region_pass.py, venno/engine/regions.py)."""

import base64
import io
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw

from venno.engine import regions
from venno.pipeline import region_pass

W, H = 200, 100


def mask(box):
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).rectangle(box, fill=255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def job(*boxes, kinds=None):
    kinds = kinds or ["auto"] * len(boxes)
    return SimpleNamespace(regions=[SimpleNamespace(mask=mask(b), kind=k) for b, k in zip(boxes, kinds)], resize_mode=0)


SMALL, BACKGROUND = (10, 10, 40, 40), (0, 0, W - 1, H - 1)


def test_scenery_is_skipped():
    assert [k for k, _ in region_pass.areas(job(BACKGROUND, SMALL), None, (W, H))] == [1]


def test_declared_kind_wins_over_the_size_rule():
    assert [k for k, _ in region_pass.areas(job(BACKGROUND, SMALL, kinds=["subject", "scenery"]), None, (W, H))] == [0]


def test_areas_grow_and_respect_the_limit():
    (_, area), = region_pass.areas(job(SMALL), None, (W, H))
    x1, y1, x2, y2 = area.getbbox()
    assert x1 < 10 and x2 > 41  # grown past the region
    left = Image.new("L", (W, H), 0)
    ImageDraw.Draw(left).rectangle((0, 0, 24, H), fill=255)
    (_, area), = region_pass.areas(job(SMALL), None, (W, H), limit_mask=left)
    assert area.getbbox()[2] <= 25
    assert region_pass.areas(job(SMALL), None, (W, H), limit_mask=Image.new("L", (W, H), 0)) == []


def _region(box, kind="auto"):
    m = torch.zeros(H, W)
    m[box[1]:box[3] + 1, box[0]:box[2] + 1] = 1
    return regions.Region(plan=None, contexts={}, mask=m, kind=kind)


def test_attention_backgrounds_follow_the_kind():
    assert regions.RegionalAttention([_region(BACKGROUND), _region(SMALL)]).backgrounds() == [0]
    assert regions.RegionalAttention([_region(BACKGROUND, "subject"), _region(SMALL, "scenery")]).backgrounds() == [1]
