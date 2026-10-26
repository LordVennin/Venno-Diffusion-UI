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


def _det(box, size=(W, H)):
    from venno.engine.detect import Detection

    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle(box, fill=255)
    return Detection(box, m, 0.9)


def test_characters_by_face_or_person():
    img = Image.new("RGB", (W, H))
    boxes = {0: (10, 10, 60, 90), 1: (120, 10, 190, 90)}
    # a face in box 1, a person mostly in box 0, and a face outside both
    hit, extra = region_pass.characters(img, boxes, faces=[_det((140, 20, 160, 40)), _det((90, 2, 100, 12))], persons=[_det((5, 15, 55, 85))])
    assert hit == {0, 1} and extra == 1
    # one big figure sprawling over both boxes counts for one box at most; a face near a box (within 5%) counts
    hit, extra = region_pass.characters(img, boxes, faces=[_det((62, 20, 70, 30))], persons=[_det((0, 0, 199, 99))])
    assert hit == {0} and extra == 0


def test_missing_only_checks_character_regions(monkeypatch):
    j = SimpleNamespace(regions=[SimpleNamespace(mask=mask(BACKGROUND), kind="auto", prompt="living room"),
                                 SimpleNamespace(mask=mask(SMALL), kind="auto", prompt="1girl, sitting"),
                                 SimpleNamespace(mask=mask((100, 10, 150, 60)), kind="auto", prompt="fireplace"),
                                 SimpleNamespace(mask=mask((150, 60, 190, 95)), kind="auto", prompt="1boy, lying")], resize_mode=0)
    monkeypatch.setattr(region_pass, "characters", lambda image, boxes: ({1}, 0))
    assert region_pass.missing(j, None, Image.new("RGB", (W, H))) == [3]


def test_composed_prompt_names_places_and_counts():
    from venno.pipeline import compose

    text = compose.compose("masterpiece, safe", [
        ("living room, night", "scenery", (0, 0, 1, 1)),
        ("fireplace, fire", "subject", (0.61, 0.29, 0.80, 0.64)),
        ("1girl, black hair, sitting on couch", "subject", (0.12, 0.24, 0.35, 0.80)),
        ("1girl, blonde hair, lying on stomach", "subject", (0.45, 0.64, 0.86, 0.80)),
        ("bat \\(animal\\), flying", "subject", (0.70, 0.08, 0.96, 0.29)),
        ("", "subject", None),
    ])
    import json
    d = json.loads(text)
    assert d["tags"] == "masterpiece, safe, 2girls"
    assert d["characters"] == [{"girl1": "black hair, sitting on couch"}, {"girl2": "blonde hair, lying on stomach"}]
    assert "objects" not in d
    assert d["background"] == "living room, night, fireplace, fire, bat \\(animal\\), flying"
    assert d["composition"] == "the fireplace on the right, girl1 on the left, girl2 at the bottom right, the bat at the top right, small"
    assert compose.compose("x", [("1boy, kobold", "subject", (0.4, 0.4, 0.6, 0.6))], escape=True).startswith("\\{\n")
    assert '"tags": "x, 1boy"' in compose.compose("x", [("1boy, kobold", "subject", None)])
