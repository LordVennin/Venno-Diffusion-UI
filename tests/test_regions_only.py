"""Regions only (Canvas): which areas the region pass re-renders, and job validation."""

import base64
import io
from types import SimpleNamespace

from PIL import Image, ImageDraw

from venno.pipeline import region_pass
from venno.server import jobs

W, H = 200, 100


def mask(box):
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).rectangle(box, fill=255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def job(*boxes):
    return SimpleNamespace(regions=[SimpleNamespace(mask=mask(b)) for b in boxes], resize_mode=0)


SMALL, BACKGROUND, EMPTY = (10, 10, 40, 40), (0, 0, W - 1, H - 1), (0, 0, -1, -1)


def test_backgrounds_only_when_asked():
    j = job(BACKGROUND, SMALL)
    assert [k for k, _ in region_pass.areas(j, None, (W, H))] == [1]
    assert [k for k, _ in region_pass.areas(j, None, (W, H), backgrounds=True)] == [0, 1]


def test_areas_grow_and_respect_the_limit():
    (_, area), = region_pass.areas(job(SMALL), None, (W, H))
    x1, y1, x2, y2 = area.getbbox()
    assert x1 < 10 and x2 > 41  # grown past the region
    left = Image.new("L", (W, H), 0)
    ImageDraw.Draw(left).rectangle((0, 0, 24, H), fill=255)
    (_, area), = region_pass.areas(job(SMALL), None, (W, H), limit_mask=left)
    assert area.getbbox()[2] <= 25
    assert region_pass.areas(job(SMALL), None, (W, H), limit_mask=Image.new("L", (W, H), 0)) == []


def test_coverage_is_the_union():
    found = region_pass.areas(job(SMALL, (150, 50, 190, 90)), None, (W, H))
    cover = region_pass.coverage(found, (W, H))
    assert cover.getpixel((25, 25)) == 255 and cover.getpixel((170, 70)) == 255 and cover.getpixel((100, 5)) == 0
    assert region_pass.coverage([], (W, H)) is None


def test_job_validation():
    region = jobs.Region(prompt="a", mask=mask(SMALL))
    assert jobs.unsupported(jobs.Job(mode="img2img", init_image=mask(SMALL), regions=[region], regions_only=True)) is None
    assert "regions" in jobs.unsupported(jobs.Job(mode="img2img", init_image=mask(SMALL), regions_only=True))
    assert "regions" in jobs.unsupported(jobs.Job(mode="txt2img", regions=[region], regions_only=True))
