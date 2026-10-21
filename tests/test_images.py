import numpy as np
import pytest
from PIL import Image

from venno.engine import images


def test_snap():
    assert images.snap(1010) == 1008 and images.snap(1) == 16 and images.snap(1024) == 1024


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_resize_modes_give_the_size(mode):
    img = Image.new("RGB", (300, 100), (255, 0, 0))
    assert images.resize(img, 128, 128, mode).size == (128, 128)
    assert images.resize(img.convert("L"), 64, 96, mode).size == (64, 96)


def test_fit_mode_masks_get_empty_borders():
    mask = Image.new("L", (300, 100), 255)
    out = np.asarray(images.resize(mask, 100, 100, 2))
    assert out[0, 50] == 0 and out[50, 50] == 255


def test_crop_region_keeps_aspect_and_bounds():
    mask = Image.new("L", (512, 512), 0)
    mask.paste(255, (200, 200, 260, 240))
    x1, y1, x2, y2 = images.crop_region(mask, 16, 1.5)
    assert 0 <= x1 < x2 <= 512 and 0 <= y1 < y2 <= 512
    assert (x2 - x1) / (y2 - y1) == pytest.approx(1.5, rel=0.02)
    assert x1 <= 184 and x2 >= 276 and y1 <= 184 and y2 >= 256


def test_latent_mask_shape():
    m = images.latent_mask(Image.new("L", (256, 128), 255), 32, 16)
    assert tuple(m.shape) == (1, 1, 1, 16, 32) and float(m.min()) == pytest.approx(1.0)


def test_composite_keeps_unmasked_pixels_exact():
    a = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    b = Image.new("RGB", (64, 64), (0, 0, 0))
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (16, 16, 48, 48))
    out = np.asarray(images.composite(a, b, mask))
    assert (out[:8] == np.asarray(a)[:8]).all()
    assert (out[24:40, 24:40] == 0).all()


def test_fill_masked_only_changes_the_masked_area():
    img = Image.new("RGB", (64, 64), (200, 50, 50))
    img.paste((0, 0, 255), (20, 20, 44, 44))
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (20, 20, 44, 44))
    out = np.asarray(images.fill_masked(img, mask)).astype(int)
    assert (out[:10] == [200, 50, 50]).all()
    assert abs(out[32, 32] - [200, 50, 50]).max() < 40, "the hole takes the surrounding colour"


def test_load_data_urls():
    import base64
    import io

    buf = io.BytesIO()
    Image.new("RGBA", (8, 8), (255, 0, 0, 0)).save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    assert images.load_rgb(url).getpixel((0, 0)) == (255, 255, 255), "transparent → background"
    assert images.load_mask(url).size == (8, 8)
