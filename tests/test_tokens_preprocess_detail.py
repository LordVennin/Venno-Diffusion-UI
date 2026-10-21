import numpy as np
import pytest
from PIL import Image, ImageDraw

from venno.engine import preprocess, tokens
from venno.engine.detect import Detection
from venno.pipeline import detail


def test_token_count():
    assert tokens.count("") == 0 and tokens.count("<lora:x:1>") == 0
    base = tokens.count("masterpiece, 1girl")
    assert base > 0
    assert tokens.count("masterpiece, 1girl <lora:abc:0.5>") == base, "LoRA tags don't count"
    assert tokens.count("[short:a much longer prompt with many more words:0.5]") > tokens.count("short"), "the longest variant counts"
    assert tokens.count("word, " * 300) > tokens.LIMIT


@pytest.fixture
def drawing():
    img = Image.new("RGB", (128, 128), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((30, 30, 98, 98), fill=(200, 60, 60), outline="black", width=3)
    return img


@pytest.mark.parametrize("name", ["canny", "lineart", "softedge"])
def test_line_maps_follow_the_models_polarity(drawing, name):
    dark_on_white = np.asarray(preprocess.run(name, drawing, "anima-lllite-any-test-like-v2").convert("L"))
    white_on_black = np.asarray(preprocess.run(name, drawing, "anima-lllite-scribble-1").convert("L"))
    assert dark_on_white.shape == (128, 128)
    assert dark_on_white.mean() > 180 and white_on_black.mean() < 75
    assert dark_on_white.min() < 100, "the outline is drawn"


def test_none_passes_the_picture_through(drawing):
    assert np.array_equal(np.asarray(preprocess.run("none", drawing)), np.asarray(drawing))


def test_unknown_preprocessor(drawing):
    with pytest.raises(ValueError):
        preprocess.run("pose", drawing)


def test_suggested_model():
    models = ["anima-lllite-depth-1", "anima-lllite-lineart-1", "anima-lllite-any-test-like-v2", "anima-lllite-scribble-1"]
    assert preprocess.suggested_model("lineart", models) == "anima-lllite-any-test-like-v2"
    assert preprocess.suggested_model("depth", models) == "anima-lllite-depth-1"
    assert preprocess.suggested_model("depth", ["anima-lllite-lineart-1"]) == ""


def _det(box, score):
    mask = Image.new("L", (200, 160), 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return Detection(box=box, mask=mask, score=score)


def test_detection_report_and_preview():
    found = [_det((20, 20, 80, 90), 0.91), _det((120, 30, 180, 100), 0.54)]
    report = [
        dict(model="face_yolov8n.pt", confidence=0.3, found=found, kept=[True, False]),
        dict(model="hand_yolov8n.pt", confidence=0.3, found=[], kept=[]),
    ]
    assert detail.summary(report) == "face_yolov8n.pt: 1 found (0.91), 1 outside the mask; hand_yolov8n.pt: nothing found"
    picture = Image.new("RGB", (200, 160), (90, 90, 90))
    out = detail.draw_detections(picture, report)
    assert out.size == picture.size and out.mode == "RGB"
    assert out.getpixel((20, 50)) != picture.getpixel((20, 50)), "the box is drawn"
    assert np.asarray(picture).std() == 0, "the original is left alone"
    assert detail.preview_png(picture, report)[:4] == b"\x89PNG"
