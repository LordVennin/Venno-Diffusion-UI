from venno.pipeline import infotext, metadata

PARAMS = {
    "Steps": 30, "Sampler": "Euler", "Schedule type": "Beta", "CFG scale": 4.5, "Shift": 3.0, "Seed": 1234, "Size": "832x1216",
    "Model": "anima", "Denoising strength": 0.4, "Hires resize": "1248x1824", "Hires upscaler": "Latent", "Hires steps": 12,
    "Prompt switch": "a dog", "Prompt switch at": 0.5,
    "Regions": 2, "Region base": 0.3, "Region separation": 0.7, "Region pass": 0.35,
    "Region 1": "1girl, red", "Region 2": "1boy, blue", "Region 2 negative": "hat",
    "Frames": 3, "Frame 1 prompt": "smile", "Frame 3 prompt": "angry", "Frame separation": 0.2,
    "Reference images": 1, "Reference noise": 0.5, "Reference strength": 0.85,
    "ADetailer model": "face_yolov8n.pt", "ADetailer confidence": 0.3, "ADetailer denoising strength": 0.4, "ADetailer prompt": "smile",
}


def test_round_trip():
    text = metadata.infotext("masterpiece, 1girl <lora:x:0.6>", "worst quality", PARAMS)
    s = infotext.settings(text)
    assert s["prompt"].startswith("masterpiece, 1girl") and s["negative"] == "worst quality"
    assert s["loras"] == [{"name": "x", "weight": 0.6}]
    assert (s["steps"], s["sampler"], s["scheduler"], s["cfg"], s["seed"]) == (30, "Euler", "Beta", 4.5, 1234)
    assert (s["width"], s["height"]) == (832, 1216)
    assert s["hires"]["resize_x"] == 1248 and s["hires"]["steps"] == 12 and s["hires"]["denoise"] == 0.4
    assert s["switch_prompt"] == "a dog" and s["switch_at"] == 0.5
    assert s["regions"] == ["1girl, red", "1boy, blue"] and s["region_negatives"] == ["", "hat"]
    assert s["region_separation"] == 0.7 and s["region_refine"] == 0.35
    assert s["frames"] == 3 and s["frame_prompts"] == ["smile", "", "angry"] and s["frame_separation"] == 0.2
    assert s["reference_noise"] == 0.5 and s["reference_strength"] == 0.85 and s["reference_follow"] is False
    assert s["detailer"][0]["model"] == "face_yolov8n.pt" and s["detailer"][0]["prompt"] == "smile"


def test_reference_follow():
    s = infotext.settings(metadata.infotext("x", "", {"Steps": 4, "Reference images": 1, "Reference noise": "follow"}))
    assert s["reference_follow"] is True and "reference_noise" not in s


def test_a1111_text():
    text = "1girl, smile\nNegative prompt: bad hands\nSteps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 42, Size: 512x768, Model: x"
    s = infotext.settings(text)
    assert s["prompt"] == "1girl, smile" and s["negative"] == "bad hands"
    assert (s["steps"], s["seed"], s["width"], s["height"]) == (20, 42, 512, 768)
    assert s["sampler"] == "DPM++ 2M" and s["scheduler"] == "Karras", "old combined sampler names are split"


import pytest


@pytest.mark.parametrize("value", ["a, b", "a: b", 'say "hi", smile', '"OPEN" sign, 1girl', "two\nlines, here", 'a, b: "c"', "back\\slash, x"])
def test_awkward_values_survive(value):
    text = metadata.infotext("x", "", {"Steps": 4, "Region 1": value, "Seed": 1, "Sampler": "Euler"})
    s = infotext.settings(text)
    assert s["regions"] == [value] and s["seed"] == 1 and s["sampler"] == "Euler"


def test_png_round_trip():
    from PIL import Image

    data = metadata.png_bytes(Image.new("RGB", (8, 8)), "a cat\nSteps: 4, Seed: 7")
    text, _ = infotext.read_text(data)
    assert text == "a cat\nSteps: 4, Seed: 7"
