"""The Upscale tab's work (venno/pipeline/upscaler.py): tiles, passes, CPU mode, kept settings, the API."""

import base64
import io
import os
import time

import numpy as np
import pytest
import torch
from PIL import Image, PngImagePlugin

from venno.pipeline import upscaler

pytest.importorskip("spandrel")


@pytest.fixture
def model(tmp_path):
    """A tiny 2× upscaler (RealESRGAN Compact) in an upscaler folder."""
    from spandrel.architectures.Compact import Compact

    torch.manual_seed(0)
    folder = tmp_path / "ESRGAN"
    folder.mkdir()
    torch.save(Compact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=2, upscale=2).state_dict(), folder / "2x-Tiny.pth")
    (folder / "notes.txt").write_text("not a model")
    upscaler._loaded.clear()
    return str(folder)


def picture(width=96, height=64, text=None) -> bytes:
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
    info = PngImagePlugin.PngInfo()
    for k, v in (text or {}).items():
        info.add_text(k, v)
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def run(model, source, **kw) -> upscaler.Task:
    path = upscaler.list_models([model])["2x-Tiny"]
    task = upscaler.Task(upscaler.Request(source=source, name="2x-Tiny", path=path, **kw))
    upscaler._work(task)
    return task


def test_tiles_and_passes():
    assert upscaler.tile_starts(200, 256) == [0] and upscaler.tile_starts(256, 256) == [0]
    assert upscaler.tile_starts(600, 256) == [0, 224, 344]  # the last tile ends at the edge
    assert [upscaler.passes_needed(s, 2) for s in (1, 2, 2.5, 4, 8, 16)] == [1, 1, 2, 2, 3, 3]
    assert upscaler.passes_needed(3, 4) == 1 and upscaler.passes_needed(8, 4) == 2 and upscaler.passes_needed(4, 1) == 1


def test_upscale_keeps_settings_and_saves(model, tmp_path):
    assert list(upscaler.list_models([model, str(tmp_path / "missing")])) == ["2x-Tiny"]
    source = picture(text={"parameters": "a fox\nSteps: 8, Seed: 7, Size: 96x64", "venno": '{"seed": 7}'})
    task = run(model, source, scale=2, save_dir=str(tmp_path / "out"))
    assert task.status == "done" and task.size == (192, 128) and task.source_size == (96, 64) and task.device == "cpu", task.error
    assert (task.done, task.total) == (1, 1)
    out = Image.open(io.BytesIO(task.png))
    assert out.size == (192, 128)
    assert out.info["parameters"] == "a fox\nSteps: 8, Seed: 7, Size: 96x64, Postprocess upscale by: 2, Postprocess upscaler: 2x-Tiny"
    assert out.info["venno"] == '{"seed": 7}' and out.info["upscale"].endswith("2x-Tiny")
    assert os.path.isfile(task.saved) and os.path.basename(task.saved) == "00000-2x-Tiny.png" and task.req.source == b""

    # a picture without settings gets none invented; scale 3 with a 2× model: two passes, then down to the size asked for
    task = run(model, picture(), scale=3)
    assert task.status == "done" and task.size == (288, 192) and (task.done, task.total) == (2, 2) and task.saved == ""
    assert "parameters" not in Image.open(io.BytesIO(task.png)).info


def test_tiles_blend_into_the_untiled_result(model):
    source = picture(300, 280)
    whole = run(model, source, scale=2, tile=1024)
    tiled = run(model, source, scale=2, tile=256)
    assert (whole.total, tiled.total) == (1, 4) and tiled.done == 4
    a, b = (np.asarray(Image.open(io.BytesIO(t.png)), dtype=int) for t in (whole, tiled))
    assert a.shape == (560, 600, 3) and np.abs(a - b).mean() < 1.0


def test_cpu_mode_leaves_the_graphics_card_alone(model):
    gpu = torch.device("cuda")
    with upscaler.gpu_lock:  # a picture is being generated
        task = run(model, picture(), scale=2, cpu=True, device=gpu)
        assert task.status == "done" and task.device == "cpu"  # CPU mode doesn't wait and doesn't move the model
        waiting = upscaler.Task(upscaler.Request(source=picture(), name="2x-Tiny", path=upscaler.list_models([model])["2x-Tiny"], device=gpu))
        assert waiting.device == "cuda"
        import threading

        worker = threading.Thread(target=upscaler._work, args=(waiting,))
        worker.start()
        deadline = time.time() + 5
        while "graphics card" not in waiting.phase and time.time() < deadline:
            time.sleep(0.02)
        assert waiting.status == "queued" and "Waiting for the graphics card" in waiting.phase
        waiting.cancel.set()
        worker.join(5)
        assert waiting.status == "cancelled"
    assert upscaler.gpu_lock.acquire(timeout=1) and upscaler._one_at_a_time.acquire(timeout=1)  # both were given back
    upscaler.gpu_lock.release()
    upscaler._one_at_a_time.release()


def test_errors(model):
    path = upscaler.list_models([model])["2x-Tiny"]
    task = run(model, b"not a picture")
    assert task.status == "error" and task.error
    task = run(model, picture(4096, 8), scale=8)
    assert task.status == "error" and "16384" in task.error
    with pytest.raises(ValueError, match="Tile size"):
        upscaler.start(upscaler.Request(source=picture(), name="x", path=path, tile=100))
    with pytest.raises(ValueError, match="Scale"):
        upscaler.start(upscaler.Request(source=picture(), name="x", path=path, scale=12))


def test_api(model, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    upscaler.add_routes(app, "/x/v1", folders=lambda: [model], device=lambda: torch.device("cpu"), save_dir=lambda: str(tmp_path / "saved"))
    client = TestClient(app)
    listing = client.get("/x/v1/upscale/models").json()
    assert listing["models"] == ["2x-Tiny"] and listing["gpu"] == "" and 512 in listing["tiles"]
    image = "data:image/png;base64," + base64.b64encode(picture()).decode()
    assert client.post("/x/v1/upscale", json={"image": image, "upscaler": "nope"}).status_code == 400
    assert client.post("/x/v1/upscale", json={"image": "", "upscaler": "2x-Tiny"}).status_code == 400
    assert client.post("/x/v1/upscale", json={"image": image, "upscaler": "2x-Tiny", "scale": 20}).status_code == 400
    started = client.post("/x/v1/upscale", json={"image": image, "upscaler": "2x-Tiny", "scale": 2, "cpu": True, "tile": 256, "save": False}).json()
    deadline = time.time() + 20
    while (state := client.get(f"/x/v1/upscale/{started['id']}").json())["status"] in ("queued", "running") and time.time() < deadline:
        time.sleep(0.05)
    assert state["status"] == "done" and (state["width"], state["height"]) == (192, 128) and state["saved"] == "" and state["device"] == "cpu"
    got = client.get(f"/x/v1/upscale/{started['id']}/image")
    assert got.status_code == 200 and Image.open(io.BytesIO(got.content)).size == (192, 128)
    assert client.get("/x/v1/upscale/unknown").status_code == 404 and not (tmp_path / "saved").exists()
    assert client.post(f"/x/v1/upscale/{started['id']}/cancel").json()["status"] == "done"
