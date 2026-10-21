import json

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from venno.engine import lora, lora_library as lib


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setattr(lora, "folders", lambda: [str(tmp_path)])
    t = {"w": torch.zeros(2)}
    (tmp_path / "chars" / "genshin").mkdir(parents=True)
    save_file(t, str(tmp_path / "top.safetensors"), metadata={"ss_base_model_version": "anima"})
    save_file(t, str(tmp_path / "chars" / "genshin" / "furina.safetensors"))
    save_file(t, str(tmp_path / "chars" / "oc.safetensors"))
    Image.new("RGB", (40, 50), "red").save(tmp_path / "chars" / "oc.png")  # A1111 preview name
    (tmp_path / "chars" / "genshin" / "furina.json").write_text(json.dumps({"activation text": "furina", "preferred weight": 0.7, "sd version": "Unknown"}))
    (tmp_path / "top.civitai.info").write_text(json.dumps({"trainedWords": ["sparkle", " glow "]}))
    lora.scan()
    return tmp_path


def by_name(entries):
    return {e["name"]: e for e in entries}


def test_entries(library):
    e = by_name(lib.entries())
    assert e["top"]["folder"] == "" and e["furina"]["folder"] == "chars/genshin" and e["oc"]["folder"] == "chars"
    assert e["top"]["base"] == "anima" and e["top"]["activation"] == "sparkle, glow", "Civitai Helper trigger words"
    assert e["furina"]["activation"] == "furina" and e["furina"]["weight"] == 0.7
    assert e["oc"]["preview"] and e["furina"]["preview"] is None


def test_preview_bytes(library):
    data, media = lib.preview_bytes("oc")
    assert media == "image/png" and data[:4] == b"\x89PNG"
    with pytest.raises(FileNotFoundError):
        lib.preview_bytes("furina")


def test_set_preview_writes_forge_name_and_shrinks(library):
    lib.set_preview("furina", Image.new("RGB", (2000, 1000), "blue"))
    file = library / "chars" / "genshin" / "furina.preview.png"
    assert file.exists() and max(Image.open(file).size) == lib.PREVIEW_SIZE
    assert by_name(lib.entries())["furina"]["preview"]
    lib.set_preview("oc", None)
    assert not (library / "chars" / "oc.png").exists()


def test_set_meta_keeps_unknown_keys(library):
    out = lib.set_meta("furina", tags=["character", " genshin ", "", "character"], notes="good at 0.7")
    assert out["tags"] == ["character", "genshin"] and out["notes"] == "good at 0.7" and out["activation"] == "furina"
    saved = json.loads((library / "chars" / "genshin" / "furina.json").read_text())
    assert saved["sd version"] == "Unknown" and saved["tags"] == ["character", "genshin"]


def test_embedded_thumbnail(library):
    import base64
    import io

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "green").save(buf, format="PNG")
    thumb = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    save_file({"w": torch.zeros(2)}, str(library / "thumb.safetensors"), metadata={"modelspec.thumbnail": thumb, "modelspec.architecture": "anima/lora"})
    lora.scan()
    e = by_name(lib.entries())["thumb"]
    assert e["preview"] and e["base"] == "anima/lora"
    assert lib.preview_bytes("thumb")[1] == "image/png"


def test_unknown_lora(library):
    with pytest.raises(ValueError):
        lib.set_meta("nope", notes="x")
