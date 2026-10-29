"""First-run setup (GET /venno/v1/setup): what a new install still needs, and the memory mode it suggests."""

import pytest

from venno import config
from venno.server import app


@pytest.fixture
def models(tmp_path, monkeypatch):
    for name in ("ckpt_dirs", "text_encoder_dirs", "vae_dirs"):
        monkeypatch.setattr(config.args, name, [])
    monkeypatch.setattr(config.args, "models_dir", str(tmp_path))
    monkeypatch.setattr(app, "_memory", lambda: {"saved": "auto", "from_command_line": False, "total_vram_mb": 0})
    return tmp_path


def put(folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(b"")


def test_empty_install(models):
    s = app._setup()
    assert not s["complete"] and s["have"] == {"checkpoint": False, "text_encoder": False, "vae": False}
    assert [m["id"] for m in s["models"]] == ["anima-turbo", "anima-aesthetic", "anima-base"]
    assert {k: r["id"] for k, r in s["required"].items()} == {"text_encoder": "qwen3-te", "vae": "qwen-vae"}
    assert s["folders"]["checkpoint"].startswith(str(models)) and s["suggested_vram_mode"] == "auto"


def test_other_encoders_and_vaes_dont_count(models):
    put(models / "Stable-diffusion", "anima-turbo-v1.1.safetensors")
    put(models / "text_encoder", "clip_l.safetensors")
    put(models / "text_encoder", "qwen35_4b.safetensors")  # Anima 3.8B's second encoder
    put(models / "VAE", "sdxl_vae.safetensors")
    s = app._setup()
    assert s["have"] == {"checkpoint": True, "text_encoder": False, "vae": False}
    assert next(m for m in s["models"] if m["id"] == "anima-turbo")["installed"]


def test_complete_from_comfyui_folders(models):
    put(models / "diffusion_models", "anima-base-v1.0.safetensors")
    put(models / "text_encoders", "qwen_3_06b_base.safetensors")
    put(models / "vae", "qwen_image_vae.safetensors")
    assert app._setup()["complete"]


def test_low_vram_suggested_for_small_cards(models, monkeypatch):
    for vram, mode in ((6144, "lowvram"), (12288, "auto"), (0, "auto")):
        monkeypatch.setattr(app, "_memory", lambda v=vram: {"total_vram_mb": v})
        assert app._setup()["suggested_vram_mode"] == mode
