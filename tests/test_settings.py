"""Settings saved in the Studio that apply from the next start (venno/config.py)."""

import pytest

from venno import config


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(config.args, "data_dir", str(tmp_path))
    monkeypatch.setattr(config, "startup", {"vram_mode": "auto", "vram_from": "default"})
    return tmp_path


def test_saved_mode_is_added_at_start(data):
    config.save_settings(vram_mode="lowvram")
    argv = ["venno", "--port", "7861"]
    config.apply_saved_settings(argv)
    assert argv[-1] == "--lowvram" and config.startup == {"vram_mode": "lowvram", "vram_from": "settings"}
    assert config.load_settings() == {"vram_mode": "lowvram"}


def test_command_line_wins(data):
    config.save_settings(vram_mode="novram")
    argv = ["venno", "--highvram"]
    config.apply_saved_settings(argv)
    assert argv == ["venno", "--highvram"] and config.startup == {"vram_mode": "highvram", "vram_from": "command line"}


def test_automatic_adds_nothing(data):
    argv = ["venno"]
    config.apply_saved_settings(argv)  # no settings file yet
    config.save_settings(vram_mode="auto")
    config.apply_saved_settings(argv)
    assert argv == ["venno"] and config.startup["vram_mode"] == "auto"


def test_settings_merge_and_survive_bad_files(data):
    (data / config.SETTINGS_FILE).write_text("not json")
    assert config.load_settings() == {}
    config.save_settings(vram_mode="lowvram")
    assert config.save_settings(other=1) == {"vram_mode": "lowvram", "other": 1}
