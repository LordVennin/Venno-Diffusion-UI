"""Folders, saved settings and command-line options of Venno MOSS (the MOSS-Image LoRA tester).

It lives next to Venno in the same install and uses the same data folder, so models, LoRAs and outputs sit
beside Venno's: models/moss (MOSS-Image models), models/moss-lora (LoRAs), outputs/moss (results). More model
and LoRA folders (e.g. venno-trainer's models and output folders) are added in Settings and saved in
venno-moss-settings.json.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE = "venno-moss-settings.json"
MEMORY_MODES = ("auto", "full", "low")  # full: bf16 DiT (about 24 GB cards); low: fp8 DiT (about 12 GB cards)

parser = argparse.ArgumentParser(prog="python -m venno_moss", description="Venno MOSS: test MOSS-Image LoRAs (Studio at http://127.0.0.1:7870/)")
parser.add_argument("--port", type=int, default=7870)
parser.add_argument("--listen", action="store_true", help="accept connections from other computers (0.0.0.0)")
parser.add_argument("--autolaunch", action="store_true", help="open the browser when ready")
parser.add_argument("--data-dir", default=None, help="base folder for models and outputs (default: the install folder, like Venno)")
parser.add_argument("--models-dir", default=None, help="where MOSS-Image models are downloaded (default: <data-dir>/models/moss)")
parser.add_argument("--lora-dir", action="append", default=[], help="a LoRA folder (repeat for more; also settable in Settings)")
parser.add_argument("--outputs-dir", default=None, help="where results are saved (default: <data-dir>/outputs/moss)")
parser.add_argument("--memory", choices=MEMORY_MODES, default=None, help="full: bf16 transformer (24 GB cards); low: fp8 transformer (12 GB cards); default: saved setting, else auto")
parser.add_argument("--cpu", action="store_true", help="run on the CPU (very slow; for testing)")
parser.add_argument("--skip-install", action="store_true", help="don't check or install packages (you manage venv-moss yourself)")
args, _unknown = parser.parse_known_args([])  # defaults until main() parses the real command line


def parse(argv: list[str]):
    global args
    args = parser.parse_args(argv)
    return args


def data_path() -> str:
    return os.path.abspath(args.data_dir or ROOT)


def models_path() -> str:
    path = os.path.abspath(args.models_dir or os.path.join(data_path(), "models", "moss"))
    os.makedirs(path, exist_ok=True)
    return path


def default_lora_path() -> str:
    path = os.path.join(data_path(), "models", "moss-lora")
    os.makedirs(path, exist_ok=True)
    return path


def outputs_path() -> str:
    return os.path.abspath(args.outputs_dir or os.path.join(data_path(), "outputs", "moss"))


# ---------------------------------------------------------------- saved settings


def settings_path() -> str:
    return os.path.join(data_path(), SETTINGS_FILE)


def load_settings() -> dict:
    try:
        with open(settings_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(**updates) -> dict:
    """Merge `updates` into the settings file (written atomically)."""
    data = {**load_settings(), **updates}
    folder = os.path.dirname(settings_path())
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".venno-moss-settings-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, settings_path())
    return data


def _folders(saved_key: str, first: list[str]) -> list[str]:
    seen, out = set(), []
    for d in [*first, *load_settings().get(saved_key, [])]:
        d = os.path.abspath(os.path.expanduser(d)) if d else ""
        if d and os.path.isdir(d) and os.path.realpath(d) not in seen:
            seen.add(os.path.realpath(d))
            out.append(d)
    return out


def model_folders() -> list[str]:
    """Folders searched for MOSS-Image models: the download folder first, then the ones added in Settings."""
    return _folders("model_dirs", [models_path()])


def lora_folders() -> list[str]:
    return _folders("lora_dirs", [*args.lora_dir, default_lora_path()])


def memory_mode() -> str:
    """The memory mode asked for: command line, else saved setting, else auto."""
    mode = args.memory or load_settings().get("memory", "auto")
    return mode if mode in MEMORY_MODES else "auto"
