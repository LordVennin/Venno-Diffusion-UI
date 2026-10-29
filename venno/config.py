"""Venno settings: command line options and folders.

Folders keep the familiar layout (models/Stable-diffusion, models/Lora, …), so an existing Forge / A1111
models folder can be used as it is (`--models-dir`). Options are read from the command line with
`parse_known_args`, so the engine's own options (`--cpu`, `--lowvram`, …, see backend/args.py) can be
given on the same line.
"""

from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(prog="venno", description="Venno Diffusion UI", add_help=False)
parser.add_argument("-h", "--help", action="store_true", help="show this help and exit")
server = parser.add_argument_group("server")
server.add_argument("--port", type=int, default=7860, help="port to serve on (default 7860)")
server.add_argument("--listen", action="store_true", help="serve on the network (0.0.0.0), not just this computer")
server.add_argument("--host", type=str, default=None, help="address to bind to (overrides --listen)")
server.add_argument("--autolaunch", action="store_true", help="open the Studio in the browser on start")
server.add_argument("--skip-install", action="store_true", help="launcher: don't check / install packages on start")
server.add_argument("--allow-custom-downloads", action="store_true", help="allow downloading models from any URL when serving on the network (always allowed locally)")
folders = parser.add_argument_group("folders")
folders.add_argument("--data-dir", type=str, default=None, help="base folder for models, outputs, wildcards and tags (default: the install folder)")
folders.add_argument("--models-dir", type=str, default=None, help="models folder (default: <data-dir>/models; a Forge / A1111 models folder works)")
folders.add_argument("--outputs-dir", type=str, default=None, help="where results are saved (default: <data-dir>/outputs)")
folders.add_argument("--ckpt-dirs", type=str, nargs="*", default=[], help="extra checkpoint folders")
folders.add_argument("--text-encoder-dirs", type=str, nargs="*", default=[], help="extra text encoder folders")
folders.add_argument("--vae-dirs", type=str, nargs="*", default=[], help="extra VAE folders")
folders.add_argument("--lora-dir", type=str, default=None, help="LoRA folder (default: <models>/Lora)")
folders.add_argument("--lora-dirs", type=str, nargs="*", default=[], help="extra LoRA folders")
folders.add_argument("--controlnet-dir", type=str, default=None, help="guide model folder (default: <models>/ControlNet)")
folders.add_argument("--controlnet-dirs", type=str, nargs="*", default=[], help="extra guide model folders")
folders.add_argument("--esrgan-models-path", type=str, default=None, help="extra upscaler folder")

args, _unknown = parser.parse_known_args()


# ---------------------------------------------------------------- settings made in the Studio

SETTINGS_FILE = "venno-settings.json"  # in the data folder; settings that apply from the next start
VRAM_MODES = {"auto": None, "lowvram": "--lowvram", "novram": "--novram"}  # Settings → System → Memory
VRAM_FLAGS = ("--gpu-only", "--highvram", "--lowvram", "--novram", "--cpu")  # the engine's mutually exclusive memory options
startup = {"vram_mode": "auto", "vram_from": "default"}  # what this run started with (set by apply_saved_settings)


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
    """Merge `updates` into the settings file (written atomically) and return all settings."""
    data = {**load_settings(), **updates}
    tmp = settings_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, settings_path())
    return data


def apply_saved_settings(argv: list[str]) -> None:
    """At server start, before the engine reads its options: add the saved memory mode to `argv`, unless the
    command line already sets one (the command line wins)."""
    given = next((a for a in argv[1:] if a in VRAM_FLAGS), None)
    if given:
        startup.update(vram_mode=given.lstrip("-"), vram_from="command line")  # e.g. lowvram, highvram, cpu
        return
    mode = load_settings().get("vram_mode", "auto")
    flag = VRAM_MODES.get(mode)
    if flag:
        argv.append(flag)
        startup.update(vram_mode=mode, vram_from="settings")


def data_path() -> str:
    return os.path.abspath(args.data_dir or ROOT)


def models_path() -> str:
    return os.path.abspath(args.models_dir or os.path.join(data_path(), "models"))


def outputs_path() -> str:
    return os.path.abspath(args.outputs_dir or os.path.join(data_path(), "outputs"))


def _existing(dirs) -> list[str]:
    seen, out = set(), []
    for d in dirs:
        if d and os.path.isdir(d) and os.path.realpath(d) not in seen:
            seen.add(os.path.realpath(d))
            out.append(d)
    return out


def model_folder(kind: str) -> str:
    """The folder new files of a kind go to (created on demand)."""
    m = models_path()
    folder = {
        "checkpoint": (args.ckpt_dirs or [None])[0] or os.path.join(m, "Stable-diffusion"),
        "text_encoder": (args.text_encoder_dirs or [None])[0] or os.path.join(m, "text_encoder"),
        "vae": (args.vae_dirs or [None])[0] or os.path.join(m, "VAE"),
        "lora": args.lora_dir or os.path.join(m, "Lora"),
        "controlnet": args.controlnet_dir or os.path.join(m, "ControlNet"),
        "upscaler": args.esrgan_models_path or os.path.join(m, "ESRGAN"),
        "adetailer": os.path.join(m, "adetailer"),
        "reference_adapter": os.path.join(m, "reference_adapter"),
    }[kind]
    os.makedirs(folder, exist_ok=True)
    return folder


def model_folders(kind: str) -> list[str]:
    """Every existing folder searched for a kind of model."""
    m = models_path()
    dirs = {
        "checkpoint": [*args.ckpt_dirs, os.path.join(m, "Stable-diffusion"), os.path.join(m, "diffusion_models"), os.path.join(m, "unet")],
        "text_encoder": [*args.text_encoder_dirs, os.path.join(m, "text_encoder"), os.path.join(m, "text_encoders"), os.path.join(m, "clip")],
        "vae": [*args.vae_dirs, os.path.join(m, "VAE"), os.path.join(m, "vae")],
        "lora": [args.lora_dir, *args.lora_dirs, os.path.join(m, "Lora"), os.path.join(m, "loras")],
        "controlnet": [args.controlnet_dir, *args.controlnet_dirs, os.path.join(m, "ControlNet"), os.path.join(m, "controlnet")],
        "upscaler": [args.esrgan_models_path, *(os.path.join(m, d) for d in ("ESRGAN", "RealESRGAN", "DAT", "SwinIR", "HAT", "upscale_models"))],
        "adetailer": [os.path.join(m, "adetailer")],
        "reference_adapter": [os.path.join(m, "reference_adapter")],
    }[kind]
    return _existing(dirs)


def listen_host() -> str:
    return args.host or ("0.0.0.0" if args.listen else "127.0.0.1")


def is_shared() -> bool:
    """Served beyond this computer: some actions (downloads from arbitrary URLs) need opting in."""
    return listen_host() not in ("127.0.0.1", "localhost", "::1")
