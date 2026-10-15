"""
Venno: the classic Gradio interface has been removed (Venno Studio is the only front-end).
This module only keeps the small helpers that core code and extension scripts still import.
"""

import math
import mimetypes
import warnings

import gradio as gr
import gradio.utils

from modules import gradio_extensions, sd_models, ui_common
from modules.shared import cmd_opts, opts

if not opts.show_warnings:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
if not opts.show_gradio_deprecation_warnings:
    warnings.filterwarnings("ignore", category=gradio_extensions.GradioDeprecationWarning)

# explicit content types (Windows registry can map .js to text/html)
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("image/avif", ".avif")

if not cmd_opts.share and not cmd_opts.listen:
    # stop gradio phoning home
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: "127.0.0.1"

if cmd_opts.ngrok is not None:
    import modules.ngrok as ngrok

    print("ngrok authtoken detected, trying to connect...")
    ngrok.connect(cmd_opts.ngrok, cmd_opts.port if cmd_opts.port is not None else 7860, cmd_opts.ngrok_options)


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def use_cfg(val: float | None):
    return gr.skip() if val is None else gr.update(interactive=(val > 1.0))


def no_config(*comps: gr.components.Component):
    for comp in comps:
        setattr(comp, "_internal_preset_param", True)


def cleanup():
    from modules_forge.main_thread import last_exception

    if last_exception == "OOM":
        sd_models.unload_model_weights()


# symbols still referenced by extension scripts
random_symbol = "\U0001f3b2️"  # 🎲️
reuse_symbol = "♻️"  # ♻️
paste_symbol = "↙️"  # ↙
refresh_symbol = "\U0001f504"  # 🔄
save_style_symbol = "\U0001f4be"  # 💾
apply_style_symbol = "\U0001f4cb"  # 📋
clear_prompt_symbol = "\U0001f5d1️"  # 🗑️
extra_networks_symbol = "\U0001f3b4"  # 🎴
switch_values_symbol = "\U000021c5"  # ⇅
restore_progress_symbol = "\U0001f300"  # 🌀
detect_image_size_symbol = "\U0001f4d0"  # 📐


plaintext_to_html = ui_common.plaintext_to_html


_STEP = int(opts.res_step)


def sRound(val: int | float) -> int:
    return math.floor(val / _STEP + 0.5) * _STEP
