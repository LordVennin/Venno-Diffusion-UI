"""
Venno Studio: serves the canvas front-end and a few helper endpoints.

Generation goes through the standard /sdapi/v1/txt2img and /sdapi/v1/img2img
endpoints (with the "venno regional prompting" and "controlnet" always-on scripts),
so queueing, progress and interruption behave exactly like the rest of Forge.
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from modules import paths, script_callbacks, shared
from modules.shared_cmd_options import cmd_opts

from lib_venno import downloads

STUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studio")

# Recommended settings from the Anima model card
PRESETS = {
    "turbo": {"label": "Anima Turbo", "sampler": "Euler", "scheduler": "Simple", "steps": 10, "cfg": 1.0, "shift": 3.0},
    "standard": {"label": "Anima Base / Aesthetic", "sampler": "ER SDE", "scheduler": "Beta", "steps": 32, "cfg": 4.0, "shift": 3.0},
}
PROMPT_DEFAULTS = {
    "positive_prefix": "masterpiece, best quality, score_7, safe, ",
    "negative": "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration",
}


def _loras() -> list[dict]:
    try:
        import networks  # sd_forge_lora

        if not networks.available_networks:
            networks.list_available_networks()
        return sorted(
            ({"name": name, "alias": getattr(n, "alias", name)} for name, n in networks.available_networks.items()),
            key=lambda x: x["name"].lower(),
        )
    except Exception as e:  # LoRA extension disabled or broken
        print(f"[Venno] could not list LoRAs: {e}")
        return []


def _controlnets() -> list[str]:
    try:
        from lib_controlnet import global_state

        return [m for m in global_state.get_all_controlnet_names() if m != "None"]
    except Exception:
        return []


def _adetailer_models() -> list[str]:
    folder = os.path.join(paths.models_path, "adetailer")
    try:
        return sorted(f for f in os.listdir(folder) if f.lower().endswith((".pt", ".tflite", ".task")) and f != "face_landmarker.task")
    except OSError:
        return []


def _info() -> dict:
    from modules import sd_models, sd_samplers, sd_schedulers
    from modules_forge.main_entry import module_list

    return {
        "checkpoints": [{"title": x.title, "name": x.model_name} for x in sd_models.checkpoints_list.values()],
        "modules": [{"name": k, "filename": v} for k, v in module_list.items()],
        "current_checkpoint": shared.opts.sd_model_checkpoint,
        "current_modules": list(shared.opts.forge_additional_modules or []),
        "samplers": [s.name for s in sd_samplers.visible_samplers()],
        "schedulers": [s.label for s in sd_schedulers.schedulers],
        "loras": _loras(),
        "controlnets": _controlnets(),
        "upscalers": [*shared.latent_upscale_modes.keys(), *(u.name for u in shared.sd_upscalers if u.name != "None")],
        "adetailer_models": _adetailer_models(),
        "presets": PRESETS,
        "prompt_defaults": PROMPT_DEFAULTS,
    }


def _refresh():
    from modules import shared_items
    from modules_forge import main_entry

    shared.refresh_checkpoints()
    shared_items.refresh_vae_list()
    try:
        main_entry.refresh_models()
    except Exception:
        pass
    try:
        from modules import modelloader

        modelloader.load_upscalers()
    except Exception:
        pass
    try:
        import networks

        networks.list_available_networks()
    except Exception:
        pass
    try:
        from lib_controlnet import global_state

        global_state.update_controlnet_filenames()
    except Exception:
        pass


def on_app_started(_demo, app: FastAPI):
    app.add_api_route("/venno/api/info", _info, methods=["GET"])

    def refresh():
        _refresh()
        return _info()

    app.add_api_route("/venno/api/refresh", refresh, methods=["POST"])

    # ---- model downloads ----
    def on_download_done(_job):
        try:
            _refresh()
        except Exception as e:
            print(f"[Venno] refresh after download failed: {e}")

    def catalog():
        return {"catalog": downloads.catalog(), "kinds": downloads.KINDS, "jobs": downloads.jobs(), "custom_urls": not cmd_opts.disable_extension_access}

    def download(req: dict):
        try:
            if req.get("id"):
                entry = next((c for c in downloads.CATALOG if c[0] == req["id"]), None)
                if entry is None:
                    raise ValueError("Unknown catalog entry")
                cid, kind, name, _desc, url, _size = entry
                job = downloads.start(url, kind, name=name, on_done=on_download_done)
            else:
                # arbitrary URLs write files to disk: same rule as installing extensions when listening on the network
                if cmd_opts.disable_extension_access:
                    raise ValueError("Downloading from custom URLs is disabled when the server is shared on the network (use --enable-insecure-extension-access)")
                job = downloads.start(str(req.get("url", "")).strip(), str(req.get("kind", "")), filename=req.get("filename") or None, on_done=on_download_done)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return job.to_dict()

    def download_jobs():
        return downloads.jobs()

    def download_cancel(req: dict):
        downloads.cancel(str(req.get("id", "")))
        return downloads.jobs()

    app.add_api_route("/venno/api/models/catalog", catalog, methods=["GET"])
    app.add_api_route("/venno/api/models/download", download, methods=["POST"])
    app.add_api_route("/venno/api/models/jobs", download_jobs, methods=["GET"])
    app.add_api_route("/venno/api/models/cancel", download_cancel, methods=["POST"])

    app.mount("/studio/assets", StaticFiles(directory=STUDIO_DIR), name="venno_studio_assets")

    def studio_index():
        return FileResponse(os.path.join(STUDIO_DIR, "index.html"), headers={"Cache-Control": "no-cache"})

    app.add_api_route("/studio", studio_index, methods=["GET"], include_in_schema=False)
    app.add_api_route("/studio/", studio_index, methods=["GET"], include_in_schema=False)

    app.add_api_route("/", lambda: RedirectResponse("/studio/"), methods=["GET"], include_in_schema=False)
    port = cmd_opts.port or 7860
    print(f"[Venno] Studio running on: http://127.0.0.1:{port}/")
    _maybe_open_browser(port)


def _maybe_open_browser(port: int):
    if os.getenv("SD_WEBUI_RESTARTING") == "1":
        return
    launch = cmd_opts.autolaunch or shared.opts.auto_launch_browser == "Remote" or (shared.opts.auto_launch_browser == "Local" and not cmd_opts.webui_is_non_local)
    if not launch:
        return

    import threading
    import webbrowser

    # the API server starts listening right after the app_started callbacks
    threading.Timer(2.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()


script_callbacks.on_app_started(on_app_started, name="venno_studio")
