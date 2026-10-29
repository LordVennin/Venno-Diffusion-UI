"""
Venno server: the Studio front-end and the Venno API (/venno/v1/*).

Generation runs through Venno jobs (venno/server/jobs.py): one request schema for every mode,
a single worker queue, and a WebSocket event stream for progress and live previews.
"""

import asyncio
import base64
import hashlib
import math
import os
import re

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from venno import config

from . import downloads, jobs

STUDIO_DIR = os.path.join(config.ROOT, "studio")

# Recommended settings from the Anima model card
PRESETS = {
    "turbo": {"label": "Anima Turbo", "sampler": "Euler", "scheduler": "Simple", "steps": 10, "cfg": 1.0, "shift": 3.0},
    "standard": {"label": "Anima Base / Aesthetic", "sampler": "ER SDE", "scheduler": "Beta", "steps": 32, "cfg": 4.0, "shift": 3.0},
    # Anima 2.9B model card: 28-50 steps, CFG 3.5-5, Euler + SGM Uniform
    "anima29": {"label": "Anima 2.9B", "sampler": "Euler", "scheduler": "SGM Uniform", "steps": 32, "cfg": 4.0, "shift": 3.0},
    # Anima 3.8B model card: 28-50 steps, CFG 4-7, res_multistep + Beta
    "anima38": {"label": "Anima 3.8B", "sampler": "Res Multistep", "scheduler": "Beta", "steps": 32, "cfg": 5.0, "shift": 3.0},
}
PROMPT_DEFAULTS = {
    "positive_prefix": "masterpiece, best quality, score_7, safe, ",
    "negative": "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration",
}


def _loras() -> list[dict]:
    try:
        from venno.engine import lora

        return lora.available()
    except Exception as e:
        print(f"[Venno] could not list LoRAs: {e}")
        return []


def _controlnets() -> list[str]:
    try:
        from venno.engine import guides

        return list(guides.models())
    except Exception as e:
        print(f"[Venno] could not list guide models: {e}")
        return []


def _reference_adapter() -> str:
    """File name of the installed reference adapter ("" when there is none)."""
    from venno.engine import ref_adapter

    path = ref_adapter.find()
    return os.path.basename(path) if path else ""


def _adetailer_models() -> list[str]:
    from venno.engine import detect

    return [m for m in detect.models() if m != "mediapipe_face_mesh"]


class RevalidatedStaticFiles(StaticFiles):
    """The Studio's files with `Cache-Control: no-cache`: without it browsers may keep using cached scripts for
    days after an update, and old scripts with a new page fail to start (empty model list). Unchanged files still
    come from the browser cache (the server answers 304 Not Modified)."""

    def get_path(self, scope) -> str:
        """`v/<stamp>/js/main.js` → `js/main.js`: the page links its files under a version (see _studio_stamp)."""
        parts = re.split(r"[\\/]", super().get_path(scope))
        return os.path.join(*parts[2:]) if len(parts) > 2 and parts[0] == "v" else os.path.join(*parts)

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def _studio_stamp() -> str:
    """Changes whenever a Studio file changes, so an updated page loads all its scripts anew, even when the
    browser still holds old copies (old scripts with a new page fail to start)."""
    h = hashlib.sha1()
    for folder, _, files in sorted(os.walk(STUDIO_DIR)):
        for f in sorted(files):
            st = os.stat(os.path.join(folder, f))
            h.update(f"{folder}/{f}:{st.st_mtime_ns}:{st.st_size};".encode())
    return h.hexdigest()[:12]


def _code_stamp() -> tuple:
    """Fingerprint of the server's Python code on disk (newest change, file count)."""
    newest, count = 0.0, 0
    for top in ("venno", "backend"):
        for dirpath, dirnames, files in os.walk(os.path.join(config.ROOT, top)):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for f in files:
                if f.endswith(".py"):
                    count += 1
                    newest = max(newest, os.path.getmtime(os.path.join(dirpath, f)))
    return newest, count


STARTED_WITH = _code_stamp()  # Venno updated (e.g. git pull) without a restart → the Studio says so


def _memory() -> dict:
    """The memory mode this run uses and the one saved for the next start (Settings → System)."""
    from backend import memory_management

    saved = config.load_settings().get("vram_mode", "auto")
    running = config.startup["vram_mode"]
    from_command_line = config.startup["vram_from"] == "command line"
    return {
        "saved": saved,
        "running": running,
        "from_command_line": from_command_line,
        "state": memory_management.vram_state.name,
        "total_vram_mb": round(memory_management.total_vram) if memory_management.cpu_state is memory_management.CPUState.GPU else 0,
        "total_ram_mb": round(memory_management.total_ram),
        "restart_needed": not from_command_line and saved != running,
    }


def _restart_reason() -> str:
    """Why the running server is out of date: "update" (code changed on disk), "settings" (saved for the next start) or ""."""
    if _code_stamp() != STARTED_WITH:
        return "update"
    return "settings" if _memory()["restart_needed"] else ""


def _info() -> dict:
    from venno.engine import models

    loaded = models.manager.info
    reason = _restart_reason()
    return {
        "restart_needed": bool(reason),
        "restart_reason": reason,
        "checkpoints": [{"title": t, "name": models.checkpoint_name(os.path.basename(t))} for t in models.checkpoints()],
        "modules": [{"name": k, "filename": v} for k, v in models.modules().items()],
        "current_checkpoint": loaded.get("name", ""),
        "current_modules": loaded.get("modules", []),
        "samplers": _engine_list("samplers"),
        "schedulers": _engine_list("schedulers"),
        "loras": _loras(),
        "controlnets": _controlnets(),
        "preprocessors": _preprocessors(),
        "upscalers": _upscalers(),
        "adetailer_models": _adetailer_models(),
        "reference_adapter": _reference_adapter(),
        "wildcards": _wildcards(),
        "engine_samplers": _engine_list("samplers"),
        "engine_schedulers": _engine_list("schedulers"),
        "presets": PRESETS,
        "prompt_defaults": PROMPT_DEFAULTS,
    }


def _preprocessors() -> list[dict]:
    from venno.engine import preprocess

    return [{"key": k, "label": v, "suits": list(preprocess.SUITS.get(k, ()))} for k, v in preprocess.PREPROCESSORS.items()]


def _engine_list(kind: str) -> list[str]:
    """Samplers / schedulers the engine runs."""
    try:
        from venno.engine import sampling, schedulers

        return list(sampling.SAMPLERS) if kind == "samplers" else list(schedulers.SCHEDULERS)
    except Exception:
        return []


def _wildcards() -> list[str]:
    try:
        from venno.engine import wildcards

        return wildcards.files()
    except Exception as e:
        print(f"[Venno] could not list wildcards: {e}")
        return []


def _upscalers() -> list[str]:
    try:
        from venno.engine import upscale

        return upscale.available()
    except Exception as e:
        print(f"[Venno] could not list upscalers: {e}")
        return ["Latent", "Lanczos"]


KIND_LABELS = {"checkpoint": "Checkpoints", "text_encoder": "Text encoders", "vae": "VAE", "lora": "LoRAs", "controlnet": "Guides", "upscaler": "Upscalers", "adetailer": "Detectors", "reference_adapter": "Reference adapter"}


def _system() -> dict:
    """Settings → System: version, device and the folders in use."""
    import platform
    import sys

    import torch

    from backend import memory_management
    from venno.engine import models
    from venno.pipeline import metadata

    device = memory_management.get_torch_device()
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else ""
    folders = [{"kind": label, "new_files": config.model_folder(kind), "searched": config.model_folders(kind)} for kind, label in KIND_LABELS.items()]
    version = metadata.VERSION.removeprefix("venno ")
    try:  # the commit, when running from a git checkout
        import subprocess

        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
        if commit:
            version += f" ({commit})"
    except Exception:
        pass
    return {
        "version": version,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": f"{device}" + (f" ({gpu})" if gpu else ""),
        "loaded_model": models.manager.info.get("name", ""),
        "data": config.data_path(),
        "outputs": config.outputs_path(),
        "models": config.models_path(),
        "folders": folders,
        "listen": config.listen_host(),
        "port": config.args.port,
        "command_line": " ".join(sys.argv[1:]),
        "memory": _memory(),
    }


def custom_downloads_allowed() -> bool:
    """Arbitrary URLs write files to disk: allowed locally, opt-in when served on the network."""
    return not config.is_shared() or config.args.allow_custom_downloads


def create_app() -> FastAPI:
    app = FastAPI(title="Venno", docs_url="/venno/docs", openapi_url="/venno/openapi.json")
    V = "/venno/v1"
    app.add_api_route(f"{V}/info", _info, methods=["GET"])

    def refresh():  # model folders are scanned on every request
        return _info()

    app.add_api_route(f"{V}/refresh", refresh, methods=["POST"])
    app.add_api_route(f"{V}/system", _system, methods=["GET"])

    # ---- updates (only when the user asks: Settings → System) ----
    def update_check():
        from venno import updates

        try:
            return updates.check()
        except Exception as e:  # offline, GitHub unreachable, git missing
            raise HTTPException(502, f"Could not check for updates: {e}")

    def update_apply(request: Request):
        from venno import updates

        if request.client is None or request.client.host not in ("127.0.0.1", "::1", "localhost"):
            raise HTTPException(403, "Update Venno from the computer it runs on")
        try:
            return updates.apply()
        except Exception as e:
            raise HTTPException(409, str(e))

    # ---- settings that apply from the next start (Settings → System) ----
    def settings_get():
        return {"settings": config.load_settings(), "memory": _memory(), "vram_modes": list(config.VRAM_MODES)}

    def settings_set(req: dict, request: Request):
        if request.client is None or request.client.host not in ("127.0.0.1", "::1", "localhost"):
            raise HTTPException(403, "Change Venno's settings from the computer it runs on")
        mode = req.get("vram_mode")
        if mode is not None and mode not in config.VRAM_MODES:
            raise HTTPException(400, f"Unknown memory mode {mode!r}")
        if mode is not None:
            config.save_settings(vram_mode=mode)
        return settings_get()

    app.add_api_route(f"{V}/settings", settings_get, methods=["GET"])
    app.add_api_route(f"{V}/settings", settings_set, methods=["POST"])

    app.add_api_route(f"{V}/update", update_check, methods=["GET"])
    app.add_api_route(f"{V}/update", update_apply, methods=["POST"])

    # ---- generation jobs ----
    def submit_job(job: jobs.Job):
        return jobs.submit(job).summary()

    def get_job(job_id: str):
        rec = jobs.get(job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return rec.summary()

    def job_image(job_id: str, index: int):
        rec = jobs.get(job_id)
        if rec is None or not (0 <= index < len(rec.images)):
            raise HTTPException(status_code=404, detail="Unknown image")
        return Response(rec.images[index], media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    def job_detections(job_id: str, index: int):
        """ADetailer's detection preview of an image (boxes and scores), like Forge's ADetailer preview."""
        rec = jobs.get(job_id)
        if rec is None or not (0 <= index < len(rec.detections)) or rec.detections[index] is None:
            raise HTTPException(status_code=404, detail="No detection preview")
        return Response(rec.detections[index], media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    def cancel_job(job_id: str):
        if not jobs.cancel(job_id):
            raise HTTPException(status_code=404, detail="Unknown or finished job")
        return {"ok": True}

    def list_jobs():
        return [r.summary() for r in reversed(jobs.all_records())]

    def prompt_schedule(req: jobs.ScheduleRequest):
        """What a prompt means step by step (Studio's preview line). Same parser as generation."""
        from venno.engine import prompts
        from venno.engine.conditioning import extract_lora_tags

        text = extract_lora_tags(req.prompt)[0]
        switch = extract_lora_tags(req.switch_prompt)[0] if req.switch_prompt else None
        plan = prompts.PromptPlan(req.steps, text, switch=switch, switch_at=req.switch_at, allow_and=req.allow_and)
        segments = plan.segments()
        constructs = prompts.describe(text, req.steps) if len(prompts.split_and(text)) == 1 else []
        return {"segments": segments, "constructs": constructs, "changes": len(segments) > 1 or any(len(sg["parts"]) > 1 or sg["parts"][0]["weight"] != 1.0 for sg in segments)}

    app.add_api_route(f"{V}/prompt/schedule", prompt_schedule, methods=["POST"])

    def image_params(req: jobs.ImageRequest):
        """Generation settings stored in an image (A1111 / Forge / Venno), for "use settings from image"."""
        from venno.pipeline import infotext

        data = req.image.split(",", 1)[1] if req.image.lstrip().startswith("data:") else req.image
        try:
            text, note = infotext.read_text(base64.b64decode(data))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Not a readable image: {e}")
        if not text:
            return {"found": False, "note": note}
        return {"found": True, "text": text, "settings": infotext.settings(text)}

    app.add_api_route(f"{V}/image/params", image_params, methods=["POST"])

    def guide_preview(req: jobs.GuidePreview):
        """A guide picture after its resize mode and preprocessor, as the job will use it (shown on the ControlNet card)."""
        import io

        from PIL import Image

        from venno.engine import images, preprocess

        try:
            img = images.load_rgb(req.image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Not a readable image: {e}")
        if req.width and req.height:
            scale = min(1.0, 1024 / max(req.width, req.height))  # a preview: at most 1024 px
            img = images.resize(img, max(16, round(req.width * scale)), max(16, round(req.height * scale)), req.resize_mode)
        elif max(img.size) > 1024:
            img.thumbnail((1024, 1024), Image.LANCZOS)
        out = preprocess.run(req.preprocessor, img, req.model, req.canny_low, req.canny_high)
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return {"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}

    app.add_api_route(f"{V}/guides/preview", guide_preview, methods=["POST"])

    def tag_search(q: str = "", limit: int = 12):
        """Danbooru tags for autocomplete, in Anima's format (spaces, @artist, escaped parentheses)."""
        from venno.engine import tags

        return tags.search(q, max(1, min(limit, 50)))

    app.add_api_route(f"{V}/tags", tag_search, methods=["GET"])

    # LoRA browser (venno/engine/lora_library.py)
    def lora_library():
        from venno.engine import lora_library as lib

        return lib.entries()

    def lora_preview(name: str):
        from venno.engine import lora_library as lib

        try:
            data, media = lib.preview_bytes(name)
        except (ValueError, FileNotFoundError):
            raise HTTPException(status_code=404, detail="No preview")
        return Response(data, media_type=media, headers={"Cache-Control": "private, max-age=86400"})  # the URL carries a version

    def lora_set_preview(req: jobs.LoraPreview):
        from venno.engine import images, lora_library as lib

        try:
            lib.set_preview(req.name, images.load_rgb(req.image) if req.image else None)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Could not save the preview: {e}")
        return next((e for e in lib.entries() if e["name"] == req.name), {})

    def lora_set_meta(req: jobs.LoraMeta):
        from venno.engine import lora_library as lib

        try:
            lib.set_meta(req.name, tags=req.tags, activation=req.activation, weight=req.weight, description=req.description, notes=req.notes)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Could not save: {e}")
        return next((e for e in lib.entries() if e["name"] == req.name), {})

    app.add_api_route(f"{V}/loras", lora_library, methods=["GET"])
    app.add_api_route(f"{V}/loras/preview", lora_preview, methods=["GET"])
    app.add_api_route(f"{V}/loras/preview", lora_set_preview, methods=["POST"])
    app.add_api_route(f"{V}/loras/meta", lora_set_meta, methods=["POST"])

    def token_count(req: jobs.TokenRequest):
        """Context length of prompts (T5 tokens, as Anima's adapter sees them), for the counters on the prompt boxes."""
        from venno.engine import tokens

        return {"counts": [tokens.count(t) for t in req.texts], "limit": tokens.LIMIT}

    app.add_api_route(f"{V}/tokens", token_count, methods=["POST"])

    def grid_check(req: jobs.Grid):
        """Parse grid axes for Studio's preview ("3 × 4 = 12 images") without running anything."""
        from venno.pipeline import grid

        try:
            axes = grid.resolve_axes(req)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "axes": [{"title": a.title, "values": [a.label(v) for v in a.values]} for a in axes], "cells": math.prod(len(a.values) for a in axes)}

    def grid_axes():
        from venno.pipeline import grid

        return [{"key": a.key, "label": a.label, "kind": a.kind} for a in grid.AXES.values()]

    app.add_api_route(f"{V}/grid/check", grid_check, methods=["POST"])
    app.add_api_route(f"{V}/grid/axes", grid_axes, methods=["GET"])
    app.add_api_route(f"{V}/jobs", submit_job, methods=["POST"])
    app.add_api_route(f"{V}/jobs", list_jobs, methods=["GET"])
    app.add_api_route(f"{V}/jobs/{{job_id}}", get_job, methods=["GET"])
    app.add_api_route(f"{V}/jobs/{{job_id}}/images/{{index}}.png", job_image, methods=["GET"])
    app.add_api_route(f"{V}/jobs/{{job_id}}/detections/{{index}}.png", job_detections, methods=["GET"])
    app.add_api_route(f"{V}/jobs/{{job_id}}/cancel", cancel_job, methods=["POST"])

    @app.websocket(f"{V}/events")
    async def events(ws: WebSocket):
        """Pushes {"type": "job", ...summary} on every job change and {"type": "preview", id, image} for live previews."""
        await ws.accept()
        sent: dict[str, int] = {}
        last_preview: dict[str, int] = {}
        try:
            while True:
                for rec in jobs.all_records():
                    if sent.get(rec.id) != rec.version:
                        sent[rec.id] = rec.version
                        await ws.send_json({"type": "job", **rec.summary()})
                running = jobs.running_record()
                if running is not None and running.preview is not None and running.preview_id != last_preview.get(running.id):
                    last_preview[running.id] = running.preview_id
                    await ws.send_json({"type": "preview", "id": running.id, "image": "data:image/jpeg;base64," + base64.b64encode(running.preview).decode()})
                await asyncio.sleep(0.3)
        except (WebSocketDisconnect, RuntimeError):
            return

    # ---- model downloads ----
    def on_download_done(_job):
        pass  # new files show up on the next /info

    def catalog():
        return {"catalog": downloads.catalog(), "kinds": downloads.KINDS, "jobs": downloads.jobs(), "custom_urls": custom_downloads_allowed()}

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
                if not custom_downloads_allowed():
                    raise ValueError("Downloading from custom URLs is disabled when the server is shared on the network (start with --allow-custom-downloads)")
                job = downloads.start(str(req.get("url", "")).strip(), str(req.get("kind", "")), filename=req.get("filename") or None, on_done=on_download_done)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return job.to_dict()

    def download_jobs():
        return downloads.jobs()

    def download_cancel(req: dict):
        downloads.cancel(str(req.get("id", "")))
        return downloads.jobs()

    app.add_api_route(f"{V}/models/catalog", catalog, methods=["GET"])
    app.add_api_route(f"{V}/models/download", download, methods=["POST"])
    app.add_api_route(f"{V}/models/jobs", download_jobs, methods=["GET"])
    app.add_api_route(f"{V}/models/cancel", download_cancel, methods=["POST"])

    app.mount("/studio/assets", RevalidatedStaticFiles(directory=STUDIO_DIR), name="venno_studio_assets")

    def studio_index():
        with open(os.path.join(STUDIO_DIR, "index.html"), encoding="utf-8") as f:
            page = f.read().replace("/studio/assets/", f"/studio/assets/v/{_studio_stamp()}/")
        return Response(page, media_type="text/html", headers={"Cache-Control": "no-cache"})

    app.add_api_route("/studio", studio_index, methods=["GET"], include_in_schema=False)
    app.add_api_route("/studio/", studio_index, methods=["GET"], include_in_schema=False)

    app.add_api_route("/", lambda: RedirectResponse("/studio/"), methods=["GET"], include_in_schema=False)
    return app
