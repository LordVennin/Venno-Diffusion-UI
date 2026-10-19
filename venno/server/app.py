"""
Venno server: the Studio front-end and the Venno API (/venno/v1/*).

Generation runs through Venno jobs (venno/server/jobs.py): one request schema for every mode,
a single worker queue, and a WebSocket event stream for progress and live previews.
"""

import asyncio
import base64
import math
import os

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse, Response
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


def _adetailer_models() -> list[str]:
    from venno.engine import detect

    return [m for m in detect.models() if m != "mediapipe_face_mesh"]


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


def _info() -> dict:
    from venno.engine import models

    loaded = models.manager.info
    return {
        "restart_needed": _code_stamp() != STARTED_WITH,
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


KIND_LABELS = {"checkpoint": "Checkpoints", "text_encoder": "Text encoders", "vae": "VAE", "lora": "LoRAs", "controlnet": "Guides", "upscaler": "Upscalers", "adetailer": "Detectors"}


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
    version = metadata.VERSION
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

    app.mount("/studio/assets", StaticFiles(directory=STUDIO_DIR), name="venno_studio_assets")

    def studio_index():
        return FileResponse(os.path.join(STUDIO_DIR, "index.html"), headers={"Cache-Control": "no-cache"})

    app.add_api_route("/studio", studio_index, methods=["GET"], include_in_schema=False)
    app.add_api_route("/studio/", studio_index, methods=["GET"], include_in_schema=False)

    app.add_api_route("/", lambda: RedirectResponse("/studio/"), methods=["GET"], include_in_schema=False)
    return app
