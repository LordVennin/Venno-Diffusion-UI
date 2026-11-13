"""Venno MOSS's web server: the job API under /moss/v1 (shaped like Venno's /venno/v1, so Studio's job client is
shared), settings, model downloads, and the Studio page with Venno Studio's styles."""

from __future__ import annotations

import asyncio
import base64
import os

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from . import VERSION, config, downloads, grid, jobs, lora, metadata, upscale
from . import engine as engine_module

V = "/moss/v1"
HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_DIR = os.path.join(HERE, "studio")
SHARED_DIR = os.path.join(config.ROOT, "studio")  # Venno Studio's styles and shared modules
LOCAL = {"127.0.0.1", "::1", "localhost"}


class NoCacheStatic(StaticFiles):
    """Revalidated on every load, so an update never leaves the browser on old scripts."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def _local_only(request: Request):
    if request.client is None or request.client.host not in LOCAL:
        raise HTTPException(403, "Only from the computer Venno MOSS runs on")


def info() -> dict:
    eng = engine_module.get()
    found = eng.found
    return {
        "version": VERSION,
        "models": [{"name": m.name, "size": m.size, "single_file": m.root is None} for m in found.models],
        "complete": found.complete,
        "have": {"model": bool(found.models), "text_encoder": bool(found.text_encoder), "tokenizer": bool(found.tokenizer), "vae": bool(found.vae)},
        "loras": [{k: v for k, v in item.items() if k != "path"} for item in lora.scan()],
        "defaults": engine_module.DEFAULTS,
        "engine": eng.info(),
        "folders": {"models": config.model_folders(), "loras": config.lora_folders(), "outputs": config.outputs_path(), "download": config.models_path()},
        "axes": [{"key": a.key, "label": a.label} for a in grid.AXES.values()],
        "upscalers": upscale.available(),
        "turbo": lora.find_turbo() or "",
        "fast": jobs.FAST,
    }


def create_app() -> FastAPI:
    app = FastAPI(title="Venno MOSS", version=VERSION)

    @app.get(f"{V}/info")
    def get_info():
        return info()

    @app.post(f"{V}/refresh")
    def refresh():
        engine_module.get().rescan()
        return info()

    # ---- jobs ----
    @app.post(f"{V}/jobs")
    def submit(job: jobs.Job):
        if job.grid is not None:
            try:
                grid.check(job, grid.resolve_axes(job.grid))
            except ValueError as e:
                raise HTTPException(400, str(e))
        why = jobs.unsupported(job)
        if why is not None:
            raise HTTPException(400, why)
        return jobs.submit(job).summary()

    @app.get(f"{V}/jobs")
    def list_jobs():
        return [r.summary() for r in reversed(jobs.all_records())]

    @app.get(f"{V}/jobs/{{job_id}}")
    def job_state(job_id: str):
        rec = jobs.get(job_id)
        if rec is None:
            raise HTTPException(404, "Unknown job")
        return rec.summary()

    @app.get(f"{V}/jobs/{{job_id}}/images/{{index}}.png")
    def job_image(job_id: str, index: int):
        rec = jobs.get(job_id)
        if rec is None or not 0 <= index < len(rec.images):
            raise HTTPException(404, "No such image")
        return Response(rec.images[index], media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    @app.post(f"{V}/jobs/{{job_id}}/cancel")
    def cancel(job_id: str):
        if not jobs.cancel(job_id):
            raise HTTPException(404, "Unknown or finished job")
        return {"ok": True}

    @app.post(f"{V}/jobs/{{job_id}}/skip")
    def skip(job_id: str):
        if not jobs.skip(job_id):
            raise HTTPException(404, "Unknown or finished job")
        return {"ok": True}

    @app.websocket(f"{V}/events")
    async def events(ws: WebSocket):
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

    # ---- LoRA browser: Venno's LoRA library (previews, tags, notes next to the files) over Venno MOSS's LoRAs ----
    from venno.engine import images as venno_images
    from venno.engine import lora_library

    lora_library.configure(lora.LibrarySource, V)

    def library() -> list[dict]:
        """The library's entries; the line under each card says what the file records about its training."""
        about = {item["name"]: item for item in lora.scan()}
        entries = lora_library.entries()
        for e in entries:
            a = about.get(e["name"], {})
            e["base"] = " · ".join(x for x in (a.get("trained_on"), f"rank {a['rank']}" if a.get("rank") else "") if x) or e["base"]
        return entries

    def library_entry(name: str) -> dict:
        return next((e for e in library() if e["name"] == name), {})

    @app.get(f"{V}/loras")
    def loras():
        return library()

    @app.get(f"{V}/loras/preview")
    def lora_preview(name: str):
        try:
            data, media = lora_library.preview_bytes(name)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "No preview")
        return Response(data, media_type=media, headers={"Cache-Control": "private, max-age=86400"})  # the URL carries a version

    @app.post(f"{V}/loras/preview")
    def lora_set_preview(req: dict, request: Request):
        _local_only(request)
        name, image = str(req.get("name", "")), req.get("image")
        try:
            lora_library.set_preview(name, venno_images.load_rgb(image) if image else None)
        except ValueError as e:
            raise HTTPException(404, str(e))
        except OSError as e:
            raise HTTPException(500, f"Could not save the preview: {e}")
        return library_entry(name)

    @app.post(f"{V}/loras/meta")
    def lora_set_meta(req: dict, request: Request):
        _local_only(request)
        name = str(req.get("name", ""))
        try:
            lora_library.set_meta(name, tags=req.get("tags"), activation=req.get("activation"), weight=req.get("weight"),
                                  description=req.get("description"), notes=req.get("notes"))
        except ValueError as e:
            raise HTTPException(404, str(e))
        except OSError as e:
            raise HTTPException(500, f"Could not save: {e}")
        return library_entry(name)

    # ---- helpers for the page ----
    @app.post(f"{V}/tokens")
    def tokens(req: dict):
        try:
            return {"count": engine_module.get().token_count(str(req.get("text", "")))}
        except Exception as e:  # the tokenizer may not be downloaded yet
            return {"count": 0, "error": str(e)}

    @app.post(f"{V}/grid/check")
    def grid_check(g: jobs.Grid):
        try:
            axes = grid.resolve_axes(g)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        counts = [len(a.values) for a in axes]
        return {"ok": True, "counts": counts, "total": int(__import__("math").prod(counts))}

    @app.post(f"{V}/image/read")
    def image_read(req: dict):
        data = str(req.get("image", ""))
        try:
            return metadata.read(base64.b64decode(data.split(",", 1)[-1]))
        except Exception as e:
            raise HTTPException(400, f"Not a PNG with settings: {e}")

    # ---- settings (used from the next model load; no restart needed) ----
    @app.get(f"{V}/settings")
    def settings_get():
        return {"settings": config.load_settings(), "memory_modes": list(config.MEMORY_MODES), "engine": engine_module.get().info()}

    @app.post(f"{V}/settings")
    def settings_set(req: dict, request: Request):
        _local_only(request)
        updates = {}
        for key in ("model_dirs", "lora_dirs"):
            if key in req:
                folders = [str(d).strip() for d in req[key] if str(d).strip()]
                missing = [d for d in folders if not os.path.isdir(os.path.expanduser(d))]
                if missing:
                    raise HTTPException(400, f"Folder not found: {missing[0]}")
                updates[key] = folders
        if "memory" in req:
            if req["memory"] not in config.MEMORY_MODES:
                raise HTTPException(400, f"Memory must be one of {', '.join(config.MEMORY_MODES)}")
            updates["memory"] = req["memory"]
        config.save_settings(**updates)
        engine_module.get().rescan()
        return {"settings": config.load_settings(), "info": info()}

    # ---- model downloads ----
    @app.get(f"{V}/models/catalog")
    def models_catalog():
        return {"catalog": downloads.catalog(), "jobs": downloads.jobs()}

    @app.post(f"{V}/models/download")
    def models_download(req: dict, request: Request):
        _local_only(request)
        try:
            return downloads.start(str(req.get("variant", ""))).to_dict()
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get(f"{V}/models/jobs")
    def models_jobs():
        done = any(j["status"] == "done" for j in downloads.jobs())
        if done:
            engine_module.get().rescan()
        return downloads.jobs()

    # ---- page ----
    app.mount("/moss/assets", NoCacheStatic(directory=PAGE_DIR), name="moss_assets")
    app.mount("/studio/assets", NoCacheStatic(directory=SHARED_DIR), name="venno_shared")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/moss/")

    @app.get("/moss/", include_in_schema=False)
    def page():
        return FileResponse(os.path.join(PAGE_DIR, "index.html"), headers={"Cache-Control": "no-cache"})

    return app
