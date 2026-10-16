"""
Venno job API (Phase 0 of docs/PLAN_VENNO_SERVER.md).

One request schema for every generation mode, executed by a single worker thread.
For now the adapter (`_run_forge`) translates jobs into Forge's processing calls;
in Phase 1 it is replaced by the Venno engine while this module's API stays the same.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import random
import threading
import time
import uuid
from collections import OrderedDict
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from modules import shared

# ---------------------------------------------------------------- schema


class ModelSelection(BaseModel):
    checkpoint: str = ""
    text_encoder: str = ""
    vae: str = ""


class LoraRef(BaseModel):
    name: str
    weight: float = 0.8


class Region(BaseModel):
    prompt: str
    mask: str  # PNG (data URL or base64); white / opaque = region
    strength: float = 1.0


class Guide(BaseModel):
    model: str
    image: str  # PNG, already in the polarity the model expects
    weight: float = 0.8
    start: float = 0.0
    end: float = 0.8


class Hires(BaseModel):
    upscaler: str = "Latent"
    scale: float = 1.5
    resize_x: int = 0
    resize_y: int = 0
    steps: int = 0  # 0 = same as first pass
    denoise: float = 0.5
    cfg: Optional[float] = None  # None = same as first pass
    shift: Optional[float] = None


class Detailer(BaseModel):
    model: str = "face_yolov8n.pt"
    prompt: str = ""
    negative: str = ""
    confidence: float = 0.3
    denoise: float = 0.4
    mask_blur: int = 4
    dilate: int = 4
    padding: int = 32


class GridAxis(BaseModel):
    type: str = "none"  # see venno/pipeline/grid.py AXES
    values: Union[str, list] = ""  # "1-5", "1-10 (+2)", "a, b, \"c, d\"" or a list
    target: str = ""  # lora_weight: the LoRA name


class Grid(BaseModel):
    x: Optional[GridAxis] = None
    y: Optional[GridAxis] = None
    z: Optional[GridAxis] = None
    include_cells: bool = False  # also return / save each image


class ImageRequest(BaseModel):
    image: str  # data URL or base64


class ScheduleRequest(BaseModel):
    prompt: str = ""
    switch_prompt: str = ""
    switch_at: float = Field(0.4, ge=0.0, le=1.0)
    steps: int = Field(20, ge=1, le=150)
    allow_and: bool = True


class Job(BaseModel):
    mode: Literal["txt2img", "img2img"] = "txt2img"
    model: ModelSelection = Field(default_factory=ModelSelection)
    prompt: str = ""
    negative: str = ""
    switch_prompt: str = ""  # "then switch to …": replaces the prompt after switch_at of the steps
    switch_at: float = Field(0.4, ge=0.0, le=1.0)
    loras: list[LoraRef] = []
    width: int = 1024
    height: int = 1024
    batch: int = Field(1, ge=1, le=8)  # images per run
    count: int = Field(1, ge=1, le=100)  # runs
    seed: int = -1
    sampler: str = "Euler"
    scheduler: str = "Simple"
    steps: int = Field(10, ge=1, le=150)
    cfg: float = 1.0
    shift: float = 3.0
    # img2img / inpaint
    init_image: Optional[str] = None
    strength: float = Field(0.6, ge=0.0, le=1.0)
    resize_mode: int = 0
    mask: Optional[str] = None
    mask_blur: int = 4
    fill: Literal["fill", "original", "latent_noise", "latent_nothing"] = "original"
    only_masked: bool = False
    only_masked_padding: int = 32
    soft_inpaint: bool = True
    # extras
    regions: list[Region] = []
    region_base: float = 0.25
    region_end: float = 1.0
    guides: list[Guide] = []
    hires: Optional[Hires] = None
    detailer: list[Detailer] = []
    grid: Optional[Grid] = None  # comparison grid (X/Y/Z plot): runs every combination
    save: bool = True
    tag: str = ""  # free-form client label (e.g. "live")
    engine: Literal["default", "venno", "forge"] = "default"  # Phases 1-3: pick the pipeline (parity testing)


FILL_MODES = {"fill": 0, "original": 1, "latent_noise": 2, "latent_nothing": 3}

# ---------------------------------------------------------------- job records


class JobRecord:
    def __init__(self, job: Job):
        self.id = uuid.uuid4().hex[:16]
        self.job = job
        self.status = "queued"  # queued | running | done | error | cancelled
        self.error = ""
        self.warnings: list[str] = []
        self.created = time.time()
        self.started = 0.0
        self.finished = 0.0
        self.images: list[bytes] = []
        self.seeds: list[int] = []
        self.infotexts: list[str] = []
        self.progress = 0.0
        self.step = 0
        self.steps = 0
        self.job_no = 0
        self.job_count = 0
        self.eta = 0.0
        self.engine = ""  # pipeline that ran the job: "venno" | "forge"
        self.preview: Optional[bytes] = None  # latest live preview (JPEG)
        self.preview_id = 0
        self.cancel_event = threading.Event()
        self.version = 0  # bumped on every change (for event streaming)

    def set_preview(self, jpeg: bytes):
        self.preview = jpeg
        self.preview_id += 1

    def summary(self) -> dict:
        return {
            "id": self.id,
            "tag": self.job.tag,
            "mode": self.job.mode,
            "status": self.status,
            "error": self.error,
            "warnings": self.warnings,
            "progress": round(self.progress, 4),
            "step": self.step,
            "steps": self.steps,
            "job_no": self.job_no,
            "job_count": self.job_count,
            "eta": round(self.eta, 1),
            "images": [f"/venno/v1/jobs/{self.id}/images/{i}.png" for i in range(len(self.images))],
            "seeds": self.seeds,
            "infotexts": self.infotexts,
            "queue_position": queue_position(self.id),
            "engine": self.engine,
        }


_records: "OrderedDict[str, JobRecord]" = OrderedDict()
_queue: list[str] = []
_cond = threading.Condition()
_running: Optional[str] = None
MAX_RECORDS = 60


def _lift_lora_tags(job: Job):
    """<lora:name:w> typed into any prompt becomes a job LoRA (the list wins over a tag naming the same LoRA)."""
    from venno.engine.conditioning import extract_lora_tags

    have = {lr.name for lr in job.loras}

    def lift(text: str) -> str:
        text, found = extract_lora_tags(text, keep_random=True)  # LoRAs inside {a|b} choices are picked per image
        for name, weight in found:
            if name not in have:
                have.add(name)
                job.loras.append(LoraRef(name=name, weight=weight))
        return text

    job.prompt = lift(job.prompt)
    job.negative = lift(job.negative)
    for region in job.regions:
        region.prompt = lift(region.prompt)


def submit(job: Job) -> JobRecord:
    _lift_lora_tags(job)
    rec = JobRecord(job)
    with _cond:
        _records[rec.id] = rec
        while len(_records) > MAX_RECORDS:
            oldest = next(iter(_records))
            if _records[oldest].status in ("queued", "running"):
                break
            _records.pop(oldest)
        _queue.append(rec.id)
        _cond.notify_all()
    _ensure_worker()
    return rec


def get(job_id: str) -> Optional[JobRecord]:
    return _records.get(job_id)


def all_records() -> list[JobRecord]:
    return list(_records.values())


def queue_position(job_id: str) -> int:
    try:
        return _queue.index(job_id) + 1
    except ValueError:
        return 0


def cancel(job_id: str) -> bool:
    with _cond:
        rec = _records.get(job_id)
        if rec is None:
            return False
        if rec.status == "queued":
            _queue.remove(job_id)
            rec.status = "cancelled"
            rec.version += 1
            _cond.notify_all()
            return True
        if rec.status == "running":
            rec.cancel_event.set()
            shared.state.interrupt()
            return True
    return False


# ---------------------------------------------------------------- worker

_worker: Optional[threading.Thread] = None


def _ensure_worker():
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_work_loop, daemon=True, name="venno-jobs")
        _worker.start()


def _work_loop():
    global _running
    while True:
        with _cond:
            while not _queue:
                _cond.wait()
            job_id = _queue.pop(0)
            rec = _records[job_id]
            rec.status = "running"
            rec.started = time.time()
            rec.version += 1
            _running = job_id
        try:
            if _choose_engine(rec) == "venno":
                cancelled = _run_venno(rec)
            else:
                _run_forge(rec)
                cancelled = shared.state.interrupted
            rec.status = "cancelled" if cancelled or rec.cancel_event.is_set() else "done"
        except Exception as e:  # report, keep the worker alive
            rec.status = "error"
            rec.error = getattr(e, "detail", None) or str(e) or type(e).__name__
            print(f"[Venno] job {job_id} failed: {rec.error}")
        finally:
            rec.finished = time.time()
            rec.progress = 1.0 if rec.status == "done" else rec.progress
            rec.version += 1
            with _cond:
                _running = None
                _cond.notify_all()


def running_record() -> Optional[JobRecord]:
    return _records.get(_running) if _running else None


def update_progress():
    """Forge path: copy Forge's global progress state (and preview) into the running job record."""
    rec = running_record()
    st = shared.state
    if rec is None or rec.engine != "forge" or st.job_count == 0:
        return rec
    if st.current_image is not None and st.id_live_preview != getattr(rec, "_forge_preview_id", -1):
        rec._forge_preview_id = st.id_live_preview
        buf = io.BytesIO()
        st.current_image.convert("RGB").save(buf, format="JPEG", quality=80)
        rec.set_preview(buf.getvalue())
    progress = 0.01
    if st.job_count > 0:
        progress += st.job_no / st.job_count
    if st.sampling_steps > 0:
        progress += 1 / st.job_count * st.sampling_step / st.sampling_steps
    progress = min(progress, 1.0)
    elapsed = time.time() - st.time_start
    new = (round(progress, 3), st.sampling_step, st.job_no)
    if new != (round(rec.progress, 3), rec.step, rec.job_no):
        rec.progress = progress
        rec.step = st.sampling_step
        rec.steps = st.sampling_steps
        rec.job_no = st.job_no
        rec.job_count = st.job_count
        rec.eta = max(0.0, elapsed / progress - elapsed)
        rec.version += 1
    return rec


# ---------------------------------------------------------------- engine selection


def default_engine() -> str:
    return getattr(shared.cmd_opts, "engine", "forge") or "forge"


def venno_unsupported(job: Job) -> Optional[str]:
    """Why the Venno engine can't run this job yet (None = it can). Grows smaller each phase."""
    from venno.engine import sampling, schedulers

    if job.mode == "img2img" and not job.init_image:
        return "image to image without an input image"
    if job.hires and job.mode == "txt2img":
        from venno.engine import upscale

        name = job.hires.upscaler
        if not (upscale.is_latent(name) or name in upscale.RESAMPLE or name in upscale.models()):
            return f'hires upscaler "{name}"'
    if job.sampler not in sampling.SAMPLERS:
        return f'sampler "{job.sampler}"'
    if job.scheduler not in schedulers.SCHEDULERS and job.scheduler not in ("", "Automatic"):
        return f'scheduler "{job.scheduler}"'
    return None


def _choose_engine(rec: JobRecord) -> str:
    wanted = rec.job.engine if rec.job.engine != "default" else default_engine()
    if rec.job.grid is not None:
        wanted = "venno"  # only the Venno pipeline runs comparison grids
    if wanted == "venno":
        why = venno_unsupported(rec.job)
        if why is not None:
            if rec.job.engine == "venno" or rec.job.grid is not None:  # explicitly requested / grids: fail loudly
                raise ValueError(f"The Venno engine does not support {why} yet")
            print(f"[Venno] job {rec.id}: {why} not on the Venno engine yet, using Forge")
            wanted = "forge"
    rec.engine = wanted
    rec.version += 1
    return wanted


# ---------------------------------------------------------------- Venno engine


def _run_venno(rec: JobRecord) -> bool:
    from venno.pipeline import generate

    total_runs = rec.job.count

    def on_step(step, steps, run, runs):
        rec.step, rec.steps, rec.job_no, rec.job_count = step, steps, run, runs
        rec.progress = min(1.0, (run + step / max(steps, 1)) / max(runs, 1))
        elapsed = time.time() - rec.started
        rec.eta = elapsed / rec.progress - elapsed if rec.progress > 0.01 else 0.0
        rec.version += 1

    progress = generate.Progress(on_step=on_step, on_preview=rec.set_preview, is_cancelled=rec.cancel_event.is_set)
    rec.job_count = total_runs
    result = generate.run(rec.job, progress)
    rec.images, rec.seeds, rec.infotexts, rec.warnings = result.images, result.seeds, result.infotexts, result.warnings
    return result.cancelled


# ---------------------------------------------------------------- Forge adapter (replaced as the Venno engine grows)

def _b64(data: Optional[str]) -> Optional[str]:
    if not data:
        return data
    return data.split(",", 1)[1] if data.lstrip().startswith("data:") else data


def _forge_switch(job: Job) -> str:
    """The "then switch to" prompt as A1111 syntax for the Forge path, when it parses back exactly."""
    from venno.engine import prompts

    if not job.switch_prompt:
        return job.prompt
    when = max(1, int(job.switch_at * job.steps))
    candidate = f"[{job.prompt}:{job.switch_prompt}:{when}]"
    if [t for _, t in prompts.schedule(candidate, job.steps)] == ([job.prompt, job.switch_prompt] if when < job.steps else [job.prompt]):
        return candidate
    print("[Venno] the switch prompt can't be expressed for the Forge path (brackets/colons in the prompt); using the main prompt only")
    return job.prompt


def _forge_payload(job: Job) -> dict:
    from venno.engine import wildcards

    if any(wildcards.has_syntax(t) for t in (job.prompt, job.negative, job.switch_prompt)):  # Forge has no wildcards: pick once
        seed = job.seed if job.seed >= 0 else random.randrange(2**31)
        job = job.model_copy(update={k: wildcards.for_seed(getattr(job, k), seed, k) for k in ("prompt", "negative", "switch_prompt")})
    prompt = _forge_switch(job)
    if job.loras:
        prompt += "".join(f" <lora:{lr.name}:{lr.weight:g}>" for lr in job.loras)

    payload = {
        "prompt": prompt,
        "negative_prompt": job.negative,
        "steps": job.steps,
        "cfg_scale": job.cfg,
        "distilled_cfg_scale": job.shift,
        "sampler_name": job.sampler,
        "scheduler": job.scheduler,
        "seed": job.seed,
        "width": job.width,
        "height": job.height,
        "batch_size": job.batch,
        "n_iter": job.count,
        "send_images": True,
        "save_images": job.save,
        "do_not_save_grid": True,
        "override_settings": {},
        "override_settings_restore_afterwards": False,
        "alwayson_scripts": {},
    }
    if job.model.checkpoint:
        payload["override_settings"]["sd_model_checkpoint"] = job.model.checkpoint
    modules = [m for m in (job.model.text_encoder, job.model.vae) if m]
    if modules:
        payload["override_settings"]["forge_additional_modules"] = modules

    scripts = payload["alwayson_scripts"]
    if job.mode == "txt2img" and job.hires:
        h = job.hires
        payload.update(
            enable_hr=True,
            hr_upscaler=h.upscaler,
            hr_scale=h.scale,
            hr_resize_x=h.resize_x,
            hr_resize_y=h.resize_y,
            hr_second_pass_steps=h.steps,
            denoising_strength=h.denoise,
            hr_cfg=h.cfg if h.cfg else job.cfg,
            hr_distilled_cfg=h.shift if h.shift else job.shift,
        )
    if job.mode == "img2img":
        if not job.init_image:
            raise ValueError("img2img needs init_image")
        payload.update(init_images=[_b64(job.init_image)], denoising_strength=job.strength, resize_mode=job.resize_mode)
        if job.mask:
            payload.update(
                mask=_b64(job.mask),
                mask_blur=job.mask_blur,
                inpainting_fill=FILL_MODES[job.fill],
                inpaint_full_res=job.only_masked,
                inpaint_full_res_padding=job.only_masked_padding,
                inpainting_mask_invert=0,
            )
            if job.soft_inpaint:
                scripts["soft inpainting"] = {"args": [True, 1, 0.5, 4, 0, 0.5, 2]}

    if job.regions:
        regions = [{"prompt": r.prompt, "mask": _b64(r.mask), "strength": r.strength} for r in job.regions]
        scripts["venno regional prompting"] = {"args": [True, json.dumps({"base_strength": job.region_base, "end_at": job.region_end, "regions": regions})]}

    if job.guides:
        scripts["controlnet"] = {
            "args": [
                {
                    "enabled": True,
                    "module": "None",
                    "model": g.model,
                    "weight": g.weight,
                    "image": _b64(g.image),
                    "resize_mode": "Just Resize",
                    "guidance_start": g.start,
                    "guidance_end": g.end,
                    "pixel_perfect": False,
                    "control_mode": "Balanced",
                    "save_detected_map": False,
                }
                for g in job.guides
            ]
        }

    units = [d for d in job.detailer if d.model and d.model != "None"]
    if units:
        scripts["adetailer"] = {
            "args": [
                True,
                False,
                *(
                    {
                        "ad_model": d.model,
                        "ad_prompt": d.prompt,
                        "ad_negative_prompt": d.negative,
                        "ad_confidence": d.confidence,
                        "ad_denoising_strength": d.denoise,
                        "ad_mask_blur": d.mask_blur,
                        "ad_dilate_erode": d.dilate,
                        "ad_inpaint_only_masked": True,
                        "ad_inpaint_only_masked_padding": d.padding,
                        "ad_use_inpaint_width_height": False,
                    }
                    for d in units
                ),
            ]
        }
    return payload


def _run_forge(rec: JobRecord):
    from modules.api import models

    from venno.engine.models import manager

    manager.release_for_forge()  # never hold the model twice while both pipelines exist
    forge_api = getattr(shared, "forge_api", None)  # set in webui.create_api
    if forge_api is None:
        raise RuntimeError("generation backend is not ready yet")
    payload = _forge_payload(rec.job)
    if rec.job.mode == "txt2img":
        res = forge_api.text2imgapi(models.StableDiffusionTxt2ImgProcessingAPI(**payload))
    else:
        res = forge_api.img2imgapi(models.StableDiffusionImg2ImgProcessingAPI(**payload))

    info = json.loads(res.info) if isinstance(res.info, str) else (res.info or {})
    seeds = info.get("all_seeds") or [rec.job.seed]
    first = info.get("index_of_first_image", 0)
    count = rec.job.batch * rec.job.count
    images = (res.images or [])[first : first + count]  # drop grids and script preview images
    out = []
    for b64 in images:
        try:
            out.append(base64.b64decode(b64))
        except (binascii.Error, TypeError):
            continue
    rec.images = out
    rec.seeds = list(seeds[: len(out)])
    rec.infotexts = list((info.get("infotexts") or [])[: len(out)])
