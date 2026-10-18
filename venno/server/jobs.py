"""
Venno jobs: one request schema for every generation mode, executed by a single worker thread
(venno/pipeline/generate.py does the work).
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import OrderedDict
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

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
    negative: str = ""  # the region's own negative prompt ("" = the main negative)


class Guide(BaseModel):
    model: str
    image: str  # PNG: a map in the polarity the model expects, or any picture with a preprocessor
    weight: float = 0.8
    start: float = 0.0
    end: float = 0.8
    preprocessor: Literal["none", "canny", "lineart", "softedge", "depth"] = "none"  # venno/engine/preprocess.py
    mode: Literal["balanced", "prompt", "guide"] = "balanced"  # which is more important: neither / the prompt / the guide
    resize_mode: Optional[int] = None  # an uploaded picture: 0 stretch, 1 crop, 2 fit to the job's frame (None: drawn in its coordinates)
    canny_low: int = Field(100, ge=0, le=255)
    canny_high: int = Field(200, ge=0, le=255)


class GuidePreview(BaseModel):
    image: str
    preprocessor: Literal["none", "canny", "lineart", "softedge", "depth"] = "none"
    model: str = ""  # decides the line polarity
    width: int = 0  # the job's frame (0 = the picture's own size)
    height: int = 0
    resize_mode: int = 1
    canny_low: int = Field(100, ge=0, le=255)
    canny_high: int = Field(200, ge=0, le=255)


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
    region_refine: float = 0.0  # region pass denoise: re-render each region on its own afterwards (0 = off)
    region_separation: float = 0.0  # 0..1: keep regions apart in self-attention in the middle steps
    guides: list[Guide] = []
    # Cosmos time axis (venno/engine/reference.py)
    frames: int = Field(1, ge=1, le=4)  # consistent set: frames generated together, as one sequence (text to image)
    frame_prompts: list[str] = []  # per frame, added in front of the prompt ("" = the prompt alone)
    frame_separation: float = Field(0.2, ge=0.0, le=1.0)  # with frame prompts: keep frames apart (prompts ↔ consistency)
    references: list[str] = Field([], max_length=3)  # reference pictures (data URL or base64)
    reference_noise: float = Field(0.5, ge=0.0, le=1.0)  # the timestep the clean references are given
    reference_follow: bool = False  # their timestep follows the steps instead (Anima Edit LoRAs)
    reference_strength: float = Field(1.0, ge=0.0, le=1.0)  # how much the picture attends to them (1 = fully)
    hires: Optional[Hires] = None
    detailer: list[Detailer] = []
    grid: Optional[Grid] = None  # comparison grid (X/Y/Z plot): runs every combination
    save: bool = True
    tag: str = ""  # free-form client label (e.g. "live")

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
        region.negative = lift(region.negative)


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
            cancelled = _run(rec)
            rec.status = "cancelled" if cancelled or rec.cancel_event.is_set() else "done"
        except Exception as e:  # report, keep the worker alive
            rec.status = "error"
            rec.error = getattr(e, "detail", None) or str(e) or type(e).__name__
            print(f"[Venno] job {job_id} failed: {rec.error}")
            traceback.print_exc()  # full trace in the console, for bug reports
        finally:
            rec.finished = time.time()
            rec.progress = 1.0 if rec.status == "done" else rec.progress
            rec.version += 1
            with _cond:
                _running = None
                _cond.notify_all()


def running_record() -> Optional[JobRecord]:
    return _records.get(_running) if _running else None


# ---------------------------------------------------------------- running


def unsupported(job: Job) -> Optional[str]:
    """Why this job can't run (None = it can): names the engine doesn't know."""
    from venno.engine import sampling, schedulers

    if job.mode == "img2img" and not job.init_image:
        return "image to image without an input image"
    if job.hires and job.mode == "txt2img":
        from venno.engine import upscale

        name = job.hires.upscaler
        if not (upscale.is_latent(name) or name in upscale.RESAMPLE or name in upscale.models()):
            return f'hires upscaler "{name}"'
    if job.frames > 1 and job.mode != "txt2img":
        return "consistent sets (several frames) outside text to image"
    if job.sampler not in sampling.SAMPLERS:
        return f'sampler "{job.sampler}"'
    if job.scheduler not in schedulers.SCHEDULERS and job.scheduler not in ("", "Automatic"):
        return f'scheduler "{job.scheduler}"'
    return None


def _run(rec: JobRecord) -> bool:
    from venno.pipeline import generate

    why = unsupported(rec.job)
    if why is not None:
        raise ValueError(f"Not supported: {why}")

    total_runs = rec.job.count

    def on_step(step, steps, run, runs):
        rec.step, rec.steps, rec.job_no, rec.job_count = step, steps, run, runs
        rec.progress = min(1.0, (run + step / max(steps, 1)) / max(runs, 1))
        elapsed = time.time() - rec.started
        rec.eta = elapsed / rec.progress - elapsed if rec.progress > 0.01 else 0.0
        rec.version += 1

    progress = generate.Progress(on_step=on_step, on_preview=rec.set_preview, is_cancelled=rec.cancel_event.is_set)
    rec.job_count = total_runs
    import torch

    with torch.inference_mode():  # the whole job, so weights moved / patched during it are consistent
        result = generate.run(rec.job, progress)
    rec.images, rec.seeds, rec.infotexts, rec.warnings = result.images, result.seeds, result.infotexts, result.warnings
    return result.cancelled
