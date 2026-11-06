"""Jobs: the request schema, one worker that runs them in order, and the record Studio follows (same shape as
Venno's, so Studio's job client works unchanged): progress, finished images as they come, Cancel (keeps the
finished images) and Skip (drops the image being made)."""

from __future__ import annotations

import io
import random
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional

from pydantic import BaseModel, Field

from . import engine as engine_module
from . import metadata


class LoraRef(BaseModel):
    name: str
    weight: float = 1.0


class Axis(BaseModel):
    type: str = "none"
    values: str = ""
    target: str = ""


class Grid(BaseModel):
    x: Optional[Axis] = None
    y: Optional[Axis] = None
    z: Optional[Axis] = None
    include_cells: bool = False


class Job(BaseModel):
    prompt: str = ""
    negative: str = ""
    model: str = ""
    loras: list[LoraRef] = []
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    steps: int = Field(50, ge=1, le=200)
    cfg: float = Field(3.0, ge=1.0, le=20.0)
    shift: float = Field(5.0, gt=0, le=20.0)
    skip_layers: list[int] = [22]
    seed: int = -1
    count: int = Field(1, ge=1, le=100)
    grid: Optional[Grid] = None
    save: bool = True
    tag: str = ""


# ---------------------------------------------------------------- running


@dataclass
class Result:
    images: list[bytes] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    infotexts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cancelled: bool = False


@dataclass
class Progress:
    on_step: Callable[[int, int, int, int], None] = lambda step, steps, run, runs: None
    on_preview: Optional[Callable[[object], None]] = None
    on_image: Callable[[Result], None] = lambda result: None
    is_cancelled: Callable[[], bool] = lambda: False  # also true while a skip is pending
    take_skip: Callable[[], bool] = lambda: False

    def stop(self, result: Result) -> bool:
        """After a cancel: True when the job stops, False when only the current image was skipped."""
        if self.take_skip():
            return False
        result.cancelled = True
        return True


def resolve_seed(seed: int) -> int:
    return random.randrange(4294967294) if seed is None or seed < 0 else int(seed)


def run(job: Job, progress: Progress) -> Result:
    if job.grid is not None:
        from . import grid

        return grid.run(job, progress)
    eng = engine_module.get()
    base = resolve_seed(job.seed)
    result = Result()
    for k in range(job.count):
        seed = base + k
        try:
            image, warnings = eng.generate(
                model=job.model, prompt=job.prompt, negative=job.negative, width=job.width, height=job.height,
                steps=job.steps, cfg=job.cfg, shift=job.shift, seed=seed, skip_layers=job.skip_layers,
                loras=[(lr.name, lr.weight) for lr in job.loras],
                on_step=lambda i, n, k=k: progress.on_step(i, n, k, job.count),
                on_preview=progress.on_preview, preview_every=max(1, job.steps // 8) if progress.on_preview else 0,
                is_cancelled=progress.is_cancelled,
            )
        except engine_module.Cancelled:
            if progress.stop(result):
                break
            continue
        result.warnings += [w for w in warnings if w not in result.warnings]
        info = metadata.infotext(job, seed)
        png = metadata.png_bytes(image, info, metadata.settings_json(job, seed))
        if job.save:
            metadata.save(png, seed)
        result.images.append(png)
        result.seeds.append(seed)
        result.infotexts.append(info)
        progress.on_image(result)
    return result


# ---------------------------------------------------------------- records


class JobRecord:
    def __init__(self, job: Job):
        self.id = uuid.uuid4().hex[:16]
        self.job = job
        self.status = "queued"  # queued | running | done | error | cancelled
        self.error = ""
        self.warnings: list[str] = []
        self.started = self.finished = 0.0
        self.images: list[bytes] = []
        self.seeds: list[int] = []
        self.infotexts: list[str] = []
        self.progress = 0.0
        self.step = self.steps = self.job_no = self.job_count = 0
        self.eta = 0.0
        self.preview: Optional[bytes] = None
        self.preview_id = 0
        self.cancel_event = threading.Event()
        self.skip_event = threading.Event()
        self.skipped = 0
        self.version = 0

    def summary(self) -> dict:
        return {
            "id": self.id, "tag": self.job.tag, "mode": "txt2img", "status": self.status, "error": self.error,
            "warnings": self.warnings, "progress": round(self.progress, 4), "step": self.step, "steps": self.steps,
            "job_no": self.job_no, "job_count": self.job_count, "eta": round(self.eta, 1),
            "images": [f"/moss/v1/jobs/{self.id}/images/{i}.png" for i in range(len(self.images))],
            "seeds": self.seeds, "infotexts": self.infotexts, "detections": [None] * len(self.images), "detection_notes": [""] * len(self.images),
            "queue_position": queue_position(self.id), "skipped": self.skipped, "batch": 1,
        }


_records: "OrderedDict[str, JobRecord]" = OrderedDict()
_queue: list[str] = []
_cond = threading.Condition()
_running: Optional[str] = None
_worker: Optional[threading.Thread] = None
MAX_RECORDS = 60


def submit(job: Job) -> JobRecord:
    rec = JobRecord(job)
    with _cond:
        _records[rec.id] = rec
        while len(_records) > MAX_RECORDS and _records[next(iter(_records))].status not in ("queued", "running"):
            _records.popitem(last=False)
        _queue.append(rec.id)
        _cond.notify_all()
    _ensure_worker()
    return rec


def get(job_id: str) -> Optional[JobRecord]:
    return _records.get(job_id)


def all_records() -> list[JobRecord]:
    return list(_records.values())


def running_record() -> Optional[JobRecord]:
    return _records.get(_running) if _running else None


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
            return True
        if rec.status == "running":
            rec.cancel_event.set()
            return True
    return False


def skip(job_id: str) -> bool:
    with _cond:
        rec = _records.get(job_id)
        if rec is None or rec.status != "running":
            return False
        rec.skip_event.set()
        return True


def _ensure_worker():
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_work_loop, daemon=True, name="venno-moss-jobs")
        _worker.start()


def _is_oom(e: BaseException) -> bool:
    import torch

    return isinstance(e, getattr(torch, "OutOfMemoryError", ())) or "out of memory" in str(e).lower()


def out_of_memory_message() -> str:
    tips = []
    if engine_module.get().resolved_memory() == "full":
        tips.append("switch Memory to Low (12 GB cards) in ⚙ Settings, then restart Venno MOSS")
    tips += ["make the picture smaller", "close other programs that use the graphics card"]
    return "Out of graphics memory. Try: " + "; ".join(tips) + ". The memory was freed, so the next job can run."


def _work_loop():
    global _running
    while True:
        with _cond:
            while not _queue:
                _cond.wait()
            job_id = _queue.pop(0)
            rec = _records[job_id]
            rec.status, rec.started = "running", time.time()
            rec.version += 1
            _running = job_id
        oom = False
        try:
            cancelled = _run(rec)
            rec.status = "cancelled" if cancelled or rec.cancel_event.is_set() else "done"
        except Exception as e:  # report, keep the worker alive
            oom = _is_oom(e)
            if oom:
                print(f"[Venno MOSS] job {job_id} ran out of graphics memory: {e}")
                rec.error = out_of_memory_message()
            else:
                rec.error = str(e) or type(e).__name__
                print(f"[Venno MOSS] job {job_id} failed: {rec.error}")
                traceback.print_exc()
                rec.status = "error"
        if oom:  # outside the except block, so the failed job's tensors are released first
            engine_module.get().unload_model()
            rec.status = "error"
        rec.finished = time.time()
        rec.progress = 1.0 if rec.status == "done" else rec.progress
        rec.version += 1
        with _cond:
            _running = None
            _cond.notify_all()


def _run(rec: JobRecord) -> bool:
    import torch

    def on_step(step, steps, run_no, runs):
        rec.step, rec.steps, rec.job_no, rec.job_count = step, steps, run_no, runs
        rec.progress = min(1.0, (run_no + step / max(steps, 1)) / max(runs, 1))
        elapsed = time.time() - rec.started
        rec.eta = elapsed / rec.progress - elapsed if rec.progress > 0.01 else 0.0
        rec.version += 1

    def on_preview(image):
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=80)
        rec.preview = buf.getvalue()
        rec.preview_id += 1

    def take_skip() -> bool:
        if not rec.skip_event.is_set() or rec.cancel_event.is_set():
            return False
        rec.skip_event.clear()
        rec.skipped += 1
        rec.version += 1
        return True

    def store(result: Result):
        rec.seeds, rec.infotexts, rec.warnings = list(result.seeds), list(result.infotexts), list(result.warnings)
        rec.images = list(result.images)  # last: the other lists are never shorter than the images listed
        rec.version += 1

    rec.job_count = rec.job.count
    progress = Progress(on_step=on_step, on_preview=on_preview, on_image=store, take_skip=take_skip,
                        is_cancelled=lambda: rec.cancel_event.is_set() or rec.skip_event.is_set())
    with torch.inference_mode():
        result = run(rec.job, progress)
    store(result)
    return result.cancelled
