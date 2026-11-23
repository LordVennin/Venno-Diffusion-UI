"""The Upscale tab's work, shared by Venno and Venno MOSS: a picture through an upscaler model from the upscaler
folders (models/ESRGAN, …; Spandrel reads ESRGAN, DAT, SwinIR, HAT, …), in feathered tiles, as a task with
progress that can be cancelled.

CPU mode keeps the whole run on the processor: the model is loaded to system RAM and never touches the graphics
card, so it doesn't compete for VRAM and runs while pictures are being generated. Without it the run takes the
graphics card when no picture is being made (``gpu_lock``, which both job runners hold while they work).

The result keeps the source picture's PNG text (its generation settings), with the upscale noted behind them
the way Forge's Extras tab does.
"""

from __future__ import annotations

import base64
import datetime
import io
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
from fastapi import HTTPException
from fastapi import Request as HttpRequest  # at module level: the routes' annotations are resolved from here
from fastapi.responses import Response
from PIL import Image, PngImagePlugin

EXTENSIONS = (".safetensors", ".pth", ".pt", ".ckpt", ".bin")
TILES = (256, 384, 512, 768, 1024)
OVERLAP = 32
MAX_SIDE = 16384  # of the result
MAX_PASSES = 3  # a 2× model run twice reaches 4×
KEEP = 6  # finished tasks kept in memory (their PNGs)

gpu_lock = threading.Lock()  # held by a job runner while it makes a picture, and by an upscale on the graphics card
_one_at_a_time = threading.Lock()  # upscales take turns: they share the loaded model
_cache_lock = threading.Lock()
_loaded: dict = {}  # path → spandrel descriptor (one at a time; apart from hires fix's copy)
TASKS: "dict[str, Task]" = {}
_save_lock = threading.Lock()


class Cancelled(Exception):
    pass


def list_models(folders: list[str]) -> dict[str, str]:
    """{name: path} of the upscaler models in `folders` (file names without extension)."""
    found: dict[str, str] = {}
    for root in folders:
        if not os.path.isdir(root):
            continue
        for f in sorted(os.listdir(root), key=str.lower):
            if f.lower().endswith(EXTENSIONS):
                found.setdefault(os.path.splitext(f)[0], os.path.join(root, f))
    return found


def decode(data: str) -> bytes:
    """A data URL or bare base64 → the file's bytes."""
    if data.startswith("data:"):
        data = data.split(",", 1)[1] if "," in data else ""
    try:
        return base64.b64decode(data)
    except ValueError:
        raise ValueError("The picture could not be read") from None


def tile_starts(n: int, tile: int, overlap: int = OVERLAP) -> list[int]:
    step = max(1, tile - overlap)
    return [0] if n <= tile else sorted({*range(0, n - tile, step), n - tile})  # the last tile flush with the edge


def passes_needed(scale: float, model_scale: int) -> int:
    """How often the model runs to reach `scale` (at least once; a 1× model once)."""
    count, reached = 1, model_scale
    while reached < scale - 1e-6 and count < MAX_PASSES and model_scale > 1:
        count, reached = count + 1, reached * model_scale
    return count


def _feather(h: int, w: int, overlap: int, device) -> torch.Tensor:
    if overlap <= 0:
        return torch.ones(1, 1, h, w, device=device)
    ramp = lambda n: torch.clamp(torch.minimum(torch.arange(n, device=device) + 1, torch.arange(n, 0, -1, device=device)) / overlap, max=1.0)
    return (ramp(h)[:, None] * ramp(w)[None, :])[None, None]


def _load(path: str):
    try:
        import spandrel
    except ImportError:
        raise RuntimeError("Upscaler models need the spandrel package: start with the launcher (without --skip-install) to install it") from None
    with _cache_lock:
        if path not in _loaded:
            _loaded.clear()
            desc = spandrel.ModelLoader().load_from_file(path)
            if not isinstance(desc, spandrel.ImageModelDescriptor):
                raise ValueError(f"{os.path.basename(path)} is not an image upscaler")
            desc.model.eval()
            _loaded[path] = desc
        return _loaded[path]


def kept_text(source: bytes) -> dict[str, str]:
    """The text a PNG carries (generation settings), to be written into the upscaled picture."""
    try:
        info = Image.open(io.BytesIO(source)).info
    except Exception:
        return {}
    return {k: v for k, v in info.items() if isinstance(k, str) and isinstance(v, str)}


def to_png(image: Image.Image, text: dict[str, str], note: str) -> bytes:
    pnginfo = PngImagePlugin.PngInfo()
    for key, value in text.items():
        if key == "parameters" and value.strip():
            value = f"{value.rstrip()}, {note}"  # as Forge's Extras tab: behind the picture's own settings
        pnginfo.add_text(key, value)
    pnginfo.add_text("upscale", note)
    buf = io.BytesIO()
    image.save(buf, format="PNG", pnginfo=pnginfo)
    return buf.getvalue()


def save(png: bytes, folder: str, name: str) -> str:
    """folder/<date>/<number>-<name>.png"""
    folder = os.path.join(folder, datetime.date.today().isoformat())
    os.makedirs(folder, exist_ok=True)
    safe = re.sub(r"[^\w.-]+", "_", name)[:60] or "upscale"
    with _save_lock:
        numbers = [int(m.group(1)) for f in os.listdir(folder) if (m := re.match(r"(\d+)-", f))]
        path = os.path.join(folder, f"{(max(numbers) + 1) if numbers else 0:05d}-{safe}.png")
        with open(path, "wb") as f:
            f.write(png)
    return path


@dataclass
class Request:
    source: bytes  # the picture file
    name: str  # the upscaler model's name
    path: str
    scale: float = 2.0  # final size over the picture's size
    cpu: bool = False  # the processor only
    tile: int = 512
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))  # the graphics card, when not in CPU mode
    save_dir: Optional[str] = None
    prepare: Optional[Callable[[], None]] = None  # called with the graphics card held, before the model moves there (free VRAM)


class Task:
    def __init__(self, req: Request):
        self.id = uuid.uuid4().hex[:12]
        self.req = req
        self.status = "queued"  # queued | running | done | error | cancelled
        self.phase = "Starting…"
        self.done = self.total = 0  # tiles
        self.error = ""
        self.started = time.time()
        self.finished: Optional[float] = None
        self.png: Optional[bytes] = None
        self.size = self.source_size = (0, 0)
        self.device = "cpu" if req.cpu or req.device.type == "cpu" else req.device.type
        self.saved = ""
        self.cancel = threading.Event()

    def to_dict(self) -> dict:
        return {"id": self.id, "status": self.status, "phase": self.phase, "done": self.done, "total": self.total, "error": self.error,
                "seconds": round((self.finished or time.time()) - self.started, 1), "device": self.device, "upscaler": self.req.name,
                "width": self.size[0], "height": self.size[1], "source_width": self.source_size[0], "source_height": self.source_size[1],
                "saved": self.saved}


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, (getattr(torch, "OutOfMemoryError", ()), MemoryError)) or "out of memory" in str(e).lower() or "not enough memory" in str(e).lower()


@torch.inference_mode()
def _pass(task: Task, desc, image: torch.Tensor, device, dtype, tile: int) -> torch.Tensor:
    """One run of the model over `image` (1, 3, H, W in 0..1, on the CPU), in tiles; the result stays on the CPU
    (a 4× picture is bigger than most graphics memory)."""
    s = desc.scale
    _, _, H, W = image.shape
    out = torch.zeros(1, 3, H * s, W * s)
    weight = torch.zeros(1, 1, H * s, W * s)
    for ya in tile_starts(H, tile):
        for xa in tile_starts(W, tile):
            if task.cancel.is_set():
                raise Cancelled
            y1, x1 = min(ya + tile, H), min(xa + tile, W)
            part = desc(image[:, :, ya:y1, xa:x1].to(device=device, dtype=dtype)).float().cpu()
            w = _feather(part.shape[2], part.shape[3], OVERLAP * s, "cpu")
            out[:, :, ya * s : y1 * s, xa * s : x1 * s] += part * w
            weight[:, :, ya * s : y1 * s, xa * s : x1 * s] += w
            task.done += 1
    return (out / weight.clamp_min(1e-6)).clamp(0, 1)


def _work(task: Task):
    req = task.req
    on_gpu = task.device != "cpu"
    held = turn = False
    desc = None
    try:
        picture = Image.open(io.BytesIO(req.source))
        picture.load()
        if picture.mode in ("RGBA", "LA", "P"):  # transparency onto white, as the other tabs do
            rgba = picture.convert("RGBA")
            picture = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
        picture = picture.convert("RGB")
        task.source_size = picture.size
        target = (max(1, round(picture.width * req.scale)), max(1, round(picture.height * req.scale)))
        if max(target) > MAX_SIDE:
            raise ValueError(f"The result would be {target[0]}×{target[1]}: more than {MAX_SIDE} px a side. Use a smaller scale.")
        task.phase = "Waiting for another upscale to finish…"
        while not _one_at_a_time.acquire(timeout=0.25):
            if task.cancel.is_set():
                raise Cancelled
        turn = True
        if on_gpu:
            task.phase = "Waiting for the graphics card (a picture is being made)…"
            while not gpu_lock.acquire(timeout=0.25):
                if task.cancel.is_set():
                    raise Cancelled
            held = True
        task.status, task.phase = "running", f"Loading {req.name}…"
        desc = _load(req.path)
        device = req.device if on_gpu else torch.device("cpu")
        dtype = torch.float16 if device.type == "cuda" and desc.supports_half else torch.float32
        if on_gpu and req.prepare is not None:
            req.prepare()
        desc.to(device).to(dtype)
        count = passes_needed(req.scale, desc.scale)
        w, h, task.total = picture.width, picture.height, 0
        for _ in range(count):
            task.total += len(tile_starts(h, req.tile)) * len(tile_starts(w, req.tile))
            w, h = w * desc.scale, h * desc.scale
        x = torch.from_numpy(np.asarray(picture, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        for k in range(count):
            task.phase = f"Upscaling on the {'graphics card' if on_gpu else 'processor'}" + (f" (pass {k + 1} of {count})" if count > 1 else "")
            x = _pass(task, desc, x, device, dtype, req.tile)
        result = Image.fromarray((x[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8))
        del x
        if result.size != target:
            result = result.resize(target, Image.LANCZOS)
        task.phase = "Saving…"
        task.size = result.size
        task.png = to_png(result, kept_text(req.source), f"Postprocess upscale by: {req.scale:g}, Postprocess upscaler: {req.name}")
        if req.save_dir:
            task.saved = save(task.png, req.save_dir, req.name)
        task.status, task.phase = "done", "Done"
    except Cancelled:
        task.status, task.phase = "cancelled", "Cancelled"
    except Exception as e:
        task.status = "error"
        if _is_oom(e):
            task.error = ("Not enough graphics memory for this tile size: pick a smaller tile, or turn on CPU mode." if on_gpu
                          else "Not enough system RAM for this picture: pick a smaller tile or a smaller scale.")
        else:
            task.error = str(e) or type(e).__name__
        print(f"[upscale] {req.name} failed: {e}", flush=True)
    finally:
        if desc is not None and on_gpu:
            try:
                desc.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        if held:
            gpu_lock.release()
        if turn:
            _one_at_a_time.release()
        task.finished = time.time()
        task.req.source = b""  # the task stays for its result, not for the upload


def start(req: Request) -> Task:
    if req.tile not in TILES:
        raise ValueError(f"Tile size must be one of {', '.join(map(str, TILES))}")
    if not 1.0 <= req.scale <= 8.0:
        raise ValueError("Scale must be between 1 and 8")
    task = Task(req)
    finished = sorted((t for t in TASKS.values() if t.finished), key=lambda t: t.finished)
    for old in finished[: max(0, len(finished) - KEEP + 1)]:
        del TASKS[old.id]
    TASKS[task.id] = task
    threading.Thread(target=_work, args=(task,), daemon=True, name=f"upscale-{task.id}").start()
    return task


def add_routes(app, base: str, *, folders: Callable[[], list[str]], device: Callable[[], torch.device], save_dir: Callable[[], str],
               gpu_name: Callable[[], str] = lambda: "", prepare: Optional[Callable[[], None]] = None, guard: Optional[Callable] = None):
    """The Upscale tab's API under `base` (/venno/v1 or /moss/v1): what a server gives is its upscaler folders, its
    graphics device and where results go."""
    @app.get(f"{base}/upscale/models")
    def upscale_models():
        dev = device()
        return {"models": list(list_models(folders())), "folders": folders(), "tiles": list(TILES), "gpu": gpu_name() if dev.type != "cpu" else "", "device": dev.type}

    @app.post(f"{base}/upscale")
    def upscale_start(req: dict, request: HttpRequest):
        if guard is not None:
            guard(request)
        known = list_models(folders())
        name = str(req.get("upscaler", ""))
        if name not in known:
            raise HTTPException(400, f'Upscaler "{name}" not found' if name else "Pick an upscaler model")
        try:
            source = decode(str(req.get("image", "")))
            if not source:
                raise ValueError("Add a picture first")
            task = start(Request(source=source, name=name, path=known[name], scale=float(req.get("scale", 2)), cpu=bool(req.get("cpu", False)),
                                 tile=int(req.get("tile", 512)), device=device(), save_dir=save_dir() if req.get("save", True) else None, prepare=prepare))
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        return task.to_dict()

    def find(task_id: str) -> Task:
        task = TASKS.get(task_id)
        if task is None:
            raise HTTPException(404, "No such upscale (results are kept for the last few only)")
        return task

    @app.get(f"{base}/upscale/{{task_id}}")
    def upscale_status(task_id: str):
        return find(task_id).to_dict()

    @app.get(f"{base}/upscale/{{task_id}}/image")
    def upscale_image(task_id: str):
        task = find(task_id)
        if task.png is None:
            raise HTTPException(404, "Not finished")
        return Response(task.png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    @app.post(f"{base}/upscale/{{task_id}}/cancel")
    def upscale_cancel(task_id: str):
        task = find(task_id)
        task.cancel.set()
        return task.to_dict()
