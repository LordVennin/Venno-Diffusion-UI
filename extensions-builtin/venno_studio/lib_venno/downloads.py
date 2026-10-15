"""
Model downloads for Venno Studio: a curated catalog of Anima-related files plus
"download from URL", saved straight into the folders the loaders scan.
"""

import os
import re
import threading
import time
import urllib.parse
import urllib.request
import uuid

from modules import paths, shared

HF = "https://huggingface.co"
ANIMA = f"{HF}/circlestone-labs/Anima/resolve/main/split_files"
LLLITE = f"{HF}/kohya-ss/Anima-LLLite/resolve/main"

MODEL_EXTENSIONS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf")

KINDS = {
    "checkpoint": "Checkpoint",
    "text_encoder": "Text encoder",
    "vae": "VAE",
    "lora": "LoRA",
    "controlnet": "Guide (ControlNet / ControlLLLite)",
    "upscaler": "Upscaler",
    "adetailer": "ADetailer detector",
}

# (id, kind, name, description, url, size in bytes)
CATALOG = [
    ("anima-turbo", "checkpoint", "Anima Turbo v1.1", "Distilled, 8–12 steps at CFG 1. Recommended to start with; best for live mode.", f"{ANIMA}/diffusion_models/anima-turbo-v1.1.safetensors", 4182230656),
    ("anima-aesthetic", "checkpoint", "Anima Aesthetic v1.1", "Fine-tuned for a consistent, high quality default style. 30–50 steps, CFG 4–5.", f"{ANIMA}/diffusion_models/anima-aesthetic-v1.1.safetensors", 4182230656),
    ("anima-base", "checkpoint", "Anima Base v1.0", "Unrefined base model: most flexible, best for style LoRAs.", f"{ANIMA}/diffusion_models/anima-base-v1.0.safetensors", 4182218328),
    ("anima-29b", "checkpoint", "Anima 2.9B preview (community)", "Layer-expanded Anima (40 blocks) by Gazingstars123. Same text encoder / VAE. 28–50 steps, CFG 3.5–5.", f"{HF}/Gazingstars123/Anima-2.9B/resolve/main/Anima-2.9B-preview-v1.safetensors", 5843204206),
    ("qwen3-te", "text_encoder", "Qwen3 0.6B text encoder", "Required by every Anima checkpoint.", f"{ANIMA}/text_encoders/qwen_3_06b_base.safetensors", 1192135096),
    ("qwen-vae", "vae", "Qwen-Image VAE", "Required by every Anima checkpoint.", f"{ANIMA}/vae/qwen_image_vae.safetensors", 253806246),
    ("lllite-anytest", "controlnet", "LLLite any-test-like v2", "Best all-round sketch guide: line art, scribbles and grayscale. Trained on Anima Base v1.0.", f"{LLLITE}/anima-lllite-any-test-like-v2.safetensors", 16403008),
    ("lllite-lineart", "controlnet", "LLLite lineart", "Clean line art (black lines on white). Preview3-era weights.", f"{LLLITE}/anima-lllite-lineart-1.safetensors", 7787888),
    ("lllite-scribble", "controlnet", "LLLite scribble", "Rough scribbles (Venno inverts the lines automatically). Preview3-era weights.", f"{LLLITE}/anima-lllite-scribble-1.safetensors", 7864128),
    ("lllite-depth", "controlnet", "LLLite depth", "Depth maps (white = near). Preview3-era weights.", f"{LLLITE}/anima-lllite-depth-1.safetensors", 7864128),
    ("lllite-pose", "controlnet", "LLLite pose", "DWPose skeletons. Preview3-era weights.", f"{LLLITE}/anima-lllite-pose-1.safetensors", 23310040),
    ("lllite-inpaint", "controlnet", "LLLite inpainting v2", "Inpainting helper for large masked areas.", f"{LLLITE}/anima-lllite-inpainting-v2.safetensors", 65772872),
    ("up-animesharp-v3", "upscaler", "2x AnimeSharp V3", "Fast 2× anime upscaler for hires fix.", f"{HF}/Kim2091/AnimeSharpV3/resolve/main/2x-AnimeSharpV3.safetensors", 33478206),
    ("up-animesharp", "upscaler", "4x AnimeSharp", "Sharp 4× anime upscaler.", f"{HF}/Kim2091/AnimeSharp/resolve/main/4x-AnimeSharp.pth", 67010245),
    ("up-realesrgan-anime", "upscaler", "4x RealESRGAN anime 6B", "Small, soft 4× anime upscaler.", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth", 17938799),
    ("up-ultrasharp", "upscaler", "4x UltraSharp", "General purpose 4× upscaler.", f"{HF}/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth", 66961958),
    ("ad-face-v9c", "adetailer", "face_yolov9c", "More accurate face detector than the bundled yolov8 ones.", f"{HF}/Bingsu/adetailer/resolve/main/face_yolov9c.pt", 51648019),
]


def folder_for(kind: str) -> str:
    opts = shared.cmd_opts
    models = paths.models_path
    folder = {
        "checkpoint": (opts.ckpt_dirs or [None])[0] or os.path.join(models, "Stable-diffusion"),
        "text_encoder": (opts.text_encoder_dirs or [None])[0] or os.path.join(models, "text_encoder"),
        "vae": (opts.vae_dirs or [None])[0] or os.path.join(models, "VAE"),
        "lora": opts.lora_dir,
        "controlnet": opts.controlnet_dir,
        "upscaler": opts.esrgan_models_path,
        "adetailer": os.path.join(models, "adetailer"),
    }[kind]
    os.makedirs(folder, exist_ok=True)
    return folder


def filename_from_url(url: str) -> str:
    name = os.path.basename(urllib.parse.urlparse(url).path)
    return sanitize(urllib.parse.unquote(name))


def sanitize(name: str) -> str:
    name = os.path.basename(name.replace("\\", "/"))
    name = re.sub(r"[^\w.\-+ ()\[\]]", "_", name).strip(" .")
    return name


def catalog() -> list[dict]:
    out = []
    for cid, kind, name, desc, url, size in CATALOG:
        fname = filename_from_url(url)
        out.append({"id": cid, "kind": kind, "kind_label": KINDS[kind], "name": name, "description": desc, "filename": fname, "size": size, "installed": os.path.isfile(os.path.join(folder_for(kind), fname))})
    return out


class Job:
    def __init__(self, url: str, kind: str, filename: str, name: str):
        self.id = uuid.uuid4().hex[:12]
        self.url = url
        self.kind = kind
        self.filename = filename
        self.name = name
        self.total = 0
        self.done = 0
        self.status = "queued"  # queued | downloading | done | error | cancelled
        self.error = ""
        self.started = time.time()
        self.cancel = threading.Event()

    def to_dict(self):
        return {k: getattr(self, k) for k in ("id", "url", "kind", "filename", "name", "total", "done", "status", "error")}


JOBS: dict[str, Job] = {}
_lock = threading.Lock()


def start(url: str, kind: str, filename: str | None = None, name: str | None = None, on_done=None) -> Job:
    if kind not in KINDS:
        raise ValueError(f"Unknown model type: {kind}")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("Only https:// URLs are supported")
    filename = sanitize(filename or filename_from_url(url))
    if not filename:
        raise ValueError("Could not determine a file name; add one")
    if not filename.lower().endswith(MODEL_EXTENSIONS):
        raise ValueError(f"File name must end in one of: {', '.join(MODEL_EXTENSIONS)}")

    with _lock:
        for j in JOBS.values():
            if j.filename == filename and j.kind == kind and j.status in ("queued", "downloading"):
                return j
        job = Job(url, kind, filename, name or filename)
        JOBS[job.id] = job

    threading.Thread(target=_run, args=(job, on_done), daemon=True, name=f"venno-download-{job.id}").start()
    return job


def _run(job: Job, on_done):
    dest = os.path.join(folder_for(job.kind), job.filename)
    tmp = dest + ".part"
    try:
        job.status = "downloading"
        req = urllib.request.Request(job.url, headers={"User-Agent": "VennoStudio/1.0"})
        token = os.environ.get("HF_TOKEN")
        if token and urllib.parse.urlparse(job.url).hostname == "huggingface.co":
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
            job.total = int(resp.headers.get("Content-Length") or 0)
            while True:
                if job.cancel.is_set():
                    raise InterruptedError
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                job.done += len(chunk)
        if job.total and job.done != job.total:
            raise IOError(f"incomplete download ({job.done} of {job.total} bytes)")
        os.replace(tmp, dest)
        job.status = "done"
        print(f"[Venno] Downloaded {job.filename} -> {dest}")
        if on_done:
            on_done(job)
    except InterruptedError:
        job.status = "cancelled"
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        print(f"[Venno] Download of {job.filename} failed: {e}")
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def jobs() -> list[dict]:
    return [j.to_dict() for j in sorted(JOBS.values(), key=lambda j: j.started, reverse=True)]


def cancel(job_id: str):
    job = JOBS.get(job_id)
    if job:
        job.cancel.set()
