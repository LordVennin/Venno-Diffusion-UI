"""The MOSS-Image engine: loads a model, encodes prompts and samples, the way venno-trainer's samples are made
(the vendor pipeline's Euler on its snapped sigma grid, CFG with DiT block 22 skipped on the negative branch,
fp32 noise from a CPU generator), so a seed and settings that made a training sample make the same picture.

Memory (``config.memory_mode()``):
- full: the transformer in bf16 on the GPU (about 10 GB); for cards with about 20 GB or more.
- low: the transformer's block weights in fp8 (about 5 GB), dequantized per layer; for 12 GB cards.
- auto: full with 20 GB of VRAM or more, else low.
The 9B text encoder (about 17 GB in bf16) stays in system RAM and only its layers visit the GPU, one at a
time, when a prompt is encoded; encoded prompts are cached, so it only runs for new text.
"""

from __future__ import annotations

import gc
import json
import math
import os
import threading
import time
from collections import OrderedDict
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from . import config, lora, models, upscale
from .model.fp8 import quantize_modules
from .model.transformer import MossImageTransformer
from .model.vae import AutoEncoder, convert_diffusers_state_dict

NUM_TRAIN_TIMESTEPS = 1000
MAX_TEXT_TOKENS = 4096
DEFAULTS = {"steps": 50, "cfg": 3.0, "shift": 5.0, "skip_layers": [22], "width": 1024, "height": 1024}
FULL_VRAM_GB = 20
PROMPT_CACHE = 64


class Cancelled(Exception):
    pass


def run_steps(steps: int, denoise: float = 1.0) -> int:
    """Steps actually sampled in a hires pass: the last `denoise` part of the schedule (as in Venno)."""
    return steps if denoise >= 1.0 else max(1, min(steps, math.floor(steps * denoise + 0.5)))


def hires_size(width: int, height: int, scale: float = 1.5, resize_x: int = 0, resize_y: int = 0) -> tuple[int, int]:
    """Final size of a hires job: explicit width/height (one may be 0 = keep aspect), else the scale; multiples of 16."""
    r = lambda v: max(16, round(v / 16) * 16)
    if resize_x and resize_y:
        return r(resize_x), r(resize_y)
    if resize_x:
        return r(resize_x), r(height * resize_x / width)
    if resize_y:
        return r(width * resize_y / height), r(resize_y)
    return r(width * scale), r(height * scale)


def moss_sigmas(num_steps, shift=5.0, training_shift=1.0, total=NUM_TRAIN_TIMESTEPS):
    """The vendor MOSSImageScheduler grid (as venno-trainer): shifted, snapped to the nearest training sigma,
    duplicates removed, terminal 0 appended."""

    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(num_steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


# ---------------------------------------------------------------- loading helpers


def load_weights(path: str, key_filter: Optional[Callable[[str], bool]] = None) -> dict[str, torch.Tensor]:
    """One .safetensors file, or every shard of a folder."""
    if os.path.isfile(path):
        files = [path]
    else:
        index = [n for n in os.listdir(path) if n.endswith(".safetensors.index.json")]
        if index:
            with open(os.path.join(path, index[0]), encoding="utf-8") as f:
                files = sorted({os.path.join(path, v) for v in json.load(f)["weight_map"].values()})
        else:
            files = sorted(os.path.join(path, n) for n in os.listdir(path) if n.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors weights in {path}")
    sd: dict[str, torch.Tensor] = {}
    for file in files:
        if key_filter is None:
            sd.update(load_file(file))
            continue
        with safe_open(file, "pt") as f:
            for k in f.keys():
                if key_filter(k):
                    sd[k] = f.get_tensor(k)
    return sd


def strip_prefixes(sd: dict, prefixes) -> dict:
    out = {}
    for k, v in sd.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p) :]
                break
        out[k] = v
    return out


def assert_materialized(module: torch.nn.Module, what: str):
    meta = [n for n, t in [*module.named_parameters(), *module.named_buffers()] if t.is_meta]
    if meta:
        raise RuntimeError(f"{what}: weights missing from the files, e.g. {meta[:5]}")


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class LayerStreamer:
    """Keeps a stack of layers in system RAM and copies each one to the GPU only for its own forward."""

    def __init__(self, layers, device):
        self.device = device
        self.cpu = {}
        for layer in layers:
            self.cpu[id(layer)] = {n: p.data for n, p in [*layer.named_parameters(), *layer.named_buffers()]}
            layer.register_forward_pre_hook(self._load)
            layer.register_forward_hook(self._unload)

    def _tensors(self, layer):
        return [*layer.named_parameters(), *layer.named_buffers()]

    def _load(self, layer, _args):
        for name, t in self._tensors(layer):
            t.data = self.cpu[id(layer)][name].to(self.device, non_blocking=True)

    def _unload(self, layer, _args, _out):
        for name, t in self._tensors(layer):
            t.data = self.cpu[id(layer)][name]  # the GPU copy is simply dropped


# ---------------------------------------------------------------- engine


class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.token_lock = threading.Lock()
        self.device = torch.device("cpu") if config.args.cpu or not torch.cuda.is_available() else torch.device("cuda")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.found = models.Found()
        self.dit = None
        self.dit_key = None  # (transformer path, memory mode)
        self.dit_parked = False  # moved to system RAM to make room for a VAE call
        self.loras: Optional[lora.LoraStack] = None
        self.te = self.tokenizer = None
        self.te_streamed = False
        self.vae = None
        self.prompts: "OrderedDict[str, torch.Tensor]" = OrderedDict()  # prompt → (tokens, 4096) on the CPU
        self.rescan()

    # -- state --
    def rescan(self) -> models.Found:
        self.found = models.scan()
        return self.found

    def vram_gb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.get_device_properties(self.device).total_memory / 1024**3

    def resolved_memory(self) -> str:
        mode = config.memory_mode()
        if mode != "auto":
            return mode
        return "full" if self.device.type != "cuda" or self.vram_gb() >= FULL_VRAM_GB else "low"

    def info(self) -> dict:
        return {
            "device": self.device.type,
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "",
            "vram_gb": round(self.vram_gb(), 1),
            "memory": config.memory_mode(),
            "memory_resolved": self.resolved_memory(),
            "loaded": os.path.basename(os.path.dirname(self.dit_key[0])) if self.dit_key and os.path.isdir(self.dit_key[0]) else (os.path.basename(self.dit_key[0]) if self.dit_key else ""),
            "prompt_cache": len(self.prompts),
        }

    # -- parts --
    def _load_te(self):
        if self.te is not None:
            return
        from transformers import AutoConfig, AutoTokenizer, Qwen3_5TextModel

        if not (self.found.text_encoder and self.found.tokenizer):
            raise FileNotFoundError("The MOSS-Image text encoder isn't there yet: download a model in ⚙ Settings → Models")
        t0 = time.time()
        print(f"[Venno MOSS] loading the Qwen3.5 9B text encoder from {self.found.text_encoder}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.found.tokenizer)
        cfg = AutoConfig.from_pretrained(self.found.text_encoder)
        cfg = getattr(cfg, "text_config", cfg)
        dropped = ("visual.", "model.visual.", "lm_head.", "mtp.")  # the language tower only
        sd = load_weights(self.found.text_encoder, key_filter=lambda k: not k.startswith(dropped))
        sd = strip_prefixes(sd, ("model.language_model.", "language_model.", "model."))
        with torch.device("meta"):
            model = Qwen3_5TextModel(cfg)
        model.load_state_dict({k: v.to(self.dtype) for k, v in sd.items()}, strict=False, assign=True)
        del sd
        rotary = getattr(model, "rotary_emb", None)
        if rotary is not None and any(b.is_meta for b in rotary.buffers()):  # computed in __init__, not saved
            with torch.device("cpu"):
                model.rotary_emb = type(rotary)(config=cfg)
        assert_materialized(model, "Qwen3.5 text encoder")
        model = model.eval().requires_grad_(False)
        if self.device.type == "cuda":
            # embeddings (2 GB) stay on the CPU (only looked up); the layers visit the GPU one at a time
            for name, child in model.named_children():
                if name not in ("layers", "embed_tokens"):
                    child.to(self.device)
            LayerStreamer(model.layers, self.device)
            self.te_streamed = True
        self.te = model
        print(f"[Venno MOSS] text encoder ready in {time.time() - t0:.1f}s", flush=True)

    def _load_vae(self):
        if self.vae is not None:
            return
        if not self.found.vae:
            raise FileNotFoundError("The MOSS-Image VAE isn't there yet: download a model in ⚙ Settings → Models")
        sd = load_weights(self.found.vae)
        if any(k.startswith("encoder.down_blocks.") or k.startswith("decoder.up_blocks.") for k in sd):
            sd = convert_diffusers_state_dict(sd)
        params = AutoEncoder.aitk_config_from_state_dict(sd)
        with torch.device("meta"):
            vae = AutoEncoder(params)
        vae.load_state_dict(sd, assign=True, strict=False)
        if vae.bn.num_batches_tracked.is_meta:  # a bookkeeping counter, not always saved
            vae.bn.num_batches_tracked = torch.tensor(0, dtype=torch.long)
        assert_materialized(vae, "FLUX.2 VAE")
        self.vae = vae.eval().requires_grad_(False).to(self.device, dtype=torch.float32)  # the vendor decodes in fp32

    def load_model(self, name: str):
        """The transformer of model `name` (Base, SFT or a fine-tune), in the memory mode asked for."""
        with self.lock:
            model = self.found.model(name)
            key = (model.transformer, self.resolved_memory())
            if self.dit is not None and self.dit_key == key:
                return self.dit
            self.unload_model()
            t0 = time.time()
            print(f"[Venno MOSS] loading {model.name} ({key[1]} memory)", flush=True)
            sd = load_weights(model.transformer)
            sd = strip_prefixes(sd, ("model.diffusion_model.", "diffusion_model."))
            sd = MossImageTransformer.convert_state_dict_on_load(sd)
            params = MossImageTransformer.aitk_config_from_state_dict(sd)
            with torch.device("meta"):
                dit = MossImageTransformer(params)
            dit.load_state_dict({k: v.to(self.dtype) for k, v in sd.items()}, strict=True, assign=True)
            del sd
            dit = dit.eval().requires_grad_(False)
            if key[1] == "low":
                quantize_modules(dit, ["layers."], compute_dtype=self.dtype)  # on the CPU, before it moves
                free_memory()
            self.dit = dit.to(self.device)
            self.dit_parked = False
            self.dit_key = key
            self.loras = lora.LoraStack(self.dit, self.device, self.dtype)
            print(f"[Venno MOSS] {model.name} ready in {time.time() - t0:.1f}s", flush=True)
            return self.dit

    def unload_model(self):
        with self.lock:
            if self.loras is not None:
                self.loras.clear()
            self.dit = self.dit_key = self.loras = None
            free_memory()

    def unload_all(self):
        with self.lock:
            self.unload_model()
            self.te = self.tokenizer = self.vae = None
            self.te_streamed = False
            free_memory()

    # -- prompts --
    def _token_ids(self, text: str) -> list[int]:
        prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        return self.tokenizer(prompt, truncation=True, max_length=MAX_TEXT_TOKENS)["input_ids"]

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """(tokens, 4096) hidden states of a prompt (the chat-templated prompt's last hidden state), cached."""
        with self.lock:
            if text in self.prompts:
                self.prompts.move_to_end(text)
                return self.prompts[text]
            self._load_te()
            ids = torch.tensor([self._token_ids(text)])
            embeds = self.te.embed_tokens(ids).to(self.device)  # looked up on the CPU
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
                states = self.te(inputs_embeds=embeds, use_cache=False).last_hidden_state
            out = states[0].to(torch.bfloat16).cpu()
            self.prompts[text] = out
            while len(self.prompts) > PROMPT_CACHE:
                self.prompts.popitem(last=False)
            return out

    def token_count(self, text: str) -> int:
        with self.token_lock:  # not the engine lock: counting must not wait for a picture being made
            if self.tokenizer is None:
                if not self.found.tokenizer:
                    return 0
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.found.tokenizer)
            return len(self._token_ids(text))

    # -- sampling --
    def _collate(self, embeds: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        length = max(e.shape[0] for e in embeds)
        stacked = torch.stack([F.pad(e.to(self.dtype), (0, 0, 0, length - e.shape[0])) for e in embeds])
        mask = torch.stack([torch.arange(length) < e.shape[0] for e in embeds])
        return stacked.to(self.device), mask.to(self.device)

    def _vae(self, fn, x: torch.Tensor, tile: int, overlap: int, factor: float) -> torch.Tensor:
        """A VAE call on the whole picture. When that runs out of graphics memory (big hires pictures): again with
        the transformer parked in system RAM (it returns for the next sampling step), and only then in large
        overlapping tiles, which shift the tones a little (the VAE normalizes over what it sees)."""
        oom = lambda e: isinstance(e, getattr(torch, "OutOfMemoryError", ())) or "out of memory" in str(e).lower()
        try:
            return fn(x).float()
        except Exception as e:
            if not oom(e):
                raise
        free_memory()
        if self.dit is not None and self.device.type == "cuda" and not self.dit_parked:
            print("[Venno MOSS] not enough graphics memory for the VAE: moving the transformer to system RAM meanwhile", flush=True)
            self.dit.to("cpu")
            self.dit_parked = True
            free_memory()
            try:
                return fn(x).float()
            except Exception as e:
                if not oom(e):
                    raise
            free_memory()
        print("[Venno MOSS] not enough graphics memory for the VAE on the whole picture: working in tiles", flush=True)
        return upscale.tiled(fn, x, tile, overlap, factor)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor):
        from PIL import Image

        self._load_vae()
        pixels = self._vae(self.vae.decode, latents.to(self.device, torch.float32), 64, 16, 16)  # tiles of 1024 px
        img = ((pixels[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        return Image.fromarray(img.permute(1, 2, 0).cpu().numpy())

    @torch.no_grad()
    def encode_image(self, image) -> torch.Tensor:
        """A picture (sides multiples of 16) as latents (1, 128, H/16, W/16)."""
        import numpy as np

        self._load_vae()
        x = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
        return self._vae(self.vae.encode, x.to(self.device, torch.float32), 1024, 256, 1 / 16)

    @torch.no_grad()
    def resize_latents(self, latents: torch.Tensor, width: int, height: int, mode: str) -> torch.Tensor:
        """Latent upscale. The model's latents pack 2×2 VAE latents into the channels, so they are unpacked (and
        un-normalized) to the VAE's own grid, resized there, and packed again."""
        from einops import rearrange

        self._load_vae()
        z = self.vae.inv_normalize(latents.to(self.device, torch.float32))
        z = rearrange(z, "b (c pi pj) i j -> b c (i pi) (j pj)", pi=2, pj=2)
        z = F.interpolate(z, size=(height // 8, width // 8), mode=mode)
        return self.vae.normalize(rearrange(z, "b c (i pi) (j pj) -> b (c pi pj) i j", pi=2, pj=2))

    def upscale_latents(self, latents: torch.Tensor, upscaler: str, width: int, height: int) -> torch.Tensor:
        """Hires fix's upscale: latent resize, or decode → upscale (model / resampling) → encode."""
        if upscale.is_latent(upscaler):
            return self.resize_latents(latents, width, height, upscale.latent_mode(upscaler))
        return self.encode_image(upscale.upscale(self.decode(latents), upscaler, width, height, self.device))

    def _denoise(self, dit, latents, sig, cond, uncond, cfg: float, skip, *, done: int, total: int, on_step, on_preview, preview_every, is_cancelled):
        """Euler over the sigmas `sig` (the trainer's sampler); `done` steps of `total` ran before (progress)."""
        if self.dit_parked:  # a VAE call needed its room (see _vae)
            dit.to(self.device)
            self.dit_parked = False
        n = len(sig) - 1
        for i in range(n):
            if is_cancelled():
                raise Cancelled
            t = (sig[i] * NUM_TRAIN_TIMESTEPS).reshape(1)
            pred = dit(latents, t, *cond).float()
            if uncond is not None:
                pu = dit(latents, t, *uncond, skip_layers=skip).float()
                pred = pu + cfg * (pred - pu)
            if on_preview is not None and preview_every and (done + i + 1) % preview_every == 0 and i + 1 < n:
                on_preview(self.decode(latents - sig[i] * pred))  # the picture this step points at
            latents = latents + (sig[i + 1] - sig[i]) * pred
            on_step(done + i + 1, total)
        return latents

    @torch.inference_mode()
    def generate(self, *, model: str, prompt: str, negative: str, width: int, height: int, steps: int, cfg: float,
                 shift: float, seed: int, skip_layers: list[int], loras: list[tuple[str, float]], hires: Optional[dict] = None,
                 on_step: Callable[[int, int], None] = lambda i, n: None,
                 on_preview: Optional[Callable[[object], None]] = None, preview_every: int = 0,
                 is_cancelled: Callable[[], bool] = lambda: False) -> tuple[object, list[str]]:
        """One picture. Returns (PIL image, warnings).

        hires: {upscaler, width, height, steps (0 = as the first pass), denoise, cfg / shift (None = as the first
        pass)}: the first pass's latents are upscaled to width × height, noised to the `denoise` point of the
        hires schedule and sampled from there (the last `denoise` part of its steps), like Venno's hires fix."""
        with self.lock:
            width, height = width // 16 * 16, height // 16 * 16
            cond = self.encode(prompt)
            hires_cfg = (hires.get("cfg") or cfg) if hires else cfg
            uncond = self.encode(negative) if cfg != 1.0 or hires_cfg != 1.0 else None
            dit = self.load_model(model)
            warnings = self.loras.set([(lora.resolve(name), weight) for name, weight in loras])
            c = self._collate([cond])
            u = self._collate([uncond]) if uncond is not None else None
            noise = lambda w, h: torch.randn(1, dit.params.latent_channels, h // 16, w // 16, generator=torch.Generator("cpu").manual_seed(int(seed))).to(self.device)  # fp32, as the trainer
            sig = moss_sigmas(steps, shift=shift).to(self.device)
            total = len(sig) - 1
            if hires:
                full = moss_sigmas(hires.get("steps") or steps, shift=hires.get("shift") or shift)
                sig2 = full[-(run_steps(len(full) - 1, hires.get("denoise", 0.5)) + 1) :].to(self.device)
                total += len(sig2) - 1
            skip = [k for k in skip_layers if 0 <= k < len(dit.layers)]
            common = dict(total=total, on_step=on_step, on_preview=on_preview, preview_every=preview_every, is_cancelled=is_cancelled)
            latents = self._denoise(dit, noise(width, height), sig, c, u if cfg != 1.0 else None, cfg, skip, done=0, **common)
            if hires:
                tw, th = hires["width"] // 16 * 16, hires["height"] // 16 * 16
                if tw * th > 4.3e6:
                    warnings = [*warnings, f"{tw}×{th} is above the 4 megapixels MOSS-Image is made for: expect repeated or broken details"]
                latents = self.upscale_latents(latents, hires.get("upscaler", "Lanczos"), tw, th)
                latents = (1 - sig2[0]) * latents + sig2[0] * noise(tw, th)  # flow matching: back to the schedule's `denoise` point
                latents = self._denoise(dit, latents, sig2, c, u if hires_cfg != 1.0 else None, hires_cfg, skip, done=len(sig) - 1, **common)
            return self.decode(latents), warnings


engine: Optional[Engine] = None


def get() -> Engine:
    global engine
    if engine is None:
        engine = Engine()
    return engine
