"""
Venno Regional Prompting

Latent-couple style regional conditioning for Anima:
every denoising step, the positive branch is evaluated once with the root prompt
and once per region prompt; the predictions are blended per-pixel using the
region masks. Works with any sampler, CFG, LoRA, inpainting and img2img.

API usage (alwayson_scripts):
    "venno regional prompting": {"args": [true, "<json>"]}

where <json> is
    {
        "base_strength": 0.25,   # how much the root prompt still applies inside regions (0..1)
        "end_at": 1.0,           # stop regional guidance after this fraction of steps
        "regions": [
            {"prompt": "...", "mask": "<base64 png, white = region>", "strength": 1.0},
            ...
        ]
    }

Masks are interpreted relative to the full generation canvas (p.width x p.height).
"""

import base64
import io
import json

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend import memory_management
from backend.sampling.sampling_function import compute_cond_indices, compute_cond_mark
from modules import extra_networks, scripts
from modules.prompt_parser import SdConditioning

TITLE = "Venno Regional Prompting"


def decode_mask(data: str) -> Image.Image:
    if "," in data and data.lstrip().startswith("data:"):
        data = data.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    if img.mode in ("RGBA", "LA"):
        # painted region layers: use alpha as the mask
        return img.getchannel("A")
    return img.convert("L")


def encode_prompt(sd_model, text: str, width: int, height: int) -> torch.Tensor:
    text, _ = extra_networks.parse_prompt(text)
    # is_negative_prompt=True keeps Anima's reference-latent bookkeeping untouched
    conds = sd_model.get_learned_conditioning(SdConditioning([text], is_negative_prompt=True, width=width, height=height))
    cond = conds[0] if isinstance(conds, (list, tuple)) else conds
    if isinstance(cond, dict):
        cond = cond["crossattn"]
    if cond.ndim == 2:
        cond = cond.unsqueeze(0)
    return cond.reshape(-1, cond.shape[-2], cond.shape[-1])[:1].detach()


class Region:
    def __init__(self, cond: torch.Tensor, mask: torch.Tensor, strength: float):
        self.cond = cond  # (1, L, D)
        self.mask = mask  # (H, W) float 0..1 at pixel resolution
        self.strength = strength
        self._cache = {}

    def mask_for(self, x: torch.Tensor) -> torch.Tensor:
        """Mask broadcastable against a latent chunk x (B, C, [T,] H, W)."""
        h, w = x.shape[-2], x.shape[-1]
        key = (h, w, x.ndim, x.device, x.dtype)
        if key not in self._cache:
            m = F.interpolate(self.mask[None, None].float(), size=(h, w), mode="area")
            # soften edges by ~1 latent pixel so neighbouring regions blend instead of leaving a seam
            m = F.avg_pool2d(F.pad(m, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
            if x.ndim == 5:
                m = m.unsqueeze(2)
            self._cache[key] = (m * self.strength).to(device=x.device, dtype=x.dtype)
        return self._cache[key]


def _slice_batch(value, i: int, n: int, total: int):
    """Take chunk i of n from anything batched along dim 0 (tensors, nested lists / dicts)."""
    if isinstance(value, torch.Tensor):
        return value.chunk(n)[i] if value.ndim > 0 and value.shape[0] == total else value
    if isinstance(value, (list, tuple)):
        return type(value)(_slice_batch(v, i, n, total) for v in value)
    if isinstance(value, dict):
        return {k: _slice_batch(v, i, n, total) for k, v in value.items()}
    return value


def chunk_conditioning(c: dict, i: int, n: int, total: int, sigma: torch.Tensor) -> dict:
    """Conditioning dict for a single cond chunk, with transformer_options describing just that chunk."""
    out = {k: (_slice_batch(v, i, n, total) if k not in ("transformer_options", "control_model") else v) for k, v in c.items()}
    if "transformer_options" in c:
        to = dict(c["transformer_options"])
        to["cond_or_uncond"] = [0]
        to["sigmas"] = sigma
        to["cond_mark"] = compute_cond_mark(cond_or_uncond=[0], sigmas=sigma)
        to["cond_indices"], to["uncond_indices"] = compute_cond_indices(cond_or_uncond=[0], sigmas=sigma)
        out["transformer_options"] = to
    return out


def build_wrapper(regions: list[Region], base_strength: float, sigma_end: float, previous_wrapper=None):
    COND = 0

    def call_model(apply_model, x, t, c, cond_or_uncond):
        if previous_wrapper is not None:
            return previous_wrapper(apply_model, {"input": x, "timestep": t, "c": c, "cond_or_uncond": cond_or_uncond})
        return apply_model(x, t, **c)

    def wrapper(apply_model, args):
        x: torch.Tensor = args["input"]
        t: torch.Tensor = args["timestep"]
        c: dict = args["c"]
        cond_or_uncond: list[int] = args["cond_or_uncond"]

        out = call_model(apply_model, x, t, c, cond_or_uncond)

        if COND not in cond_or_uncond or float(t.max()) < sigma_end:
            return out

        n = len(cond_or_uncond)
        outs = list(out.chunk(n))
        xs = x.chunk(n)
        ts = t.chunk(n)
        crossattn = c["c_crossattn"].chunk(n)

        for i, kind in enumerate(cond_or_uncond):
            if kind != COND:
                continue

            xi, ti = xs[i], ts[i]
            base = outs[i]
            bsz = xi.shape[0]

            masks = [r.mask_for(xi) for r in regions]
            coverage = torch.clamp(sum(masks), 0.0, 1.0)
            w_base = 1.0 - coverage * (1.0 - base_strength)

            acc = base * w_base
            wsum = w_base.clone()

            base_c = chunk_conditioning(c, i, n, x.shape[0], ti)
            for region, m in zip(regions, masks):
                ci = dict(base_c)
                ci["c_crossattn"] = region.cond.to(device=crossattn[i].device, dtype=crossattn[i].dtype).expand(bsz, -1, -1)
                pred = call_model(apply_model, xi, ti, ci, [COND])
                acc = acc + pred * m
                wsum = wsum + m

            outs[i] = acc / wsum.clamp_min(1e-6)

        return torch.cat(outs, dim=0)

    return wrapper


class VennoRegionalPrompting(scripts.Script):
    sorting_priority = 5

    def title(self):
        return TITLE

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(open=False, label=TITLE):
            gr.Markdown("Regions are normally painted in **Venno Studio**. Advanced: paste the JSON payload here.")
            enabled = gr.Checkbox(False, label="Enabled")
            payload = gr.Textbox("", label="Regions JSON", lines=4)
        return [enabled, payload]

    def process_before_every_sampling(self, p, enabled=False, payload="", **kwargs):
        if not enabled or not payload:
            return

        data = json.loads(payload) if isinstance(payload, str) else payload
        entries = [r for r in data.get("regions", []) if r.get("prompt", "").strip() and r.get("mask")]
        if not entries:
            return

        base_strength = float(data.get("base_strength", 0.25))
        end_at = float(data.get("end_at", 1.0))

        width, height = p.width, p.height
        regions: list[Region] = []
        for entry in entries:
            mask_img = decode_mask(entry["mask"]).resize((width, height), Image.BILINEAR)
            mask = torch.from_numpy(np.asarray(mask_img, dtype=np.float32) / 255.0)
            if float(mask.max()) <= 0.0:
                continue
            cond = encode_prompt(p.sd_model, entry["prompt"], width, height)
            regions.append(Region(cond, mask, float(entry.get("strength", 1.0))))

        if not regions:
            return

        unet = p.sd_model.forge_objects.unet.clone()
        predictor = unet.model.predictor
        sigma_end = float(predictor.percent_to_sigma(end_at)) if end_at < 1.0 else -1.0

        previous = unet.model_options.get("model_function_wrapper", None)
        unet.set_model_unet_function_wrapper(build_wrapper(regions, base_strength, sigma_end, previous))
        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params["Venno Regions"] = len(regions)
        p.extra_generation_params["Venno Region Base"] = base_strength
        memory_management.logger.info(f"[Venno] Regional prompting with {len(regions)} region(s)")
