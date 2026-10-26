"""Regional prompting by cross-attention blending.

Each transformer block runs its text cross-attention once with the main prompt and once per region prompt
(every prompt with its own normal 512-token context, so the model sees exactly what it was trained on),
then blends the outputs per image token with the region masks. Self-attention and the MLP run once on the
shared image, so regions stay coherent with each other, and a step costs one model pass instead of one
pass per region (Forge's latent couple).

Weights per token (same rule as Venno's Forge-side script):
    base   = 1 - coverage * (1 - base_strength)          coverage = min(1, sum of region masks)
    region = mask * strength
    out    = (base * main + sum(region * region_out)) / (base + sum(region))

A region with its own negative prompt gets the same blend on the negative (CFG) rows; regions without one
use the main negative there. Without a negative pass (CFG 1) region negatives have no effect.

Separation: cross-attention alone tells each area what to be, but self-attention still mixes every image
token with every other, so with many regions the characters average out (hair colours, outfits, faces
swap or blend). In the middle of the steps, `separation` damps self-attention between tokens of
different regions (by up to 1000×), so each character's look forms from its own prompt. The first steps
run unmasked so the picture's layout forms as one image (separating from step 1 makes every region
compose its own little picture, e.g. a group of people per region), and the last steps run unmasked to
blend lighting and style. Regions whose box spans most of the picture count as background
and are never separated from the others (a background region behind characters stays shared).

Frames (consistent sets, reference images): tokens are ordered (t h w). A region with `frame` set is a
frame prompt: on its frame it takes the main prompt's share of the blend (its prompt already contains the
main prompt), and it is absent on the other frames. Spatial regions apply to every generated frame.
Frames past `frames` are reference pictures (venno/engine/reference.py): they see the main prompt only.
Joint self-attention averages the frames' differences away (three expressions come out as one in-between
face), so with frame prompts attention between different generated frames is damped by `frame_separation`
at every step (`frame_bias`); reference frames stay fully visible. Separating only some of the steps
doesn't work here: early steps fix the expression, and switching separation off late leaves blocky patches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class Region:
    plan: object  # prompts.PromptPlan of the region prompt (scheduling works here too)
    contexts: dict  # text → (1, L, D) encoded, for every text the plan uses
    mask: Optional[torch.Tensor]  # (H, W) float 0..1 at image resolution (None: frame prompts)
    strength: float = 1.0
    negative_plan: object = None  # PromptPlan of the region's own negative, None = the main negative
    negative_contexts: dict = field(default_factory=dict)
    frame: Optional[int] = None  # a frame prompt for this frame of a consistent set (0-based)

    def context(self, step: int, negative: bool = False):
        """The encoded prompt for a step: a (1, L, D) tensor, or an Anima 3.8B placeholder dict resolved per model call."""
        if negative:
            return self.negative_contexts[self.negative_plan.parts(step)[0][0]]
        return self.contexts[self.plan.parts(step)[0][0]]


@dataclass
class RegionalAttention:
    regions: list[Region]
    base_strength: float = 0.25
    separation: float = 0.0  # 0..1, how strongly regions are kept apart in self-attention
    separate_from: int = 1  # first step (1-based) with separation
    separate_until: int = 0  # last step (1-based) with separation
    frames: int = 1  # generated frames; tokens of later frames belong to reference pictures
    frame_separation: float = 0.0  # 0..1, how strongly frames with their own prompts are kept apart
    step: int = 1  # set by the denoiser before each model call
    _contexts: dict = field(default_factory=dict, repr=False)
    _weights: dict = field(default_factory=dict, repr=False)
    _bias: dict = field(default_factory=dict, repr=False)
    resolve_context: object = field(default=None, repr=False)  # set per model call for Anima 3.8B prompts

    MAX_DAMPING = -6.9  # log(1/1000): separation 1 damps attention between regions 1000×
    BACKGROUND_BOX = 0.6  # a region whose bounding box covers more of the picture than this is background

    def token_masks(self, h: int, w: int) -> torch.Tensor:
        """(R, h, w) region masks at the block's token resolution."""
        masks = []
        for r in self.regions:
            if r.mask is None:  # frame prompts cover their whole frame
                masks.append(torch.ones(h, w))
                continue
            m = F.interpolate(r.mask[None, None].float(), size=(h, w), mode="area")
            m = F.avg_pool2d(F.pad(m, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)  # soften by one token
            masks.append(m[0, 0])
        masks = torch.stack(masks)
        background = self.backgrounds()
        subjects = [k for k, r in enumerate(self.regions) if r.mask is not None and k not in background]
        if background and subjects:  # a background never shares a subject's area, whatever the overlap mode:
            masks[background] = masks[background] * (1.0 - masks[subjects].sum(0).clamp(max=1.0))  # blended, it halves the subject's prompt
        return masks

    def backgrounds(self) -> list[int]:
        """Spatial regions whose bounding box spans most of the picture (BACKGROUND_BOX): scenery, not subjects."""
        if not hasattr(self, "_background_list"):
            out = []
            for k, r in enumerate(self.regions):
                if r.frame is not None or r.mask is None:
                    continue
                ys, xs = torch.nonzero(r.mask > 0.5, as_tuple=True)
                box = 0.0 if not len(ys) else float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) / r.mask.numel()
                if box > self.BACKGROUND_BOX:
                    out.append(k)
            self._background_list = out
        return self._background_list

    def self_bias(self, thw: tuple[int, int, int], like: torch.Tensor):
        """Additive self-attention bias (1, 1, N, N) that keeps regions apart, or None when not separating now."""
        if self.separation <= 0 or not self.separate_from <= self.step <= self.separate_until or thw[0] != 1:
            return None
        key = (thw, like.device, like.dtype)
        if key not in self._bias:
            _, h, w = thw
            masks = self.token_masks(h, w)
            background = self.backgrounds()
            keep = [k for k, r in enumerate(self.regions) if r.frame is None and k not in background]
            if len(keep) < 2:
                self._bias[key] = None
            else:
                m = masks[keep].reshape(len(keep), h * w).to(like.device, torch.float32)  # soft membership per token
                any_ = m.sum(0)
                cross = (any_[:, None] * any_[None, :] - m.T @ m).clamp_(0, 1)  # sum over a≠b of m_a(i)·m_b(j)
                bias = cross * (self.MAX_DAMPING * self.separation)  # even steps: 0.5 → ~30×, 1 → 1000×
                self._bias[key] = bias.to(like.dtype)[None, None]
        return self._bias[key]

    def frame_bias(self, thw: tuple[int, int, int], like: torch.Tensor):
        """Self-attention key bias per query frame, (t, N), or None. It depends only on the key's frame,
        so a frame's queries can be run with a (1, 1, 1, N) mask instead of an (N, N) one."""
        t, h, w = thw
        if self.frame_separation <= 0 or t < 2 or self.frames < 2:
            return None
        if not any(r.frame is not None for r in self.regions):
            return None
        key = ("frames", thw, like.device, like.dtype)
        if key not in self._bias:
            frame = torch.arange(t)
            generated = frame < self.frames
            other = generated[:, None] & generated[None, :] & (frame[:, None] != frame[None, :])  # query frame × key frame
            bias = other.float() * (self.MAX_DAMPING * self.frame_separation)
            self._bias[key] = bias.repeat_interleave(h * w, dim=1).to(device=like.device, dtype=like.dtype)
        return self._bias[key]

    def contexts(self, like: torch.Tensor, regions: list[Region] | None = None, negative: bool = False) -> torch.Tensor:
        """(R, L, D) region contexts, zero-padded to a common length (Anima pads prompts with zeros)."""
        current = [r.context(self.step, negative) for r in (self.regions if regions is None else regions)]
        current = [self.resolve_context(c) if isinstance(c, dict) else c for c in current]
        key = (like.device, like.dtype, like.shape[1], tuple(id(c) for c in current))
        if key not in self._contexts:
            if len(self._contexts) > 8:  # 3.8B contexts are new tensors every step: keep the cache small
                self._contexts.clear()
            length = max(max(c.shape[1] for c in current), like.shape[1])
            ctx = [F.pad(c[:1], (0, 0, 0, length - c.shape[1])) for c in current]
            # the entry holds its source tensors, so their ids can't be reused by new tensors while it exists
            self._contexts[key] = (current, torch.cat(ctx).to(device=like.device, dtype=like.dtype))
        return self._contexts[key][1]

    def weights(self, thw: tuple[int, int, int], like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token blend weights for the block's token grid: base (N, 1) and regions (R, N, 1)."""
        key = (thw, like.device, like.dtype)
        if key not in self._weights:
            t, h, w = thw
            R = len(self.regions)
            spatial = [k for k, r in enumerate(self.regions) if r.frame is None]
            masks = torch.zeros(R, t, h, w)
            if spatial:
                masks[spatial, : self.frames] = self.token_masks(h, w)[spatial][:, None]
            strength = torch.tensor([r.strength for r in self.regions])[:, None, None, None]
            coverage = masks.sum(0).clamp(max=1.0)
            base = 1.0 - coverage * (1.0 - self.base_strength)
            region = masks * strength
            total = (base + region.sum(0)).clamp_min(1e-6)
            base, region = base / total, region / total  # (t, h, w), (R, t, h, w)
            for k, r in enumerate(self.regions):
                if r.frame is not None and r.frame < min(t, self.frames):  # the frame prompt replaces the main prompt there
                    region[k, r.frame] = base[r.frame]
                    base[r.frame] = 0.0
            base = base.reshape(-1, 1)
            region = region.reshape(R, -1, 1)
            self._weights[key] = (base.to(device=like.device, dtype=like.dtype), region.to(device=like.device, dtype=like.dtype))
        return self._weights[key]

    def __call__(self, attn, tokens: torch.Tensor, context: torch.Tensor, thw, transformer_options: dict) -> torch.Tensor:
        out = attn(tokens, context, transformer_options=transformer_options)
        w_base, w_region = self.weights(thw, out)  # (N, 1), (R, N, 1)
        out = self._blend(attn, tokens, context, out, transformer_options.get("cond_indices") or [], self.regions, False, w_base, w_region, transformer_options)
        negative = [k for k, r in enumerate(self.regions) if r.negative_plan is not None]
        if negative:  # the main negative's output stands in for regions without their own
            out = self._blend(attn, tokens, context, out, transformer_options.get("uncond_indices") or [], [self.regions[k] for k in negative], True, w_base, w_region, transformer_options, negative)
        return out

    def _blend(self, attn, tokens, context, out, rows, regions, negative, w_base, w_region, transformer_options, only=None):
        """Replace `rows` of `out` by the weighted mix of the main output and each region's attention output."""
        if not rows:
            return out
        rows_t = torch.tensor(rows, device=tokens.device)
        x = tokens.index_select(0, rows_t)  # (n, N, D)
        main = out.index_select(0, rows_t)
        n, R = x.shape[0], len(regions)
        ctx = self.contexts(context, regions, negative)  # (R, L, D)
        region_out = attn(
            x.repeat(R, 1, 1),  # region-major: [r0 rows…, r1 rows…]
            ctx.repeat_interleave(n, dim=0),
            transformer_options=transformer_options,
        ).reshape(R, n, *out.shape[1:])
        if only is None:
            blended = main * w_base + (region_out * w_region[:, None]).sum(0)
        else:
            idx = torch.tensor(only, device=w_region.device)
            w_own = w_region.index_select(0, idx)  # regions with their own negative
            w_main = w_base + w_region.sum(0) - w_own.sum(0)  # everything else keeps the main negative
            blended = main * w_main + (region_out * w_own[:, None]).sum(0)
        return out.index_copy(0, rows_t, blended)
