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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class Region:
    plan: object  # prompts.PromptPlan of the region prompt (scheduling works here too)
    contexts: dict  # text → (1, L, D) encoded, for every text the plan uses
    mask: torch.Tensor  # (H, W) float 0..1 at image resolution
    strength: float = 1.0
    negative_plan: object = None  # PromptPlan of the region's own negative, None = the main negative
    negative_contexts: dict = field(default_factory=dict)

    def context(self, step: int, negative: bool = False) -> torch.Tensor:
        if negative:
            return self.negative_contexts[self.negative_plan.parts(step)[0][0]]
        return self.contexts[self.plan.parts(step)[0][0]]


@dataclass
class RegionalAttention:
    regions: list[Region]
    base_strength: float = 0.25
    step: int = 1  # set by the denoiser before each model call
    _contexts: dict = field(default_factory=dict, repr=False)
    _weights: dict = field(default_factory=dict, repr=False)

    def contexts(self, like: torch.Tensor, regions: list[Region] | None = None, negative: bool = False) -> torch.Tensor:
        """(R, L, D) region contexts, zero-padded to a common length (Anima pads prompts with zeros)."""
        current = [r.context(self.step, negative) for r in (self.regions if regions is None else regions)]
        key = (like.device, like.dtype, tuple(id(c) for c in current))
        if key not in self._contexts:
            length = max(max(c.shape[1] for c in current), like.shape[1])
            ctx = [F.pad(c[:1], (0, 0, 0, length - c.shape[1])) for c in current]
            self._contexts[key] = torch.cat(ctx).to(device=like.device, dtype=like.dtype)
        return self._contexts[key]

    def weights(self, thw: tuple[int, int, int], like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token blend weights for the block's token grid: base (N, 1) and regions (R, N, 1)."""
        key = (thw, like.device, like.dtype)
        if key not in self._weights:
            t, h, w = thw
            masks = []
            for r in self.regions:
                m = F.interpolate(r.mask[None, None].float(), size=(h, w), mode="area")
                m = F.avg_pool2d(F.pad(m, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)  # soften by one token
                masks.append(m[0, 0])
            masks = torch.stack(masks)  # (R, h, w)
            coverage = masks.sum(0).clamp(max=1.0)
            base = 1.0 - coverage * (1.0 - self.base_strength)
            region = masks * torch.tensor([r.strength for r in self.regions])[:, None, None]
            total = (base + region.sum(0)).clamp_min(1e-6)
            base, region = base / total, region / total
            # token order is (t h w); every frame shares the spatial mask
            base = base.reshape(1, h * w).repeat(t, 1).reshape(-1, 1)
            region = region.reshape(len(self.regions), 1, h * w).repeat(1, t, 1).reshape(len(self.regions), -1, 1)
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
