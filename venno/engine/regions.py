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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class Region:
    context: torch.Tensor  # (1, L, D) encoded region prompt
    mask: torch.Tensor  # (H, W) float 0..1 at image resolution
    strength: float = 1.0


@dataclass
class RegionalAttention:
    regions: list[Region]
    base_strength: float = 0.25
    _contexts: dict = field(default_factory=dict, repr=False)
    _weights: dict = field(default_factory=dict, repr=False)

    def contexts(self, like: torch.Tensor) -> torch.Tensor:
        """(R, L, D) region contexts, zero-padded to a common length (Anima pads prompts with zeros)."""
        key = (like.device, like.dtype)
        if key not in self._contexts:
            length = max(max(r.context.shape[1] for r in self.regions), like.shape[1])
            ctx = [F.pad(r.context[:1], (0, 0, 0, length - r.context.shape[1])) for r in self.regions]
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
        rows = transformer_options.get("cond_indices") or []
        if not rows:
            return out

        rows_t = torch.tensor(rows, device=tokens.device)
        x = tokens.index_select(0, rows_t)  # (n, N, D) positive-prompt rows only; the negative stays global
        n, R = x.shape[0], len(self.regions)
        ctx = self.contexts(context)  # (R, L, D)
        region_out = attn(
            x.repeat(R, 1, 1),  # region-major: [r0 rows…, r1 rows…]
            ctx.repeat_interleave(n, dim=0),
            transformer_options=transformer_options,
        ).reshape(R, n, *out.shape[1:])

        w_base, w_region = self.weights(thw, out)  # (N, 1), (R, N, 1)
        blended = out.index_select(0, rows_t) * w_base + (region_out * w_region[:, None]).sum(0)
        return out.index_copy(0, rows_t, blended)
