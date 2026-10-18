"""Anima 3.8B text connectors: they replace Anima's native LLM adapter (Qwen3 0.6B features + T5 ids → context).

Both wrap the native adapter's own blocks and add cross-attention to four hidden layers of Qwen3.5 4B:
- `CrossAdapter` (v1, separate `…expanded_adapter.safetensors`): after every native block, attend to a learned
  mix of the four Qwen3.5 layers. Static: computed once per prompt.
- `SemanticConnectorV2` (v1.1 release, bundled inside the checkpoint): the v1 path ("quality anchor") plus a
  timestep-aware resampler that condenses the Qwen3.5 layers into 64 tokens the native blocks also attend to.
  It depends on the diffusion timestep, so it runs at every denoising step.

Parameter names follow the released weights. Ported from GumGum10/forge-anima-3.8B (MIT, see LICENSE).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from backend.nn.anima import Attention

LAYERS = (7, 15, 23, 31)  # Qwen3.5 blocks whose outputs condition the connector
QWEN_DIM = 2560


def _bool_mask(mask):
    if mask is None:
        return None
    mask = mask.to(torch.bool)
    return mask[:, None, None] if mask.ndim == 2 else mask


class CrossAdapter(nn.Module):
    """v1: native adapter blocks, each followed by attention to a per-block mix of the Qwen3.5 layers."""

    def __init__(self, native: nn.Module, qwen_dim: int = QWEN_DIM, layers=LAYERS):
        super().__init__()
        object.__setattr__(self, "native", native)  # shared with the text encoder, not part of these weights
        n, dim = len(native.blocks), native.embed.weight.shape[1]
        heads = native.blocks[0].self_attn.n_heads
        self.query_norms = nn.ModuleList(nn.RMSNorm(dim, eps=1e-6) for _ in range(n))
        self.source_norms = nn.ModuleList(nn.RMSNorm(qwen_dim, eps=1e-6) for _ in range(n))
        self.semantic_attentions = nn.ModuleList(Attention(dim, qwen_dim, heads, dim // heads) for _ in range(n))
        self.layer_mix_logits = nn.Parameter(torch.zeros(n, len(layers)))

    def mixed(self, semantic: list[torch.Tensor], mix: torch.Tensor, block: int) -> torch.Tensor:
        return sum(state * mix[block, i] for i, state in enumerate(semantic))

    def anchor(self, x, block: int, semantic, mix, semantic_mask, query_rope, semantic_rope):
        source = self.source_norms[block](self.mixed(semantic, mix, block))
        return x + self.semantic_attentions[block](self.query_norms[block](x), mask=semantic_mask, context=source, position_embeddings=query_rope, position_embeddings_context=semantic_rope)

    def forward(self, source, target_ids, semantic, semantic_mask=None):
        native = self.native
        x = native.in_proj(native.embed(target_ids))
        rope = lambda length: native.rotary_emb(x, torch.arange(length, device=x.device)[None])
        query_rope, source_rope, semantic_rope = rope(x.shape[1]), rope(source.shape[1]), rope(semantic[0].shape[1])
        mix = self.layer_mix_logits.float().softmax(-1).to(x.device, x.dtype)
        semantic_mask = _bool_mask(semantic_mask)
        for i, block in enumerate(native.blocks):
            x = block(x, source, position_embeddings=query_rope, position_embeddings_context=source_rope)
            x = self.anchor(x, i, semantic, mix, semantic_mask, query_rope, semantic_rope)
        return native.norm(native.out_proj(x))


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Sinusoidal embedding of diffusion timesteps in [0, 1] (scaled by 1000)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1))
    angles = t.float().reshape(-1, 1) * 1000.0 * freqs[None]
    emb = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return F.pad(emb, (0, dim % 2)) if dim % 2 else emb


class _MHA(nn.Module):
    def __init__(self, dim: int, context_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(context_dim, dim, bias=False)
        self.v_proj = nn.Linear(context_dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, q, context, context_mask=None):
        b, nq, dim = q.shape
        split = lambda t: t.reshape(b, t.shape[1], self.heads, dim // self.heads).transpose(1, 2)
        mask = None if context_mask is None else context_mask.to(torch.bool).reshape(b, 1, 1, -1)
        out = F.scaled_dot_product_attention(split(self.q_proj(q)), split(self.k_proj(context)), split(self.v_proj(context)), attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(b, nq, dim))


def _modulated(x, scale, shift):
    return F.layer_norm(x, x.shape[-1:]) * (1.0 + scale[:, None]) + shift[:, None]


class _ResamplerBlock(nn.Module):
    def __init__(self, dim: int, qwen_dim: int, heads: int, mlp_dim: int):
        super().__init__()
        self.source_norm = nn.LayerNorm(qwen_dim)
        self.cross_attention = _MHA(dim, qwen_dim, heads)
        self.self_attention = _MHA(dim, dim, heads)
        self.mlp_in = nn.Linear(dim, 2 * mlp_dim, bias=False)
        self.mlp_out = nn.Linear(mlp_dim, dim, bias=False)
        self.time_modulation = nn.Linear(dim, 6 * dim, bias=True)

    def forward(self, q, qwen, t_emb, qwen_mask):
        c_scale, c_shift, s_scale, s_shift, m_scale, m_shift = self.time_modulation(F.silu(t_emb)).chunk(6, dim=-1)
        q = q + self.cross_attention(_modulated(q, c_scale, c_shift), self.source_norm(qwen), qwen_mask)
        h = _modulated(q, s_scale, s_shift)
        q = q + self.self_attention(h, h)
        gate, value = self.mlp_in(_modulated(q, m_scale, m_shift)).chunk(2, dim=-1)
        return q + self.mlp_out(F.silu(gate) * value)


class Resampler(nn.Module):
    """Condenses the four Qwen3.5 layers into a few query tokens, conditioned on the timestep."""

    def __init__(self, qwen_dim: int, out_dim: int, layers: int, queries: int, blocks: int, dim: int, heads: int, mlp_dim: int):
        super().__init__()
        self.dim = dim
        self.query_tokens = nn.Parameter(torch.zeros(1, queries, dim))
        self.layer_embeddings = nn.Parameter(torch.zeros(layers, 1, qwen_dim))
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(_ResamplerBlock(dim, qwen_dim, heads, mlp_dim) for _ in range(blocks))
        self.output_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, out_dim, bias=False)

    def forward(self, semantic: list[torch.Tensor], t: torch.Tensor, mask=None):
        qwen = torch.cat([h + self.layer_embeddings[i].to(h)[None] for i, h in enumerate(semantic)], dim=1)
        qwen_mask = None if mask is None else torch.cat([mask] * len(semantic), dim=1)
        t_emb = self.time_mlp(timestep_embedding(t, self.dim).to(qwen.dtype))
        q = self.query_tokens.to(qwen).expand(qwen.shape[0], -1, -1)
        for block in self.blocks:
            q = block(q, qwen, t_emb, qwen_mask)
        return self.output_projection(self.output_norm(q))


class SemanticConnectorV2(nn.Module):
    def __init__(self, native: nn.Module, queries: int = 64, blocks: int = 6, dim: int = 2048, heads: int = 16, mlp_dim: int = 5632, qwen_dim: int = QWEN_DIM, layers=LAYERS):
        super().__init__()
        n, model_dim = len(native.blocks), native.embed.weight.shape[1]
        native_heads = native.blocks[0].self_attn.n_heads
        self.quality_anchor = CrossAdapter(native, qwen_dim, layers)
        self.semantic_resampler = Resampler(qwen_dim, model_dim, len(layers), queries, blocks, dim, heads, mlp_dim)
        self.v2_query_norms = nn.ModuleList(nn.RMSNorm(model_dim, eps=1e-6) for _ in range(n))
        self.v2_semantic_norms = nn.ModuleList(nn.RMSNorm(model_dim, eps=1e-6) for _ in range(n))
        self.v2_attentions = nn.ModuleList(Attention(model_dim, model_dim, native_heads, model_dim // native_heads) for _ in range(n))

    def forward(self, source, target_ids, semantic, timesteps, semantic_mask=None):
        anchor = self.quality_anchor
        native = anchor.native
        bank = self.semantic_resampler(semantic, timesteps, semantic_mask)
        x = native.in_proj(native.embed(target_ids))
        rope = lambda length: native.rotary_emb(x, torch.arange(length, device=x.device)[None])
        query_rope, source_rope, semantic_rope, bank_rope = rope(x.shape[1]), rope(source.shape[1]), rope(semantic[0].shape[1]), rope(bank.shape[1])
        mix = anchor.layer_mix_logits.float().softmax(-1).to(x.device, x.dtype)
        mask = _bool_mask(semantic_mask)
        for i, block in enumerate(native.blocks):
            x = block(x, source, position_embeddings=query_rope, position_embeddings_context=source_rope)
            x = anchor.anchor(x, i, semantic, mix, mask, query_rope, semantic_rope)
            x = x + self.v2_attentions[i](self.v2_query_norms[i](x), context=self.v2_semantic_norms[i](bank), position_embeddings=query_rope, position_embeddings_context=bank_rope)
        return native.norm(native.out_proj(x))


class SamplingUnit(nn.Module):
    """What has to be on the GPU while sampling: the connector and the native adapter it wraps."""

    def __init__(self, native: nn.Module, connector: nn.Module):
        super().__init__()
        self.native = native
        self.connector = connector
