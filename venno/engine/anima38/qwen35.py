"""Qwen3.5 4B text backbone, as used by Anima 3.8B (only its hidden states are needed).

A hybrid stack of 32 blocks: Gated DeltaNet linear attention in most blocks and gated full self-attention
in every fourth, each followed by a SwiGLU MLP (the last block has none). Parameter names follow the
`qwen35_4b.safetensors` checkpoint. Ported from GumGum10/forge-anima-3.8B (MIT, see LICENSE in this folder).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

HIDDEN = 2560
LAYERS = 32
FULL_ATTENTION = {3, 7, 11, 15, 19, 23, 27, 31}
VOCAB = 248320
ROPE_THETA = 10_000_000.0
ROTARY_DIM = 64


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    """Plain RMSNorm (weight is the scale)."""

    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class OffsetRMSNorm(RMSNorm):
    """Qwen3.5's RMSNorm: the stored weight is an offset from one."""

    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__(dim, eps, device, dtype)
        nn.init.zeros_(self.weight)

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).to(x.dtype)


class GatedDeltaNet(nn.Module):
    """Linear attention with the gated delta rule (16 key heads, 32 value heads, 128 dims each)."""

    def __init__(self, ops, device=None, dtype=None):
        super().__init__()
        self.k_heads, self.v_heads, self.head_dim = 16, 32, 128
        self.key_dim = self.k_heads * self.head_dim
        self.value_dim = self.v_heads * self.head_dim
        conv_dim = 2 * self.key_dim + self.value_dim
        kw = dict(bias=False, device=device, dtype=dtype)
        self.in_proj_qkv = ops.Linear(HIDDEN, conv_dim, **kw)
        self.in_proj_z = ops.Linear(HIDDEN, self.value_dim, **kw)
        self.in_proj_a = ops.Linear(HIDDEN, self.v_heads, **kw)
        self.in_proj_b = ops.Linear(HIDDEN, self.v_heads, **kw)
        self.conv1d = ops.Conv1d(conv_dim, conv_dim, 4, groups=conv_dim, padding=3, **kw)
        self.out_proj = ops.Linear(self.value_dim, HIDDEN, **kw)
        self.norm = RMSNorm(self.head_dim, device=device, dtype=dtype)
        self.A_log = nn.Parameter(torch.zeros(self.v_heads, device=device, dtype=dtype))
        self.dt_bias = nn.Parameter(torch.zeros(self.v_heads, device=device, dtype=dtype))

    @staticmethod
    def _scan(q, k, v, g, beta):
        """Sequential delta-rule recurrence over the tokens (prompts are short), in float32."""
        dtype = q.dtype
        q = _l2norm(q).transpose(1, 2).float()
        k = _l2norm(k).transpose(1, 2).float()
        v = v.transpose(1, 2).float()
        g = g.transpose(1, 2).float()
        beta = beta.transpose(1, 2).float()
        b, h, length, dk = k.shape
        q = q / dk**0.5
        state = torch.zeros(b, h, dk, v.shape[-1], device=q.device, dtype=torch.float32)
        out = []
        for t in range(length):
            state = state * g[:, :, t].exp()[..., None, None]
            k_t = k[:, :, t]
            delta = (v[:, :, t] - (state * k_t[..., None]).sum(-2)) * beta[:, :, t][..., None]
            state = state + k_t[..., None] * delta[..., None, :]
            out.append((state * q[:, :, t][..., None]).sum(-2))
        return torch.stack(out, dim=2).transpose(1, 2).to(dtype)

    def forward(self, x, mask=None):
        if mask is not None and mask.shape[0] > 1 and mask.shape[1] > 1:
            x = x * mask[:, :, None].to(x.dtype)
        b, length, _ = x.shape
        z = self.in_proj_z(x).reshape(b, length, self.v_heads, self.head_dim)
        qkv = F.silu(self.conv1d(self.in_proj_qkv(x).transpose(1, 2))[:, :, :length]).transpose(1, 2)
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(b, length, self.k_heads, self.head_dim)
        k = k.reshape(b, length, self.k_heads, self.head_dim)
        v = v.reshape(b, length, self.v_heads, self.head_dim)
        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp().to(x.device) * F.softplus(self.in_proj_a(x).float() + self.dt_bias.to(x.device).float())
        rep = self.v_heads // self.k_heads
        q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        y = self.norm(self._scan(q, k, v, g, beta).reshape(-1, self.head_dim))
        y = y * F.silu(z.reshape(-1, self.head_dim).float()).to(y.dtype)
        return self.out_proj(y.reshape(b, length, self.value_dim))


def _rope(x, cos, sin):
    d = cos.shape[-1]
    rot, rest = x[..., :d], x[..., d:]
    half = d // 2
    rotated = torch.cat((-rot[..., half:], rot[..., :half]), dim=-1)
    return torch.cat((rot * cos + rotated * sin, rest), dim=-1)


def _rope_tables(length: int, device, dtype):
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, ROTARY_DIM, 2, device=device, dtype=torch.float32) / ROTARY_DIM))
    freqs = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv)
    cos = freqs.cos()[None, None].repeat(1, 1, 1, 2).to(dtype)
    sin = freqs.sin()[None, None].repeat(1, 1, 1, 2).to(dtype)
    return cos, sin


class GatedAttention(nn.Module):
    """Full causal self-attention: 16 query heads (with an output gate packed into q_proj), 4 KV heads, 256 dims."""

    def __init__(self, ops, device=None, dtype=None):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = 16, 4, 256
        inner = self.heads * self.head_dim
        kw = dict(bias=False, device=device, dtype=dtype)
        self.q_proj = ops.Linear(HIDDEN, 2 * inner, **kw)
        self.k_proj = ops.Linear(HIDDEN, self.kv_heads * self.head_dim, **kw)
        self.v_proj = ops.Linear(HIDDEN, self.kv_heads * self.head_dim, **kw)
        self.o_proj = ops.Linear(inner, HIDDEN, **kw)
        self.q_norm = OffsetRMSNorm(self.head_dim, device=device, dtype=dtype)
        self.k_norm = OffsetRMSNorm(self.head_dim, device=device, dtype=dtype)

    def forward(self, x, bias=None, rope=None):
        b, length, _ = x.shape
        q, gate = self.q_proj(x).view(b, length, self.heads, 2 * self.head_dim).chunk(2, dim=-1)
        gate = gate.reshape(b, length, -1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, length, self.kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, length, self.kv_heads, self.head_dim).transpose(1, 2)
        if rope is not None:
            q, k = _rope(q, *rope), _rope(k, *rope)
        rep = self.heads // self.kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None if bias is None else bias.to(q.dtype), is_causal=bias is None)
        return self.o_proj(out.transpose(1, 2).reshape(b, length, -1) * torch.sigmoid(gate))


class MLP(nn.Module):
    def __init__(self, ops, device=None, dtype=None):
        super().__init__()
        kw = dict(bias=False, device=device, dtype=dtype)
        self.gate_proj = ops.Linear(HIDDEN, 9216, **kw)
        self.up_proj = ops.Linear(HIDDEN, 9216, **kw)
        self.down_proj = ops.Linear(9216, HIDDEN, **kw)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, index: int, ops, device=None, dtype=None):
        super().__init__()
        self.linear = index not in FULL_ATTENTION
        self.input_layernorm = OffsetRMSNorm(HIDDEN, device=device, dtype=dtype)
        if self.linear:
            self.linear_attn = GatedDeltaNet(ops, device, dtype)
        else:
            self.self_attn = GatedAttention(ops, device, dtype)
        self.has_mlp = index != LAYERS - 1
        if self.has_mlp:
            self.post_attention_layernorm = OffsetRMSNorm(HIDDEN, device=device, dtype=dtype)
            self.mlp = MLP(ops, device, dtype)

    def forward(self, x, bias=None, rope=None, mask=None):
        h = self.input_layernorm(x)
        x = x + (self.linear_attn(h, mask) if self.linear else self.self_attn(h, bias, rope))
        if self.has_mlp:
            x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen35(nn.Module):
    """Returns the hidden states after the requested blocks (0-based)."""

    def __init__(self, ops=nn, device=None, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.embed_tokens = ops.Embedding(VOCAB, HIDDEN, device=device, dtype=dtype)
        self.layers = nn.ModuleList(Block(i, ops, device, dtype) for i in range(LAYERS))
        # the checkpoint's output head (unused here: Anima reads intermediate layers), kept so the weights load strictly
        self.norm = nn.Sequential(
            ops.Linear(HIDDEN, 1024, bias=True, device=device, dtype=dtype),
            OffsetRMSNorm(1024, device=device, dtype=dtype),
            nn.SiLU(),
            ops.Linear(1024, 1024, bias=True, device=device, dtype=dtype),
        )

    def forward(self, input_ids: torch.Tensor, layers: tuple[int, ...]) -> list[torch.Tensor]:
        x = self.embed_tokens(input_ids)
        rope = _rope_tables(x.shape[1], x.device, x.dtype)
        wanted, out = set(layers), {}
        for i, block in enumerate(self.layers):
            x = block(x, None, rope, None)  # one unpadded prompt: causal attention, no padding mask
            if i in wanted:
                out[i] = x
            if len(out) == len(wanted):
                break
        return [out[i] for i in layers]
