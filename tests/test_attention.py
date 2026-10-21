"""Attention back ends agree on layout (backend/attention.py)."""

import pytest
import torch

from backend import attention as A
from backend import memory_management as M

B, H, S, D = 2, 4, 16, 8


def qkv(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, H, S, D, generator=g) for _ in range(3)]


@pytest.mark.parametrize("skip_output_reshape", [False, True])
@pytest.mark.parametrize("skip_reshape", [False, True])
def test_xformers_fallback_matches_pytorch(skip_reshape, skip_output_reshape):
    """Without a working xformers the call falls back to PyTorch attention, in the caller's layout."""
    q, k, v = qkv()
    if not skip_reshape:  # (B, S, H*D)
        q, k, v = (t.transpose(1, 2).reshape(B, S, H * D) for t in (q, k, v))
    want = A.attention_pytorch(q, k, v, H, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape)
    got = A.attention_xformers(q, k, v, H, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape)
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-5)


def test_basic_matches_pytorch():
    q, k, v = qkv(1)
    mask = torch.randn(1, 1, S, S) * 0.5
    assert torch.allclose(A.attention_basic(q, k, v, H, mask=mask, skip_reshape=True), A.attention_pytorch(q, k, v, H, mask=mask, skip_reshape=True), atol=1e-5)


def test_attn_precision_upcast(monkeypatch):
    monkeypatch.setattr(M, "force_upcast_attention_dtype", lambda: {torch.float16: torch.float32})
    assert A.get_attn_precision(None, torch.float16) is torch.float32
    assert A.get_attn_precision(None, torch.bfloat16) is None


def test_cpu_uses_pytorch_attention():
    assert M.cpu_mode() and A.attention_function is A.attention_pytorch
