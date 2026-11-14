"""GGUF files: quantized MOSS-Image transformers and Qwen3.5 text encoders (the small downloads).

A GGUF file (the format of ggml / llama.cpp, MIT) stores each weight in blocks of a few bits per value.
This module reads the file and turns blocks back into numbers in torch, on the graphics card and one layer at
a time: the weights stay quantized in memory (3 to 6 GB for the transformer instead of 10) and a layer's
weight exists in full precision only while that layer runs. LoRAs are added on top as usual.

Written for Venno from the block layouts in ggml's ``ggml-quants.c`` (dequantize_row_*); the unit tests
compare every type with the reference implementation in the ``gguf`` Python package. Types read: F32, F16,
BF16 (stored as they are) and Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K, Q5_1, Q5_0, Q4_1, Q4_0. The "I-quants"
(IQ…) and other codebook types are not read: pick a Q file.
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

MAGIC = 0x46554747  # "GGUF"
KINDS = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
         14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
         24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16"}
PLAIN = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "F64": torch.float64,
         "I8": torch.int8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64}
BLOCKS = {  # values per block, bytes per block
    "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24), "Q8_0": (32, 34),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210),
}
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


@dataclass
class Tensor:
    name: str
    kind: str  # "Q4_K", "BF16", …
    shape: tuple[int, ...]  # torch order (GGUF lists dimensions the other way round)
    offset: int  # in the file
    nbytes: int

    @property
    def quantized(self) -> bool:
        return self.kind not in PLAIN


class File:
    """A GGUF file's metadata and tensors; tensor data is read on request."""

    def __init__(self, path: str, arrays: bool = False):
        self.path = path
        self.metadata: dict[str, object] = {}
        self.tensors: dict[str, Tensor] = {}
        with open(path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                self._parse(m, arrays)

    def _parse(self, m, arrays: bool):
        pos = 0

        def read(fmt):
            nonlocal pos
            (v,) = struct.unpack_from(fmt, m, pos)
            pos += struct.calcsize(fmt)
            return v

        def string():
            nonlocal pos
            n = read("<Q")
            s = bytes(m[pos : pos + n]).decode("utf-8", "replace")
            pos += n
            return s

        def value(kind):
            nonlocal pos
            if kind in _SCALARS:
                return read(_SCALARS[kind])
            if kind == _STRING:
                return string()
            if kind == _ARRAY:
                item, n = read("<I"), read("<Q")
                if item in _SCALARS and not arrays:  # e.g. token types: skipped
                    pos += struct.calcsize(_SCALARS[item]) * n
                    return None
                items = [value(item) for _ in range(n)]
                return items if arrays else None
            raise ValueError(f"{os.path.basename(self.path)}: unknown GGUF metadata type {kind}")

        if len(m) < 24 or read("<I") != MAGIC:
            raise ValueError(f"{os.path.basename(self.path)} is not a GGUF file")
        version = read("<I")
        if version not in (2, 3):
            raise ValueError(f"{os.path.basename(self.path)}: GGUF version {version} is not supported")
        tensor_count, kv_count = read("<Q"), read("<Q")
        for _ in range(kv_count):
            key = string()
            v = value(read("<I"))
            if v is not None:
                self.metadata[key] = v
        infos = []
        for _ in range(tensor_count):
            name = string()
            dims = [read("<Q") for _ in range(read("<I"))]
            kind, offset = read("<I"), read("<Q")
            infos.append((name, tuple(reversed(dims)), KINDS.get(kind, f"type {kind}"), offset))
        align = int(self.metadata.get("general.alignment", 32) or 32)
        start = (pos + align - 1) // align * align
        for name, shape, kind, offset in infos:
            count = int(np.prod(shape)) if shape else 1
            if kind in PLAIN:
                nbytes = count * torch.empty(0, dtype=PLAIN[kind]).element_size()
            elif kind in BLOCKS:
                values, size = BLOCKS[kind]
                nbytes = count // values * size
            else:
                nbytes = 0  # a type this module doesn't read; reported when the tensor is asked for
            self.tensors[name] = Tensor(name, kind, shape, start + offset, nbytes)

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", ""))

    def kinds(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.tensors.values():
            out[t.kind] = out.get(t.kind, 0) + 1
        return out

    def raw(self, name: str) -> torch.Tensor:
        """A tensor's bytes as they are in the file (uint8, in memory)."""
        t = self.tensors[name]
        check(t.kind, f"{os.path.basename(self.path)}: {name}")
        with open(self.path, "rb") as f:
            f.seek(t.offset)
            data = bytearray(t.nbytes)
            if f.readinto(data) != t.nbytes:
                raise ValueError(f"{os.path.basename(self.path)} is cut short (an unfinished download?)")
        return torch.frombuffer(data, dtype=torch.uint8)

    def plain(self, name: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """A tensor in full precision (quantized ones are dequantized here, on the CPU)."""
        t = self.tensors[name]
        return dequantize(self.raw(name), t.kind, t.shape, dtype or (PLAIN[t.kind] if t.kind in PLAIN else torch.float32))


def check(kind: str, what: str = "this file"):
    if kind not in PLAIN and kind not in BLOCKS:
        raise ValueError(f"{what} is stored as {kind}, which Venno MOSS can't read. Use a Q4_K, Q5_K, Q6_K or Q8_0 file.")


# ---------------------------------------------------------------- dequantizing


def _f16(b: torch.Tensor) -> torch.Tensor:
    """(n, 2) bytes → (n, 1) float32."""
    return b.contiguous().view(torch.float16).to(torch.float32)


def _nibbles(qs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return qs & 0xF, qs >> 4


def _q4_0(b):
    d = _f16(b[:, :2])
    lo, hi = _nibbles(b[:, 2:18])
    return d * (torch.cat([lo, hi], 1).to(torch.float32) - 8)


def _q4_1(b):
    d, m = _f16(b[:, :2]), _f16(b[:, 2:4])
    lo, hi = _nibbles(b[:, 4:20])
    return d * torch.cat([lo, hi], 1).to(torch.float32) + m


def _fifth_bit(qh: torch.Tensor) -> torch.Tensor:
    """(n, 4) bytes (a little-endian uint32) → (n, 32) with 16 where the value's bit is set."""
    bits = (qh.unsqueeze(-1) >> torch.arange(8, device=qh.device, dtype=torch.uint8)) & 1  # (n, 4, 8)
    return bits.reshape(qh.shape[0], 32) << 4


def _q5_0(b):
    d = _f16(b[:, :2])
    lo, hi = _nibbles(b[:, 6:22])
    q = torch.cat([lo, hi], 1) | _fifth_bit(b[:, 2:6])
    return d * (q.to(torch.float32) - 16)


def _q5_1(b):
    d, m = _f16(b[:, :2]), _f16(b[:, 2:4])
    lo, hi = _nibbles(b[:, 8:24])
    q = torch.cat([lo, hi], 1) | _fifth_bit(b[:, 4:8])
    return d * q.to(torch.float32) + m


def _q8_0(b):
    return _f16(b[:, :2]) * b[:, 2:].contiguous().view(torch.int8).to(torch.float32)


def _shifts(count: int, device) -> torch.Tensor:
    return (2 * torch.arange(count, device=device, dtype=torch.uint8)).view(1, 1, count, 1)


def _q2_k(b):
    n = b.shape[0]
    scales, qs = b[:, :16], b[:, 16:80]
    d, dmin = _f16(b[:, 80:82]), _f16(b[:, 82:84])
    # two halves of 128 values: 32 bytes each, read four times with a growing shift (2 bits per value)
    q = ((qs.reshape(n, 2, 1, 32) >> _shifts(4, b.device)) & 3).reshape(n, 16, 16).to(torch.float32)
    dl = (d * (scales & 0xF).to(torch.float32)).unsqueeze(-1)
    ml = (dmin * (scales >> 4).to(torch.float32)).unsqueeze(-1)
    return (dl * q - ml).reshape(n, 256)


def _q3_k(b):
    n = b.shape[0]
    hmask, qs, sc = b[:, :32], b[:, 32:96], b[:, 96:108]
    d = _f16(b[:, 108:110])
    # 16 six-bit scales: low 4 bits in bytes 0-7, high 2 bits packed in bytes 8-11
    low = torch.cat([sc[:, 0:8] & 0xF, sc[:, 0:8] >> 4], 1)  # scales 0-7, 8-15
    tmp = sc[:, 8:12]
    high = torch.cat([(tmp >> s) & 3 for s in (0, 2, 4, 6)], 1)  # (n, 16)
    scales = (low | (high << 4)).view(torch.int8).to(torch.float32) - 32
    q = ((qs.reshape(n, 2, 1, 32) >> _shifts(4, b.device)) & 3).reshape(n, 8, 32)
    bit = (hmask.unsqueeze(1) >> torch.arange(8, device=b.device, dtype=torch.uint8).view(1, 8, 1)) & 1  # (n, 8, 32)
    q = q.to(torch.float32) - (1 - bit.to(torch.float32)) * 4
    return (d.unsqueeze(-1) * scales.unsqueeze(-1) * q.reshape(n, 16, 16)).reshape(n, 256)


def _scale_min_k4(sc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The 8 six-bit scales and 8 six-bit minimums packed in 12 bytes (Q4_K, Q5_K)."""
    a, b, c = sc[:, 0:4], sc[:, 4:8], sc[:, 8:12]
    scales = torch.cat([a & 63, (c & 0xF) | ((a >> 6) << 4)], 1)
    mins = torch.cat([b & 63, (c >> 4) | ((b >> 6) << 4)], 1)
    return scales.to(torch.float32), mins.to(torch.float32)


def _q4_k(b):
    n = b.shape[0]
    d, dmin = _f16(b[:, :2]), _f16(b[:, 2:4])
    scales, mins = _scale_min_k4(b[:, 4:16])
    qs = b[:, 16:144].reshape(n, 4, 32)
    q = torch.stack([qs & 0xF, qs >> 4], 2).reshape(n, 8, 32).to(torch.float32)  # per 64 values: low nibbles, then high
    return ((d * scales).unsqueeze(-1) * q - (dmin * mins).unsqueeze(-1)).reshape(n, 256)


def _q5_k(b):
    n = b.shape[0]
    d, dmin = _f16(b[:, :2]), _f16(b[:, 2:4])
    scales, mins = _scale_min_k4(b[:, 4:16])
    qh, qs = b[:, 16:48], b[:, 48:176].reshape(n, 4, 32)
    q = torch.stack([qs & 0xF, qs >> 4], 2).reshape(n, 8, 32)
    bit = (qh.unsqueeze(1) >> torch.arange(8, device=b.device, dtype=torch.uint8).view(1, 8, 1)) & 1  # sub-block j uses bit j
    q = (q | (bit << 4)).to(torch.float32)
    return ((d * scales).unsqueeze(-1) * q - (dmin * mins).unsqueeze(-1)).reshape(n, 256)


def _q6_k(b):
    n = b.shape[0]
    ql, qh = b[:, :128].reshape(n, 2, 2, 32), b[:, 128:192].reshape(n, 2, 1, 32)
    scales = b[:, 192:208].contiguous().view(torch.int8).to(torch.float32)
    d = _f16(b[:, 208:210])
    # per half of 128 values: four runs of 32 = low nibbles of bytes 0-31, of 32-63, then their high nibbles;
    # the two top bits of each come from one qh byte
    low = torch.cat([ql & 0xF, ql >> 4], 2)  # (n, 2, 4, 32)
    high = (qh >> _shifts(4, b.device)) & 3
    q = (low | (high << 4)).to(torch.float32) - 32
    return ((d * scales).unsqueeze(-1) * q.reshape(n, 16, 16)).reshape(n, 256)


_DECODERS = {"Q4_0": _q4_0, "Q4_1": _q4_1, "Q5_0": _q5_0, "Q5_1": _q5_1, "Q8_0": _q8_0,
             "Q2_K": _q2_k, "Q3_K": _q3_k, "Q4_K": _q4_k, "Q5_K": _q5_k, "Q6_K": _q6_k}


def dequantize(raw: torch.Tensor, kind: str, shape, dtype=torch.float32) -> torch.Tensor:
    """Bytes of a tensor (uint8, any device) → the tensor in `dtype`."""
    check(kind)
    if kind in PLAIN:
        return raw.view(PLAIN[kind]).reshape(shape).to(dtype)
    values, size = BLOCKS[kind]
    out = _DECODERS[kind](raw.reshape(-1, size))
    return out.reshape(shape).to(dtype)


# ---------------------------------------------------------------- layers that keep their weight quantized


def _linear_forward(self, x):
    if self.gguf_input_index is not None:
        x = x.index_select(-1, self.gguf_input_index)
    parts = [dequantize(getattr(self, f"gguf_{i}"), kind, shape, self.compute_dtype) for i, (kind, shape) in enumerate(self.gguf_parts)]
    weight = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
    bias = self.bias.to(self.compute_dtype) if self.bias is not None else None
    return F.linear(x.to(self.compute_dtype), weight, bias)


def attach_linear(module: nn.Linear, parts: list[tuple[str, tuple, torch.Tensor]], compute_dtype=torch.bfloat16, input_index: Optional[torch.Tensor] = None):
    """Give a Linear its weight as quantized bytes: `parts` are (kind, shape, bytes) stacked along the output
    (one, or the q, k and v of a fused projection, which files store separately and in different types).
    `input_index` reorders the layer's input first, for a file whose weight has its columns in another order."""
    rows = sum(shape[0] for _, shape, _ in parts)
    if rows != module.out_features or any(shape[1] != module.in_features for _, shape, _ in parts):
        raise ValueError(f"quantized weight of shape {[tuple(s) for _, s, _ in parts]} doesn't fit a {module.out_features}×{module.in_features} layer")
    module.weight = None
    module.gguf_parts = [(kind, tuple(shape)) for kind, shape, _ in parts]
    for i, (_, _, raw) in enumerate(parts):
        module.register_buffer(f"gguf_{i}", raw, persistent=False)
    module.register_buffer("gguf_input_index", input_index, persistent=False)
    module.compute_dtype = compute_dtype
    module.forward = _linear_forward.__get__(module, nn.Linear)


def _embedding_forward(self, ids):
    rows = self.gguf_rows[ids.reshape(-1).to(self.gguf_rows.device)]
    out = dequantize(rows, self.gguf_kind, (rows.shape[0], self.embedding_dim), self.compute_dtype)
    return out.reshape(*ids.shape, self.embedding_dim)


def attach_embedding(module: nn.Embedding, kind: str, shape, raw: torch.Tensor, compute_dtype=torch.bfloat16):
    """An Embedding whose table stays quantized: only the rows looked up are dequantized."""
    if tuple(shape) != (module.num_embeddings, module.embedding_dim):
        raise ValueError(f"quantized embedding of shape {tuple(shape)} doesn't fit {module.num_embeddings}×{module.embedding_dim}")
    module.weight = None
    module.gguf_kind = kind
    module.register_buffer("gguf_rows", raw.reshape(module.num_embeddings, -1), persistent=False)
    module.compute_dtype = compute_dtype
    module.forward = _embedding_forward.__get__(module, nn.Embedding)
