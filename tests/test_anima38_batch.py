"""Anima 3.8B (v1.1): negatives carry run id 0, so prompt and negative batch together (venno/engine/anima38)."""

import torch

from backend.sampling.condition import compile_conditions
from backend.sampling.sampling_function import get_area_and_mult, can_concat_cond
from venno.engine import anima38, conditioning

L, D = 6, 4


class FakeRuntime(anima38.Runtime):
    def __init__(self):
        self.connector = object()
        self.runs = {}

    def encode(self, text):
        return {"crossattn": torch.zeros(1, L, D), "vector": torch.tensor([[7]])}

    def _context(self, run_id, timesteps, device, dtype):
        return torch.full((timesteps.numel(), L, D), float(run_id), dtype=dtype)


class FakeEngine:
    venno_anima38 = FakeRuntime()

    def get_learned_conditioning(self, c):
        return torch.ones(1, L, D)


def test_negative_batches_with_positive():
    engine = FakeEngine()
    pos = conditioning.encode(engine, "a", 1, negative=False, width=64, height=64)
    neg = conditioning.encode(engine, "b", 1, negative=True, width=64, height=64)
    assert int(neg["vector"]) == anima38.NATIVE and torch.equal(neg["crossattn"], torch.ones(1, L, D))
    x, t = torch.zeros(1, 16, 1, 8, 8), torch.ones(1)
    p = [get_area_and_mult(compile_conditions(c)[0], x, t) for c in (pos, neg)]
    assert can_concat_cond(p[0], p[1])


def test_expand_keeps_native_rows():
    rt = FakeRuntime()
    context = torch.stack([torch.zeros(L, D), torch.ones(L, D)])
    out = rt.expand(context, torch.tensor([0.5, 0.5]), torch.tensor([[7], [anima38.NATIVE]]))
    assert torch.all(out[0] == 7) and torch.all(out[1] == 1)
    native = {"crossattn": torch.full((1, L, D), 3.0), "vector": torch.tensor([[anima38.NATIVE]])}
    assert torch.all(rt.expand_one(native, torch.tensor([0.5]), "cpu", torch.float32) == 3)
