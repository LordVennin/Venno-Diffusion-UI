"""Prompt → conditioning tensors through the Anima text engine (Qwen3 + LLM adapter)."""

from __future__ import annotations

import re

import torch

# <lora:name:weight> tags are taken out of the text by extract_lora_tags before a job runs; scheduling
# ([a:b:when], [a|b]) and AND are resolved per step by PromptConditioning (venno/engine/prompts.py).
_LORA = re.compile(r"<lora:([^>:]+)(?::([^>:]*))?(?::[^>]*)?>", re.I)


def extract_lora_tags(prompt: str) -> tuple[str, list[tuple[str, float]]]:
    """'a cat <lora:x:0.6>' → ('a cat', [('x', 0.6)])"""
    found: list[tuple[str, float]] = []

    def take(m):
        try:
            weight = float(m.group(2)) if m.group(2) not in (None, "") else 1.0
        except ValueError:
            weight = 1.0
        found.append((m.group(1).strip(), weight))
        return ""

    text = _LORA.sub(take, prompt)
    text = re.sub(r"\s*,(\s*,)+", ",", text)  # ", ," left behind by a removed tag
    return text.strip().strip(",").strip(), found


def encode(engine, prompt: str, batch: int, *, negative: bool, width: int, height: int) -> torch.Tensor:
    """(batch, L, D) conditioning for one prompt repeated over a batch."""
    from modules.prompt_parser import SdConditioning  # list + flags; replaced by a Venno type in Phase 4

    conds = engine.get_learned_conditioning(SdConditioning([prompt], is_negative_prompt=negative, width=width, height=height))
    c = conds[0] if isinstance(conds, (list, tuple)) else conds
    if isinstance(c, dict):
        c = c["crossattn"]
    if c.ndim == 2:
        c = c.unsqueeze(0)
    c = c.reshape(-1, c.shape[-2], c.shape[-1])[:1]
    return c.repeat(batch, 1, 1)


class PromptConditioning:
    """Per-step conditioning for scheduled / composed prompts.

    Every distinct text is encoded once, up front (encoding mid-sampling would swap the text encoder back in).
    """

    def __init__(self, engine, positive, negative, *, width: int, height: int, cache: dict | None = None):
        from backend.sampling.condition import compile_conditions

        self.positive, self.negative = positive, negative  # prompts.PromptPlan
        self.needs_uncond = positive.composed  # AND / weights: CFG composition needs the negative even at CFG 1
        self._compile = compile_conditions
        self._tensors: dict[tuple[str, bool], torch.Tensor] = {} if cache is None else cache  # shared between passes
        self._steps: dict[tuple, tuple] = {}
        for step in range(1, positive.steps + 1):
            for text, _ in positive.parts(step):
                self.tensor(engine, text, False, width, height)
            for text, _ in negative.parts(step):
                self.tensor(engine, text, True, width, height)

    def tensor(self, engine, text: str, negative: bool, width: int = 0, height: int = 0) -> torch.Tensor:
        key = (text, negative)
        if key not in self._tensors:
            self._tensors[key] = encode(engine, text, 1, negative=negative, width=width, height=height)
        return self._tensors[key]

    @property
    def texts(self) -> int:
        return len(self._tensors)

    def at(self, step: int):
        """(cond list, uncond list) in the backend's compiled form for a 1-based step."""
        key = (self.positive.parts(step), self.negative.parts(step))
        if key not in self._steps:
            cond = []
            for text, weight in key[0]:
                c = self._compile(self._tensors[(text, False)])[0]
                if weight != 1.0 or len(key[0]) > 1:
                    c = {**c, "strength": weight}
                cond.append(c)
            uncond = [c for text, _ in key[1] for c in self._compile(self._tensors[(text, True)])]
            self._steps[key] = (cond, uncond)
        return self._steps[key]
