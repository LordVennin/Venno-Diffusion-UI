"""Prompt → conditioning tensors through the Anima text engine (Qwen3 + LLM adapter)."""

from __future__ import annotations

import re

import torch

# Phase 1 handles plain prompts (incl. (word:1.2) weights). LoRA tags, prompt scheduling
# ([a:b:when], [a|b]) and AND arrive in Phase 2 - until then such jobs run on the Forge path.
_LORA = re.compile(r"<lora:[^>]+>", re.I)
_SCHEDULE = re.compile(r"\[[^\[\]]*[:|][^\[\]]*\]")
_AND = re.compile(r"\bAND\b")


def unsupported_syntax(prompt: str) -> str | None:
    if _LORA.search(prompt):
        return "LoRA tags"
    if _SCHEDULE.search(prompt.replace("\\[", "").replace("\\]", "")):
        return "prompt scheduling"
    if _AND.search(prompt):
        return "AND composition"
    return None


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
