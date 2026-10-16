"""Prompt → conditioning tensors through the Anima text engine (Qwen3 + LLM adapter)."""

from __future__ import annotations

import re

import torch

# Plain prompts incl. (word:1.2) weights. Prompt scheduling ([a:b:when], [a|b]) and AND arrive in
# Phase 2 - until then such jobs run on the Forge path. <lora:name:weight> tags are taken out of the
# text by extract_lora_tags before a job runs.
_LORA = re.compile(r"<lora:([^>:]+)(?::([^>:]*))?(?::[^>]*)?>", re.I)
_SCHEDULE = re.compile(r"\[[^\[\]]*[:|][^\[\]]*\]")
_AND = re.compile(r"\bAND\b")


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


def unsupported_syntax(prompt: str) -> str | None:
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
