"""Prompt token counts for the Studio's prompt boxes, without loading the text encoder.

Anima reads a prompt twice: Qwen3 0.6B encodes it, and the LLM adapter turns that into one context entry
per T5 token. The T5 count is therefore the length of the context the model sees, which Anima pads to (and
was trained up to) 512. Counted like the engine tokenizes: LoRA tags removed, emphasis such as
`(word:1.2)` parsed out, and for scheduled prompts or AND the longest variant.
"""

from __future__ import annotations

import os
import threading

from . import conditioning, prompts

LIMIT = 512
STEPS = 20  # enough to see every variant of a scheduled prompt
_lock = threading.Lock()
_tokenizer = None


def _t5():
    global _tokenizer
    with _lock:
        if _tokenizer is None:
            from transformers import T5TokenizerFast

            from backend import huggingface  # unpacks the bundled tokenizer files on first use
            from backend.text_processing._comfy import INF, SDTokenizer

            huggingface.process()
            path = os.path.join(os.path.dirname(huggingface.__file__), "circlestone-labs", "Anima", "tokenizer_2")
            tok = T5TokenizerFast.from_pretrained(path)
            tok._eventual_warn_about_too_long_sequence = lambda *args, **kwargs: None
            _tokenizer = SDTokenizer(tok, pad_with_end=False, has_start_token=False, pad_to_max_length=False, max_length=INF, min_length=1)
        return _tokenizer


def count(text: str) -> int:
    """Context length of a prompt (0 for an empty one)."""
    text, _ = conditioning.extract_lora_tags(text or "", keep_random=True)
    if not text.strip():
        return 0
    tok = _t5()
    plan = prompts.PromptPlan(STEPS, text)
    variants = {t for step in range(1, STEPS + 1) for t, _ in plan.parts(step)}
    return max(sum(len(chunk) for chunk in tok.tokenize_with_weights(v)) for v in variants)
