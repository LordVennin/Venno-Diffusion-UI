"""Runtime options the engine reads (formerly A1111's shared.opts / shared)."""

emphasis = "Original"  # how (word:1.2) weights are applied, see backend/text_processing/emphasis.py
batch_cond_uncond = True  # run the positive and negative prompt in one batch


class Conditioning(list):
    """Prompts for the text encoder, with the flags the engines read."""

    def __init__(self, prompts, is_negative_prompt=False, width=None, height=None, copy_from=None, distilled_cfg_scale=None):
        super().__init__(prompts)
        copy_from = prompts if copy_from is None else copy_from
        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, "is_negative_prompt", False)
        self.width = width or getattr(copy_from, "width", None)
        self.height = height or getattr(copy_from, "height", None)
        self.distilled_cfg_scale = distilled_cfg_scale or getattr(copy_from, "distilled_cfg_scale", None)
