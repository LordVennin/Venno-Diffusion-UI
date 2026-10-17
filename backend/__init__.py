"""Venno's diffusion engine (forked from Forge Neo's backend)."""

import os
import sys

# third-party code vendored as top-level packages (comfy, gguf, huggingface_guess, k_diffusion)
_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
