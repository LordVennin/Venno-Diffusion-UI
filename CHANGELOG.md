# Changelog

Releases are the `main` branch; each one raises `venno/version.py`. Settings → System → *Check for updates*
shows the sections newer than the installed version.

## 0.9.1 — 2026-11-01

- First-run setup: a welcome window downloads a checkpoint with the text encoder and VAE in one go, selects it
  with its settings, and offers Low VRAM mode on small graphics cards (again from Settings → Models).
- Cancel keeps the images already finished; a new Skip button drops only the image being made. The Canvas makes
  its ×N images one after another (the same pictures as before), so both work there too.
- Downloading Anima 3.8B or its Qwen3.5 4B encoder also downloads whichever of its two text encoders is
  missing; with a 3.8B checkpoint selected the text encoder picker shows both. Cancelling a download stops it at
  once (it used to retry for ~20 seconds first).
- Memory mode (Automatic / Low VRAM / No VRAM) in Settings → System, used from the next start; Venno falls back
  to the CPU instead of stopping when no graphics card is usable.
- The console warns at start when installed packages differ from `requirements.txt` (e.g. a newer
  transformers), with the command that fixes it.

## 0.9.0 — 2026-10-28

First numbered release, and the start of the `main` branch.

- Venno Studio: Canvas (layers, regions, sketch guides, live mode), Text to Image, Image to Image, Inpaint and
  Image Info tabs; Settings with model downloads, Studio preferences and system information.
- Venno's own server and pipeline for Anima: samplers and schedulers, prompt scheduling and AND, wildcards,
  hires fix and upscalers, ADetailer-style detailer with detection previews, X/Y/Z comparison grids, PNG settings
  that load back into Studio.
- Regional prompting: one tuned method (region own passes in the first steps), empty character regions redrawn,
  the main prompt composed from the regions; the region lab (`python -m venno.regionlab`) for comparing layouts.
- ControlNet / ControlLLLite guides with preprocessors (Canny, line art, soft edge, depth) and control modes.
- Anima's time axis: consistent sets and reference pictures, and the community reference adapter (one picture
  for the character, the prompt for the scene).
- Anima 2B, 2.9B (community) and 3.8B (community, with Qwen3.5 4B) checkpoints; LoRA browser with folders,
  previews and tags.
- Check for updates in Settings → System.
