# Changelog

Releases are the `main` branch; each one raises `venno/version.py`. Settings → System → *Check for updates*
shows the sections newer than the installed version.

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
