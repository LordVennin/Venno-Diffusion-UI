# Changelog

Releases are the `main` branch; each one raises `venno/version.py`. Settings → System → *Check for updates*
shows the sections newer than the installed version.

## Unreleased (development branch)

- **Venno MOSS**, a separate Studio for testing MOSS-Image LoRAs (`moss.sh` / `moss.bat`, port 7870, its own
  `venv-moss`): SFT / Base downloads, LoRAs from venno-trainer and other trainers, a comparison grid that puts a
  training run's epochs side by side, settings saved in the PNGs, Full (24 GB) and Low (12 GB) memory modes.
  Kromoss (lodestones' fine-tune of Base) is in its download list; a line under the LoRA list shows each
  LoRA's rank, alpha and the model it was trained on when its file records it.
- Venno MOSS: **hires fix** (Lanczos, Latent or Venno's upscaler models; denoising, own steps / CFG / shift; grid
  axes) and **Fast preview** (8 steps at CFG 1 with lodestones' turbo LoRA, downloadable in its Settings).
- Venno MOSS: **Image to Image** (with Outpaint) and **Inpaint** tabs like Venno's, each with its own prompt and
  settings: mask painting with zoom, masked content modes, only masked, soft inpainting; pictures go between
  tabs with Send to / Use as input.
- Venno MOSS's Image Info tab has Venno's layout and also reads pictures from Venno, Forge and A1111 (prompt,
  negative prompt, seed and size carry over).
- Venno MOSS: **LoRA browser** (🗂 Browse next to the LoRA list), the same one as Venno's: folders, previews, search,
  tags, sorting, notes and trigger words per LoRA, for LoRA folders too big for a dropdown. Listing a large folder
  is also faster (each file is read once, not on every refresh).
- Venno MOSS: *Send to Image to Image / Inpaint* takes the picture's prompts, LoRAs, seed and settings along (it
  sent only the picture).
- **Batch sheet** (Venno's form tabs and Venno MOSS): two or more pictures of a job are also stitched into one
  picture for comparison, like Forge's grid; shown first, saved with the grids, switched off per tab.
- Inpaint and Image to Image: the wheel zooms the input image where the pointer is (up to 8×); drag with the middle
  button or hold Space and drag to move around; the zoom button fits it again. Shift + wheel changes the brush
  size (the plain wheel did).
- The hires card shows the final size as the server computes it (it rounded to 64).
- Images of a several-image job appear in the results as each one finishes (Canvas and form tabs), so you can look
  at, apply, Skip or Cancel early; a result you click meanwhile stays shown instead of the next image's progress.

- Out of graphics memory: the job says what to change (Low VRAM mode, hires scale, batch size, picture size) and
  the memory is freed, so the next job runs without restarting Venno.
- A missing or wrong text encoder is named plainly ("Anima needs the Qwen3 0.6B text encoder …") instead of a
  weight-loading error; switching to Anima 3.8B says which of its two encoders is missing.

- No more console warnings on every step with the SDE samplers ("Should have tb<=t1") or on every upscale
  ("torch.meshgrid: in an upcoming release …"); neither changed any picture.

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
