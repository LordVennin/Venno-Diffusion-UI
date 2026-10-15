# Venno Diffusion UI

A painting-first UI for the **[Anima](https://huggingface.co/circlestone-labs/Anima)** image model, in the spirit of Krita AI Diffusion. You sketch, select and paint on a layered canvas. The model then generates, refines or inpaints exactly the part you point it at.

The generation engine is **[Stable Diffusion WebUI Forge – Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02, cut down to Anima only. Its memory management, samplers, LoRA support and API are unchanged. Venno adds a new front end (**Venno Studio**) and a regional prompting module on top.

![Venno Studio](docs/studio.png)

## Features

| | |
|---|---|
| **Sketch → image** | Draw on any paint layer with pressure-sensitive brushes, fill and eraser, then **Refine** at 40–80 % strength. Composition and colours from the sketch are kept. |
| **Sketch guides** | *Guide* layers are passed to ControlNet / **Anima ControlLLLite** models (line art, scribble…) instead of being painted into the image, so loose line art steers the result even at 100 % strength. |
| **Inpainting** | Select an area with the rectangle, lasso or selection brush and press **Generate**. Only the selection changes. Venno adds surrounding context automatically, feathers the edges, and uses Forge's soft inpainting for seamless blends. |
| **Image to image** | Open, paste or drag in an image and refine the whole canvas or just a selection at any strength. |
| **Regional prompting** | *Region* layers are painted masks, each with its own prompt (e.g. one character per region). Regions apply at every denoising step, alongside the main prompt, with any sampler or LoRA. Inpainting works with regions too. |
| **Live mode** | Re-generates automatically after each stroke (fixed seed, so changes stay comparable). Works best with **Anima Turbo** and the 512²/768² resolutions. |
| **Layers & history** | Paint / region / guide layers, opacity, visibility, reorder, duplicate, merge down, undo/redo for everything, export to PNG. |
| **Results tray** | Every generation lands in a tray. Preview a result on the canvas, then **Apply** it as a new layer (it is never baked in automatically). |

## Install

1. Install [git](https://git-scm.com/downloads) and [uv](https://github.com/astral-sh/uv#installation) (recommended) or Python 3.13.
2. Clone and create a venv:
   ```bash
   git clone https://github.com/LordVennin/Venno-Diffusion-UI
   cd Venno-Diffusion-UI
   uv venv venv --python 3.13 --seed
   ```
3. Download the Anima files ([model card](https://huggingface.co/circlestone-labs/Anima)):

   | File | Put it in |
   |---|---|
   | `anima-turbo-v1.1.safetensors` (or `anima-aesthetic-…`, `anima-base-…`) | `models/Stable-diffusion/` |
   | `qwen_3_06b_base.safetensors` | `models/text_encoder/` |
   | `qwen_image_vae.safetensors` | `models/VAE/` |
   | *(optional)* LoRAs | `models/Lora/` |
   | *(optional)* Anima ControlLLLite / ControlNet models | `models/ControlNet/` |

4. Run `webui-user.bat` (Windows) or `./webui-user.sh` (Linux/macOS). The first start installs dependencies. Studio then opens at **http://127.0.0.1:7860/**.

The text encoder and VAE are picked automatically in the top bar. Choosing a checkpoint with `turbo` in its name switches to the Turbo preset (Euler · Simple · 10 steps · CFG 1). Other checkpoints get the model card's settings (ER SDE · Beta · 32 steps · CFG 4). Both use shift 3.

Only Anima checkpoints are accepted. Loading any other architecture fails with a clear error.

### Classic UI

Add `--classic-ui` to `COMMANDLINE_ARGS` to start the original Forge (Gradio) interface instead. Studio is then still available at `/studio/`. All of Forge's [command-line flags](docs/FORGE_NEO_README.md#commandline) still work, e.g. `--uv`, `--sage`, `--listen`, `--port`, `--ckpt-dirs`.

## Using the Studio

**The one rule:** *selection decides where, strength decides how much.*

| Selection | Strength | What happens |
|---|---|---|
| none | 100 % | **Generate** a new image at the canvas aspect ratio |
| none | < 100 % | **Refine** the whole canvas (img2img / sketch → image) |
| yes | 100 % | **Generate inside the selection** (inpaint) |
| yes | < 100 % | **Refine the selection** (partial inpaint) |

Results are rendered at about the chosen *Resolution* (default 1024², Anima's native size), whatever the size of the area you work on. A small selection is upscaled for detail and the result scaled back in.

### Shortcuts

| Key | Action | Key | Action |
|---|---|---|---|
| `B` | Brush | `Ctrl+Enter` | Generate / Refine |
| `E` | Eraser | `Enter` / `Esc` | Apply / discard previewed result |
| `G` | Fill | `Ctrl+Z`, `Ctrl+Shift+Z` | Undo / redo |
| `I` (or `Alt`+click) | Colour picker | `Ctrl+A`, `Ctrl+D`, `Ctrl+Shift+I` | Select all / none / invert |
| `M` / `L` / `S` | Rect / lasso / brush select | `Delete`, `Alt+Backspace` | Clear / fill selection |
| `V` | Move layer | `[` `]` | Brush size |
| `H` or `Space` drag | Pan | `X` | Swap colours |
| wheel | Zoom | `0` / `1` | Fit / 100 % |

`Shift` adds to a selection, `Alt` subtracts. Pen pressure controls brush size.

### Tips for Anima

- Danbooru-style tags, natural language, or both. Prefix artists with `@`.
- *Add Anima quality tags* prepends `masterpiece, best quality, score_7, safe, `. Turn it off for Anima-Aesthetic, whose model card advises against `score_*` tags.
- Prompt weights need to be stronger than on SDXL, e.g. `(chibi:2)`.
- Regions: describe each character fully in its region prompt. The main prompt is appended to each region by default; *Main prompt inside regions* controls how much the main prompt still shapes the regions.

## API

Studio uses the standard Forge endpoints (`/sdapi/v1/txt2img`, `/sdapi/v1/img2img`, `/sdapi/v1/progress`, …), so scripts can use them directly. Regional prompting is an always-on script:

```json
"alwayson_scripts": {
  "venno regional prompting": {
    "args": [true, "{\"base_strength\": 0.25, \"end_at\": 1.0, \"regions\": [{\"prompt\": \"1girl, red kimono\", \"mask\": \"<base64 png>\", \"strength\": 1.0}]}"]
  }
}
```

Masks are white (or opaque) where the region applies, relative to the full generation size. `GET /venno/api/info` lists models, modules, LoRAs, samplers, schedulers and presets.

## Project layout

| Path | |
|---|---|
| `extensions-builtin/venno_studio/studio/` | Studio front end (plain ES modules, no build step) |
| `extensions-builtin/venno_studio/scripts/venno_studio.py` | Serves Studio, `/venno/api/*` |
| `extensions-builtin/venno_studio/scripts/venno_regional.py` | Regional prompting (masked per-region conditioning) |
| `backend/`, `modules/`, `modules_forge/` | Forge Neo engine (Anima only) |

## Credits & licence

- Engine: [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) by Haoming02, based on [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) by lllyasviel and [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) by AUTOMATIC1111. The upstream README is kept in [`docs/FORGE_NEO_README.md`](docs/FORGE_NEO_README.md).
- Model: [Anima](https://huggingface.co/circlestone-labs/Anima) by CircleStone Labs & Comfy Org (non-commercial model licence; generated images may be used commercially).
- Inspired by [Krita AI Diffusion](https://github.com/Acly/krita-ai-diffusion).

Licensed under the [AGPL-3.0](LICENSE), same as the upstream projects.
