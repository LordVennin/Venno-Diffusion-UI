# Plan: replace the A1111 layer with a lean Venno server ("option 2")

Status: proposal · written after the Forge trim (commit `36ac0b4`)

## Goal

Keep Forge Neo's **engine** (`backend/`: Anima model, Qwen3 text encoder, Qwen-Image VAE, VRAM management, quantization/GGUF, LoRA patching, CFG/conditioning core). Replace everything between that engine and Venno Studio with a small Venno-owned server. That means removing:
- the A1111 `modules/` package
- Gradio
- the extension script system
- the `/sdapi` API

## Where we are

| Layer | Python lines | Fate |
|---|---|---|
| `backend/` engine | ~14,100 | **keep** (synced with upstream Forge Neo where useful) |
| `modules_forge/packages/` (k-diffusion samplers, comfy LoRA formats, GGUF, model detection) | ~11,500 | **keep**, moved to `venno/vendor/` |
| `modules/` (A1111 pipeline, API, settings, scripts, Gradio helpers) | ~21,100 | **replace** |
| `modules_forge/` glue | ~1,800 | **replace** |
| extensions (ControlNet, ControlLLLite, LoRA, soft inpainting, ADetailer-Neo, Never OOM) | ~8,900 | **replace** (ControlLLLite's Anima module is ported as-is) |
| Venno (Studio JS + `venno_studio`) | ~3,700 | grows by ~3,500–4,500 lines of Python |

The coupling is small in the direction that matters. `backend/` only reaches into the A1111 layer for:
- `opts.anima_do_reference` and `opts.emphasis`
- `shared.batch_cond_uncond` and `devices.device`
- `SdConditioning` (a list subclass carrying `is_negative_prompt`/size)
- the vendored packages (which we keep)

A ~40-line `venno/engine/compat.py` shim covers all of it, and `backend/` itself needs almost no changes.

## Target architecture

```
Venno Studio (browser)  ──HTTP/WebSocket──▶  venno/server   (FastAPI: jobs, models, downloads, static Studio)
                                                  │
                                                  ▼
                                            venno/pipeline  (what A1111's processing.py did, Anima-only)
                                              txt2img · img2img · inpaint · hires · regions · guides · detailer · LoRA
                                                  │
                                                  ▼
                                            venno/engine    (thin layer over backend/)
                                              ModelManager · Conditioner · Sampler · VAE · Upscaler
                                                  │
                                                  ▼
                                            backend/  +  venno/vendor/ (k_diffusion, comfy lora, gguf, huggingface_guess)
```

```
venno/
  server/        app.py (FastAPI + static), jobs.py (single GPU worker, queue, cancel), api.py (schemas), downloads.py (moved), settings.py
  pipeline/      job.py (Job schema → plan), txt2img.py, img2img.py, inpaint.py, hires.py, regions.py, guides.py, detailer.py, metadata.py
  engine/        models.py, conditioning.py, sampling.py, schedulers.py, rng.py, vae.py, lora.py, upscale.py, preview.py, compat.py
  vendor/        k_diffusion/, comfy/, gguf/, huggingface_guess/, controllllite/ (ported Anima module)
  launcher/      launch.py (venv/uv install, torch build selection), requirements
studio/          (moved from extensions-builtin/venno_studio/studio)
backend/         (unchanged engine)
```

## What each A1111 piece becomes

| Today | Replacement | Notes / est. lines |
|---|---|---|
| `sd_models` + `main_entry` model switching, `--*-dirs` scanning | `engine/models.py` `ModelManager`: scans folders, calls `backend.loader.forge_loader(ckpt, [te, vae])`, keeps one engine loaded, unloads on switch | ~200 |
| `prompt_parser`, `extra_networks` | `engine/conditioning.py`: strip `<lora:…>` tags, parse schedules/`AND` (ported `prompt_parser.py`), encode positive/negative/region prompts through the Anima text engine (which already handles weights/emphasis) | ~150 + ported 370 |
| `sd_samplers*`, `sd_schedulers`, `forge_alter_samplers` | `engine/sampling.py`: build sigmas (Simple, Beta, Karras, Normal, SGM, …), wrap the model as a k-diffusion denoiser calling `backend.sampling.sampling_function_inner`, run `vendor/k_diffusion` samplers (Euler, Euler a, ER SDE, DPM++ 2M/SDE, …) | ~400 |
| `rng.py`, `rng_philox` | `engine/rng.py`: per-image seeded CPU noise (+ subseed variation if wanted) | ~60. **Old seeds won't reproduce bit-for-bit** |
| `processing.py` (txt2img / img2img / inpaint / hires) | `pipeline/*`: init latent, mask prep (blur, fill modes, "only masked" crop + paste-back), denoise, decode, overlay | ~900 |
| soft-inpainting extension | `pipeline/inpaint.py`: per-step latent blending (differential diffusion), same seamless result, far less code | ~120 |
| `venno_regional.py` (script) | `pipeline/regions.py`: same wrapper, now a plain function | moved, ~250 |
| ControlNet extension + `supported_controlnet` + ControlLLLite script | `pipeline/guides.py` + `vendor/controllllite/` (Anima module unchanged) | ~200 + ported 555 |
| sd_forge_lora `networks` | `engine/lora.py`: load file → `backend.patcher.lora` patches on the UNet/text-encoder patchers, cached by (file, weight) | ~150 |
| ADetailer-Neo | `pipeline/detailer.py`: ultralytics YOLO / mediapipe detection → mask → inpaint pass (only-masked crop) → paste back; uses the same detector files in `models/adetailer` | ~300 |
| `upscaler*`, `esrgan_model`, `modelloader` | `engine/upscale.py`: spandrel load + tiled upscale; latent upscale modes | ~180 |
| `images.py` (save + infotext) | `pipeline/metadata.py`: PNG with an A1111-compatible `parameters` text chunk (Civitai etc. still read it), output folders by date | ~120 |
| `shared_state`, `progress`, API queue | `server/jobs.py`: one worker thread, FIFO queue, cancel, progress + live previews (latent→RGB from `LatentFormat.latent_rgb_factors`) pushed over WebSocket | ~250 |
| `api.py` (`/sdapi/v1/*`, dropped) | `server/api.py`: `POST /v1/jobs` (one schema for all modes), `GET /v1/jobs/{id}`, `WS /v1/events`, `/v1/models`, `/v1/loras`, `/v1/downloads` | ~300 |
| `options`, `shared_options`, `config.json` | `server/settings.py`: small typed settings (paths, VRAM mode, attention backend, output dir) in `venno.json` | ~120 |
| `launch_utils` (torch/CUDA/ROCm install, uv) | `launcher/`: keep its torch-build selection logic, drop extension installers | ~250, mostly ported |

**Estimated total:** ~3,500–4,500 new Python lines, replacing ~32,000.

## Job schema (one request for every mode)

```jsonc
{
  "mode": "txt2img" | "img2img",
  "model": { "checkpoint": "…", "text_encoder": "…", "vae": "…" },
  "prompt": "…", "negative": "…", "loras": [{ "name": "…", "weight": 0.8 }],
  "width": 1024, "height": 1024, "batch": 2, "seed": 123,
  "sampler": "Euler", "scheduler": "Simple", "steps": 10, "cfg": 1.0, "shift": 3.0,
  "init_image": "<png>", "strength": 0.6,                        // img2img
  "mask": "<png>", "mask_blur": 8, "fill": "original", "only_masked": false, "soft_inpaint": true,
  "regions": [{ "prompt": "…", "mask": "<png>", "strength": 1.0 }], "region_base": 0.25, "region_end": 1.0,
  "guides": [{ "model": "…", "image": "<png>", "weight": 0.8, "start": 0, "end": 0.8 }],
  "hires": { "upscaler": "Latent", "scale": 1.5, "steps": 0, "denoise": 0.5, "cfg": null, "shift": null },
  "detailer": [{ "model": "face_yolov8n.pt", "denoise": 0.4, "confidence": 0.3, "padding": 32, "prompt": "" }],
  "plugins": { "<plugin name>": { /* plugin-defined fields */ } },
  "save": true
}
```

Studio already builds exactly these pieces; Phase 0 just changes where it sends them.

## Phases (each one ships a working app)

**Phase 0: decouple the front end** (1–2 days, low risk)
- Add `POST /venno/v1/jobs` with the schema above, implemented *on top of the current Forge pipeline* (an adapter that builds the `/sdapi` payload server-side).
- Studio, Text to Image and Image to Image switch to it; progress moves to a WebSocket.
- Result: the front end no longer knows Forge exists, and every later phase is invisible to it.

**Phase 1: Venno engine, text to image** (3–5 days)
- `engine/` (ModelManager, conditioning, sampling, schedulers, rng, VAE, preview) and `pipeline/txt2img`, behind `--engine venno` (Forge path stays default).
- Parity check: same prompt/sampler/steps on both engines. Results should match closely, not bit-exact, because the noise generator differs.

**Phase 2: img2img, inpainting, regions, LoRA, prompt syntax** (5–7 days)
- Mask handling, fill modes, only-masked crop, differential (soft) inpainting, regional prompting, LoRA patching and caching.
- Prompt scheduling / `AND` (ported parser) incl. inside region prompts; syntax hint in Studio.

**Phase 2b: Anima 3.8B** (3–5 days)
- Load Qwen3.5 4B as a second text encoder and implement the 3.8B conditioning path (from the 3.8B model card and the companion Forge extension).
- Variant-aware presets for 2B / 2.9B / 3.8B.

**Phase 3: hires fix, upscalers, guides, detailer, saving** (4–6 days)
- Spandrel upscalers, latent upscale, ControlLLLite port, YOLO/mediapipe detailer, PNG metadata, output folders.

**Phase 4: switch over and delete** (2–3 days)
- `--engine venno` becomes the default.
- Remove `modules/`, Forge extensions, Gradio, the scripts system, `webui.py` and `/sdapi` (no compatibility layer).
- Move `modules_forge/packages` to `venno/vendor` and the Studio folder to `studio/`.
- New launcher (`venno.bat` / `venno.sh`) plus requirements without Gradio.

**Phase 5: hardening** (ongoing)
- Unit tests for pure parts: schedulers, mask ops, region masks, job validation, metadata.
- A local smoke script that runs every mode on Anima Turbo at 512² × 4 steps (CPU-capable, ~10 min).
- Low-VRAM checks (`--lowvram`, Never-OOM equivalent: always-offload + tiled VAE) on a real GPU.

**Phase 6: plugin system** (after the rewrite)
- Manifest, loader and hook API as described under Decisions.
- First plugins written by us (e.g. tiled upscale).

**Total: roughly 4–5 weeks of focused work for Phases 0–4 + 2b.** Phases 0 and 1 are where most of the risk shows up early.

## Decisions

| # | Topic | Decision |
|---|---|---|
| 1 | `/sdapi` (A1111/Forge API) | **Dropped.** Only the Venno API exists after Phase 4. |
| 2 | Prompt scheduling (`[a:b:when]`, `[a\|b]`, `AND`) | **Keep.** Port `prompt_parser.py` as-is and surface the syntax in Studio (see below) |
| 3 | Seed compatibility with the Forge-based builds | **Not needed.** Use a simple seeded noise generator. |
| 4 | Anima variants | **All versions.** Must-have: Base / Aesthetic / Turbo (2B) and **2.9B**. Also planned: 3.8B (see below) |
| 5 | Upstream sync of `backend/` | *Open*: recommended to keep it close to Forge Neo |
| 6 | Extensions | **No Forge/A1111 extension compatibility.** Venno gets its own plugin system after the rewrite (Phase 6). |

### Prompt scheduling (decision 2)

A1111 prompt syntax that changes the prompt *during* sampling:
- `[cat:dog:0.4]` uses "cat" for the first 40 % of steps, then "dog"
- `[cat:dog:12]` switches at step 12
- `[hat:0.5]` adds "hat" halfway; `[hat::0.5]` removes it halfway
- `[cat|dog]` alternates every step
- `prompt A AND prompt B` mixes two separately-encoded prompts

It is little-used mostly because few people know it exists. Keeping it is cheap:
- The parser is self-contained (only `lark` + `torch`) and ports unchanged.
- The sampler needs ~30 lines to pick the per-step conditioning.
- `AND` maps onto the weighted-conditions support already in `backend/`.
- Hires schedules are built in.
- Region prompts go through the same parser.

All schedule variants are encoded once up front and cached. Estimated ~1 day in Phase 2 plus unit tests (nesting, escapes, LoRA tags inside schedules).

Studio gets a prompt-syntax hint and, later, highlighting that shows where switches happen.

### Anima variants (decision 4)

| Variant | Diffusion model | Text encoder(s) | Status in Venno |
|---|---|---|---|
| Anima 2B Base / Aesthetic / Turbo | 28 blocks | Qwen3 0.6B | works today |
| Anima 2.9B (Gazingstars123) | 40 blocks, layer-expanded 2B | Qwen3 0.6B | engine detects the 40 blocks and remaps 2B LoRAs / ControlLLLite; generation verified (incl. region + remapped guide); preset + Models-tab download added |
| Anima 3.8B (lylogummy) | 52 blocks + bundled "Semantic Connector v2" | Qwen3 0.6B **+ Qwen3.5 4B** | needs work: Forge Neo only loads the extra Qwen3.5 encoder through a third-party extension (`GumGum10/forge-anima-3.8B`); Venno must implement the dual-encoder conditioning itself (Phase 2b) |

Requirements that follow:
- Presets come from the detected variant, not the file name. 2.9B recommends ~28–50 steps, CFG 3.5–5, Euler + SGM Uniform or Res Multistep + Linear Quadratic.
- The Models tab offers the 2.9B (and later 3.8B) downloads.
- The parity/smoke tests run on 2B Turbo and 2.9B.

### Plugins (decision 6)

Venno plugins are Python packages in `plugins/<name>/` with a `venno_plugin.toml` manifest. They can:
- register pipeline hooks: `before_encode`, `model_wrapper` (per-step, like regions/guides), `after_decode`, `post_process`
- add job fields with validation
- add server routes
- ship a Studio ES module that adds a panel or tool

Built-in features (regions, guides, detailer, soft inpaint) are written against the same hook points, so the plugin API is exercised from day one.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Subtle sampling differences (sigma schedules, shift, CFG edge cases) | Phase 1 parity runs against the Forge path with the same seeds before anything is deleted |
| VRAM regressions on small GPUs | Engine memory management is unchanged; we only call it. Test `--lowvram` on a real 6–8 GB card before Phase 4 |
| Detailer quality vs ADetailer-Neo | Same detectors and the same crop/inpaint/paste algorithm; compare side-by-side in Phase 3 |
| Windows install / torch selection | Port `launch_utils.prepare_environment` logic rather than rewrite it |
| RAM blow-up on CPU / weak machines: the engine upcasts the DiT to fp32 when no GPU is present (2.9B → 10.6 GB) | Default the storage dtype to bf16 when system RAM is tight (verified: 2.9B then fits in 5.3 GB) |
| Losing upstream engine fixes | Keep `backend/` layout and names; record upstream commit in `docs/` for periodic merges |
