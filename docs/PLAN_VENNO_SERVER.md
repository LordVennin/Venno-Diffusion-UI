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
| `prompt_parser`, `extra_networks` | `engine/conditioning.py`: strip `<lora:…>` tags, encode positive/negative/region prompts through the Anima text engine (which already handles weights/emphasis) | ~150. Decision: drop `[a:b:when]` scheduling and `AND`, or port `prompt_parser.py` (~370) |
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
| `api.py` (`/sdapi/v1/*`) | `server/api.py`: `POST /v1/jobs` (one schema for all modes), `GET /v1/jobs/{id}`, `WS /v1/events`, `/v1/models`, `/v1/loras`, `/v1/downloads` | ~300 |
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

**Phase 2: img2img, inpainting, regions, LoRA** (4–6 days)
- Mask handling, fill modes, only-masked crop, differential (soft) inpainting, regional prompting, LoRA patching and caching.

**Phase 3: hires fix, upscalers, guides, detailer, saving** (4–6 days)
- Spandrel upscalers, latent upscale, ControlLLLite port, YOLO/mediapipe detailer, PNG metadata, output folders.

**Phase 4: switch over and delete** (2–3 days)
- `--engine venno` becomes the default.
- Remove `modules/`, Forge extensions, Gradio, the scripts system, `webui.py` and `/sdapi`.
- Move `modules_forge/packages` to `venno/vendor` and the Studio folder to `studio/`.
- New launcher (`venno.bat` / `venno.sh`) plus requirements without Gradio.

**Phase 5: hardening** (ongoing)
- Unit tests for pure parts: schedulers, mask ops, region masks, job validation, metadata.
- A local smoke script that runs every mode on Anima Turbo at 512² × 4 steps (CPU-capable, ~10 min).
- Low-VRAM checks (`--lowvram`, Never-OOM equivalent: always-offload + tiled VAE) on a real GPU.

**Total: roughly 3–4 weeks of focused work.** Phases 0 and 1 are where most of the risk shows up early.

## Decisions needed from you

1. **`/sdapi` compatibility:** drop it (simplest), or keep a small compatibility shim for scripts, bots or plugins written against the A1111/Forge API?
2. **Prompt syntax:** keep `[from:to:when]` prompt scheduling and `AND` composition (port `prompt_parser.py`), or support only weights/emphasis, LoRA tags and Venno regions?
3. **Seeds:** accept that seeds from the Forge-based versions won't reproduce exactly after the switch? (Porting A1111's RNG bit-for-bit is possible but adds complexity.)
4. **Anima variants:** besides Anima 2B (Turbo/Aesthetic/Base), do the 2.9B/3.8B community variants and Anima Edit/reference mode need first-class support? (The engine already supports them; it's a testing and UI question.)
5. **Upstream sync:** keep `backend/` structurally close to Forge Neo, so engine fixes and speedups can still be merged from upstream, or let it diverge freely?

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Subtle sampling differences (sigma schedules, shift, CFG edge cases) | Phase 1 parity runs against the Forge path with the same seeds before anything is deleted |
| VRAM regressions on small GPUs | Engine memory management is unchanged; we only call it. Test `--lowvram` on a real 6–8 GB card before Phase 4 |
| Detailer quality vs ADetailer-Neo | Same detectors and the same crop/inpaint/paste algorithm; compare side-by-side in Phase 3 |
| Windows install / torch selection | Port `launch_utils.prepare_environment` logic rather than rewrite it |
| Losing upstream engine fixes | Keep `backend/` layout and names; record upstream commit in `docs/` for periodic merges |
