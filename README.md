# Venno Diffusion UI

A painting-first UI for the **[Anima](https://huggingface.co/circlestone-labs/Anima)** image model, in the spirit of Krita AI Diffusion. You sketch, select and paint on a layered canvas. The model then generates, refines or inpaints exactly the part you point it at.

Venno is a standalone app built for Anima: its own server and generation pipeline (`venno/`: job queue, sampling, regions, guides, detailer, consistent sets, reference pictures, LoRA library) and its own front end, Venno Studio (`studio/`). Underneath is a slim diffusion engine (`backend/`) that started from [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)'s and keeps its model loading, memory management, attention and LoRA patching, trimmed to Anima only (see [Credits](#credits--licence)). There is no A1111/Forge layer: no Gradio, no extensions, no `/sdapi`. Studio has these tabs:

- **Canvas**: the layered painting workspace.
- **Text to Image**, **Image to Image** and **Inpaint**: classic form-style generation, with hires fix, ADetailer, comparison grids and a results gallery. Image to Image and Inpaint show the input (and mask) large, next to the result.
- **Image Info**: drop any image to read its generation parameters and send them (with the image) to another tab.
- **⚙ Settings** (top right): *Models* with one-click downloads of Anima, guide (ControlLLLite) models, upscalers and detectors; *System* with the version, device and the folders Venno reads and writes; *Plugins* (coming).

![Venno Studio](docs/studio.png)

## Features

| | |
|---|---|
| **Sketch → image** | Draw on any paint layer with pressure-sensitive brushes, fill and eraser, then **Refine** at 40–80 % strength. Composition and colours from the sketch are kept. |
| **Sketch guides** | *Guide* layers steer generation through **Anima ControlLLLite** models (line art, scribble, depth, pose…) instead of being painted into the image, so loose line art shapes the result even at 100 % strength. Start/end are real steps, and guides also steer the hires pass. |
| **ControlNet** | Text to Image, Image to Image and Inpaint: up to three units, each guiding the picture with another one. A **preprocessor** extracts what the guide model needs (Canny edges, line art, soft edge / scribble, or depth with Depth Anything V2) and shows the map; picking one selects the matching Anima ControlLLLite model. Strength, start/end, resize mode and Forge's **control modes** (Balanced, My prompt is more important, ControlNet is more important). |
| **Outpainting** | Image to Image: extend the picture by any number of pixels on each side; the new border is generated to continue the image and the original stays untouched. |
| **Inpainting** | Select an area with the rectangle, lasso or selection brush and press **Generate**. Only the selection changes. Venno adds surrounding context automatically, feathers the edges, and uses soft (differential) inpainting for seamless blends. |
| **Image to image** | Open, paste or drag in an image and refine the whole canvas or just a selection at any strength. |
| **Regional prompting** | *Region* layers are painted masks, each with its own prompt (e.g. one character per region, one for the scenery). Each region's prompt steers its own area inside the model's text attention at every step, and in the first steps every region forms on its own (*own passes*) so each subject lands where its region is; then the picture is finished as one. Works with any sampler or LoRA, also in live mode and inpainting. |
| **Prompted sketch layers** | Give any paint or guide layer its own prompt and its strokes become a region. Venno grows the strokes and fills closed outlines, so sketching a figure on its own layer is enough to prompt it separately. |
| **Prompt scheduling** | Forge/A1111 syntax works as-is (`[a:b:0.4]`, `[a:0.5]`, `[a::0.5]`, `[a\|b]`, `AND`, also in region prompts), and a preview line under the prompt says what happens at which step. For the common case, *Then switch to…* takes a second prompt and a slider: composition from the first prompt, details from the second. |
| **Tag autocomplete** | Every prompt box suggests Danbooru tags (140k, with post counts and aliases) while you type, inserted the way Anima wants them: lowercase with spaces, `@artist`, escaped parentheses. `@` lists artists only, `__` lists wildcard files. ↑/↓ and Enter/Tab; extra tag lists can go in `tags/*.csv`. A counter in the corner of each prompt box shows its length in tokens (what Anima's adapter sees, quality tags included; orange past the 512 Anima is trained on). Both can be switched off in ⚙ Settings → Studio. |
| **Wildcards** | `{red\|blue\|green} hair`, `{2$$a\|b\|c}`, `{0.2::rare\|common}` and `__file__` (lines from `wildcards/file.txt`, nested folders and `*` allowed). Each image picks from its own seed, so results are reproducible; the template is kept in the image metadata. |
| **Settings from images** | Drop a PNG from Venno, Forge or A1111 (or an A1111 JPEG/WebP) onto a tab, or press *From image…*, to load its prompt, LoRAs, seed, sampler, size, hires, switch prompt and detailer settings. Opening such an image on the Canvas offers the same. Results have a *Use settings* button. |
| **Comparison grid (X/Y/Z)** | Every generation tab and the Canvas: up to three axes over artist, prompt S/R, CFG, steps, shift, seed, sampler, scheduler, checkpoint, LoRA weight, denoising, hires upscaler, size, switch point, detailer denoise and (Canvas regions) region strength settings. A1111 value syntax (`1-5`, `1-10 (+2)`, `0.5-1.5 [5]`), a live cell count, labelled grid images (one per Z value) saved to `outputs/txt2img-grids`. |
| **LoRA browser** | 🗂 next to the LoRA list opens a browser: your LoRA sub-folders are tiles you open (with a folder tree and breadcrumbs), every LoRA is a card with its preview. Search names, tags, trigger words and notes across all folders, filter by tags (combine several), sort by name / newest / folder, resize the cards. Click a card to add or remove the LoRA (at its preferred weight, with its trigger words added to the prompt). Right-click (or ✎) edits a LoRA: preview picture (upload, paste, drop, or *Use current result*), tags, trigger words, preferred weight and notes. Saved next to the LoRA file in Forge / A1111's format (`name.preview.png`, `name.json`), so previews made in Forge show up and the other way round; Civitai Helper's trigger words are read too. |
| **LoRAs** | Picked LoRAs appear as chips under the prompts instead of `<lora:…>` text. Click a chip to remove it; edit or scroll its number to change the weight (0 = off). Pasted `<lora:name:weight>` tags turn into chips automatically. LoRAs made for a smaller Anima are re-mapped onto larger ones (e.g. 2B → 2.9B). |
| **ADetailer** | On by default (except live mode). Detects faces (and optionally hands / people) with ADetailer's YOLO / mediapipe models (the standard face / hand / person detectors download by themselves on first use, into `models/adetailer`) and re-renders each at a resolution suited to its size, blended in seamlessly. Several faces are refined in one batched pass; when inpainting, only faces inside the selection are touched. `<lora:…>` in a detailer prompt applies to the detailer only. Like Forge's ADetailer, it shows what it detected: each box with its score in the live preview, a report under the result ("face_yolov8n.pt: 2 found (0.91, 0.54)", in orange when nothing was found) with *Show detections* on the form tabs, and on the Canvas a message plus right-click on the result thumbnail. |
| **Hires. fix** | Text to Image and Canvas: generate, upscale (latent, Lanczos or any ESRGAN/DAT/SwinIR-style model via Spandrel), refine. Upscale factor or target size, separate steps, denoising, CFG and shift. LoRAs, regions, guides and prompt scheduling all carry into the second pass. |
| **Consistent sets** | Text to Image: generate 2–4 pictures **together** as frames of one sequence on Anima's Cosmos time axis. They share the character, outfit and style, with small changes between them, like frames of a clip. Each frame can get its own text (an expression, a pose). |
| **Reference pictures** | Every tab: show the model up to three pictures next to the one it generates, and the result takes their character, outfit and style, without a LoRA. *Follow the steps* runs references the way the **Anima Edit LoRAs** expect. |
| **Live mode** | Re-generates automatically after each stroke (fixed seed, so changes stay comparable). Works best with **Anima Turbo** and the 512²/768² resolutions. |
| **Layers & history** | Paint / region / guide layers, opacity, visibility, reorder, duplicate, merge down, undo/redo for everything, export to PNG. |
| **Results tray** | Every generation lands in a tray. Preview a result on the canvas, then **Apply** it as a new layer (it is never baked in automatically). |

## Install

1. Install [git](https://git-scm.com/downloads) and Python 3.10 or newer (3.12 / 3.13 recommended).
2. Clone:
   ```bash
   git clone https://github.com/LordVennin/Venno-Diffusion-UI
   cd Venno-Diffusion-UI
   ```
3. Run `venno.bat` (Windows) or `./venno.sh` (Linux/macOS). The first start creates `venv/` and installs PyTorch (CUDA 13 build; set `TORCH_INDEX_URL` / `TORCH_COMMAND` for other GPUs) and the requirements, then Studio runs at **http://127.0.0.1:7860/** (add `--autolaunch` to open the browser). Later starts only check that everything is in place.
   - **Windows:** keep Venno in a short folder such as `C:\Venno`. PyTorch installs files nested ~140 characters deep, and Windows stops at 260 characters per path unless long paths are enabled; with a long folder path the install fails with *"The filename or extension is too long"* (WinError 206). Venno checks this before installing and says what to do: move the folder (then delete `venv`), or enable long paths once in an administrator PowerShell with `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`.
4. With no Anima checkpoint installed, Studio opens **⚙ Settings → Models**. Download *Anima Turbo* (or Aesthetic/Base), the *Qwen3 text encoder* and the *Qwen-Image VAE*, plus any guides and upscalers you want. *Download from URL* takes any Hugging Face / Civitai / direct link (LoRAs too). You can also place files manually:

   | File | Folder |
   |---|---|
   | `anima-turbo-v1.1.safetensors` (or `anima-aesthetic-…`, `anima-base-…`) | `models/Stable-diffusion/` |
   | `qwen_3_06b_base.safetensors` | `models/text_encoder/` |
   | `qwen_image_vae.safetensors` | `models/VAE/` |
   | *(optional)* LoRAs | `models/Lora/` |
   | *(optional)* Anima ControlLLLite guide models, e.g. [`anima-lllite-any-test-like-v2`](https://huggingface.co/kohya-ss/Anima-LLLite) (line art / scribble / grayscale) | `models/ControlNet/` |
   | *(optional)* upscalers (`.pth` / `.safetensors`) | `models/ESRGAN/` |

The text encoder and VAE are picked automatically in the top bar. Choosing a checkpoint with `turbo` in its name switches to the Turbo preset (Euler · Simple · 10 steps · CFG 1). Other checkpoints get the model card's settings (ER SDE · Beta · 32 steps · CFG 4). Both use shift 3.

Only Anima checkpoints are accepted. Loading any other architecture fails with a clear error.

**Anima 3.8B** ([lylogummy/Anima-3.8B](https://huggingface.co/lylogummy/Anima-3.8B), community) is supported natively: a 52-block Anima whose prompt is also read by **Qwen3.5 4B**, for better prompt adherence, several characters, interactions, positions and natural-language prompts. Get *Anima 3.8B v1.1* and *Qwen3.5 4B text encoder* in ⚙ Settings → Models (or put `Anima-3.8B-v1.1.safetensors` in `models/Stable-diffusion/` and `qwen35_4b.safetensors` in `models/text_encoder/`), then pick the checkpoint and keep **Qwen3 0.6B** as the text encoder: Venno recognises the 3.8B, finds the Qwen3.5 encoder by itself and downloads Qwen's tokenizer (12 MB, once). It picks the model card's settings (Res Multistep · Beta · 32 steps · CFG 5). The bundled "Semantic Connector v2" runs at every step; both text encoders are released after the prompt is encoded, so sampling needs no more memory than the diffusion model plus the connector. Negative prompts use the native encoder, as in the reference workflows. The older v1 checkpoint works too with `Anima-3.8B-expanded_adapter.safetensors` next to the text encoders. Regions, region negatives, the region pass, hires fix, guides, ADetailer, LoRAs (2B / 2.9B LoRAs are re-mapped onto the 52 blocks) and prompt scheduling all work with it. Prompting tip from the model card: give each character their own sentence and name characters instead of using "she" / "he".

**Options** go after the launcher, e.g. `./venno.sh --listen --lowvram` (`--help` lists them all; options you always want can go in the `VENNO_ARGS` environment variable):

- `--port 7860`, `--listen` (serve on the network), `--autolaunch`
- `--models-dir PATH`: use an existing models folder, e.g. your Forge / A1111 `models` (same sub-folders: `Stable-diffusion`, `text_encoder`, `VAE`, `Lora`, `ControlNet`, `ESRGAN`, `adetailer`); `--data-dir`, `--outputs-dir`; extra folders with `--ckpt-dirs`, `--lora-dir`, `--controlnet-dir`, …
- engine: `--lowvram` / `--novram` / `--highvram` / `--cpu`, precision (`--fp8_e4m3fn-unet`, `--bf16-vae`, …), attention (`--sage`, `--flash`, `--xformers` also install the package on first use), `--cuda-malloc`, …
- `--skip-install`: don't check / install packages on start (you manage the environment yourself)

With `--listen`, downloading from custom URLs is disabled unless you add `--allow-custom-downloads`. Catalog downloads keep working.

## Using the Studio

**The one rule:** *selection decides where, strength decides how much.*

| Selection | Strength | What happens |
|---|---|---|
| none | 100 % | **Generate** a new image at the canvas aspect ratio |
| none | < 100 % | **Refine** the whole canvas (img2img / sketch → image) |
| yes | 100 % | **Generate inside the selection** (inpaint) |
| yes | < 100 % | **Refine the selection** (partial inpaint) |

Results are rendered at about the chosen *Resolution* (default 1024², Anima's native size), whatever the size of the area you work on. A small selection is upscaled for detail and the result scaled back in.

### Regions

The **Regions** list under the prompt shows every region with its prompt. *+ Region* adds one and picks the **Rectangle tool (U)**: drag boxes where it applies (or paint with the brush; Ctrl+drag erases a box). *Area* reselects a region's layer to edit it, the eye turns it off. LoRAs apply to the whole image, not to a single region.

A region's prompt is what its area shows: write each character or object fully in its own region (`1girl, black hair, frilly dress, sitting on couch`) and give the scenery a region of its own over the whole canvas (`living room, couch, fluffy rug, night`). The main prompt can stay empty (*Add Anima quality tags* puts the quality tags in front of every prompt, region prompts included); it still reaches the regions a little (10%), so a style written there colours the whole picture. Where regions overlap, the **higher layer wins**: a region covers the ones below it, with a soft edge, so a character region above the scenery region takes the character's area and the scenery prompt gets everything else. The *Regions* overlay shows exactly this.

Every region is either **scenery** or a **subject** (*Auto* in the regions list says which: a region whose box spans more than 60% of the canvas is scenery, anything smaller a subject; set it by hand when a big subject or a small scenery region gets it wrong). Scenery gives way to every subject where they overlap, is never kept apart from them and is not refined by the region pass; subjects are what the own passes place.

Each region can also have its own **negative prompt**: open *Negative* under its prompt (or use *Region negative* in the layer panel). It applies only inside that region, and the main negative is added to it, so it only needs what is specific to that area: `blonde hair` for the black-haired character so colours don't bleed, `hat` for one character only, or `1girl, people` on the scenery region to keep extra figures out of it. Regions without one use the main negative. Region negatives need a negative pass: they do nothing at CFG 1 (e.g. turbo settings).

**Region own passes** (Settings → Regions, 30% by default): the regions' prompts steer the attention, but attention alone can't add a subject that the first steps' layout left out, so a second character used to be swallowed by the first (one big figure, its legs where the second region is). In the first steps (this share of them) every region also gets a model pass of its own, on a crop around it, with only its prompt, and that prediction is used inside the region (a latent couple), so each subject starts to form where its region is; the remaining steps blend everything into one picture. Measured on an RTX 4090 with the base model at 1024², 32 steps, CFG 4 over 8 seeds of a living-room layout (scenery, fireplace, a girl sitting on a couch, a girl lying in a thin strip at the bottom): the two characters were both in their regions in 13 of 16 cases with own passes, 11 of 16 without, and with a character LoRA the plain single pass drew one big character in every seed while own passes drew both in 7 of 8. What still fails is a subject whose natural size is bigger than its region (a girl lying in a thin strip comes out small or sketchy): make the region as big as the subject, not as small as the spot. Cost: about one extra pass per region in those steps (~30% more time with four regions). 0% turns it off; 50% and more start to merge characters.

**Keep regions apart** (Settings → Regions, 70% by default) helps with three or more characters. Without it every part of the image still "looks at" every other part, so characters drift towards each other: hair colours, outfits or faces mix or swap. In the middle of the steps it damps that between different subject regions (100% = almost fully separated), so each character takes its look from its own prompt. The first steps run normally so the layout forms as one picture, and the last steps run normally to blend lighting and style. Scenery regions stay shared with everything. If characters look pasted in or ignore each other (no hand-holding, mismatched light), lower it; if they still mix, raise it. The comparison grid has a *Region separation* axis for trying values side by side.

**Region pass** (Settings → Regions, off by default) re-renders each subject region on its own after the picture is made, like inpainting it again: the region is cropped with some context, rendered at up to ~1 megapixel with only its own prompt and negative, and blended back with a soft edge. Characters get the detail of a single-character image (faces, eyes, clothing) and leftover bleed gets a second chance; it doesn't move or resize anything (measured: the same regions hit as without it). *Region pass denoising* ~0.35–0.4 sharpens and cleans up; 0.5+ can also fix a wrong colour but may change poses. Regions run one at a time from the bottom layer up; scenery regions are skipped; sketch guides follow into each crop; ADetailer runs afterwards. Cost: one partial pass per region. Not used in live mode.

To redo one region of a finished picture, select its area with the selection tools and *Refine* the selection with the usual inpainting; the region prompts apply there too.

### Prompting sketch layers

Write a prompt in a paint or guide layer's *Layer prompt* field. That layer's strokes then act as a region, which is handy for "sketch each character on its own layer". *Region spread* sets how far around the strokes the prompt reaches. *Fill enclosed shapes* makes a closed outline count as its whole interior. Tick *Show region* to see the area that will be used. Paint layers stay part of the image. Guide layers with a prompt both steer the lines and prompt the area.

### Sketch guides

Add a **Guide** layer, pick a model (Venno prefers `any-test-like` / `lineart` models), and draw dark lines. The guide steers composition without being painted into the image, so it works even at 100 % strength. You can combine it with paint layers at lower strength to control colours as well. For scribble models, *Line polarity: Auto* sends white-on-black lines automatically. Lower *Guide until* if the result follows wobbly lines too literally.

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
- Regions: describe each character fully in its region prompt (`1girl, red hair, …`) and the scenery in a region of its own; the main prompt is mostly for style and quality tags (it applies inside regions at 10%). Count tags like `2girls` go in the main prompt, if anywhere, never in a region.

## Text to Image / Image to Image

These are Forge-style form pages: prompt, negative, LoRA chips, preset/sampler/scheduler/steps/CFG/shift, size presets, batch size and count, and seed reuse.

- **Hires. fix (Text to Image):** generate small, upscale with a latent mode or an upscaler model, then refine. Set the upscale factor or a target width/height, plus separate hires steps, denoising, CFG and shift (0 = same as the first pass).
- **Image to Image / Inpaint:** drop, paste or send in an image; it is shown large next to the result. Size is *Resize by* a scale (e.g. 150 %) or *Resize to* a width/height. On **Inpaint**, paint the mask (wheel = brush size); settings include masked content, whole picture vs only masked (with padding), mask blur and seamless inpainting.
- **Sketch guide (Inpaint):** switch the image pane from *Mask* to *Sketch* and draw lines over the picture; pick a guide model under *Sketch guide* and the regenerated area follows your lines. Only the masked area changes. If you haven't painted a mask, the area around the sketch (closed shapes filled) is masked for you, or press *Mask around sketch* to see and adjust it first.
- **Outpaint (Image to Image):** tick *Outpaint* and set how many pixels to add left / right / top / bottom (or use the quick buttons). The input pane shows the new border as a checkerboard. The border is filled from the picture's edge colours and generated at the card's own denoising (1 = from scratch), blending *Overlap* pixels into the original; everything else stays exactly as it was. Use *Use as input* on a result to keep extending.
- **ControlNet (all form tabs):** tick *ControlNet*, drop a picture on a unit (or, on Image to Image / Inpaint, tick *Use the input image*) and pick a **preprocessor**: *Canny edges*, *Line art*, *Soft edge / scribble* or *Depth*. The map it makes is shown next to the picture, and the guide model trained on that kind of map is selected for you (any-test for edges, line art and soft edges, depth for depth; download them in ⚙ Settings → Models). On current Anima, *any-test v2* follows line maps much more closely than the older lineart / scribble models. *None* takes a picture that already is a map. Each unit has its strength, the part of the steps it guides (*Guide from / until*), how the picture is fitted to the output (*Crop and resize*, *Just resize*, *Resize and fill*) and a **control mode**: *Balanced*; *Prompt* (my prompt is more important: the guide fades over the model's later blocks, so it sets the layout and the prompt decides the details); *ControlNet* (the guide acts on the prompt side only, so CFG amplifies it; needs CFG above 1). Up to three units can be combined, e.g. depth for the scene and line art for a character. Depth Anything V2 Small (Apache-2.0, ~100 MB) downloads into `models/preprocessors` on first use; Canny, line art and soft edge need no download. Guides also steer the hires pass and the region pass. There is no pose preprocessor yet (draw a skeleton on a Canvas guide layer or use a ready-made DWPose map with preprocessor *None*).
- **Consistent set (Text to Image):** tick it and pick 2–4 frames. The frames are sampled as one sequence (Anima is built on NVIDIA's Cosmos video model and still has its time axis), so they come out as the same character in the same style, with small, video-like differences. Text in a frame's box is added in front of the prompt for that frame only (`smile`, `angry`, `from side`…). Frames sampled together tend to average their differences, so with frame texts *Keep frames apart* damps how much the frames look at each other: higher follows each frame's text more, lower keeps them more alike (it is also a grid axis, *Frame separation*, and a grid cell shows the whole set). Batch size counts sets: batch 2 with 3 frames gives 6 pictures. Each step costs more with every frame (about 2× for 2 frames, 5× for 4). Hires fix, guides, ADetailer and the region pass work per set or per picture as usual; the Canvas doesn't use sets.
- **Reference (every tab):** add up to three pictures (*Add picture…*, drop one on the card, or *Use as reference* on a result); on Image to Image / Inpaint you can also use the input image. Each reference becomes an extra, clean frame the generated picture attends to, so it takes over who and what is in it. Works best when the prompt also describes the character. At full *Reference strength* the result stays close to the reference, pose and framing included (good for variations and for Edit LoRAs); lower it to about 80–90 % to let the prompt set the scene and pose while the reference lends its look (hair, eyes, outfit, colours). The range is narrow: around 70 % the reference stops mattering. *Reference noise* is how noisy the model is told the references are: lower copies them more closely (below ~0.3 a dotted pattern can appear), higher follows the prompt more; 0.5 is a good start. *Follow the steps* gives the references the picture's own noise level at every step, which is how the community **Anima Edit LoRAs** were trained: add the Edit LoRA, use the input image as the reference, set Denoising to 1 and write the change in the prompt.
- **ADetailer:** up to two detectors (e.g. `face_yolov8n.pt` + `hand_yolov8n.pt`), each with its own denoising, confidence, mask blur, dilation, padding and optional detail prompt.
- **Results:** *Open in Canvas* adds the image as a layer; *Send to Image to Image* / *Send to Inpaint* pass it on **with the prompt, LoRAs, seed, sampler settings and ADetailer** of the tab it came from; *Use settings* loads its parameters; *Download* saves it. Results are also saved to `outputs/` unless you turn that off.
- **Option cards:** the checkbox turns hires fix / ADetailer / the grid on and off; clicking the title only opens its settings. The Canvas has the same three cards under its prompt panel (not used in live mode; hires fix applies when the Canvas makes a new picture, and grids open in a viewer instead of on the canvas).

## API

Studio talks to the server only through the Venno API under `/venno/v1`. Every generation mode uses one job format; see `Job` in [`venno/server/jobs.py`](venno/server/jobs.py) (interactive docs at `/venno/docs`).

```jsonc
POST /venno/v1/jobs
{
  "mode": "txt2img",                     // or "img2img" (+ init_image, strength, mask, …)
  "model": { "checkpoint": "anima-turbo-v1.1.safetensors", "text_encoder": "qwen_3_06b_base.safetensors", "vae": "qwen_image_vae.safetensors" },
  "prompt": "masterpiece, best quality, safe, 1girl, …", "negative": "…",
  "width": 1024, "height": 1024, "batch": 2, "seed": -1,
  "sampler": "Euler", "scheduler": "Simple", "steps": 10, "cfg": 1.0, "shift": 3.0,
  "regions":  [{ "prompt": "…", "negative": "", "mask": "<png>", "kind": "auto" }],   // kind: auto | scenery | subject
  "guides":   [{ "model": "anima-lllite-depth-1", "image": "<png>", "weight": 0.8, "preprocessor": "depth", "mode": "balanced", "resize_mode": 1 }],
  "hires":    { "upscaler": "Latent", "scale": 1.5, "denoise": 0.5 },
  "frames": 1, "frame_prompts": [],      // consistent set (text to image): 2–4 frames sampled together
  "references": ["<png>"], "reference_noise": 0.5, "reference_follow": false, "reference_strength": 1.0,
  "detailer": [{ "model": "face_yolov8n.pt", "denoise": 0.4 }]
}
```

- `GET /venno/v1/jobs/{id}`: status, progress, seeds, infotexts and image URLs (`/venno/v1/jobs/{id}/images/{n}.png`)
- `POST /venno/v1/jobs/{id}/cancel`: cancel a queued or running job
- `WS /venno/v1/events`: pushes every job update plus live preview frames
- `GET /venno/v1/info`: models, modules, LoRAs, samplers, schedulers, upscalers, detectors, presets
- `/venno/v1/models/catalog`, `/download`, `/jobs`, `/cancel`: model downloads

The old A1111/Forge `/sdapi` endpoints are gone (see [`docs/PLAN_VENNO_SERVER.md`](docs/PLAN_VENNO_SERVER.md)).

## Project layout

| Path | |
|---|---|
| `venno.sh`, `venno.bat` | Launchers (venv, install check, start) |
| `studio/` | Studio front end (plain ES modules, no build step) |
| `venno/__main__.py`, `venno/config.py`, `venno/install.py` | Server start-up, options and folders, package installer |
| `venno/server/` | FastAPI app (`/venno/v1`, Studio files), job queue and schema, model downloader |
| `venno/pipeline/` | Generation: passes, hires, region pass, detailer, grids, image metadata |
| `venno/engine/` | Model manager, sampling, schedulers, prompts, regions, LoRA, guides, upscalers, detectors, tags, wildcards |
| `backend/` | Diffusion engine forked from Forge Neo (Anima only); third-party code in `backend/vendor/` |
| `tests/`, `venno/smoke.py` | Unit tests and the smoke test (see below) |

## Development

- **Unit tests** (no model needed, a few seconds): `pip install -r requirements-dev.txt`, then `python -m pytest`. They cover the prompt parser, schedulers, region / frame weights, masks and resizing, job validation, saved settings (infotext), wildcards, grid values, the LoRA library, token counts, preprocessors and the detailer report.
- **Smoke test** (runs every mode for real): `python -m venno.smoke` with the usual options (e.g. `--models-dir …`, `--cpu`). It picks an Anima Turbo checkpoint when there is one and runs text to image, seeds and batches, several samplers, image to image, inpainting (whole picture and only masked), hires fix, prompt scheduling and AND, wildcards, regions (with own passes), the region pass, a LoRA, guides with preprocessors and control modes, ADetailer, consistent sets, reference pictures and a grid, at 256² × 4 steps. Each check verifies what must hold (e.g. inpainting leaves the unmasked pixels untouched, the same seed gives the same picture); checks whose models are missing are skipped. A contact sheet of the results goes to `outputs/smoke/`. About 20 minutes on a CPU (a minute or two on a GPU); `--only regions,sets` runs some checks, `--size` / `--steps` change the work per check. Exits with 1 when a check fails.
- **Region lab** (compares regional-prompting settings on one layout): `python -m venno.regionlab --scene living --variants single,own30 --seeds 1,2,3,4` writes a labelled sheet to `outputs/region_lab/` and scores every picture with the detailer's face and person detectors (how many character regions hold a character, and how many characters landed outside every region), so settings can be compared over many seeds instead of by eye; scenes can be JSON files. See `docs/REGIONS_HANDOFF.md` for the measurements and what is still open.

## Credits & licence

- Engine core (`backend/`): derived from the engine of [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) by Haoming02, which builds on [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) by lllyasviel and [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) by AUTOMATIC1111. Model loading, memory management, the attention back ends, LoRA patching and the Anima network come from there; Venno trimmed it to Anima and added the hooks its features use (regions, time-axis frames and references, guides). Third-party packages it bundles are in `backend/vendor/`. Forge Neo's README is kept in [`docs/FORGE_NEO_README.md`](docs/FORGE_NEO_README.md).
- Anima 3.8B support (`venno/engine/anima38/`): ported from [forge-anima-3.8B](https://github.com/GumGum10/forge-anima-3.8B) by GumGum10 (MIT); the model is [Anima 3.8B](https://huggingface.co/lylogummy/Anima-3.8B) by lylogummy, and Qwen3.5's tokenizer comes from [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0).
- Guide models: [Anima ControlLLLite](https://huggingface.co/Mooshie/Anima-LLLite) by Mooshie (Anima licence). Depth preprocessor: [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) (Apache-2.0), downloaded on first use.
- Model: [Anima](https://huggingface.co/circlestone-labs/Anima) by CircleStone Labs & Comfy Org (non-commercial model licence; generated images may be used commercially).
- Inspired by [Krita AI Diffusion](https://github.com/Acly/krita-ai-diffusion).

Licensed under the [AGPL-3.0](LICENSE), as are the projects its engine derives from.
