# Regional prompting: handoff (October 2026)

Notes for picking up the regional-prompting work. The first version of this file came out of a cloud session
that could only test on a CPU with Anima Turbo; everything below was then re-measured on the user's RTX 4090
with their settings, and the method and the Canvas were reworked on that basis.

## Goal

The user builds whole pictures from regions only on the Canvas: one region for the scenery over the whole
canvas, one per character and one per object, drawn as rectangles. The main prompt is left empty (with *Add
Anima quality tags* on, it is just the quality tags). What they want:

- each region shows what its prompt says, where the region is;
- subjects blend into the picture like one painting (natural size and lighting), not pasted into boxes;
- free-form composition: a region says roughly where something goes, it isn't a frame to fill;
- and all of that with the defaults, for people who don't want to tinker.

Their usual settings: Anima base, preset *Anima Base / Aesthetic*, ER SDE + Beta, 32 steps, CFG 4, shift 3,
1024², often with the LoRA *Anima Furry Domain* at 0.8 and characters described as kobolds (base Anima
doesn't know "kobold", so leave the LoRA out of tests that don't need it).

## The method (one, with tuned defaults)

1. **Regional attention** at every step: each block's text cross-attention runs once with the main prompt and
   once per region prompt, blended per image token by the region masks (`venno/engine/regions.py`,
   `RegionalAttention`, hooked in `backend/nn/anima.py`). The main prompt applies inside regions at 10%
   (`region_base`, fixed in the Canvas).
2. **Own passes in the first 30% of the steps**: every region also gets a model pass with only its prompt on a
   crop around it (grown by half its size, at least 32 latents a side), used inside the region
   (`venno/engine/sampling.py`, `Denoiser.couple`, `_couple_box`; job field `region_couple`). This is what makes
   a second character appear where its region is. On in live mode too.
3. **Keep regions apart** (`separation`, 70%): self-attention between different subject regions damped in the
   middle steps.
4. **Scenery vs subject**: a region is scenery when its box spans more than 60% of the picture
   (`RegionalAttention.BACKGROUND_BOX`) or when the job says so (`Region.kind`: `auto` / `scenery` /
   `subject`, the Canvas's *Auto* select). Scenery gives way to subjects where they overlap, is never kept
   apart from them, gets no own-pass crop (its pass runs on the whole picture) and is skipped by the region
   pass.
5. **Region pass** (optional, off): each subject region re-rendered on its own afterwards at a moderate
   denoise (`venno/pipeline/region_pass.py`). Detail only; it doesn't move anything.

What the Canvas sends (`studio/js/generate.js` `buildJob`): region prompt = the region's text with the quality
tags in front (when *Add Anima quality tags* is on), never the main prompt; region negative = its own text
plus the main negative; masks with *higher layer wins* (upper regions cut out of lower ones, feathered);
`kind` from the layer. Settings left: *Region own passes*, *Keep regions apart*, *Region pass* and its
denoise. Removed in the rework: *Main prompt inside regions*, *Region guidance until*, *Add the main prompts
to region prompts*, *Overlapping regions* (blend), *Region edges*, *Only regenerate the regions* (and the
server's `regions_only` / `region_context` / `region_edges` fields, `region_pass.falloff_mask`, `blank_under`,
`surroundings`, `_steering`).

## Measurements (RTX 4090, Anima base, 1024², 32 steps, CFG 4, ER SDE / Beta)

Scored by `venno.regionlab` (see below): characters found in their regions / character regions, plus
characters outside every region. `living` layout: scenery, fireplace, a girl sitting on a couch (left), a
girl lying on a rug in a thin strip at the bottom. 8 seeds unless noted.

| Variant | In their regions | Extra | Note |
|---|---|---|---|
| single pass (own passes 0) | 11/16 | 0 | the sitting girl grows, her legs fill the strip, or a cat/fox appears there |
| **own30** (default) | **13/16** | 0 | |
| own30 + quality tags on region prompts | 13/16 | 1 | neutral; the Canvas adds them (a bare region prompt is otherwise unstyled) |
| own40 | 12/16 | 0 | |
| own30, crops at least 48 latents and aspect ≤ 1.6 | ~11/16 | 1 | bigger crops → smaller subjects; twice 0/2 |
| own30, small crops upscaled to 64 latents | 11/16 | 0 | odd poses |
| own30 fading out over the next 20% of the steps | 11/16 | 0 | |
| own30, own pass reaching 25% past the box | 9/16 | 2 | subjects wander and duplicate |
| own30, 25% past the box with a soft edge | 11/16 | 1 | |
| own30 + region pass 0.4 | 13/16 | 0 | same placements, sharper |

With the furry LoRA at 0.8 on the user's reported layout (2 seeds × 2 scenes): single pass drew one big
character from a low angle in 4/4 (exactly the complaint), own30 drew both characters in their boxes in 7/8.
`witch` (moon, witch, a small bat): every region right in 4/4 for both variants, so "small regions are
ignored" from the CPU tests does not hold on the real model. A main prompt describing the layout in words
("two girls, one sitting on the left, …") added the second girl in the single pass but didn't place her, and
changed nothing with own passes: not worth building.

The remaining failure is a subject whose natural size is bigger than its region: a girl lying in a strip 16%
of the picture high comes out small, sketchy or child-like, and wide-shot scenery makes it worse (the rug is
far away). Everything tried to let her grow past the box scored worse. The honest advice is in the README:
make the region as big as the subject.

## Open ideas (untested)

- Own passes with a higher CFG, or with `solo` / a count negative, for the subject's crop.
- A per-region own-pass share: longer for small regions, shorter for big ones (own40 overall was no better).
- Attribute drift with own passes (a blonde witch's hair turned silver in one seed): a region negative is the
  workaround.
- The detector score misses anime faces from behind / side; the person detector at 0.2 covers some of it.
  Furry characters: score by eye.

## How to test

```
python -m venno.regionlab --scene living --variants single,own30 --seeds 1,2,3,4,5,6,7,8
python -m venno.regionlab --scene witch --variants own30,own30-r40 --seeds 1,2 --with-lora "Anima Furry Domain .11:0.8"
python -m venno.regionlab --scene my_layout.json --variants own30,own30-q --seeds 1,2,3
python -m venno.regionlab --help-lab
```

- Defaults match the user's settings (1024², 32 steps, CFG 4, ER SDE / Beta, main prompt inside regions 10%,
  keep regions apart 75%); `--checkpoint` picks the model (the first found otherwise; here the 2.9B sorts first,
  so pass `anima_baseV10.safetensors`). `--lora` can't be used: the server's own parser reads it as `--lora-dir`.
- Variants: `single`, `ownNN` with suffixes `-l48` (crop at least 48 latents), `-s0.5` (crop grow share), `-q`
  (quality tags on region prompts), `-rNN` (region pass at NN%). The Canvas sends own30 with `-q`.
- Scenes: `witch`, `living`, `reported` (the user's layout as reported), or a JSON file (format in the script's
  docstring; boxes are fractions of the picture, bottom layer first, the first region usually the scenery).
- Every picture is scored with the detailer's face and person detectors (`face_yolov8n.pt`,
  `person_yolov8n-seg.pt` in `models/adetailer`, downloaded on first use): a character region counts when a
  face lies in its box (grown by 5%) or at least 30% of a detected person's pixels do; a face outside every
  character region is an extra. `--no-score` skips it. The summary line per variant is the number to compare;
  ±2 over 16 is noise.
- Results are cached per scene, variant, seed and settings in `outputs/region_lab/cache`, and each run writes
  a labelled sheet with the region boxes drawn in to `outputs/region_lab/`. About 20 s per picture on the 4090.

Before committing: `python -m pytest` (unit tests) and `python -m venno.smoke --only regions,region_pass`
(or the whole smoke test, about a minute on a GPU).

## Conventions

- Work on the branch `ccr-7369d750-va6fhe`: the repository's only branch, so `git pull` gets it.
- Commit messages end with the `Co-Authored-By` / `Claude-Session` lines used throughout the history.
- After pulling, restart Venno; the Studio shows a banner when the server's code changed. The Studio's files
  are versioned per change, so a reload is enough on the browser side.
- Keep README's *Regions* section in step with behaviour changes, and tell the user what the test sheets show,
  including what still fails.
