# Regional prompting: handoff (October 2026)

Notes for picking up the regional-prompting work, written at the end of a cloud session that could only test on a
CPU. Everything measured so far needs re-checking on a real GPU with the user's settings (see *Caveats*).

## Goal

The user builds whole pictures from regions only on the Canvas: one region for the background (scenery over
the whole canvas), one per character and one per object, drawn as rectangles. The main prompt is left empty
(with *Add Anima quality tags* on, it is just the quality tags). What they want:

- each region shows what its prompt says, where the region is;
- subjects blend into the picture like one painting (natural size and lighting), not pasted into boxes;
- free-form composition: a region says roughly where something goes, it isn't a frame to fill.

Their usual settings: Anima base, preset *Anima Base / Aesthetic*, ER SDE + Beta, 32 steps, CFG 4, shift 3,
1024², the LoRA *Anima Furry Domain* at 0.8, often characters described as kobolds. Base Anima doesn't know
"kobold", so don't use it in tests without the LoRA.

Decision so far: the single-pass method (regional attention) is the one to improve. The multi-pass methods
below stay as tools, not as the way to build pictures.

## Where things are

| What | Where |
|---|---|
| Single pass: per-region cross-attention blend, background rule, *Keep regions apart* | `venno/engine/regions.py` (`RegionalAttention`: `token_masks`, `backgrounds`, `weights`, `self_bias`, `contexts`), called from `backend/nn/anima.py` |
| Region own passes (cropped latent couple in the first steps) | `venno/engine/sampling.py` (`Denoiser.couple`, `_couple_box`); job field `region_couple`; Canvas *Region own passes (first steps)*, default 30% |
| Only regenerate the regions (one region at a time, bottom up) | `venno/pipeline/region_pass.py` (`areas`, `surroundings`, `_steering`, `falloff_mask`, `blank_under`), `regions_only` branch in `venno/pipeline/generate.py`; job fields `regions_only`, `region_context`, `region_edges` |
| Region pass (refine each region after generating) | `region_pass.run` with `job.region_refine` |
| What the Canvas sends | `studio/js/generate.js` `buildJob` (region masks via `regionMasks`: *Top layer wins* cuts upper regions out of lower ones, *Blend* sends full shapes), settings in `studio/js/main.js` and `studio/index.html` |
| Comparison script | `venno/regionlab.py` (below) |

## What was tried, and what came out

Test pictures: Anima Turbo at CFG 1, 8 steps, 512², one or two seeds, no LoRA (see *Caveats*).

1. **Only regenerate the regions, hard squares at 100%.** Regions drew random scene content. The crop around
   each region was tiny (5%) and the prompt was region + whole main prompt, so the model drew a little copy
   of the main scene in the box. Fixed: more context around each region (the Canvas *Context*), the crop is
   prompted with the main prompt while the region's own prompt steers the region through regional attention
   (main prompt inside the region: 0), and region prompts don't get the main prompt added in this mode.
2. **Building a picture region by region** (background first, then each region): with only quality tags as
   the main prompt, Anima put a girl in the moon and bat boxes. Fixed by describing each crop's surroundings
   with the main prompt plus the regions already painted below (`surroundings`). Still box-shaped: a subject
   is stretched to fill its square, what was under the square is lost, and the squares show.
3. **Soft region edges** (`falloff_mask`: full strength in the middle, 35% at the border): subjects take a
   natural size and blend in. But they come out small, and the background decides their pose and size.
   Single pass followed by the regions at 75% with soft edges looked best of the multi-pass variants.
4. **User tests of 2–3 went badly:**
   - Soft edges on an empty canvas came out washed out. The background region was faded around the holes of
     the regions above it, and a white canvas was "kept". Fixed: a background region is always repainted in
     full, soft edges are skipped over a blank canvas, and regions are sent in their full shape.
   - 75% strength on a white canvas came out pale. The Canvas label now warns about that.
   - The user then decided to go back to the single pass.
5. **Region own passes** (latent couple in the first steps): every region also gets a model pass with only
   its prompt, used inside the region, so its subject starts forming there.
   - On the whole picture: things landed in their boxes more often, but small subjects were still missed.
   - On a crop around the region (grown by half its size, at least 32 latents): a second character that the
     plain single pass never drew appeared in both seeds, and in one seed all four regions were right.
   - 30% of the steps is the sweet spot; 50% merged two characters into one.
   - Cost: about +30% time with four regions.
6. **Blend prompts with a background region over everything:** the background was blended 50/50 into every
   character region, which halved the character prompts. Fixed: background regions (box over 60% of the
   picture) always give way to the other regions. A background prompt that contradicts the layout ("couch
   right" while the region sitting on the couch is on the left) clearly hurt; a matching one put the couch,
   fireplace and rug where the regions said.

## Open problems

- **Small or thin regions are still ignored** in the single pass, even with own passes: a bat in the sky, a
  girl lying in a thin strip at the bottom. The only runs where the bat appeared were the region-by-region
  inpainting ones, where the crop was **enlarged** before drawing. Next idea: in `Denoiser.couple`, upscale
  small crops (e.g. to at least 64 latents a side), run the region's own pass there, and scale the
  prediction back down.
- **Attribute drift with own passes:** in one seed the witch's blonde hair turned silver.
- **LoRA composition bias:** with the furry LoRA at 0.8 both of the user's pictures were one big character
  filling the frame from a low angle. Compare LoRA weights (0.4 / 0.6 / 0.8) on the same layout and seed.
- **No count in the main prompt:** with only quality tags, nothing says "two characters". An untested idea is
  to add a count (`2girls`, …) to the main prompt from the character regions.
- **Own passes with a Guide layer or reference pictures** run on the whole picture, not a crop (guides and
  references are sized for the whole picture).

## Caveats

Everything above was measured on a CPU with Anima Turbo at CFG 1 (negative prompts have no effect there),
8 steps, 512², one or two seeds per variant, without the user's LoRA. Regions are twice as big in pixels at
1024², CFG 4 follows prompts more closely, and the base model behaves differently from Turbo. Treat the
findings as leads and re-measure them with the user's settings before building on them.

## How to test

```
python -m venno.regionlab --scene living --variants single,own30 --seeds 1,2,3
python -m venno.regionlab --scene witch --variants single,own30,refine75,build --seeds 1,2 --lora "LoRA name:0.8"
python -m venno.regionlab --scene my_layout.json --variants single,own30 --seeds 1,2,3 --overlap blend
python -m venno.regionlab --help-lab
```

- Defaults match the user's settings (1024², 32 steps, CFG 4, ER SDE / Beta, main prompt inside regions 10%,
  keep regions apart 75%); `--checkpoint` picks the model.
- Scenes: `witch`, `living`, `reported`, or a JSON file (format in the script's docstring; boxes are
  fractions of the picture, bottom layer first, the first region usually the scenery).
- Results are cached per scene, variant, seed and settings in `outputs/region_lab/cache`, and each run writes
  a labelled sheet with the region boxes drawn in to `outputs/region_lab/`.
- Compare variants on the same seeds and show the user the sheets; judge by whether each region shows its
  subject, in place, blended in.

Before committing: `python -m pytest` (unit tests) and `python -m venno.smoke --only regions,region_pass,regions_only`
(or the whole smoke test, about a minute on a GPU).

## Conventions

- Work on the branch `ccr-7369d750-va6fhe`: the repository's only branch, so `git pull` gets it.
- Commit messages end with the `Co-Authored-By` / `Claude-Session` lines used throughout the history.
- After pulling, restart Venno; the Studio shows a banner when the server's code changed. The Studio's files
  are versioned per change, so a reload is enough on the browser side.
- Keep README's sections on *Region own passes*, *Region pass* and *Only regenerate the regions* in step
  with behaviour changes, and tell the user what the test sheets show, including what still fails.
