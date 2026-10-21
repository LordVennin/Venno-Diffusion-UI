"""Smoke test: run every generation mode once on a small picture and check the results.

    python -m venno.smoke [--checkpoint NAME] [--size 256] [--steps 4] [--only txt2img,regions] [Venno's options]

Meant for Anima Turbo (a few minutes on a CPU, seconds on a GPU); it picks a checkpoint with "turbo" in its
name when there is one. Each check runs a real job through the pipeline (venno/pipeline/generate.py) and checks
what came back: the number and size of the images, that they are not blank, and per mode what must hold
(an inpaint leaves the unmasked pixels alone, the same seed gives the same picture, …). Checks whose models
are missing (guides, LoRAs, detectors that can't be downloaded) are skipped. All results go on one contact
sheet in outputs/smoke/. Exits with 1 when a check fails.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
import traceback

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], add_help=False)
ap.add_argument("--checkpoint", default="", help="checkpoint file (default: one with 'turbo' in its name)")
ap.add_argument("--size", type=int, default=256, help="picture size (default 256)")
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--only", default="", help="comma-separated check names")
ap.add_argument("--help-smoke", action="store_true", help="show this help")
opts, _rest = ap.parse_known_args()  # Venno's own options (folders, --cpu, …) are read by venno.config / backend.args

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import backend  # noqa: E402,F401
from venno import config  # noqa: E402,F401
from venno.engine import detect, guides, lora, models  # noqa: E402
from venno.pipeline import generate, infotext  # noqa: E402
from venno.server.jobs import Job  # noqa: E402

PROMPT = "masterpiece, best quality, safe, 1girl, silver hair, red eyes, school uniform, smile, upper body, simple background"


class Skip(Exception):
    pass


def data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def pictures(result) -> list[Image.Image]:
    return [Image.open(io.BytesIO(b)).convert("RGB") for b in result.images]


def assert_picture(img: Image.Image, size: tuple[int, int]):
    assert img.size == size, f"size {img.size}, expected {size}"
    assert np.asarray(img).std() > 8, "the picture is (nearly) one flat colour"


class Smoke:
    def __init__(self):
        self.S = opts.size
        self.model = self.pick_models()
        self.base = dict(model=self.model, prompt=PROMPT, negative="worst quality, low quality", width=self.S, height=self.S,
                         steps=opts.steps, sampler="Euler", scheduler="Simple", cfg=1.0, shift=3.0, seed=11, save=False)
        self.shown: list[tuple[str, Image.Image]] = []
        self.first: Image.Image | None = None

    def pick_models(self) -> dict:
        ckpts = list(models.checkpoints())
        if not ckpts:
            sys.exit("No Anima checkpoint found (use --models-dir or --ckpt-dirs)")
        ckpt = opts.checkpoint or next((c for c in ckpts if "turbo" in c.lower()), ckpts[0])
        if ckpt not in ckpts:
            sys.exit(f'Checkpoint "{ckpt}" not found; there are: {", ".join(ckpts)}')
        mods = list(models.modules())
        te = next((m for m in mods if "qwen_3_06b" in m.lower() or "qwen3_06b" in m.lower()), "")
        vae = next((m for m in mods if "vae" in m.lower()), "")
        if not te or not vae:
            sys.exit("The Qwen3 0.6B text encoder and the Qwen-Image VAE are needed (models/text_encoder, models/VAE)")
        print(f"[smoke] {ckpt} · {te} · {vae} · {self.S}² · {opts.steps} steps")
        if "turbo" not in ckpt.lower():
            print("[smoke] not a Turbo checkpoint: the pictures will be rough at this step count, the checks still apply")
        return {"checkpoint": ckpt, "text_encoder": te, "vae": vae}

    def run(self, label=None, **kw):
        with torch.inference_mode():
            result = generate.run(Job(**{**self.base, **kw}), generate.Progress())
        assert not result.cancelled
        if label:
            for k, img in enumerate(pictures(result)):
                self.shown.append((f"{label}{f' {k + 1}' if len(result.images) > 1 else ''}", img))
        return result

    def source(self) -> Image.Image:
        if self.first is None:
            self.first = pictures(self.run())[0]
        return self.first

    # ---------------------------------------------------------------- checks

    def check_txt2img(self):
        r = self.run("txt2img")
        assert len(r.images) == 1
        assert_picture(pictures(r)[0], (self.S, self.S))
        s = infotext.settings(r.infotexts[0])
        assert s["seed"] == 11 and s["steps"] == opts.steps and s["sampler"] == "Euler", "settings written to the PNG"

    def check_seed(self):
        a, b = self.run(), self.run()
        assert a.images[0] == b.images[0] or np.abs(np.asarray(pictures(a)[0], int) - np.asarray(pictures(b)[0], int)).mean() < 1, "same seed, different picture"
        c = self.run(seed=12)
        assert np.abs(np.asarray(pictures(a)[0], int) - np.asarray(pictures(c)[0], int)).mean() > 2, "different seeds, same picture"

    def check_batch(self):
        r = self.run("batch", batch=2, count=2, width=self.S, height=self.S * 3 // 4 // 16 * 16)
        assert len(r.images) == 4 and r.seeds == [11, 12, 13, 14]
        for img in pictures(r):
            assert_picture(img, (self.S, self.S * 3 // 4 // 16 * 16))

    def check_samplers(self):
        for sampler, scheduler in [("Euler a", "Normal"), ("DPM++ 2M", "Karras"), ("Res Multistep", "Beta"), ("ER SDE", "Simple")]:
            r = self.run(sampler=sampler, scheduler=scheduler, cfg=3.0)
            assert_picture(pictures(r)[0], (self.S, self.S))

    def check_img2img(self):
        r = self.run("img2img", mode="img2img", init_image=data_url(self.source()), strength=0.5, prompt=PROMPT.replace("red eyes", "blue eyes"))
        out = pictures(r)[0]
        assert_picture(out, (self.S, self.S))
        diff = np.abs(np.asarray(out, int) - np.asarray(self.source(), int)).mean()
        assert 0.5 < diff < 60, f"image to image at 0.5 changed the picture by {diff:.1f}/255 on average"

    def check_inpaint(self):
        mask = Image.new("L", (self.S, self.S), 0)
        ImageDraw.Draw(mask).rectangle((self.S // 4, self.S // 4, self.S * 3 // 4, self.S * 3 // 4), fill=255)
        for only in (False, True):
            r = self.run("inpaint" + (" (only masked)" if only else ""), mode="img2img", init_image=data_url(self.source()), mask=data_url(mask),
                         strength=0.9, fill="fill", only_masked=only, mask_blur=4, prompt=PROMPT + ", flower hair ornament")
            out = np.asarray(pictures(r)[0], int)
            src = np.asarray(self.source(), int)
            m = self.S // 8
            assert (out[:m] == src[:m]).all() and (out[-m:] == src[-m:]).all(), "pixels outside the mask changed"
            c = slice(self.S * 3 // 8, self.S * 5 // 8)
            assert np.abs(out[c, c] - src[c, c]).mean() > 1, "the masked area did not change"

    def check_hires(self):
        r = self.run("hires", hires={"upscaler": "Latent", "scale": 1.5, "denoise": 0.5, "steps": max(2, opts.steps // 2)})
        assert_picture(pictures(r)[0], (self.S * 3 // 2, self.S * 3 // 2))

    def check_prompts(self):
        r = self.run("scheduling", prompt=PROMPT + ", [red:blue:0.5] background", switch_prompt=PROMPT + ", closed eyes", switch_at=0.5)
        assert_picture(pictures(r)[0], (self.S, self.S))
        r = self.run(prompt="1girl, standing AND red background:0.6", cfg=1.0)
        assert_picture(pictures(r)[0], (self.S, self.S))

    def check_wildcards(self):
        r = self.run(prompt="1girl, {red|blue|green} hair, smile", count=2)
        assert len(r.images) == 2 and "Template" in r.infotexts[0]

    def _halves(self):
        left, right = Image.new("L", (self.S, self.S), 0), Image.new("L", (self.S, self.S), 0)
        ImageDraw.Draw(left).rectangle((0, 0, self.S // 2, self.S), fill=255)
        ImageDraw.Draw(right).rectangle((self.S // 2, 0, self.S, self.S), fill=255)
        return data_url(left), data_url(right)

    def check_regions(self):
        left, right = self._halves()
        regions = [{"prompt": "1girl, red hair, red dress", "mask": left, "negative": "blue"}, {"prompt": "1girl, blue hair, blue dress", "mask": right}]
        r = self.run("regions", prompt="masterpiece, best quality, safe, 2girls, standing, park", regions=regions, region_separation=0.7, width=self.S * 3 // 2 // 16 * 16)
        img = pictures(r)[0]
        assert_picture(img, (self.S * 3 // 2 // 16 * 16, self.S))
        s = infotext.settings(r.infotexts[0])
        assert s["regions"] == [x["prompt"] for x in regions] and s["region_negatives"] == ["blue", ""]

    def check_region_pass(self):
        left, right = self._halves()
        r = self.run("region pass", prompt="2girls", regions=[{"prompt": "1girl, red hair", "mask": left}, {"prompt": "1girl, blue hair", "mask": right}], region_refine=0.4)
        assert_picture(pictures(r)[0], (self.S, self.S))

    def check_lora(self):
        names = list(lora.scan())
        if not names:
            raise Skip("no LoRA in the LoRA folders")
        r = self.run("LoRA", loras=[{"name": names[0], "weight": 0.8}])
        assert_picture(pictures(r)[0], (self.S, self.S))
        assert f"<lora:{names[0]}:0.8>" in r.infotexts[0]

    def check_guides(self):
        known = guides.models()
        line = next((m for m in known if "any-test" in m or "anytest" in m), None) or next((m for m in known if "lineart" in m), None)
        if not line:
            raise Skip("no line-art guide model (LLLite any-test / lineart)")
        for mode in ("balanced", "prompt"):
            r = self.run(f"guide ({mode})", guides=[{"model": line, "image": data_url(self.source()), "preprocessor": "lineart", "mode": mode, "resize_mode": 1, "weight": 1.0}])
            assert_picture(pictures(r)[0], (self.S, self.S))
        r = self.run("guide + hires", guides=[{"model": line, "image": data_url(self.source()), "preprocessor": "canny"}], hires={"upscaler": "Latent", "scale": 1.25, "denoise": 0.5})
        assert pictures(r)[0].size == (self.S * 5 // 4 // 8 * 8, self.S * 5 // 4 // 8 * 8)

    def check_detailer(self):
        try:
            detect.detect("face_yolov8n.pt", self.source(), 0.3)
        except Exception as e:  # no detector and no network
            raise Skip(f"face detector unavailable: {e}")
        r = self.run("ADetailer", detailer=[{"model": "face_yolov8n.pt", "denoise": 0.4}])
        assert_picture(pictures(r)[0], (self.S, self.S))
        assert r.detections[0] and r.detection_notes[0].startswith("face_yolov8n.pt"), "detection preview and report"

    def check_sets(self):
        r = self.run("set", frames=2, frame_prompts=["smile", "angry"])
        assert len(r.images) == 2 and r.seeds == [11, 11]
        a, b = pictures(r)
        assert_picture(a, (self.S, self.S))
        assert np.abs(np.asarray(a, int) - np.asarray(b, int)).mean() > 0.5, "the frames are identical"
        assert infotext.settings(r.infotexts[1])["frame_prompts"] == ["smile", "angry"]

    def check_references(self):
        ref = data_url(self.source())
        for name, extra in [("reference", {}), ("reference (follow)", {"reference_follow": True}), ("reference (0.85)", {"reference_strength": 0.85})]:
            r = self.run(name, prompt="masterpiece, best quality, safe, 1girl, standing, city at night", references=[ref], **extra)
            assert_picture(pictures(r)[0], (self.S, self.S))

    def check_grid(self):
        r = self.run("grid", grid={"x": {"type": "cfg", "values": "1, 2"}, "y": {"type": "steps", "values": f"{max(2, opts.steps - 1)}, {opts.steps}"}})
        assert len(r.images) == 1 and 'Grid: "' in r.infotexts[0]
        w, h = pictures(r)[0].size
        assert w > 2 * self.S and h > 2 * self.S, "a 2 × 2 grid with labels"


def contact_sheet(shown, path):
    if not shown:
        return
    cell = 192
    cols = min(6, len(shown))
    rows = (len(shown) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 18)), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for k, (label, img) in enumerate(shown):
        x, y = (k % cols) * cell, (k // cols) * (cell + 18)
        thumb = img.copy()
        thumb.thumbnail((cell, cell))
        sheet.paste(thumb, (x + (cell - thumb.width) // 2, y + 18 + (cell - thumb.height) // 2))
        draw.text((x + 4, y + 3), label, fill=(230, 230, 230))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path)


def main() -> int:
    if opts.help_smoke:
        print(__doc__)
        return 0
    smoke = Smoke()
    checks = [n[len("check_"):] for n in dir(Smoke) if n.startswith("check_")]
    order = ["txt2img", "seed", "batch", "samplers", "img2img", "inpaint", "hires", "prompts", "wildcards", "regions", "region_pass", "lora", "guides", "detailer", "sets", "references", "grid"]
    checks = sorted(checks, key=lambda c: order.index(c) if c in order else 99)
    if opts.only:
        wanted = {c.strip() for c in opts.only.split(",")}
        unknown = wanted - set(checks)
        if unknown:
            sys.exit(f"Unknown checks: {', '.join(sorted(unknown))} (there are: {', '.join(checks)})")
        checks = [c for c in checks if c in wanted]
    results = []
    started = time.time()
    for name in checks:
        t = time.time()
        try:
            getattr(smoke, f"check_{name}")()
            status, note = "PASS", ""
        except Skip as e:
            status, note = "SKIP", str(e)
        except Exception as e:
            status, note = "FAIL", f"{type(e).__name__}: {e}"
            traceback.print_exc()
        results.append((name, status, time.time() - t, note))
        print(f"[smoke] {status:4}  {name:12} {time.time() - t:6.1f}s  {note}", flush=True)
    sheet = os.path.join(config.outputs_path(), "smoke", time.strftime("%Y%m%d-%H%M%S") + ".png")
    contact_sheet(smoke.shown, sheet)
    failed = [r for r in results if r[1] == "FAIL"]
    print(f"\n[smoke] {len(results) - len(failed) - sum(r[1] == 'SKIP' for r in results)} passed, {len(failed)} failed, "
          f"{sum(r[1] == 'SKIP' for r in results)} skipped in {time.time() - started:.0f}s · contact sheet: {sheet}")
    for name, _, _, note in failed:
        print(f"[smoke] FAIL {name}: {note}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
