"""Region lab: generate one region layout several ways, side by side, to compare regional-prompting methods.

    python -m venno.regionlab --scene living --variants single,own30 --seeds 1,2 [options] [Venno's options]
    python -m venno.regionlab --scene my_layout.json ...

A scene is a main prompt plus regions, bottom layer first, each a box (fractions of the picture: x0, y0, x1, y1)
with a prompt and an optional negative. Built in: witch (scenery, moon, witch, bat), living (living room,
fireplace, a girl sitting on the couch, a girl lying on the rug), reported (the living-room layout as reported,
with a background prompt that matches it). A JSON scene file looks like
    {"main": "masterpiece, best quality, safe", "negative": "worst quality",
     "regions": [{"box": [0, 0, 1, 1], "prompt": "living room, night"},
                 {"box": [0.1, 0.2, 0.35, 0.8], "prompt": "1girl, sitting on couch", "negative": ""}]}

Variants (see docs/REGIONS_HANDOFF.md):
    single      one pass, regional attention only (Region own passes 0)
    own30/own50 one pass with region own passes in the first 30% / 50% of the steps (the Canvas default is 30%)
    refine75    single, then Only regenerate the regions at 75% with soft edges
    build       Only regenerate the regions at 100% with soft edges, from an empty canvas
Results are cached per scene, variant, seed and settings (outputs/region_lab/cache), so re-running a sheet only
computes what is new; each run writes a labelled sheet to outputs/region_lab/.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], add_help=False)
ap.add_argument("--scene", default="living", help="built-in scene (witch, living, reported) or a JSON file")
ap.add_argument("--variants", default="single,own30", help="comma-separated variants")
ap.add_argument("--seeds", default="1,2", help="comma-separated seeds")
ap.add_argument("--checkpoint", default="", help="checkpoint (default: the first one found)")
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--cfg", type=float, default=4.0)
ap.add_argument("--shift", type=float, default=3.0)
ap.add_argument("--sampler", default="ER SDE")
ap.add_argument("--scheduler", default="Beta")
ap.add_argument("--lora", action="append", default=[], help="name:weight, may be repeated")
ap.add_argument("--overlap", choices=["top", "blend"], default="top", help="Overlapping regions on the Canvas")
ap.add_argument("--region-base", type=float, default=0.1, help="Main prompt inside regions")
ap.add_argument("--separation", type=float, default=0.75, help="Keep regions apart")
ap.add_argument("--context", type=float, default=0.5, help="Context (regions only)")
ap.add_argument("--help-lab", action="store_true", help="show this help")
opts, _rest = ap.parse_known_args()  # Venno's own options (folders, --cpu, …) are read by venno.config / backend.args

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import backend  # noqa: E402,F401
from venno import config  # noqa: E402,F401
from venno.engine import models  # noqa: E402
from venno.pipeline import generate  # noqa: E402
from venno.server.jobs import Job  # noqa: E402

QUALITY = "masterpiece, best quality, safe"
SCENES = {
    "witch": {"main": QUALITY, "regions": [
        {"box": [0, 0, 1, 1], "prompt": "graveyard, tombstones, pumpkins, dark blue fog, night sky"},
        {"box": [0.04, 0.04, 0.39, 0.39], "prompt": "full moon, large moon, glowing"},
        {"box": [0.37, 0.21, 0.66, 0.98], "prompt": "1girl, witch, witch hat, blonde hair, red eyes, holding staff, black cape"},
        {"box": [0.70, 0.08, 0.96, 0.29], "prompt": "bat \\(animal\\), flying"},
    ]},
    "living": {"main": QUALITY, "regions": [
        {"box": [0, 0, 1, 1], "prompt": "living room, warm lighting, couch, night, fluffy rug, scenery", "negative": "1girl"},
        {"box": [0.61, 0.29, 0.80, 0.64], "prompt": "fireplace, fire"},
        {"box": [0.12, 0.24, 0.35, 0.80], "prompt": "1girl, black hair, frilly dress, sitting on couch, legs crossed"},
        {"box": [0.45, 0.64, 0.86, 0.80], "prompt": "1girl, blonde hair, frilly dress, lying on stomach, on rug"},
    ]},
    "reported": {"main": QUALITY + ", score_7", "regions": [
        {"box": [0, 0, 1, 1], "prompt": "Living room, couch left, fluffy rug right, fireplace,"},
        {"box": [0.12, 0.15, 0.33, 0.80], "prompt": "1girl, black skin, frilly clothes, sitting on couch, legs crossed,"},
        {"box": [0.60, 0.30, 0.75, 0.65], "prompt": "fireplace", "negative": "1girl"},
        {"box": [0.40, 0.72, 0.85, 0.85], "prompt": "1girl, laying on stomach, on rug, black skin,"},
    ]},
}


def load_scene(name: str) -> dict:
    if name in SCENES:
        return SCENES[name]
    with open(name, encoding="utf-8") as f:
        return json.load(f)


def url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Lab:
    def __init__(self, scene: dict):
        self.scene, self.S = scene, opts.size
        ckpts = list(models.checkpoints())
        if not ckpts:
            sys.exit("No Anima checkpoint found")
        ckpt = opts.checkpoint or ckpts[0]
        mods = list(models.modules())
        te = next((m for m in mods if "qwen_3_06b" in m.lower() or "qwen3_06b" in m.lower()), "")
        vae = next((m for m in mods if "vae" in m.lower()), "")
        self.model = {"checkpoint": ckpt, "text_encoder": te, "vae": vae}
        loras = [{"name": n, "weight": float(w)} if n else {"name": w, "weight": 1.0} for n, _, w in (x.rpartition(":") for x in opts.lora)]
        self.base = dict(model=self.model, prompt=scene.get("main", QUALITY), negative=scene.get("negative", "worst quality, low quality"),
                         width=self.S, height=self.S, steps=opts.steps, sampler=opts.sampler, scheduler=opts.scheduler, cfg=opts.cfg,
                         shift=opts.shift, loras=loras, save=False, region_base=opts.region_base, region_separation=opts.separation)
        print(f"[lab] {ckpt} · {self.S}² · {opts.steps} steps · CFG {opts.cfg} · {opts.sampler} / {opts.scheduler} · LoRAs {loras or 'none'}")

    def box(self, b) -> tuple[int, int, int, int]:
        return tuple(round(v * (self.S - 1)) for v in b)

    def masks(self, overlap: str) -> list[str]:
        """Region masks as the Canvas sends them: with Top layer wins each upper region is cut out of the lower ones."""
        boxes = [self.box(r["box"]) for r in self.scene["regions"]]
        out = []
        for i, b in enumerate(boxes):
            m = Image.new("L", (self.S, self.S), 0)
            d = ImageDraw.Draw(m)
            d.rectangle(b, fill=255)
            if overlap == "top":
                for upper in boxes[i + 1:]:
                    d.rectangle(upper, fill=0)
            out.append(url(m))
        return out

    def regions(self, overlap: str) -> list[dict]:
        return [{"prompt": r["prompt"], "negative": r.get("negative", ""), "mask": m} for r, m in zip(self.scene["regions"], self.masks(overlap))]

    def run(self, **kw) -> Image.Image:
        with torch.inference_mode():
            result = generate.run(Job(**{**self.base, **kw}), generate.Progress())
        return Image.open(io.BytesIO(result.images[0])).convert("RGB")

    def single(self, seed: int, couple: float) -> Image.Image:
        return self.run(seed=seed, regions=self.regions(opts.overlap), region_couple=couple)

    def regions_only(self, init: Image.Image, seed: int, strength: float) -> Image.Image:
        """Only regenerate the regions: each region in its full shape, bottom up, as the Canvas sends it."""
        return self.run(seed=seed, mode="img2img", init_image=url(init), strength=strength, regions_only=True, region_edges="soft",
                        region_context=opts.context, regions=self.regions("blend"))

    def variant(self, name: str, seed: int) -> Image.Image:
        if name == "single":
            return self.single(seed, 0.0)
        if name.startswith("own") and name[3:].isdigit():
            return self.single(seed, int(name[3:]) / 100)
        if name == "refine75":
            return self.regions_only(self.cached("single", seed), seed, 0.75)
        if name == "build":
            return self.regions_only(Image.new("RGB", (self.S, self.S), (255, 255, 255)), seed, 1.0)
        sys.exit(f"Unknown variant {name}")

    def cached(self, name: str, seed: int) -> Image.Image:
        key = json.dumps([self.scene, name, seed, {k: v for k, v in vars(opts).items() if k not in ("variants", "seeds", "scene", "help_lab")}], sort_keys=True)
        path = os.path.join(config.outputs_path(), "region_lab", "cache", hashlib.sha1(key.encode()).hexdigest()[:16] + ".png")
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
        t = time.time()
        img = self.variant(name, seed)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path)
        print(f"[lab] {name} seed {seed}: {time.time() - t:.0f}s", flush=True)
        return img


def sheet(lab: Lab, cells: list[tuple[str, Image.Image]], cols: int, path: str, cell: int = 400):
    rows = (len(cells) + cols - 1) // cols
    out = Image.new("RGB", (cols * cell, rows * (cell + 18)), (30, 30, 30))
    draw = ImageDraw.Draw(out)
    for k, (label, img) in enumerate(cells):
        x, y = (k % cols) * cell, (k // cols) * (cell + 18)
        thumb = img.resize((cell, cell))
        d = ImageDraw.Draw(thumb)
        for r in lab.scene["regions"]:  # region boxes, except the scenery
            b = [v * cell / (lab.S - 1) for v in lab.box(r["box"])]
            if (b[2] - b[0]) * (b[3] - b[1]) < 0.6 * cell * cell:
                d.rectangle(b, outline=(0, 255, 0))
        out.paste(thumb, (x, y + 18))
        draw.text((x + 4, y + 3), label, fill=(230, 230, 230))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.save(path)


def main() -> int:
    if opts.help_lab:
        print(__doc__)
        return 0
    lab = Lab(load_scene(opts.scene))
    variants = [v.strip() for v in opts.variants.split(",") if v.strip()]
    cells = [(f"{v} · seed {s}", lab.cached(v, s)) for s in (int(x) for x in opts.seeds.split(",")) for v in variants]
    name = os.path.splitext(os.path.basename(opts.scene))[0]
    path = os.path.join(config.outputs_path(), "region_lab", f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.png")
    sheet(lab, cells, len(variants), path)
    print(f"[lab] sheet: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
