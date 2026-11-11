"""Comparison grid (X/Y/Z): the job once per combination of axis values, assembled into labelled grids.
Value syntax as in Venno: "3, 4, 5", "2-6 (+1)", "0-1.2 [4]", quoted values for commas.

For LoRA testing: the LoRA axis puts different files (e.g. a training run's epochs) side by side; "none" is
a cell without that LoRA. The LoRA weight axis sweeps one LoRA's weight.
"""

from __future__ import annotations

import csv
import io
import itertools
import math
import os
import re
from dataclasses import dataclass
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

from . import config, metadata

MAX_CELLS = 400
NONE_WORDS = ("none", "off", "-", "no lora")


def _lora_name(v: str) -> str:
    return os.path.splitext(os.path.basename(v))[0]


def _lora(job, value, target):
    """Cell with LoRA file `value` in place of `target` (or of the first LoRA; added at weight 1 when there is none)."""
    loras = list(job.loras)
    own = [i for i, lr in enumerate(loras) if not config.is_turbo(lr.name)]  # Fast preview's turbo LoRA stays in every cell
    slot = next((i for i in own if loras[i].name == target), own[0] if own else None)
    if value.strip().lower() in NONE_WORDS:
        return {"loras": [lr for i, lr in enumerate(loras) if i != slot]}
    ref = type(job).model_fields["loras"].annotation.__args__[0]
    if slot is None:
        return {"loras": [ref(name=value, weight=1.0), *loras]}
    loras[slot] = ref(name=value, weight=loras[slot].weight)
    return {"loras": loras}


def _lora_weight(job, value, target):
    names = [lr.name for lr in job.loras if not config.is_turbo(lr.name)]
    name = target if target in names else (names[0] if names else "")
    if not name:  # e.g. the "none" cell of a LoRA axis: nothing to weigh
        return {}
    return {"loras": [lr.model_copy(update={"weight": value}) if lr.name == name else lr for lr in job.loras]}


def _size(job, value, target):
    m = re.fullmatch(r"\s*(\d+)\s*[x×]\s*(\d+)\s*", value)
    if not m:
        raise ValueError(f'Size "{value}" should look like 832x1216')
    return {"width": int(m.group(1)), "height": int(m.group(2))}


def _skip(job, value, target):
    v = value.strip().lower()
    return {"skip_layers": [] if v in NONE_WORDS else [int(x) for x in re.split(r"[ +;]+", v) if x]}


def _hires(field: str):
    def apply(job, value, target):
        if job.hires is None:
            raise ValueError("Hires axes need hires fix switched on")
        return {"hires": job.hires.model_copy(update={field: value})}

    return apply


def _prompt_sr(job, value, target, *, search):
    return {"prompt": job.prompt.replace(search, value), "negative": job.negative.replace(search, value)}


@dataclass(frozen=True)
class AxisType:
    key: str
    label: str
    kind: str  # "int" | "float" | "str"
    apply: Callable


AXES: dict[str, AxisType] = {a.key: a for a in [
    AxisType("lora", "LoRA", "str", _lora),
    AxisType("lora_weight", "LoRA weight", "float", _lora_weight),
    AxisType("prompt_sr", "Prompt S/R", "str", None),
    AxisType("seed", "Seed", "int", lambda j, v, t: {"seed": v}),
    AxisType("steps", "Steps", "int", lambda j, v, t: {"steps": v}),
    AxisType("cfg", "CFG", "float", lambda j, v, t: {"cfg": v}),
    AxisType("shift", "Shift", "float", lambda j, v, t: {"shift": v}),
    AxisType("model", "Model", "str", lambda j, v, t: {"model": v}),
    AxisType("size", "Size", "str", _size),
    AxisType("skip_layers", "Skip layers", "str", _skip),
    AxisType("hires_denoise", "Hires denoising", "float", _hires("denoise")),
    AxisType("hires_scale", "Hires upscale by", "float", _hires("scale")),
    AxisType("hires_steps", "Hires steps", "int", _hires("steps")),
    AxisType("hires_upscaler", "Hires upscaler", "str", _hires("upscaler")),
]}

_RANGE_STEP = re.compile(r"^\s*([+-]?\d*\.?\d+)\s*-\s*([+-]?\d*\.?\d+)\s*\(\s*([+-]?\d*\.?\d+)\s*\)\s*$")
_RANGE_COUNT = re.compile(r"^\s*([+-]?\d*\.?\d+)\s*-\s*([+-]?\d*\.?\d+)\s*\[\s*(\d+)\s*\]\s*$")
_RANGE = re.compile(r"^\s*([+-]?\d+)\s*-\s*([+-]?\d+)\s*$")


def parse_values(kind: str, values) -> list:
    items = values if isinstance(values, list) else [v.strip() for v in next(csv.reader(io.StringIO(str(values)), skipinitialspace=True), [])]
    items = [str(v).strip() for v in items if str(v).strip()]
    if kind == "str":
        return items
    out = []
    for v in items:
        if m := _RANGE_COUNT.match(v):
            a, b, n = float(m.group(1)), float(m.group(2)), int(m.group(3))
            out += [a + (b - a) * i / max(1, n - 1) for i in range(n)]
        elif m := _RANGE_STEP.match(v):
            a, b, step = float(m.group(1)), float(m.group(2)), float(m.group(3))
            if step == 0 or (b - a) / step < 0:
                raise ValueError(f'Range "{v}" never reaches its end')
            out += [a + step * i for i in range(int(math.floor((b - a) / step + 1e-9)) + 1)]
        elif kind == "int" and (m := _RANGE.match(v)):
            a, b = int(m.group(1)), int(m.group(2))
            out += list(range(a, b + 1) if b >= a else range(a, b - 1, -1))
        else:
            try:
                out.append(float(v))
            except ValueError:
                raise ValueError(f'"{v}" is not a number') from None
    return [int(round(x)) if kind == "int" else round(x, 6) for x in out]


@dataclass
class ResolvedAxis:
    type: AxisType
    values: list
    target: str = ""

    def label(self, value) -> str:
        if self.type.key == "lora":
            return "no LoRA" if str(value).strip().lower() in NONE_WORDS else _lora_name(value)
        return f"{value:g}" if isinstance(value, float) else str(value)

    def heading(self, value) -> str:
        if self.type.kind == "str":
            return self.label(value)
        title = f"{self.type.label} ({_lora_name(self.target)})" if self.type.key == "lora_weight" and self.target else self.type.label
        return f"{title}: {self.label(value)}"

    def updates(self, job, value) -> dict:
        if self.type.key == "prompt_sr":
            return _prompt_sr(job, value, self.target, search=self.values[0])
        return self.type.apply(job, value, self.target)


def check(job, axes: list[ResolvedAxis]):
    keys = {a.type.key for a in axes}
    if "lora_weight" in keys and not job.loras and "lora" not in keys:
        raise ValueError("The LoRA weight axis needs a LoRA in the LoRA list")
    if job.hires is None and any(k.startswith("hires_") for k in keys):
        raise ValueError("Hires axes need hires fix switched on")


def resolve_axes(grid) -> list[ResolvedAxis]:
    axes = []
    for name in ("x", "y", "z"):
        a = getattr(grid, name)
        if a is None or not a.type or a.type == "none":
            continue
        t = AXES.get(a.type)
        if t is None:
            raise ValueError(f'Unknown grid axis "{a.type}"')
        values = parse_values(t.kind, a.values)
        if not values:
            raise ValueError(f"The {name.upper()} axis ({t.label}) has no values")
        axes.append(ResolvedAxis(t, values, a.target or ""))
    if not axes:
        raise ValueError("The grid needs at least one axis")
    total = math.prod(len(a.values) for a in axes)
    if total > MAX_CELLS:
        raise ValueError(f"The grid has {total} cells; the limit is {MAX_CELLS}")
    return axes


# ---------------------------------------------------------------- drawing


def _font(size: int):
    return ImageFont.load_default(size=size)


def _wrap(draw, text: str, font, width: int) -> list[str]:
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines[:4]


def draw_grid(cells: list[list[Image.Image]], x_labels: list[str], y_labels: list[str], title: str = "") -> Image.Image:
    rows, cols = len(cells), len(cells[0])
    cw = max(im.width for row in cells for im in row)
    ch = max(im.height for row in cells for im in row)
    fs = max(14, cw // 18)
    font, tfont = _font(fs), _font(int(fs * 1.15))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    gap = max(4, cw // 64)
    top = (max(len(_wrap(probe, t, font, cw - 8)) for t in x_labels) * (fs + 4) + 2 * gap) if any(x_labels) else 0
    left = (min(cw, max(int(probe.textlength(t, font=font)) for t in y_labels) + 3 * gap)) if any(y_labels) else 0
    head = (fs + 3 * gap) if title else 0
    W, H = left + cols * (cw + gap), head + top + rows * (ch + gap)
    out = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(out)
    if title:
        d.text((W // 2, gap + fs // 2), title, fill="black", font=tfont, anchor="mm")
    for c, t in enumerate(x_labels):
        for k, line in enumerate(_wrap(d, t, font, cw - 8)):
            d.text((left + c * (cw + gap) + cw // 2, head + gap + k * (fs + 4) + fs // 2), line, fill="black", font=font, anchor="mm")
    for r, t in enumerate(y_labels):
        lines = _wrap(d, t, font, left - 2 * gap) if left else []
        y0 = head + top + r * (ch + gap) + ch // 2 - len(lines) * (fs + 4) // 2
        for k, line in enumerate(lines):
            d.text((left - gap, y0 + k * (fs + 4) + fs // 2), line, fill="black", font=font, anchor="rm")
    for r, row in enumerate(cells):
        for c, im in enumerate(row):
            out.paste(im, (left + c * (cw + gap), head + top + r * (ch + gap)))
    return out


# ---------------------------------------------------------------- running


def run(job, progress):
    from .jobs import Progress, Result, resolve_seed, run as run_job

    axes = resolve_axes(job.grid)
    check(job, axes)
    none = AxisType("none", "", "str", lambda j, v, t: {})
    while len(axes) < 3:
        axes.append(ResolvedAxis(none, [""]))
    ax, ay, az = axes
    seed = resolve_seed(job.seed)
    base = job.model_copy(update={"grid": None, "count": 1, "seed": seed, "save": job.save and job.grid.include_cells})
    coords = list(itertools.product(range(len(az.values)), range(len(ay.values)), range(len(ax.values))))
    model_axis = next((i for i, a in enumerate((az, ay, ax)) if a.type.key == "model"), None)
    order = sorted(coords, key=lambda c: c[model_axis]) if model_axis is not None else coords  # fewer model loads

    images: dict[tuple, Image.Image] = {}
    infos: dict[tuple, tuple[bytes, int, str]] = {}
    result = Result()
    for n, (zi, yi, xi) in enumerate(order):
        sub = base
        for a, i in ((ax, xi), (ay, yi), (az, zi)):
            if a.type.key != "none":
                sub = sub.model_copy(update=a.updates(sub, a.values[i]))
        cell = Progress(on_step=lambda step, steps, r, rs, n=n: progress.on_step(step, steps, n, len(order)),
                        on_preview=progress.on_preview, is_cancelled=progress.is_cancelled, take_skip=progress.take_skip)
        part = run_job(sub, cell)
        result.warnings += [w for w in part.warnings if w not in result.warnings]
        if part.cancelled:
            result.cancelled = True
            break
        if part.images:  # a skipped cell stays blank
            images[(zi, yi, xi)] = Image.open(io.BytesIO(part.images[0])).convert("RGB")
            infos[(zi, yi, xi)] = (part.images[0], part.seeds[0], part.infotexts[0])
    if result.cancelled or not images:
        return result

    blank = Image.new("RGB", next(iter(images.values())).size, (40, 40, 40))
    text = "; ".join(f"{name}: {a.type.label} = {', '.join(a.label(v) for v in a.values)}" for name, a in zip("XYZ", (ax, ay, az)) if a.type.key != "none")
    first = next(iter(infos.values()))
    for zi, zv in enumerate(az.values):
        cells = [[images.get((zi, yi, xi), blank) for xi in range(len(ax.values))] for yi in range(len(ay.values))]
        grid_img = draw_grid(
            cells,
            [ax.heading(v) for v in ax.values] if ax.type.key != "none" else [""],
            [ay.heading(v) for v in ay.values] if ay.type.key != "none" else [""],
            az.heading(zv) if az.type.key != "none" else "",
        )
        info = first[2].rstrip() + f', Grid: "{text}"'
        png = metadata.png_bytes(grid_img, info, metadata.settings_json(base, first[1]))
        if job.save:
            metadata.save(png, seed, "grids")
        result.images.append(png)
        result.seeds.append(seed)
        result.infotexts.append(info)
    if job.grid.include_cells:
        for key in coords:
            if key in infos:
                png, s, info = infos[key]
                result.images.append(png)
                result.seeds.append(s)
                result.infotexts.append(info)
    return result
