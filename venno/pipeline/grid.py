"""Comparison grids (X/Y/Z plot): one job, every combination of up to three axes, labelled grid images.

Values (A1111-compatible): comma-separated, quotes for values with commas ("a, b"), numeric ranges
`1-5`, `1-10 (+2)` (step), `0.5-1.5 [5]` (count). Every cell uses the same seed unless seed is an axis.
"""

from __future__ import annotations

import csv
import io
import itertools
import math
import re
from dataclasses import dataclass
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

from venno.engine import rng

# ---------------------------------------------------------------- axes


@dataclass(frozen=True)
class AxisType:
    key: str
    label: str
    kind: str  # "int" | "float" | "str"
    apply: Callable  # (job, value, target) -> dict of job field updates


def _prompt_sr(job, value, target, *, search):
    rep = lambda t: t.replace(search, value) if t else t
    return {
        "prompt": rep(job.prompt), "negative": rep(job.negative), "switch_prompt": rep(job.switch_prompt),
        "regions": [r.model_copy(update={"prompt": rep(r.prompt)}) for r in job.regions],
    }


def _artist(job, value, target):
    name = value.strip()
    name = name if name.startswith("@") else "@" + name
    if "@ARTIST" in job.prompt:
        return {"prompt": job.prompt.replace("@ARTIST", name)}
    return {"prompt": f"{job.prompt.rstrip().rstrip(',')}, {name}"}


def _lora_weight(job, value, target):
    loras = [lr.model_copy(update={"weight": value}) if lr.name == target else lr for lr in job.loras]
    if not any(lr.name == target for lr in job.loras):
        lora_ref = type(job).model_fields["loras"].annotation.__args__[0]
        loras.append(lora_ref(name=target, weight=value))
    return {"loras": loras}


def _denoise(job, value, target):
    if job.hires is not None and job.mode == "txt2img":
        return {"hires": job.hires.model_copy(update={"denoise": value})}
    return {"strength": value}


def _size(job, value, target):
    m = re.fullmatch(r"\s*(\d+)\s*[x×]\s*(\d+)\s*", value)
    if not m:
        raise ValueError(f'Size "{value}" should look like 832x1216')
    return {"width": int(m.group(1)), "height": int(m.group(2))}


def _hires_upscaler(job, value, target):
    if job.hires is None:
        raise ValueError("The hires upscaler axis needs hires fix enabled")
    return {"hires": job.hires.model_copy(update={"upscaler": value})}


def _detailer_denoise(job, value, target):
    return {"detailer": [d.model_copy(update={"denoise": value}) for d in job.detailer]}


AXES: dict[str, AxisType] = {a.key: a for a in [
    AxisType("prompt_sr", "Prompt S/R", "str", None),  # apply bound per axis (needs the search term)
    AxisType("artist", "Artist", "str", _artist),
    AxisType("seed", "Seed", "int", lambda j, v, t: {"seed": v}),
    AxisType("steps", "Steps", "int", lambda j, v, t: {"steps": v}),
    AxisType("cfg", "CFG", "float", lambda j, v, t: {"cfg": v}),
    AxisType("shift", "Shift", "float", lambda j, v, t: {"shift": v}),
    AxisType("sampler", "Sampler", "str", lambda j, v, t: {"sampler": v}),
    AxisType("scheduler", "Scheduler", "str", lambda j, v, t: {"scheduler": v}),
    AxisType("checkpoint", "Checkpoint", "str", lambda j, v, t: {"model": j.model.model_copy(update={"checkpoint": v})}),
    AxisType("lora_weight", "LoRA weight", "float", _lora_weight),
    AxisType("denoise", "Denoising", "float", _denoise),
    AxisType("hires_upscaler", "Hires upscaler", "str", _hires_upscaler),
    AxisType("size", "Size", "str", _size),
    AxisType("switch_at", "Switch at", "float", lambda j, v, t: {"switch_at": v}),
    AxisType("region_base", "Region base", "float", lambda j, v, t: {"region_base": v}),
    AxisType("detailer_denoise", "Detailer denoise", "float", _detailer_denoise),
]}

_RANGE_STEP = re.compile(r"^\s*([+-]?\d*\.?\d+)\s*-\s*([+-]?\d*\.?\d+)\s*\(\s*([+-]?\d*\.?\d+)\s*\)\s*$")
_RANGE_COUNT = re.compile(r"^\s*([+-]?\d*\.?\d+)\s*-\s*([+-]?\d*\.?\d+)\s*\[\s*(\d+)\s*\]\s*$")
_RANGE = re.compile(r"^\s*([+-]?\d+)\s*-\s*([+-]?\d+)\s*$")
MAX_CELLS = 400


def parse_values(kind: str, values) -> list:
    items = values if isinstance(values, list) else [v.strip() for v in next(csv.reader(io.StringIO(str(values)), skipinitialspace=True), [])]
    items = [str(v).strip() for v in items if str(v).strip()]
    if kind == "str":
        return items
    cast = int if kind == "int" else float
    out = []
    for v in items:
        if m := _RANGE_COUNT.match(v):
            a, b, n = float(m.group(1)), float(m.group(2)), int(m.group(3))
            out += [a + (b - a) * i / max(1, n - 1) for i in range(n)]
        elif m := _RANGE_STEP.match(v):
            a, b, step = float(m.group(1)), float(m.group(2)), float(m.group(3))
            if step == 0 or (b - a) / step < 0:
                raise ValueError(f'Range "{v}" never reaches its end')
            n = int(math.floor((b - a) / step + 1e-9)) + 1
            out += [a + step * i for i in range(n)]
        elif kind == "int" and (m := _RANGE.match(v)):
            a, b = int(m.group(1)), int(m.group(2))
            out += list(range(a, b + 1) if b >= a else range(a, b - 1, -1))
        else:
            try:
                out.append(float(v))
            except ValueError:
                raise ValueError(f'"{v}" is not a number') from None
    return [int(round(x)) if cast is int else round(x, 6) for x in out]


@dataclass
class ResolvedAxis:
    type: AxisType
    values: list
    target: str = ""

    @property
    def title(self) -> str:
        return f"{self.type.label}: {self.target}" if self.type.key == "lora_weight" and self.target else self.type.label

    def label(self, value) -> str:
        return f"{value:g}" if isinstance(value, float) else str(value)

    def heading(self, value) -> str:
        """Cell label on the grid: numbers say what they are ("CFG: 4.5"); names speak for themselves."""
        return self.label(value) if self.type.kind == "str" else f"{self.title}: {self.label(value)}"

    def updates(self, job, value) -> dict:
        if self.type.key == "prompt_sr":
            return _prompt_sr(job, value, self.target, search=self.values[0])
        return self.type.apply(job, value, self.target)


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
        if t.key == "lora_weight" and not a.target:
            raise ValueError("The LoRA weight axis needs a LoRA")
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


def run(job, progress, run_job) -> "object":
    """Run every cell through `run_job(job, progress)` (the normal pipeline) and assemble the grids."""
    from . import metadata
    from .generate import Progress, Result

    axes = resolve_axes(job.grid)
    while len(axes) < 3:
        axes.append(ResolvedAxis(AxisType("none", "", "str", lambda j, v, t: {}), [""]))
    ax, ay, az = axes
    base_seed = rng.resolve_seed(job.seed)
    base = job.model_copy(update={"grid": None, "batch": 1, "count": 1, "seed": base_seed, "save": job.save and job.grid.include_cells})

    coords = list(itertools.product(range(len(az.values)), range(len(ay.values)), range(len(ax.values))))
    ckpt_axis = next((i for i, a in enumerate((az, ay, ax)) if a.type.key == "checkpoint"), None)
    order = sorted(coords, key=lambda c: c[ckpt_axis]) if ckpt_axis is not None else coords  # fewer model reloads

    images: dict[tuple, Image.Image] = {}
    result = Result()
    cell_infos: dict[tuple, tuple[bytes, int, str]] = {}
    for n, (zi, yi, xi) in enumerate(order):
        sub = base
        for a, i in ((ax, xi), (ay, yi), (az, zi)):
            if a.type.key != "none":
                sub = sub.model_copy(update=a.updates(sub, a.values[i]))
        cell_progress = Progress(
            on_step=lambda step, steps, run, runs, n=n: progress.on_step(step, steps, n, len(order)),
            on_preview=progress.on_preview, is_cancelled=progress.is_cancelled, preview_every=progress.preview_every,
        )
        part = run_job(sub, cell_progress)
        result.warnings += [w for w in part.warnings if w not in result.warnings]
        if part.cancelled or not part.images:
            result.cancelled = part.cancelled
            break
        images[(zi, yi, xi)] = Image.open(io.BytesIO(part.images[0])).convert("RGB")
        cell_infos[(zi, yi, xi)] = (part.images[0], part.seeds[0], part.infotexts[0])
    if result.cancelled or not images:
        return result

    blank = Image.new("RGB", next(iter(images.values())).size, (40, 40, 40))
    grid_text = "; ".join(f"{name}: {a.title} = {', '.join(a.label(v) for v in a.values)}" for name, a in zip("XYZ", (ax, ay, az)) if a.type.key != "none")
    first_info = next(iter(cell_infos.values()))[2]
    for zi, zv in enumerate(az.values):
        cells = [[images.get((zi, yi, xi), blank) for xi in range(len(ax.values))] for yi in range(len(ay.values))]
        title = az.heading(zv) if az.type.key != "none" else ""
        grid_img = draw_grid(
            cells,
            [ax.heading(v) for v in ax.values] if ax.type.key != "none" else [""],
            [ay.heading(v) for v in ay.values] if ay.type.key != "none" else [""],
            title,
        )
        info = first_info.rstrip() + f", Grid: \"{grid_text}\""
        if job.save:
            metadata.save(grid_img, info, f"{job.mode}-grids", base_seed)
        result.images.append(metadata.png_bytes(grid_img, info))
        result.seeds.append(base_seed)
        result.infotexts.append(info)
    if job.grid.include_cells:
        for key in coords:
            if key in cell_infos:
                png, seed, info = cell_infos[key]
                result.images.append(png)
                result.seeds.append(seed)
                result.infotexts.append(info)
    return result
