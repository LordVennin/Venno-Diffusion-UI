"""Batch sheet: the pictures of one job stitched into a single picture for comparison, like Forge's image grid
(Venno and Venno MOSS; only PIL, so both can import it).

Layout as Forge's default: rows = round(sqrt(n)), columns to fit; a consistent set's frames form the columns.
Cells keep their full size unless the sheet would pass `MAX_SIDE`, then they are scaled down together.
"""

from __future__ import annotations

import io
import math

from PIL import Image

MAX_SIDE = 8192  # pixels: above this a PNG gets too heavy for the browser
MARK = ', Grid: "'  # in a sheet's infotext (as comparison grids): Studio shows such pictures as view-only


def layout(n: int, cols: int = 0) -> tuple[int, int]:
    """(rows, columns) for n pictures; `cols` fixes the columns (e.g. the frames of a consistent set)."""
    if cols > 0:
        return math.ceil(n / cols), min(cols, n)
    rows = max(1, min(n, round(math.sqrt(n))))
    return rows, math.ceil(n / rows)


def tile(images: list[Image.Image], cols: int = 0, background=(24, 25, 28)) -> Image.Image:
    rows, cols = layout(len(images), cols)
    w, h = max(im.width for im in images), max(im.height for im in images)
    scale = min(1.0, MAX_SIDE / max(cols * w, rows * h))
    cw, ch = max(1, int(w * scale)), max(1, int(h * scale))  # rounded down: the sheet stays within MAX_SIDE
    out = Image.new("RGB", (cols * cw, rows * ch), background)
    for k, im in enumerate(images):
        im = im.convert("RGB")
        if scale < 1.0:
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
        out.paste(im, ((k % cols) * cw + (cw - im.width) // 2, (k // cols) * ch + (ch - im.height) // 2))
    return out


def note(seeds: list[int]) -> str:
    """What the sheet shows, for its infotext: '4 images, seeds 7-10'."""
    unique = sorted(set(seeds))
    if len(unique) > 1 and unique == list(range(unique[0], unique[-1] + 1)):
        which = f"seeds {unique[0]}-{unique[-1]}"
    else:
        which = "seed" + ("s " if len(unique) > 1 else " ") + ", ".join(str(s) for s in unique[:8]) + ("…" if len(unique) > 8 else "")
    return f"{len(seeds)} images, {which}"


def from_pngs(pngs: list[bytes], cols: int = 0) -> Image.Image:
    return tile([Image.open(io.BytesIO(p)) for p in pngs], cols)
