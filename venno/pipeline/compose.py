"""The main prompt composed from the regions, in the structured form Anima's LLM text encoder follows.

Regional attention tells the model what each area shows, but the picture's composition (how many subjects,
where, how big) is decided by the main prompt in the first steps, and a main prompt of quality tags says
nothing about it. Anima's Qwen encoder follows a structured prompt that names every subject, describes it
and places the names (measured in venno/regionlab.py: with the same regions, a JSON-style main prompt alone
put two characters in their places in 14 of 16 pictures, the quality tags alone in 3). So when a job asks
for it, the main prompt becomes: the user's main prompt as "tags" (plus the subject count), one named entry
per subject region, the scenery regions as "background", and a "composition" line that places every name
by where its region is.
"""

from __future__ import annotations

import json
import re
from typing import Optional

SUBJECT = re.compile(r"^\s*(\d*)\s*(girl|boy|other)s?\s*(?:,|$)", re.I)
CHARACTER = re.compile(r"\b(woman|man|person|character|witch|kobold|anthro|furry)s?\b", re.I)


def _place(box: tuple[float, float, float, float]) -> str:
    """Where a box (fractions of the picture) is, in words: "on the left", "at the bottom right", "in the middle"."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    horizontal = "left" if cx < 0.37 else "right" if cx > 0.63 else ""
    vertical = "top" if cy < 0.37 else "bottom" if cy > 0.63 else ""
    if horizontal and vertical:
        where = f"at the {vertical} {horizontal}"
    elif horizontal:
        where = f"on the {horizontal}"
    elif vertical:
        where = f"at the {vertical}"
    else:
        where = "in the middle"
    area = (x1 - x0) * (y1 - y0)
    if area < 0.06:
        where += ", small"
    elif area > 0.45:
        where += ", large"
    return where


def compose(prompt: str, regions: list[tuple[str, str, Optional[tuple[float, float, float, float]]]], *, escape: bool = False) -> str:
    """regions: (prompt, kind "scenery" | "subject", box as fractions of the picture or None) in layer order.
    escape: braces written as \\{ \\} so the prompt parser's wildcard syntax leaves them alone."""
    characters, objects, placed, counts = [], [], [], {}
    for text, kind, box in regions:
        text = text.strip().strip(",").strip()
        if not text or kind == "scenery":
            continue
        m = SUBJECT.match(text)
        if m:
            base = m.group(2).lower()
            description = text[m.end():].strip().strip(",").strip()
        elif CHARACTER.search(text):
            base, description = "character", text
        else:
            objects.append(text)
            if box is not None:
                name = re.sub(r"\\\(.*?\\\)", "", text.split(",")[0]).strip()  # "bat \(animal\)" → "bat"
                placed.append(f"the {name} {_place(box)}")
            continue
        counts[base] = counts.get(base, 0) + 1
        name = f"{base}{counts[base]}"
        characters.append({name: description or base})
        if box is not None:
            placed.append(f"{name} {_place(box)}")
    tags = [prompt.strip().strip(",").strip()] + [f"{n}{base}{'s' if n > 1 else ''}" for base, n in counts.items() if base != "character"]
    out = {"tags": ", ".join(t for t in tags if t)}
    if characters:
        out["characters"] = characters
    # objects go into the background text: as a list of their own they made the model draw comic panels
    scenery = [text.strip().strip(",").strip() for text, kind, _ in regions if kind == "scenery" and text.strip()] + objects
    if scenery:
        out["background"] = ", ".join(scenery)
    if placed:
        out["composition"] = ", ".join(placed)
    text = json.dumps(out, indent=2, ensure_ascii=False)
    return text.replace("{", "\\{").replace("}", "\\}") if escape else text
