"""Random prompt parts (Dynamic Prompts-compatible syntax), resolved per image from that image's seed.

    {red|blue|green} hair          one option
    {2$$red|blue|green}            two different options, joined by ", "
    {1-3$$a|b|c|d}                 one to three options
    {2$$ and $$a|b|c}              custom separator
    {0.2::rare|common}             weighted options (default weight 1)
    __hair/colors__                a random line of wildcards/hair/colors.txt (blank lines and # comments skipped)
    __colors*__                    any line from every matching file
    \\{  \\}  \\|                     literal characters

Options and wildcard lines may contain more of the same (up to 20 levels).
"""

from __future__ import annotations

import fnmatch
import os
import random
import re

_WILDCARD = re.compile(r"__([\w\-/*. ]+?)__")
_COUNT = re.compile(r"^\s*(\d+)?\s*(?:-\s*(\d+)\s*)?\$\$(?:(.*?)\$\$)?", re.S)
_WEIGHT = re.compile(r"^\s*(\d*\.?\d+)\s*::(.*)$", re.S)
MAX_DEPTH = 20


def folder() -> str:
    from modules import paths  # Phase 4: Venno settings

    path = os.path.join(paths.data_path, "wildcards")
    os.makedirs(path, exist_ok=True)
    return path


def files() -> list[str]:
    """Wildcard names (relative paths without .txt), e.g. 'hair/colors'."""
    root = folder()
    out = []
    for dirpath, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith(".txt"):
                rel = os.path.relpath(os.path.join(dirpath, n), root)
                out.append(os.path.splitext(rel)[0].replace(os.sep, "/"))
    return sorted(out, key=str.lower)


def has_syntax(text: str) -> bool:
    return bool(text) and (_WILDCARD.search(text) is not None or re.search(r"(?<!\\)\{", text) is not None)


def _lines(name: str) -> list[str]:
    names = [n for n in files() if fnmatch.fnmatch(n.lower(), name.lower())] if "*" in name else [name]
    out = []
    root = folder()
    for n in names:
        path = os.path.realpath(os.path.join(root, n + ".txt"))
        if not path.startswith(os.path.realpath(root) + os.sep) or not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            out += [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not out:
        raise ValueError(f'Wildcard "__{name}__" not found (put {name}.txt in {root})')
    return out


def _split_options(body: str) -> list[str]:
    """Split on top-level | (not inside nested braces, not escaped)."""
    parts, depth, cur, i = [], 0, [], 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            cur.append(body[i : i + 2])
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    parts.append("".join(cur))
    return parts


def _choose(body: str, rng: random.Random, depth: int) -> str:
    sep, lo, hi = ", ", 1, 1
    m = _COUNT.match(body)
    if m and (m.group(1) or m.group(2)):
        lo = int(m.group(1) or 1)
        hi = int(m.group(2) or lo)
        sep = m.group(3) if m.group(3) is not None else ", "
        body = body[m.end() :]
    options, weights = [], []
    for opt in _split_options(body):
        w = _WEIGHT.match(opt)
        options.append(w.group(2) if w else opt)
        weights.append(float(w.group(1)) if w else 1.0)
    n = max(0, min(rng.randint(min(lo, hi), max(lo, hi)), len(options)))
    picked = []
    pool = list(zip(options, weights))
    for _ in range(n):  # weighted, without repeats
        total = sum(w for _, w in pool)
        if total <= 0:
            break
        r = rng.uniform(0, total)
        for k, (opt, w) in enumerate(pool):
            r -= w
            if r <= 0 or k == len(pool) - 1:
                picked.append(opt)
                pool.pop(k)
                break
    return sep.join(resolve(p.strip(), rng, depth + 1) for p in picked)


def resolve(text: str, rng: random.Random, depth: int = 0) -> str:
    if not text or depth > MAX_DEPTH:
        return text
    out, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in "{}|":
            out.append(text[i + 1])  # escaped: literal
            i += 2
            continue
        if ch == "{":
            depth_b, j = 1, i + 1
            while j < len(text) and depth_b:
                if text[j] == "\\":
                    j += 2
                    continue
                depth_b += {"{": 1, "}": -1}.get(text[j], 0)
                j += 1
            if depth_b:  # unbalanced: keep as text
                out.append(text[i:])
                break
            out.append(_choose(text[i + 1 : j - 1], rng, depth))
            i = j
            continue
        m = _WILDCARD.match(text, i)
        if m:
            out.append(resolve(rng.choice(_lines(m.group(1).strip())), rng, depth + 1))
            i = m.end()
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def for_seed(text: str, seed: int, salt: str = "") -> str:
    """Resolve with an RNG derived from the image seed (same seed → same choices)."""
    return resolve(text, random.Random(f"{seed}:{salt}"))
