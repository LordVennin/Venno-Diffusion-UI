"""Danbooru tag search for Studio's autocomplete, formatted the way Anima wants tags written.

Anima (model card): lowercase, spaces instead of underscores (score_ tags keep theirs), artists prefixed with @.
Parentheses are escaped so "(" isn't read as emphasis: ganyu \\(genshin impact\\).
"""

from __future__ import annotations

import bisect
import csv
import gzip
import io
import os
import threading
from dataclasses import dataclass

BUNDLED = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "danbooru_tags.csv.gz")
CATEGORIES = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta"}


@dataclass(slots=True)
class Tag:
    name: str  # danbooru form: long_hair
    category: int
    count: int
    aliases: tuple[str, ...]


_lock = threading.Lock()
_tags: list[Tag] = []
_names: list[str] = []  # sorted keys for prefix search
_order: list[int] = []  # _names[i] → index into _tags
_aliases: list[tuple[str, int]] = []  # sorted (alias, tag index)


def folder() -> str:
    from venno import config

    return os.path.join(config.data_path(), "tags")


def _read(stream) -> list[Tag]:
    out = []
    for row in csv.reader(stream):
        if len(row) < 3 or not row[0]:
            continue
        try:
            out.append(Tag(row[0].strip().lower(), int(row[1] or 0), int(float(row[2] or 0)), tuple(a.strip().lower() for a in (row[3] if len(row) > 3 else "").split(",") if a.strip())))
        except ValueError:
            continue
    return out


def _load():
    global _tags, _names, _order, _aliases
    with _lock:
        if _tags:
            return
        tags: dict[str, Tag] = {}
        if os.path.isfile(BUNDLED):
            with gzip.open(BUNDLED, "rt", encoding="utf-8") as f:
                for t in _read(f):
                    tags[t.name] = t
        extra = folder()
        if os.path.isdir(extra):
            for fn in sorted(os.listdir(extra)):
                if fn.lower().endswith(".csv"):
                    with open(os.path.join(extra, fn), encoding="utf-8", errors="replace") as f:
                        for t in _read(f):
                            tags[t.name] = t  # user lists win
        _tags = sorted(tags.values(), key=lambda t: -t.count)
        pairs = sorted((t.name, i) for i, t in enumerate(_tags))
        _names = [p[0] for p in pairs]
        _order = [p[1] for p in pairs]
        _aliases = sorted((a, i) for i, t in enumerate(_tags) for a in t.aliases if not a.startswith("/"))


def reload():
    global _tags
    with _lock:
        _tags = []
    _load()


def insert_text(t: Tag) -> str:
    name = t.name if t.name.startswith("score_") else t.name.replace("_", " ")
    name = name.replace("(", "\\(").replace(")", "\\)")
    return ("@" + name) if t.category == 1 else name


def search(query: str, limit: int = 12) -> list[dict]:
    _load()
    q = query.strip().lower()
    artists_only = q.startswith("@")
    q = q.lstrip("@").replace("\\(", "(").replace("\\)", ")").replace(" ", "_")
    if len(q) < 1:
        return []

    hits: dict[int, str] = {}  # tag index → matched alias ("" = name)

    i = bisect.bisect_left(_names, q)
    while i < len(_names) and _names[i].startswith(q) and len(hits) < 400:
        hits[_order[i]] = ""
        i += 1
    i = bisect.bisect_left(_aliases, (q,))
    while i < len(_aliases) and _aliases[i][0].startswith(q) and len(hits) < 400:
        hits.setdefault(_aliases[i][1], _aliases[i][0])
        i += 1
    prefixed = set(hits)
    if len(q) >= 3:  # words inside tags too: "miku" → "hatsune miku" (tags are sorted by count, so the first few are the best)
        needle, found = "_" + q, 0
        for idx, t in enumerate(_tags):
            if needle in t.name and idx not in hits:
                hits[idx] = ""
                found += 1
                if found >= limit * 2:
                    break

    ranked = sorted(hits.items(), key=lambda kv: -_tags[kv[0]].count * (3 if kv[0] in prefixed else 1))
    out = []
    for idx, alias in ranked:
        t = _tags[idx]
        if artists_only and t.category != 1:
            continue
        out.append({"text": insert_text(t), "category": CATEGORIES.get(t.category, "general"), "count": t.count, "alias": alias.replace("_", " ") if alias else ""})
        if len(out) >= limit:
            break
    return out
