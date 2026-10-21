"""Studio class and id names must not look like ads: ad blockers hide elements such as `.ad-card` or
`.ad-unit` (EasyList generic rules), which once made the whole ADetailer card vanish."""

import os
import re

STUDIO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studio")
AD_LIKE = re.compile(r"^(ad|ads|adv|advert\w*|sponsor\w*|banner)$|^(ad|ads|adv|advert|sponsor|banner)[-_]|[-_](ad|ads|advert|sponsor|banner)([-_]|$)", re.I)
NAMES = re.compile(r"""(?:class|id)\s*=\s*["']([^"'$]+)["']|(?:classList\.add|\.id\s*=)\s*\(?\s*["']([^"']+)["']|[$]{1,2}\(\s*["']([^"']+)["']""")
SELECTOR_NAMES = re.compile(r"[.#]([A-Za-z][\w-]*)")


def _names():
    for folder, _, files in os.walk(STUDIO):
        for file in files:
            if not file.endswith((".html", ".js", ".css")):
                continue
            path = os.path.join(folder, file)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            for m in NAMES.finditer(text):
                attr, added, selector = m.groups()
                if selector:
                    yield from ((path, n) for n in SELECTOR_NAMES.findall(selector))
                else:
                    yield from ((path, n) for n in (attr or added).split())
            if file.endswith(".css"):
                for rule in re.findall(r"([^{}]+)\{", re.sub(r"/\*.*?\*/", "", text, flags=re.S)):
                    yield from ((path, n) for n in SELECTOR_NAMES.findall(rule))


def test_no_ad_like_names():
    names = list(_names())
    assert any(n == "det-card" for _, n in names)  # the scan does see the ADetailer card
    bad = sorted({f"{os.path.relpath(p, STUDIO)}: {n}" for p, n in names if AD_LIKE.search(n)})
    assert not bad, "ad blockers would hide these: " + ", ".join(bad)
