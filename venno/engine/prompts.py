"""Prompt syntax: A1111-compatible scheduling ([a:b:when], [a:when], [a::when], [a|b]) and AND composition.

Compatibility layer for prompts pasted from other UIs and old images. Studio also offers a plain
"then switch to … at N %" prompt, which is the same thing without the syntax (`PromptPlan.switch`).

Step numbers are 1-based, as in A1111: `[a:b:3]` uses `a` for steps 1-3 and `b` from step 4; a decimal is a
fraction of the steps (`[a:b:0.25]` at 20 steps switches after step 5). Steps are the ones actually sampled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import lark

_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER [WHITESPACE] "]"
alternate: "[" prompt ("|" [prompt])+ "]"
WHITESPACE: /\s+/
plain: /([^\\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""")

_positions = lark.Lark(_parser.source_grammar, propagate_positions=True)  # same grammar, for describe()

_AND = re.compile(r"\bAND\b")
_WEIGHT = re.compile(r"^((?:\s|.)*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")


@lru_cache(maxsize=256)
def schedule(prompt: str, steps: int) -> tuple[tuple[int, str], ...]:
    """((last_step, text), …) covering steps 1..steps. Unparsable prompts are used as plain text."""
    try:
        tree = _parser.parse(prompt)
    except lark.exceptions.LarkError:
        return ((steps, prompt),)

    ends = {steps}

    class Collect(lark.Visitor):
        def scheduled(self, t):
            s = t.children[-2]
            v = float(s)
            t.children[-2] = min(steps, int(v * steps) if "." in s else int(v))
            if t.children[-2] >= 1:
                ends.add(t.children[-2])

        def alternate(self, t):
            ends.update(range(1, steps + 1))

    Collect().visit(tree)

    def at(step: int) -> str:
        class AtStep(lark.Transformer):
            def scheduled(self, args):
                before, after, _, when, _ = args
                yield before or () if step <= when else after

            def alternate(self, args):
                args = ["" if not a else a for a in args]
                yield args[(step - 1) % len(args)]

            def start(self, args):
                def flatten(x):
                    if isinstance(x, str):
                        yield x
                    else:
                        for g in x:
                            yield from flatten(g)

                return "".join(flatten(args))

            def plain(self, args):
                yield args[0].value

            def __default__(self, data, children, meta):
                for child in children:
                    yield child

        return AtStep().transform(tree)

    return tuple((end, at(end)) for end in sorted(ends))


def describe(prompt: str, steps: int) -> list[dict]:
    """The scheduling constructs in a prompt, in reading order, for humans:
    {"kind": "switch", "from": "cat", "to": "dog", "after": 6} / {"kind": "alternate", "options": ["red", "blue"]}"""
    try:
        tree = _positions.parse(prompt)
    except lark.exceptions.LarkError:
        return []

    def text(node) -> str:
        if node is None or not hasattr(node, "meta") or getattr(node.meta, "empty", True):
            return ""
        return prompt[node.meta.start_pos : node.meta.end_pos].strip()

    out = []
    for node in tree.iter_subtrees_topdown():
        if node.data == "scheduled":
            before, after, number = node.children[0], node.children[1], str(node.children[-2])
            v = float(number)
            when = min(steps, int(v * steps) if "." in number else int(v))
            out.append({"kind": "switch", "from": text(before), "to": text(after), "after": when, "pos": node.meta.start_pos})
        elif node.data == "alternate":
            out.append({"kind": "alternate", "options": [text(c) for c in node.children], "pos": node.meta.start_pos})
    out.sort(key=lambda d: d.pop("pos"))
    return out


def split_and(prompt: str) -> list[tuple[str, float]]:
    """'a AND b:0.5' → [('a', 1.0), ('b', 0.5)]"""
    parts = []
    for sub in _AND.split(prompt):
        m = _WEIGHT.search(sub)
        text, weight = m.groups() if m else (sub, None)
        parts.append((text, float(weight) if weight is not None else 1.0))
    return parts


def has_syntax(prompt: str) -> bool:
    return len(split_and(prompt)) > 1 or len(schedule(prompt, 10)) > 1


@dataclass
class PromptPlan:
    """Which (text, weight) parts apply at each step of a run."""

    steps: int
    prompt: str
    switch: Optional[str] = None  # Studio's "then switch to …" prompt
    switch_at: float = 0.4  # fraction of the steps
    allow_and: bool = True  # negative and region prompts: AND is plain text, as in A1111
    _parts: dict = field(default_factory=dict, repr=False)

    def _top(self, step: int) -> str:
        if self.switch and step > int(self.switch_at * self.steps):
            return self.switch
        return self.prompt

    def parts(self, step: int) -> tuple[tuple[str, float], ...]:
        step = min(max(step, 1), self.steps)
        top = self._top(step)
        key = (top, step)
        if key not in self._parts:
            subs = split_and(top) if self.allow_and else [(top, 1.0)]
            out = []
            for text, weight in subs:
                for end, t in schedule(text, self.steps):
                    if step <= end:
                        out.append((t, weight))
                        break
            self._parts[key] = tuple(out)
        return self._parts[key]

    @property
    def composed(self) -> bool:
        """AND or non-unit weights anywhere (needs the negative pass even at CFG 1)."""
        return any(len(self.parts(s)) > 1 or any(w != 1.0 for _, w in self.parts(s)) for s in range(1, self.steps + 1))

    def segments(self) -> list[dict]:
        """Consecutive steps with the same parts merged: [{from, to, parts: [{text, weight}]}]."""
        out: list[dict] = []
        for step in range(1, self.steps + 1):
            parts = self.parts(step)
            if out and out[-1]["_key"] == parts:
                out[-1]["to"] = step
            else:
                out.append({"from": step, "to": step, "parts": [{"text": t, "weight": w} for t, w in parts], "_key": parts})
        for s in out:
            del s["_key"]
        return out
