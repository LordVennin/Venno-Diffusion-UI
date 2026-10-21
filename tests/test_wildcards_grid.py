import random

import pytest

from venno.engine import wildcards
from venno.pipeline import grid


@pytest.fixture
def wc_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(wildcards, "folder", lambda: str(tmp_path))
    (tmp_path / "hair").mkdir()
    (tmp_path / "hair" / "colors.txt").write_text("red\nblue\n# a comment\n\ngreen\n", encoding="utf-8")
    (tmp_path / "pose.txt").write_text("sitting\nstanding\n", encoding="utf-8")
    return tmp_path


def test_choices_are_one_of_the_options():
    for seed in range(20):
        assert wildcards.for_seed("{a|b|c} cat", seed) in ("a cat", "b cat", "c cat")


def test_same_seed_same_choice():
    text = "{red|green|blue|black} hair, {sitting|standing}"
    assert wildcards.for_seed(text, 7, "prompt") == wildcards.for_seed(text, 7, "prompt")
    assert len({wildcards.for_seed(text, s) for s in range(40)}) > 3


def test_several_without_repeats():
    for seed in range(20):
        picked = wildcards.for_seed("{2$$a|b|c}", seed).split(", ")
        assert len(picked) == 2 and len(set(picked)) == 2


def test_weights():
    rng = random.Random(1)
    hits = sum(wildcards.resolve("{0.05::rare|common}", rng) == "rare" for _ in range(400))
    assert hits < 60


def test_escapes_and_nesting():
    assert wildcards.resolve(r"\{not a choice\}", random.Random(0)) == "{not a choice}"
    assert wildcards.resolve("{x {a|a}|x a}", random.Random(0)) == "x a"


def test_files(wc_folder):
    assert wildcards.files() == ["hair/colors", "pose"]
    for seed in range(10):
        assert wildcards.for_seed("__hair/colors__ hair, __pose__", seed).split(" hair, ")[0] in ("red", "blue", "green")
    assert wildcards.for_seed("__h*/col*__", 3) in ("red", "blue", "green")
    with pytest.raises(ValueError):
        wildcards.for_seed("__missing__", 1)


def test_files_cannot_escape_the_folder(wc_folder):
    with pytest.raises(ValueError):
        wildcards.for_seed("__../../etc/passwd__", 1)


def test_has_syntax():
    assert wildcards.has_syntax("{a|b}") and wildcards.has_syntax("__x__") and not wildcards.has_syntax(r"plain \{x\}")


@pytest.mark.parametrize("kind,values,expected", [
    ("int", "1-5", [1, 2, 3, 4, 5]),
    ("int", "10-1 (-3)", [10, 7, 4, 1]),
    ("int", "1-10 (+3)", [1, 4, 7, 10]),
    ("float", "0.5-1.5 [3]", [0.5, 1.0, 1.5]),
    ("float", "1, 2.5, 4", [1.0, 2.5, 4.0]),
    ("str", 'a, "b, c", d', ["a", "b, c", "d"]),
    ("str", ["x", " y "], ["x", "y"]),
])
def test_grid_values(kind, values, expected):
    assert grid.parse_values(kind, values) == expected


def test_grid_bad_values():
    with pytest.raises(ValueError):
        grid.parse_values("float", "1-5 (-1)")
    with pytest.raises(ValueError):
        grid.parse_values("int", "abc")


def test_grid_axes_resolve():
    from venno.server.jobs import Grid, GridAxis

    axes = grid.resolve_axes(Grid(x=GridAxis(type="cfg", values="1, 2"), y=GridAxis(type="frame_separation", values="0, 0.2")))
    assert [a.type.key for a in axes] == ["cfg", "frame_separation"] and axes[1].values == [0.0, 0.2]
    with pytest.raises(ValueError):
        grid.resolve_axes(Grid(x=GridAxis(type="nope", values="1")))
