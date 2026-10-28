"""Check for updates (Settings → System): compare this install with the newest version on GitHub, and update
on request. Nothing is checked or installed unless the user asks.

Releases are the `main` branch of the repository; each one raises venno/version.py and adds a section to
CHANGELOG.md. A git checkout is updated with a fast-forward pull of `main` (never when the install has local
edits or commits that aren't on `main`); an install without git gets a download link. Either way the new code
runs after a restart, which also installs changed requirements.
"""

from __future__ import annotations

import os
import re
import subprocess
import urllib.request
from typing import Optional

from venno import config
from venno.version import VERSION

REPO = "LordVennin/Venno-Diffusion-UI"
BRANCH = "main"
REMOTE = f"https://github.com/{REPO}"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
DOWNLOAD = f"{REMOTE}/archive/refs/heads/{BRANCH}.zip"


def _git(*args: str, timeout: float = 60) -> str:
    out = subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError((out.stderr or out.stdout).strip().splitlines()[-1] if (out.stderr or out.stdout).strip() else f"git {args[0]} failed")
    return out.stdout.strip()


def _is_git_checkout() -> bool:
    if not os.path.exists(os.path.join(config.ROOT, ".git")):
        return False
    try:
        _git("rev-parse", "--is-inside-work-tree", timeout=10)
        return True
    except Exception:  # git not installed, or not usable here
        return False


def parse_version(text: str) -> Optional[str]:
    """VERSION from the text of a venno/version.py."""
    m = re.search(r'^VERSION\s*=\s*"([^"]+)"', text or "", re.M)
    return m.group(1) if m else None


def version_key(v: str) -> tuple:
    return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.\-+]", v or "0"))


def changes_since(changelog: str, version: str) -> list[dict]:
    """The CHANGELOG.md sections (## <version> …) newer than `version`, newest first."""
    out = []
    for m in re.finditer(r"^## \[?v?([0-9][^\]\s]*)\]?[^\n]*\n(.*?)(?=^## |\Z)", changelog or "", re.M | re.S):
        if version_key(m.group(1)) > version_key(version):
            out.append({"version": m.group(1), "notes": m.group(2).strip()})
    return out


def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=20) as r:  # noqa: S310 (fixed https URL)
        return r.read().decode("utf-8", "replace")


def check() -> dict:
    """Where this install stands against `main` on GitHub."""
    out = {"version": VERSION, "latest": None, "update_available": False, "changes": [], "commits": [], "can_update": False, "reason": "", "git": False}
    if not _is_git_checkout():
        latest = parse_version(_fetch(f"{RAW}/venno/version.py"))
        out.update(latest=latest, download=DOWNLOAD, reason="This install isn't a git checkout: download the new version and copy your models and outputs over, or install it with git to update from here.")
        if latest and version_key(latest) > version_key(VERSION):
            out.update(update_available=True, changes=changes_since(_fetch(f"{RAW}/CHANGELOG.md"), VERSION))
        return out

    out["git"] = True
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", timeout=10)
    _git("fetch", "--quiet", REMOTE, f"+refs/heads/{BRANCH}:refs/remotes/venno-update/{BRANCH}", timeout=120)
    target = f"refs/remotes/venno-update/{BRANCH}"
    behind = int(_git("rev-list", "--count", f"HEAD..{target}", timeout=10))
    ahead = int(_git("rev-list", "--count", f"{target}..HEAD", timeout=10))
    try:
        latest = parse_version(_git("show", f"{target}:venno/version.py", timeout=10))
    except RuntimeError:  # a main from before version numbers
        latest = None
    commits = [line.split("\t", 1) for line in _git("log", "--format=%h%x09%s", "-n", "50", f"HEAD..{target}", timeout=10).splitlines() if "\t" in line]
    try:
        changelog = _git("show", f"{target}:CHANGELOG.md", timeout=10)
    except RuntimeError:
        changelog = ""
    edited = [line[3:] for line in _git("status", "--porcelain", "--untracked-files=no", timeout=20).splitlines()]
    out.update(branch=branch, commit=_git("rev-parse", "--short", "HEAD", timeout=10), latest=latest, behind=behind, ahead=ahead,
               update_available=behind > 0, changes=changes_since(changelog, VERSION), commits=[{"id": c, "title": t} for c, t in commits])
    if behind == 0:
        out["reason"] = "Up to date." if branch == BRANCH or ahead == 0 else f"This install is on the branch '{branch}' with changes that aren't on '{BRANCH}'."
    elif edited:
        out["reason"] = f"Files of this install were edited ({', '.join(edited[:5])}{'…' if len(edited) > 5 else ''}): update by hand with git."
    elif ahead:
        out["reason"] = f"This install has {ahead} commit(s) that aren't on '{BRANCH}': update by hand with git."
    else:
        out["can_update"] = True
        if branch != BRANCH:
            out["reason"] = f"Updating also moves this install from the branch '{branch}' to '{BRANCH}', where releases are made."
    return out


def apply() -> dict:
    """Fast-forward this git checkout to `main` (after the same checks as `check`)."""
    state = check()
    if not state["git"] or not state["can_update"]:
        raise RuntimeError(state["reason"] or "Already up to date")
    target = f"refs/remotes/venno-update/{BRANCH}"
    if state["branch"] != BRANCH:
        try:
            _git("rev-parse", "--verify", "--quiet", f"refs/heads/{BRANCH}", timeout=10)
            _git("checkout", "--quiet", BRANCH, timeout=60)
        except RuntimeError:
            _git("checkout", "--quiet", "-b", BRANCH, timeout=60)
    _git("merge", "--ff-only", "--quiet", target, timeout=120)
    try:  # pull from main from now on, for anyone who updates with git by hand
        _git("fetch", "--quiet", "origin", BRANCH, timeout=120)
        _git("branch", "--quiet", f"--set-upstream-to=origin/{BRANCH}", timeout=10)
    except RuntimeError:
        pass
    return {"updated": True, "from": VERSION, "to": state["latest"], "commit": _git("rev-parse", "--short", "HEAD", timeout=10)}
