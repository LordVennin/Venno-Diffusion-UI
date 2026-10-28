"""Check for updates (venno/updates.py), against a local stand-in for the GitHub repository."""

import os
import subprocess

import pytest

from venno import updates

GIT = ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "init.defaultBranch=main"]


def git(cwd, *args):
    return subprocess.run([*GIT, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def release(work, version, note):
    os.makedirs(os.path.join(work, "venno"), exist_ok=True)
    with open(os.path.join(work, "venno", "version.py"), "w") as f:
        f.write(f'VERSION = "{version}"\n')
    with open(os.path.join(work, "CHANGELOG.md"), "a") as f:
        f.write(f"\n## {version}\n\n- {note}\n")
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", f"Release {version}")


@pytest.fixture
def repos(tmp_path, monkeypatch):
    """A 'GitHub' repository with release 0.9.0, an install cloned from it, and a newer release 0.9.1 upstream."""
    remote, author, install = tmp_path / "remote.git", tmp_path / "author", tmp_path / "install"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(tmp_path, "clone", "-q", str(remote), str(author))
    with open(author / "CHANGELOG.md", "w") as f:
        f.write("# Changelog\n")
    release(author, "0.9.0", "first")
    git(author, "push", "-q", "origin", "HEAD:main")
    git(tmp_path, "clone", "-q", "-b", "main", str(remote), str(install))
    release(author, "0.9.1", "faster regions")
    git(author, "push", "-q", "origin", "HEAD:main")
    monkeypatch.setattr(updates.config, "ROOT", str(install))
    monkeypatch.setattr(updates, "REMOTE", str(remote))
    monkeypatch.setattr(updates, "VERSION", "0.9.0")
    return install


def test_versions_and_changelog():
    assert updates.parse_version('"""x"""\n\nVERSION = "1.2.3"\n') == "1.2.3"
    assert updates.version_key("0.10.0") > updates.version_key("0.9.9")
    log = "# Changelog\n\n## 0.9.2 — 2026-11-01\n\n- b\n\n## 0.9.1\n\n- a\n\n## 0.9.0\n\n- old\n"
    assert [c["version"] for c in updates.changes_since(log, "0.9.0")] == ["0.9.2", "0.9.1"]
    assert updates.changes_since(log, "0.9.0")[1]["notes"] == "- a"


def test_check_and_update(repos):
    state = updates.check()
    assert state["git"] and state["update_available"] and state["can_update"]
    assert state["latest"] == "0.9.1" and state["behind"] == 1 and [c["version"] for c in state["changes"]] == ["0.9.1"]
    result = updates.apply()
    assert result["to"] == "0.9.1"
    assert 'VERSION = "0.9.1"' in (repos / "venno" / "version.py").read_text()
    assert not updates.check()["update_available"]


def test_moves_an_install_on_another_branch_to_main(repos):
    git(repos, "checkout", "-q", "-b", "ccr-old-branch")
    state = updates.check()
    assert state["can_update"] and "moves this install" in state["reason"]
    updates.apply()
    assert git(repos, "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_local_edits_block_the_update(repos):
    (repos / "venno" / "version.py").write_text('VERSION = "0.9.0"  # edited\n')
    state = updates.check()
    assert state["update_available"] and not state["can_update"] and "edited" in state["reason"]
    with pytest.raises(RuntimeError):
        updates.apply()
