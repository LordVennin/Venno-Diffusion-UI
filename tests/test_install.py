"""Startup check of installed packages against requirements.txt (venno/install.py)."""

import importlib.metadata

from venno import install


def test_mismatches(tmp_path):
    req = tmp_path / "requirements.txt"
    have = importlib.metadata.version("pytest")
    req.write_text(f"# comment\npytest=={have}\nnot-a-real-package-xyz==1.0\npytest==0.0.1  # wrong version\nfoo==1; sys_platform == 'nope'\n")
    assert install.mismatches(str(req)) == ["not-a-real-package-xyz (missing)", f"pytest {have} (needs 0.0.1)"]
    assert not install.requirements_met(str(req))
    req.write_text(f"pytest=={have}\n")
    assert install.requirements_met(str(req))
