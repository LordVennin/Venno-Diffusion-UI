"""python -m venno_moss.install [options]: install what Venno MOSS needs into the current Python (venv-moss; run
by moss.sh / moss.bat). PyTorch first, with Venno's command (TORCH_COMMAND / TORCH_INDEX_URL override it), then
requirements-moss.txt when anything is missing or at another version. Cheap when everything is in place;
--skip-install skips it."""

from __future__ import annotations

import os
import platform
import sys

from venno.install import TORCH_COMMAND, installed, mismatches, pip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIREMENTS = os.path.join(ROOT, "requirements-moss.txt")


def main():
    argv = set(sys.argv[1:])
    if argv & {"-h", "--help", "--skip-install"}:
        return
    if sys.version_info < (3, 10):
        raise SystemExit(f"[Venno MOSS] Python 3.10 or newer is needed (this is {platform.python_version()})")
    if not installed("torch"):
        command = os.environ.get("TORCH_COMMAND") or ("torch torchvision" if sys.platform == "darwin" else TORCH_COMMAND)
        pip(command, "PyTorch")
    wrong = mismatches(REQUIREMENTS)
    if wrong:
        print(f"[Venno MOSS] installing: {', '.join(wrong)}", flush=True)
        pip(f'-r "{REQUIREMENTS}"', "requirements")


if __name__ == "__main__":
    main()
