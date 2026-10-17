"""python -m venno.install [options]: install what Venno needs into the current Python (run by venno.sh / venno.bat).

PyTorch first (a CUDA build by default; TORCH_COMMAND / TORCH_INDEX_URL override it), then requirements.txt
when anything is missing or at another version, then the optional attention packages asked for with
--xformers / --sage / --flash. Does nothing when everything is in place, so it is cheap on every start;
--skip-install skips it altogether (you manage the environment yourself).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TORCH_INDEX_URL = os.environ.get("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu130")
TORCH_COMMAND = os.environ.get("TORCH_COMMAND", f"torch==2.13.0+cu130 torchvision==0.28.0+cu130 --extra-index-url {TORCH_INDEX_URL}")
XFORMERS = os.environ.get("XFORMERS_PACKAGE", f"xformers==0.0.35 --extra-index-url {TORCH_INDEX_URL}")


def pip(args: str, what: str):
    print(f"[Venno] installing {what}…", flush=True)
    result = subprocess.run(f'"{sys.executable}" -m pip install {args}', shell=True)
    if result.returncode != 0:
        raise SystemExit(f"[Venno] could not install {what} (pip exit code {result.returncode})")


def installed(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except ModuleNotFoundError:
        return False


def requirements_met(path: str) -> bool:
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        spec, _, marker = line.partition(";")
        if marker:
            continue  # environment-specific lines: let pip decide
        name, _, version = spec.partition("==")
        try:
            have = importlib.metadata.version(name.strip())
        except importlib.metadata.PackageNotFoundError:
            return False
        if version and have != version.strip():
            return False
    return True


def torch_versions() -> tuple[str, str]:
    """('2.13', 'cu130') of the installed torch."""
    v = importlib.metadata.version("torch")
    base, _, local = v.partition("+")
    return ".".join(base.split(".")[:2]), local or "cpu"


def main():
    argv = set(sys.argv[1:])
    if argv & {"-h", "--help", "--skip-install"}:
        return
    if sys.version_info < (3, 10):
        raise SystemExit(f"[Venno] Python 3.10 or newer is needed (this is {platform.python_version()})")

    if "--reinstall-torch" in argv or not installed("torch") or not installed("torchvision"):
        command = os.environ.get("TORCH_COMMAND") or ("torch torchvision" if sys.platform == "darwin" else TORCH_COMMAND)
        pip(command, "PyTorch")

    req = os.path.join(ROOT, "requirements.txt")
    if not requirements_met(req):
        pip(f'-r "{req}"', "requirements")

    windows = os.name == "nt"
    if "--xformers" in argv and not installed("xformers"):
        pip(f"--no-deps {XFORMERS}", "xformers")
    if "--sage" in argv or "--flash" in argv:
        v_torch, cuda = torch_versions()
        py = f"cp{sys.version_info.major}{sys.version_info.minor}"
        if "--sage" in argv:
            if not installed("triton"):
                pip("-U --no-deps " + os.environ.get("TRITON_PACKAGE", "triton-windows==3.7.1.post27" if windows else "triton==3.7.1"), "triton")
            if not installed("sageattention"):
                sage = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/" f"sageattention-2.2.0+{cuda}torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl" if windows else "sageattention==2.2.0"
                pip("-U --no-deps " + os.environ.get("SAGE_PACKAGE", sage), "sageattention")
        if "--flash" in argv and not installed("flash_attn"):
            release, plat = ("v0.9.52", "win_amd64") if windows else ("v0.9.47", "linux_x86_64")
            flash = f"https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/{release}/flash_attn-2.8.3+{cuda}torch{v_torch}-{py}-{py}-{plat}.whl"
            pip(os.environ.get("FLASH_PACKAGE", flash), "flash_attn")
    if "--pynvml" in argv and not installed("pynvml"):
        pip("nvidia-ml-py==13.610.43", "pynvml")


if __name__ == "__main__":
    main()
