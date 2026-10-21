from types import SimpleNamespace

import pytest
import torch

import backend  # noqa: F401  (puts the vendored packages on sys.path)
import k_diffusion.external
from backend.modules.k_prediction import PredictionDiscreteFlow
from venno.engine import schedulers


@pytest.fixture(scope="module")
def linker():
    """Anima's flow schedule (multiplier 1, shift 3)."""
    predictor = PredictionDiscreteFlow(SimpleNamespace(sampling_settings={"shift": 3.0, "multiplier": 1.0}))
    return k_diffusion.external.ForgeScheduleLinker(predictor)


@pytest.mark.parametrize("name", list(schedulers.SCHEDULERS))
@pytest.mark.parametrize("steps", [1, 4, 20])
def test_every_scheduler_gives_a_valid_flow_schedule(linker, name, steps):
    if name == "DDIM" and steps < 4:
        pytest.skip("DDIM (uniform from timestep 1, as A1111 / ComfyUI) starts far below sigma max at 1-3 steps")
    s = schedulers.get_sigmas(name, steps, linker)
    assert len(s) == steps + 1, name
    assert torch.isfinite(s).all(), name
    assert s[-1] == 0, name
    assert float(s[0]) <= 1.0 + 1e-4, f"{name} starts above Anima's sigma max: {float(s[0])}"
    assert float(s[0]) > 0.5, name
    assert (s[:-1] >= s[1:] - 1e-6).all(), f"{name} is not descending: {s.tolist()}"


def test_uniform_regression(linker):
    # Uniform used to hand table indices to the flow schedule: sigmas ~1.5 and NaN images
    s = schedulers.get_sigmas("Uniform", 8, linker)
    normal = schedulers.get_sigmas("Normal", 8, linker)
    assert float(s[0]) == pytest.approx(1.0, abs=1e-3)
    assert torch.allclose(s[:-2], normal[:-2], atol=0.01)


def test_turbo_and_kl_optimal_regressions(linker):
    # Turbo read table indices as fractions (sigma ~1.5); KL Optimal never reached 0
    assert float(schedulers.get_sigmas("Turbo", 4, linker)[0]) == pytest.approx(1.0, abs=1e-3)
    kl = schedulers.get_sigmas("KL Optimal", 6, linker)
    assert kl[-1] == 0 and float(kl[0]) == pytest.approx(1.0, abs=1e-3) and float(kl[-2]) == pytest.approx(float(linker.sigmas[0]), rel=1e-3)


def test_discard_penultimate(linker):
    s = schedulers.get_sigmas("Simple", 8, linker, discard_penultimate=True)
    assert len(s) == 9 and s[-1] == 0


def test_automatic_uses_the_samplers_default(linker):
    assert torch.equal(schedulers.get_sigmas("Automatic", 6, linker, default="Karras"), schedulers.get_sigmas("Karras", 6, linker))


def test_unknown_scheduler(linker):
    with pytest.raises(ValueError):
        schedulers.get_sigmas("Nope", 6, linker)
