"""backend/state_dict.load_state_dict refuses files that don't fit the model."""

import pytest
import torch
from torch import nn

from backend.state_dict import load_state_dict


def model():
    with torch.device("meta"):
        return nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64, bias=False), nn.Linear(64, 4))


def weights():
    return {k: torch.ones(v.shape) for k, v in model().state_dict().items()}


def test_complete():
    m = model()
    load_state_dict(m, weights())
    assert all(p.device.type == "cpu" and p.sum() == p.numel() for p in m.parameters())


def test_a_few_missing_are_zeroed():
    m = model()
    sd = weights()
    del sd["2.bias"]
    load_state_dict(m, sd)
    assert m[2].bias.abs().sum() == 0 and m[0].weight.sum() == m[0].weight.numel()


def test_wrong_file_is_refused():
    with pytest.raises(ValueError, match="doesn't fit"):
        load_state_dict(model(), {"2.weight": torch.ones(4, 64), "unrelated.weight": torch.ones(3)}, log_name="VAE")
