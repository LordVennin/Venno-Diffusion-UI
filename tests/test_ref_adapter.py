"""The reference adapter (venno/engine/ref_adapter.py): shifted positions, guidance, the hooked LoRA, letterboxing."""

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from torch import nn

from backend.nn.anima import VideoRopePosition3DEmb
from venno.engine import ref_adapter
from venno.engine.reference import Reference, letterbox
from venno.server import jobs


def rope():
    return VideoRopePosition3DEmb(head_dim=128, h_extrapolation_ratio=1.0, w_extrapolation_ratio=1.0, t_extrapolation_ratio=1.0)


def test_shifted_positions_match_the_plain_ones_without_offset():
    pe = rope()
    T, H, W = 2, 5, 7
    plain = pe(torch.zeros(1, T, H, W, 8), device=torch.device("cpu"))
    shifted = pe(torch.zeros(1, T, H, W, 8), device=torch.device("cpu"), h_offsets=[0, 0])
    assert shifted.shape == plain.shape
    assert torch.allclose(shifted, plain, atol=2e-3)  # the plain path rounds its frequencies through fp16


def test_reference_frame_sits_below_the_picture():
    pe = rope()
    T, H, W = 2, 4, 3
    emb = pe(torch.zeros(1, T, H, W, 8), device=torch.device("cpu"), h_offsets=[0, H]).reshape(T, H, W, -1, 2, 2)
    tall = pe(torch.zeros(1, 1, 2 * H, W, 8), device=torch.device("cpu"), h_offsets=[0]).reshape(1, 2 * H, W, -1, 2, 2)
    n_t = 128 - 2 * (128 // 6 * 2)  # temporal part of the head: t positions differ between the frames
    h_part = slice(n_t // 2, None)
    assert torch.allclose(emb[1, :, :, h_part], tall[0, H:, :, h_part])  # rows H..2H-1, as if below the picture
    assert torch.allclose(emb[0, :, :, h_part], tall[0, :H, :, h_part])


def test_guidance_plain_cfg_at_reference_guidance_1():
    g = torch.Generator().manual_seed(0)
    pos, neg, base = (torch.randn(2, 4, 1, 6, 6, generator=g) for _ in range(3))
    out = ref_adapter.guidance(pos, neg, base, cfg=4.5, ref_scale=1.0, correction="none")
    assert torch.allclose(out, neg + 4.5 * (pos - neg), atol=1e-5)


def test_guidance_corrections():
    g = torch.Generator().manual_seed(1)
    pos, neg, base = (torch.randn(2, 4, 1, 6, 6, generator=g) for _ in range(3))
    apg = ref_adapter.guidance(pos, neg, base, cfg=4.5, ref_scale=6.0, correction="apg")
    update = (apg - pos).flatten(1)
    assert torch.allclose((update * pos.flatten(1)).sum(1), torch.zeros(2), atol=1e-3)  # nothing left along pos
    full = ref_adapter.guidance(pos, neg, base, cfg=4.5, ref_scale=6.0, correction="rescale", phi=1.0)
    assert torch.allclose(full.flatten(1).std(1), pos.flatten(1).std(1), atol=1e-4)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4, bias=False)


class Dit(nn.Module):
    def __init__(self, blocks=28):
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(blocks))


def adapter(tmp_path):
    down, up = torch.randn(2, 4), torch.randn(4, 2)
    path = tmp_path / "anima_ref_adapter_test.safetensors"
    save_file({"diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight": down, "diffusion_model.blocks.0.self_attn.q_proj.lora_up.weight": up,
               "diffusion_model.blocks.0.self_attn.q_proj.alpha": torch.tensor(1.0)}, str(path))
    return ref_adapter.Adapter(str(path)), down, up


def test_lora_only_while_active(tmp_path):
    a, down, up = adapter(tmp_path)
    dit = Dit()
    a.attach(dit)
    a.attach(dit)  # once per model
    layer = dit.blocks[0].self_attn.q_proj
    x = torch.randn(3, 4)
    plain = layer(x)
    with a.active():
        boosted = layer(x)
    assert torch.allclose(layer(x), plain)
    assert torch.allclose(boosted, plain + x @ down.T @ up.T * 0.5, atol=1e-5)  # alpha 1 / rank 2
    with pytest.raises(ValueError, match="28-block"):
        a.attach(Dit(40))


def test_letterbox_and_offsets():
    tall = Image.new("RGB", (100, 200), (255, 0, 0))
    out = letterbox(tall, 160, 160)
    assert out.size == (160, 160)
    assert out.getpixel((10, 80)) == (255, 255, 255) and out.getpixel((80, 80)) == (255, 0, 0)  # white sides, picture centred
    ref = Reference([tall], offset=1.0)
    ref._current = torch.zeros(1, 16, 1, 8, 8)
    assert ref.h_offsets(2, 4) == [0, 4]
    assert Reference([tall]).h_offsets(2, 4) is None


def test_job_needs_a_reference():
    assert "reference adapter" in jobs.unsupported(jobs.Job(reference_adapter=True))
    assert jobs.unsupported(jobs.Job(reference_adapter=True, references=["data:image/png;base64,"])) is None
