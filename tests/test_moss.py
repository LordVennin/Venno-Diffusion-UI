"""Venno MOSS (MOSS-Image LoRA tester): LoRA loading, the comparison grid, model discovery and the job runner,
on a tiny random MOSS-Image transformer (the real model is 27 GB)."""

import copy
import io
import os

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from venno_moss import config, grid, jobs, lora, models
from venno_moss import engine as engine_module
from venno_moss.model.fp8 import quantize_modules
from venno_moss.model.transformer import MossImageParams, MossImageTransformer

DIM, FFN, LAYERS, RANK = 64, 128, 24, 4


@pytest.fixture
def dit():
    torch.manual_seed(1)
    model = MossImageTransformer(MossImageParams(dim=DIM, ffn_dim=FFN, num_heads=4, num_layers=LAYERS, text_dim=DIM, freq_dim=256, latent_channels=128))
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)
    return model.eval()


def trainer_lora(seed=0, shared_down=True, prefix="diffusion_model.", layers=LAYERS):
    """A LoRA in venno-trainer's export layout (ComfyUI MOSS keys; q/k/v sharing one down projection)."""
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for i in range(layers):
        p = f"{prefix}layers.{i}.proj"
        down = torch.randn(RANK, DIM, generator=g)
        for part in "qkv":
            sd[f"{p}.{part}.lora_A.weight"] = down.clone() if shared_down else torch.randn(RANK, DIM, generator=g)
            sd[f"{p}.{part}.lora_B.weight"] = torch.randn(DIM, RANK, generator=g)
            sd[f"{p}.{part}.alpha"] = torch.tensor(2.0)
        for name, (i_f, o_f) in {"o": (DIM, DIM), "ffn_up_gate": (DIM, 2 * FFN), "ffn_down": (FFN, DIM)}.items():
            sd[f"{p}.{name}.lora_A.weight"] = torch.randn(RANK, i_f, generator=g)
            sd[f"{p}.{name}.lora_B.weight"] = torch.randn(o_f, RANK, generator=g)
            sd[f"{p}.{name}.alpha"] = torch.tensor(2.0)
    return sd


def forward(model):
    torch.manual_seed(5)
    x, ctx = torch.randn(1, 128, 6, 6), torch.randn(1, 5, DIM)
    with torch.inference_mode():
        return model(x, torch.tensor([500.0]), ctx, torch.ones(1, 5, dtype=torch.bool))


def merged(model, sd, weight=1.0):
    ref = copy.deepcopy(model)
    mods = {n: m for n, m in ref.named_modules() if isinstance(m, torch.nn.Linear)}
    patches, unknown = lora.parse(sd, mods)
    assert not unknown
    with torch.no_grad():
        for name, parts in patches.items():
            for part in parts:
                delta = (part.up @ part.down) * part.scale * weight
                if part.rows:
                    mods[name].weight[part.rows[0] : part.rows[1]] += delta
                else:
                    mods[name].weight += delta
    return ref


def save(tmp_path, name, sd):
    path = tmp_path / name
    save_file(sd, str(path))
    return str(path)


# ---------------------------------------------------------------- LoRAs


@pytest.mark.parametrize("shared_down", [True, False])
def test_lora_matches_merged_weights(dit, tmp_path, shared_down):
    sd = trainer_lora(shared_down=shared_down)
    path = save(tmp_path, "a.safetensors", sd)
    base = forward(dit)
    reference = forward(merged(dit, sd, 0.7))
    stack = lora.LoraStack(dit, torch.device("cpu"), torch.float32)
    assert stack.set([(path, 0.7)]) == []
    out = forward(dit)
    assert torch.allclose(out, reference, atol=1e-5) and not torch.allclose(out, base)
    stack.set([(path, 0.0)])  # weight 0: off
    assert torch.allclose(forward(dit), base)
    stack.set([(path, 0.7)])
    stack.clear()
    assert torch.allclose(forward(dit), base)


def test_two_loras_and_key_styles(dit, tmp_path):
    a = trainer_lora(1)
    b = {k.replace("diffusion_model.", "transformer.").replace("lora_A", "lora_down").replace("lora_B", "lora_up"): v for k, v in trainer_lora(2).items()}
    reference = forward(merged(merged(dit, a, 1.0), b, 0.5))
    stack = lora.LoraStack(dit, torch.device("cpu"), torch.float32)
    stack.set([(save(tmp_path, "a.safetensors", a), 1.0), (save(tmp_path, "b.safetensors", b), 0.5)])
    assert torch.allclose(forward(dit), reference, atol=1e-5)


def test_kohya_style_keys(dit):
    sd = {}
    for k, v in trainer_lora(3, layers=2).items():
        stem, _, suffix = k.removeprefix("diffusion_model.").partition(".lora_")
        if not suffix:
            stem, suffix = k.removeprefix("diffusion_model.").rsplit(".alpha", 1)[0], "alpha"
            sd[f"lora_unet_{stem.replace('.', '_')}.alpha"] = v
        else:
            sd[f"lora_unet_{stem.replace('.', '_')}.lora_{suffix}"] = v
    patches, unknown = lora.parse(sd, {n: m for n, m in dit.named_modules() if isinstance(m, torch.nn.Linear)})
    assert not unknown and set(patches) == {f"layers.{i}.proj.{p}" for i in range(2) for p in ("qkv", "o", "ffn_up_gate", "ffn_down")}


def test_lora_for_another_model_is_refused(dit, tmp_path):
    sd = {"diffusion_model.layers.0.proj.o.lora_A.weight": torch.zeros(4, 99), "diffusion_model.layers.0.proj.o.lora_B.weight": torch.zeros(DIM, 4)}
    stack = lora.LoraStack(dit, torch.device("cpu"), torch.float32)
    with pytest.raises(ValueError, match="doesn't fit"):
        stack.set([(save(tmp_path, "x.safetensors", sd), 1.0)])
    sdxl = {"lora_unet_down_blocks_0_attn.lora_down.weight": torch.zeros(4, 8), "lora_unet_down_blocks_0_attn.lora_up.weight": torch.zeros(8, 4)}
    with pytest.raises(ValueError, match="none of its weights fit"):
        stack.set([(save(tmp_path, "y.safetensors", sdxl), 1.0)])


def test_lora_on_fp8_weights(dit, tmp_path):
    sd = trainer_lora(4)
    reference = forward(merged(dit, sd, 1.0))
    quantize_modules(dit, ["layers."], compute_dtype=torch.float32)
    stack = lora.LoraStack(dit, torch.device("cpu"), torch.float32)
    stack.set([(save(tmp_path, "a.safetensors", sd), 1.0)])
    out = forward(dit)
    assert (out - reference).abs().max() < 0.05 * reference.abs().max()  # fp8 base weights: close, not exact


# ---------------------------------------------------------------- grid


def test_grid_values_and_lora_axes():
    assert grid.parse_values("float", "0-1 [3]") == [0.0, 0.5, 1.0]
    assert grid.parse_values("int", "20-40 (+10)") == [20, 30, 40]
    assert grid.parse_values("str", 'none, "a,b.safetensors", c') == ["none", "a,b.safetensors", "c"]
    job = jobs.Job(model="m", loras=[jobs.LoraRef(name="run-000004.safetensors", weight=0.8), jobs.LoraRef(name="style.safetensors")])
    lora_axis = grid.ResolvedAxis(grid.AXES["lora"], ["run-000008.safetensors", "none"])
    cell = job.model_copy(update=lora_axis.updates(job, "run-000008.safetensors"))
    assert [(lr.name, lr.weight) for lr in cell.loras] == [("run-000008.safetensors", 0.8), ("style.safetensors", 1.0)]
    empty = job.model_copy(update=lora_axis.updates(job, "none"))
    assert [lr.name for lr in empty.loras] == ["style.safetensors"]
    weight_axis = grid.ResolvedAxis(grid.AXES["lora_weight"], [0.5], target="style.safetensors")
    assert [lr.weight for lr in job.model_copy(update=weight_axis.updates(job, 0.5)).loras] == [0.8, 0.5]
    assert lora_axis.label("dir/run-000008.safetensors") == "run-000008" and lora_axis.label("none") == "no LoRA"
    with pytest.raises(ValueError, match="needs a LoRA"):
        grid.check(jobs.Job(model="m"), [grid.ResolvedAxis(grid.AXES["lora_weight"], [1.0])])


@pytest.fixture
def fake_engine(monkeypatch):
    """engine.generate stand-in: a flat picture whose shade encodes the LoRA weights and the seed."""
    class Calls(list):
        hook = None  # called with the number of calls so far

    calls = Calls()

    class Fake:
        def generate(self, **kw):
            calls.append(kw)
            if calls.hook:
                calls.hook(len(calls))
            for step in range(1, kw["steps"] + 1):
                if kw["is_cancelled"]():
                    raise engine_module.Cancelled
                kw["on_step"](step, kw["steps"])
            shade = int(sum(w for _, w in kw["loras"]) * 100 + kw["seed"]) % 256
            return Image.new("RGB", (kw["width"] // 8, kw["height"] // 8), (shade, shade, shade)), []

        def resolved_memory(self):
            return "full"

    monkeypatch.setattr(engine_module, "get", lambda: Fake())
    return calls


def test_grid_run(fake_engine):
    job = jobs.Job(model="m", width=256, height=256, steps=2, seed=7, save=False, loras=[jobs.LoraRef(name="a-000002.safetensors")],
                   grid=jobs.Grid(x=jobs.Axis(type="lora", values="none, a-000002.safetensors, a-000004.safetensors"),
                                  y=jobs.Axis(type="lora_weight", values="0.5, 1"), include_cells=True))
    result = jobs.run(job, jobs.Progress())
    assert len(fake_engine) == 6 and {c["seed"] for c in fake_engine} == {7}
    assert [c["loras"] for c in fake_engine][:3] == [[], [("a-000002.safetensors", 0.5)], [("a-000004.safetensors", 0.5)]]
    sheet = Image.open(io.BytesIO(result.images[0]))
    assert sheet.width > 3 * 32 and sheet.height > 2 * 32 and len(result.images) == 1 + 6
    assert 'Grid: "X: LoRA = no LoRA, a-000002, a-000004; Y: LoRA weight = 0.5, 1"' in result.infotexts[0]


# ---------------------------------------------------------------- jobs


def test_count_skip_cancel_and_settings_round_trip(fake_engine, monkeypatch, tmp_path):
    from venno_moss import metadata

    monkeypatch.setattr(config.args, "outputs_dir", str(tmp_path))
    rec = jobs.JobRecord(jobs.Job(model="MOSS-Image-SFT", prompt="a cat", count=4, steps=3, seed=10, loras=[jobs.LoraRef(name="x", weight=0.6)]))
    jobs._records[rec.id] = rec
    rec.status = "running"
    seen = []

    def watch(n):  # skip image 2, cancel during image 3
        if n == 2:
            rec.skip_event.set()
        if n == 3:
            rec.cancel_event.set()
        seen.append(len(rec.images))  # images already served when the next one starts

    fake_engine.hook = watch
    try:
        cancelled = jobs._run(rec)
    finally:
        jobs._records.pop(rec.id)
    assert cancelled and rec.seeds == [10] and rec.skipped == 1 and seen == [0, 1, 1]
    data = metadata.read(rec.images[0])
    assert data["settings"]["prompt"] == "a cat" and data["settings"]["seed"] == 10 and data["settings"]["loras"] == [{"name": "x", "weight": 0.6}]
    assert "Seed: 10" in data["parameters"] and "LoRA 1: x:0.6" in data["parameters"]
    assert len(os.listdir(next(os.scandir(tmp_path)).path)) == 1  # saved: the one finished picture


# ---------------------------------------------------------------- models on disk


def test_model_discovery(tmp_path, monkeypatch, dit):
    monkeypatch.setattr(config.args, "models_dir", str(tmp_path / "moss"))
    monkeypatch.setattr(config.args, "data_dir", str(tmp_path))
    weights = {k: v.contiguous() for k, v in dit.state_dict().items()}
    sft = tmp_path / "moss" / "OpenMOSS-Team__MOSS-Image-SFT"
    for part in ("transformer", "text_encoder", "vae"):
        (sft / part).mkdir(parents=True)
    save_file(weights, str(sft / "transformer" / "diffusion_pytorch_model.safetensors"))
    save_file({"w": torch.zeros(1)}, str(sft / "text_encoder" / "model.safetensors"))
    save_file({"w": torch.zeros(1)}, str(sft / "vae" / "diffusion_pytorch_model.safetensors"))
    (sft / "tokenizer").mkdir()
    (sft / "tokenizer" / "tokenizer.json").write_text("{}")
    base = tmp_path / "moss" / "MOSS-Image-Base" / "transformer"  # transformer only: shares the parts above
    base.mkdir(parents=True)
    save_file(weights, str(base / "diffusion_pytorch_model.safetensors"))
    save_file(weights, str(tmp_path / "moss" / "my-finetune.safetensors"))
    save_file(trainer_lora(layers=1), str(tmp_path / "moss" / "a-lora.safetensors"))  # a LoRA is not a model
    found = models.scan()
    assert [m.name for m in found.models] == ["MOSS-Image-Base", "MOSS-Image-SFT", "my-finetune"] and found.complete
    assert found.text_encoder == str(sft / "text_encoder") and found.model("my-finetune").root is None


def test_sigmas_follow_the_vendor_grid():
    sig = engine_module.moss_sigmas(50)
    assert sig[0] == 1.0 and sig[-1] == 0 and len(sig) == 51 and bool((sig[:-1] > sig[1:]).all())
