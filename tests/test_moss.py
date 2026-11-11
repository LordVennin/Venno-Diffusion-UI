"""Venno MOSS (MOSS-Image LoRA tester): LoRA loading, the comparison grid, model discovery and the job runner,
on a tiny random MOSS-Image transformer (the real model is 27 GB)."""

import copy
import io
import json
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


def test_download_resumes_and_reports_a_hub_that_does_not_answer(tmp_path, monkeypatch):
    from venno_moss import downloads

    monkeypatch.setattr(config.args, "models_dir", str(tmp_path / "moss"))
    files = {"model_index.json": b"{}", "transformer/a.safetensors": b"x" * 3000, "transformer/b.safetensors": b"y" * 5000, "other/c.bin": b"z"}
    requests = []

    class Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=None):  # stands in for downloads._open
        url, rng = req.full_url, req.get_header("Range")
        requests.append((url.rsplit("/main", 1)[-1], rng))
        if "/api/models/" in url:
            return Resp(json.dumps([{"type": "file", "path": p, "size": len(b)} for p, b in files.items()]).encode())
        body = files[url.split("/resolve/main/", 1)[1]]
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        resp = Resp(body[start:])
        resp.status = 206 if start else 200
        return resp

    monkeypatch.setattr(downloads, "_open", urlopen)
    target = tmp_path / "moss" / "MOSS-Image-Base" / "transformer"
    target.mkdir(parents=True)
    (target / "b.safetensors.part").write_bytes(b"y" * 2000)  # an interrupted earlier download
    job = downloads.Download("base")
    job.run()
    state = job.to_dict()
    assert state["status"] == "done" and state["done"] == state["total"] == 2 + 3000 + 5000
    assert (target / "b.safetensors").read_bytes() == files["transformer/b.safetensors"] and not (target / "b.safetensors.part").exists()
    assert ("/transformer/b.safetensors", "bytes=2000-") in requests and not any("other" in u for u, _ in requests)

    def silent(req, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(downloads, "_open", silent)
    job = downloads.Download("base")
    job.run()
    assert job.status == "error" and "didn't answer" in job.error


def test_download_connects_over_ipv4_first(monkeypatch):
    import socket

    from venno_moss import downloads

    tried = []

    class FakeSocket:
        def __init__(self, family, kind, proto):
            self.family = family

        def settimeout(self, t):
            pass

        def connect(self, addr):
            tried.append(addr[0])
            if self.family == socket.AF_INET6:
                raise TimeoutError("timed out")  # an IPv6 route that goes nowhere

        def close(self):
            pass

    monkeypatch.setattr(socket, "getaddrinfo", lambda *a: [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2600::1", 443, 0, 0)),
                                                         (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", 443))])
    monkeypatch.setattr(socket, "socket", FakeSocket)
    sock = downloads._connect(("huggingface.co", 443), 60)
    assert sock.family == socket.AF_INET and tried == ["1.2.3.4"]


def test_lora_summary_and_quantized_files_skipped(tmp_path, dit):
    path = tmp_path / "a.safetensors"
    save_file(trainer_lora(layers=1), str(path), metadata={"ss_sd_model_name": "MOSS-Image-Base", "ss_network_alpha": "16"})
    assert lora.summary(str(path)) == {"trained_on": "MOSS-Image-Base", "rank": RANK, "alpha": 16.0}
    bare = tmp_path / "b.safetensors"
    save_file(trainer_lora(layers=1), str(bare))
    assert lora.summary(str(bare)) == {"trained_on": "", "rank": RANK, "alpha": None}
    weights = {k: v.contiguous() for k, v in dit.state_dict().items()}
    save_file(weights, str(tmp_path / "plain_bf16.safetensors"))
    save_file({**weights, "layers.0.proj.qkv.weight_scale": torch.ones(3 * DIM)}, str(tmp_path / "x_int8_convrot.safetensors"))
    assert models.is_transformer_file(str(tmp_path / "plain_bf16.safetensors"))
    assert not models.is_transformer_file(str(tmp_path / "x_int8_convrot.safetensors"))


# ---------------------------------------------------------------- fast preview, hires fix


@pytest.fixture
def folders(tmp_path, monkeypatch):
    monkeypatch.setattr(config.args, "data_dir", str(tmp_path))
    monkeypatch.setattr(config.args, "models_dir", str(tmp_path / "moss"))
    monkeypatch.setattr(config.args, "lora_dir", [])
    return tmp_path


def test_fast_preview(folders, fake_engine):
    from venno_moss import downloads, metadata

    job = jobs.Job(model="m", steps=50, cfg=3.0, fast=True, save=False, seed=1, loras=[jobs.LoraRef(name="mine.safetensors", weight=0.8)],
                   hires=jobs.Hires(cfg=4.0))
    assert "turbo LoRA" in jobs.unsupported(job)  # not downloaded yet
    turbo = downloads.Download("turbo")
    assert turbo.patterns == [config.TURBO_FILES[0]] and turbo.target == config.default_lora_path()
    save_file(trainer_lora(layers=1), os.path.join(config.default_lora_path(), "8-steps-dmd-lora.safetensors"))
    save_file(trainer_lora(layers=1), os.path.join(config.default_lora_path(), "8-steps-dmd2-lora.safetensors"))
    assert jobs.unsupported(job) is None and [c["installed"] for c in downloads.catalog() if c["kind"] == "lora"] == [True]
    ready = jobs.prepare(job)
    assert (ready.steps, ready.cfg, ready.hires.cfg) == (8, 1.0, None)
    assert [(lr.name, lr.weight) for lr in ready.loras] == [("mine.safetensors", 0.8), ("8-steps-dmd2-lora.safetensors", 1.0)]  # DMD2 preferred
    assert jobs.prepare(ready).loras == ready.loras  # not added twice
    result = jobs.run(ready, jobs.Progress())
    assert fake_engine[0]["steps"] == 8 and fake_engine[0]["cfg"] == 1.0 and "Fast preview: turbo LoRA" in result.infotexts[0]
    settings = metadata.read(result.images[0])["settings"]
    assert settings["fast"] is True and settings["loras"] == [{"name": "mine.safetensors", "weight": 0.8}]
    # the grid's LoRA axis swaps the job's own LoRA, never the turbo one
    axis = grid.ResolvedAxis(grid.AXES["lora"], ["other.safetensors", "none"])
    assert [lr.name for lr in ready.model_copy(update=axis.updates(ready, "other.safetensors")).loras] == ["other.safetensors", "8-steps-dmd2-lora.safetensors"]
    assert [lr.name for lr in ready.model_copy(update=axis.updates(ready, "none")).loras] == ["8-steps-dmd2-lora.safetensors"]
    alone = jobs.prepare(jobs.Job(model="m", fast=True))
    assert [lr.name for lr in alone.model_copy(update=axis.updates(alone, "other.safetensors")).loras] == ["other.safetensors", "8-steps-dmd2-lora.safetensors"]


def test_hires_job(folders, fake_engine):
    from venno_moss import metadata

    assert engine_module.hires_size(832, 1216, 1.5) == (1248, 1824)
    assert engine_module.hires_size(1024, 1024, resize_x=1536) == (1536, 1536) and engine_module.hires_size(832, 1216, resize_y=2048) == (1408, 2048)
    assert [engine_module.run_steps(50, d) for d in (0.0, 0.3, 0.5, 1.0)] == [1, 15, 25, 50]
    job = jobs.Job(model="m", width=832, height=1216, steps=4, seed=3, save=False, hires=jobs.Hires(upscaler="Lanczos", scale=1.5, denoise=0.4, steps=20, shift=3.0))
    assert jobs.unsupported(job) is None
    assert "not found" in jobs.unsupported(job.model_copy(update={"hires": jobs.Hires(upscaler="4x-Missing")}))
    assert "too large" in jobs.unsupported(job.model_copy(update={"width": 2048, "height": 2048, "hires": jobs.Hires(scale=2.0)}))
    result = jobs.run(job, jobs.Progress())
    assert fake_engine[0]["hires"] == dict(upscaler="Lanczos", width=1248, height=1824, steps=20, denoise=0.4, cfg=None, shift=3.0)
    for text in ("Size: 832x1216", "Hires resize: 1248x1824", "Hires upscaler: Lanczos", "Denoising strength: 0.4", "Hires steps: 20", "Hires shift: 3"):
        assert text in result.infotexts[0]
    assert metadata.read(result.images[0])["settings"]["hires"]["denoise"] == 0.4
    # grid axes over the hires settings
    axis = grid.ResolvedAxis(grid.AXES["hires_denoise"], [0.3, 0.6])
    assert job.model_copy(update=axis.updates(job, 0.6)).hires.denoise == 0.6
    with pytest.raises(ValueError, match="hires fix switched on"):
        grid.check(jobs.Job(model="m"), [axis])


def test_engine_hires_pass(folders, monkeypatch, dit):
    """The second pass: upscaled latents noised to the `denoise` point, then the last part of the hires schedule."""
    folder = folders / "moss" / "Tiny" / "transformer"
    folder.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in dit.state_dict().items()}, str(folder / "diffusion_pytorch_model.safetensors"))
    eng = engine_module.Engine()
    seen = {}

    def upscaled(latents, upscaler, width, height):
        seen.update(upscaler=upscaler, size=(width, height), first=tuple(latents.shape))
        return torch.nn.functional.interpolate(latents, size=(height // 16, width // 16))

    monkeypatch.setattr(eng, "encode", lambda text: torch.randn(5, DIM, generator=torch.Generator().manual_seed(len(text))))
    monkeypatch.setattr(eng, "upscale_latents", upscaled)
    monkeypatch.setattr(eng, "decode", lambda latents: latents)
    steps = []
    kw = dict(model="Tiny", prompt="a cat", negative="", width=128, height=128, steps=6, cfg=3.0, shift=5.0, seed=1, skip_layers=[22], loras=[])
    out, warnings = eng.generate(**kw, hires=dict(upscaler="Lanczos", width=192, height=192, steps=0, denoise=0.5, cfg=None, shift=None), on_step=lambda i, n: steps.append((i, n)))
    assert tuple(out.shape) == (1, 128, 12, 12) and seen == {"upscaler": "Lanczos", "size": (192, 192), "first": (1, 128, 8, 8)} and not warnings
    assert steps == [(i, 9) for i in range(1, 10)]  # 6 first-pass steps + the last 3 of 6
    plain, _ = eng.generate(**kw)
    again, _ = eng.generate(**kw)
    assert tuple(plain.shape) == (1, 128, 8, 8) and torch.equal(plain, again)
    full, _ = eng.generate(**kw, hires=dict(upscaler="Latent", width=192, height=192, steps=4, denoise=1.0, cfg=1.0, shift=None))
    assert tuple(full.shape) == (1, 128, 12, 12)


def test_vae_falls_back_to_tiles_when_memory_runs_out(folders):
    from venno_moss import upscale

    x = torch.arange(1 * 2 * 40 * 56, dtype=torch.float32).reshape(1, 2, 40, 56)
    double = lambda t: torch.nn.functional.interpolate(t, scale_factor=2, mode="nearest")
    assert torch.allclose(upscale.tiled(double, x, 16, 4, 2), double(x))  # tiles blend back to the whole
    calls = []

    def short_of_memory(t):
        calls.append(tuple(t.shape))
        if t.shape[-1] > 32:
            raise RuntimeError("CUDA out of memory. Tried to allocate 9.00 GiB")
        return double(t)

    eng = engine_module.Engine()
    assert torch.allclose(eng._vae(short_of_memory, x, 32, 8, 2), double(x)) and calls[0] == (1, 2, 40, 56) and len(calls) > 2
    with pytest.raises(RuntimeError, match="something else"):
        eng._vae(lambda t: (_ for _ in ()).throw(RuntimeError("something else")), x, 32, 8, 2)


def test_image_info_reads_moss_and_other_pictures():
    from PIL import PngImagePlugin

    from venno_moss import metadata

    job = jobs.Job(model="MOSS-Image-Base", prompt="a cat, on a roof\nsecond line", negative="blurry", steps=50, seed=5, loras=[jobs.LoraRef(name="a, b.safetensors", weight=0.7)])
    own = metadata.read(metadata.png_bytes(Image.new("RGB", (32, 32)), metadata.infotext(job, 5), metadata.settings_json(job, 5)))
    assert own["settings"]["prompt"] == job.prompt and "partial" not in own["settings"]
    fields = dict(own["fields"])
    assert fields["Steps"] == "50" and fields["Model"] == "MOSS-Image-Base" and fields["LoRA 1"] == "a, b.safetensors:0.7" and fields["Skip layers"] == "22"
    # a picture from Venno (Anima) / Forge / A1111: the prompt, negative, seed and size carry over
    text = "1girl, red hair, <lora:x:0.8>\nNegative prompt: worst quality,\nlowres\nSteps: 30, Sampler: Euler a, CFG scale: 4.5, Seed: 1234, Size: 830x1216, Model: anima-base-v1.0, Version: venno 0.9.1"
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", text)
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="PNG", pnginfo=meta)
    other = metadata.read(buf.getvalue())
    assert other["settings"] == {"partial": True, "prompt": "1girl, red hair, <lora:x:0.8>", "negative": "worst quality,\nlowres", "seed": 1234, "width": 832, "height": 1216}
    assert ["Sampler", "Euler a"] in other["fields"] and other["parameters"] == text
    plain = io.BytesIO()
    Image.new("RGB", (32, 32)).save(plain, format="PNG")
    assert metadata.read(plain.getvalue())["settings"] is None


# ---------------------------------------------------------------- image to image, inpaint


def data_url(img: Image.Image) -> str:
    import base64

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_engine_inpaint_keeps_the_unmasked_area(folders, monkeypatch, dit):
    folder = folders / "moss" / "Tiny" / "transformer"
    folder.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in dit.state_dict().items()}, str(folder / "diffusion_pytorch_model.safetensors"))
    eng = engine_module.Engine()
    monkeypatch.setattr(eng, "encode", lambda text: torch.randn(5, DIM, generator=torch.Generator().manual_seed(len(text))))
    monkeypatch.setattr(eng, "decode", lambda latents: latents)
    keep = torch.randn(1, 128, 8, 8, generator=torch.Generator().manual_seed(3))
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., :, 4:] = 1.0  # right half regenerated
    mask[..., :, 3] = 0.5  # a feathered column
    kw = dict(model="Tiny", prompt="a cat", negative="", width=128, height=128, steps=8, cfg=3.0, shift=5.0, seed=1, skip_layers=[22], loras=[])
    steps = []
    out, _ = eng.generate(**kw, init=dict(latent=keep, keep=keep, mask=mask, soft=True, denoise=0.75), on_step=lambda i, n: steps.append((i, n)))
    assert steps == [(i, 6) for i in range(1, 7)]  # the last 75% of 8 steps
    assert torch.allclose(out[..., :, :3], keep[..., :, :3], atol=1e-5)  # untouched where the mask is 0
    assert not torch.allclose(out[..., :, 4:], keep[..., :, 4:], atol=1e-2) and not torch.allclose(out[..., :, 3], keep[..., :, 3], atol=1e-2)
    hard, _ = eng.generate(**kw, init=dict(latent=torch.zeros_like(keep), keep=keep, mask=(mask >= 0.5).float(), soft=False, denoise=1.0))
    assert torch.allclose(hard[..., :, :3], keep[..., :, :3], atol=1e-5) and not torch.allclose(hard[..., :, 3:], keep[..., :, 3:], atol=1e-2)
    plain, _ = eng.generate(**kw, init=dict(latent=keep, keep=keep, mask=None, soft=True, denoise=0.5))  # image to image: everything moves
    assert not torch.allclose(plain[..., :, :3], keep[..., :, :3], atol=1e-2)
    with pytest.raises(ValueError, match="latents"):
        eng.generate(**kw, init=dict(latent=torch.zeros(1, 128, 4, 4), keep=keep, mask=None, soft=True, denoise=0.5))


def test_image_jobs(folders):
    """Jobs with an input picture: what reaches the engine, the paste-back, and the settings written."""
    from venno_moss import metadata, source

    calls = []

    class Fake:
        def encode_image(self, image):
            calls.append(("encode", image.size))
            return torch.zeros(1, 128, image.height // 16, image.width // 16)

        def generate(self, **kw):
            calls.append(("generate", kw))
            kw["on_step"](1, 1)
            return Image.new("RGB", (kw["width"], kw["height"]), (10, 200, 10)), []

    import venno_moss.engine as em

    original = Image.new("RGB", (512, 384), (200, 10, 10))
    mask = Image.new("L", (512, 384), 0)
    mask.paste(255, (300, 100, 400, 200))
    base = dict(model="m", mode="img2img", init_image=data_url(original), width=512, height=384, steps=4, seed=2, save=False)
    assert "input image" in jobs.unsupported(jobs.Job(model="m", mode="img2img"))
    real = em.get
    em.get = lambda: Fake()
    try:
        # image to image: the whole picture, no mask
        result = jobs.run(jobs.Job(**base, strength=0.45), jobs.Progress())
        init = calls[-1][1]["init"]
        assert init["mask"] is None and init["denoise"] == 0.45 and "Denoising strength: 0.45" in result.infotexts[0] and "Mask blur" not in result.infotexts[0]
        # inpaint, whole picture: unmasked pixels come back exactly, masked ones are the generated ones
        job = jobs.Job(**base, mask=data_url(mask), mask_blur=0, fill="fill", soft_inpaint=False, hires=jobs.Hires())
        result = jobs.run(job, jobs.Progress())
        init, out = calls[-1][1]["init"], Image.open(io.BytesIO(result.images[0]))
        assert calls[-1][1]["hires"] is None and tuple(init["mask"].shape) == (1, 1, 24, 32) and set(init["mask"].unique().tolist()) == {0.0, 1.0}
        assert out.size == (512, 384) and out.getpixel((10, 10)) == (200, 10, 10) and out.getpixel((350, 150)) == (10, 200, 10)
        assert [c for c in calls if c[0] == "encode"][-2:] == [("encode", (512, 384))] * 2  # the picture and its colour-filled copy
        for text in ("Masked content: fill", "Inpaint area: Whole picture", "Soft inpainting: False", "Mask blur: 0"):
            assert text in result.infotexts[0]
        settings = metadata.read(result.images[0])["settings"]
        assert settings["mode"] == "img2img" and settings["fill"] == "fill" and settings["hires"] is None and "init_image" not in settings
        # latent noise: the masked area runs the full schedule
        assert source.denoise(jobs.Job(**base, mask=data_url(mask), fill="latent_noise", strength=0.3), source.prepare(job, 512, 384)) == 1.0
        # only masked: a crop around the mask at the job's size, pasted back into the full-size input
        job = jobs.Job(**{**base, "width": 256, "height": 256}, mask=data_url(mask), mask_blur=4, only_masked=True, only_masked_padding=16)
        result = jobs.run(job, jobs.Progress())
        out = Image.open(io.BytesIO(result.images[0]))
        assert calls[-1][1]["width"] == 256 and out.size == (512, 384) and "Size: 512x384" in result.infotexts[0] and "Inpaint area: Only masked" in result.infotexts[0]
        assert out.getpixel((10, 10)) == (200, 10, 10) and out.getpixel((350, 150)) == (10, 200, 10) and out.getpixel((250, 150)) == (200, 10, 10)
        # an empty mask is no mask
        result = jobs.run(jobs.Job(**base, mask=data_url(Image.new("L", (512, 384), 0))), jobs.Progress())
        assert calls[-1][1]["init"]["mask"] is None
    finally:
        em.get = real
