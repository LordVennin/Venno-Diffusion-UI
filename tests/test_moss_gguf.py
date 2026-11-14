"""Venno MOSS's GGUF support (the small downloads): the block decoders against the reference implementation,
and a tiny transformer and a tiny Qwen3.5 text encoder written to GGUF and loaded back."""

import os
import struct

import numpy as np
import pytest
import torch

from venno_moss import config, engine as engine_module, gguf, lora, models
from venno_moss.model.transformer import MossImageParams, MossImageTransformer

DIM, FFN, LAYERS = 64, 128, 3
KIND_IDS = {name: number for number, name in gguf.KINDS.items()}


# ---------------------------------------------------------------- writing test files


def q8_0(values: torch.Tensor) -> bytes:
    """Q8_0 as ggml quantizes it: blocks of 32 values, a float16 scale and 32 int8."""
    a = values.detach().to(torch.float32).reshape(-1, 32).numpy()
    d = np.abs(a).max(axis=1, keepdims=True) / 127
    q = np.where(d > 0, np.round(a / np.where(d > 0, d, 1)), 0).astype(np.int8)
    return np.concatenate([d.astype(np.float16).view(np.uint8), q.view(np.uint8)], axis=1).tobytes()


def stored(values: torch.Tensor, kind: str) -> tuple[str, tuple, bytes]:
    if kind == "Q8_0":
        return kind, tuple(values.shape), q8_0(values)
    return kind, tuple(values.shape), values.detach().to(gguf.PLAIN[kind]).contiguous().view(torch.uint8).numpy().tobytes()


def write_gguf(path, tensors: dict[str, tuple[str, tuple, bytes]], metadata: dict[str, str]):
    def string(s: str) -> bytes:
        raw = s.encode()
        return struct.pack("<Q", len(raw)) + raw

    head = struct.pack("<IIQQ", gguf.MAGIC, 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        head += string(key) + struct.pack("<I", 8) + string(value)
    data = b""
    for name, (kind, shape, raw) in tensors.items():
        data += b"\0" * (-len(data) % 32)
        head += string(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in reversed(shape)) + struct.pack("<IQ", KIND_IDS[kind], len(data))
        data += raw
    with open(path, "wb") as f:
        f.write(head + b"\0" * (-len(head) % 32) + data)


# ---------------------------------------------------------------- decoders


F16_FIELDS = {"Q4_0": [0], "Q4_1": [0, 2], "Q5_0": [0], "Q5_1": [0, 2], "Q8_0": [0], "Q2_K": [80, 82], "Q3_K": [108], "Q4_K": [0, 2], "Q5_K": [0, 2], "Q6_K": [208]}


@pytest.mark.parametrize("kind", sorted(gguf.BLOCKS))
def test_decoders_match_the_reference(kind):
    reference = pytest.importorskip("gguf")  # llama.cpp's gguf-py: pip install gguf
    values, size = gguf.BLOCKS[kind]
    rng = np.random.default_rng(len(kind) + size)
    raw = rng.integers(0, 256, (48, size), dtype=np.uint8)
    for at in F16_FIELDS[kind]:  # the blocks' float16 scales: finite numbers
        raw[:, at : at + 2] = rng.uniform(-0.05, 0.05, (48, 1)).astype(np.float16).view(np.uint8).reshape(48, 2)
    expected = reference.quants.dequantize(raw.reshape(-1), getattr(reference.GGMLQuantizationType, kind)).reshape(-1)
    ours = gguf.dequantize(torch.from_numpy(raw.copy()).reshape(-1), kind, (48 * values,)).numpy()
    assert np.abs(expected - ours).max() < 1e-5


def test_q8_0_round_trip_and_plain_types(tmp_path):
    torch.manual_seed(0)
    w = torch.randn(8, 64) * 0.05
    path = tmp_path / "t.gguf"
    write_gguf(path, {"a.weight": stored(w, "Q8_0"), "b": stored(w, "BF16"), "c": stored(w[0], "F32"), "odd": ("IQ4_XS", (8, 64), b"")}, {"general.architecture": "test"})
    f = gguf.File(str(path))
    assert f.architecture == "test" and f.tensors["a.weight"].shape == (8, 64) and f.kinds() == {"Q8_0": 1, "BF16": 1, "F32": 1, "IQ4_XS": 1}
    assert torch.allclose(f.plain("a.weight"), w, atol=0.05 / 127 * 4) and torch.equal(f.plain("c"), w[0])
    assert torch.equal(f.plain("b"), w.to(torch.bfloat16)) and f.plain("b", torch.float32).dtype == torch.float32
    with pytest.raises(ValueError, match="IQ4_XS"):
        f.raw("odd")
    (tmp_path / "x.gguf").write_bytes(b"not a gguf file at all, just text")
    with pytest.raises(ValueError, match="not a GGUF"):
        gguf.File(str(tmp_path / "x.gguf"))


# ---------------------------------------------------------------- transformer


@pytest.fixture
def dit():
    torch.manual_seed(1)
    model = MossImageTransformer(MossImageParams(dim=DIM, ffn_dim=FFN, num_heads=4, num_layers=LAYERS, text_dim=DIM, freq_dim=256, latent_channels=128))
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)
    return model.eval()


def forward(model):
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 128, 4, 4, generator=g)
    text = torch.randn(1, 5, DIM, generator=g)
    with torch.no_grad():
        return model(x, torch.tensor([500.0]), text, torch.ones(1, 5, dtype=torch.bool))


def dit_gguf(path, dit, split=True):
    """The transformer as community GGUFs store it: big matrices quantized, q / k / v apart (and in different
    types) or fused, the rest plain."""
    tensors = {"rope.bands": stored(torch.arange(16.0), "F32")}
    for name, value in dit.state_dict().items():
        if name.startswith("layers.") and name.endswith(".proj.qkv.weight") and split:
            for part, rows, kind in zip("qkv", value.chunk(3), ("Q8_0", "F32", "Q8_0")):
                tensors[name.replace(".qkv.", f".{part}.")] = stored(rows, kind)
        elif value.ndim == 2 and name.endswith(".weight") and value.shape[1] % 32 == 0 and "time_embedding" not in name:
            tensors[name] = stored(value, "Q8_0")
        else:
            tensors[name] = stored(value, "BF16" if value.ndim == 2 else "F32")
    write_gguf(path, tensors, {"general.architecture": "moss_image"})


@pytest.mark.parametrize("split", [True, False])
def test_transformer_from_gguf(dit, tmp_path, split):
    path = str(tmp_path / "tiny-Q8_0.gguf")
    dit_gguf(path, dit, split)
    loaded = engine_module.gguf_transformer(path, torch.float32).eval()
    qkv = loaded.layers[0].proj["qkv"]
    assert qkv.weight is None and [kind for kind, _ in qkv.gguf_parts] == (["Q8_0", "F32", "Q8_0"] if split else ["Q8_0"])
    assert not any(p.is_meta for p in loaded.parameters())
    a, b = forward(dit), forward(loaded)
    assert torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0) > 0.999
    assert not torch.equal(a, b)  # quantized, not a copy

    # a LoRA on top of the quantized weights changes the output as it does on the full ones
    torch.manual_seed(5)
    sd = {}
    for i in range(LAYERS):
        for part in ("q", "k", "v", "o"):
            sd[f"diffusion_model.layers.{i}.proj.{part}.lora_A.weight"] = torch.randn(4, DIM) * 0.5
            sd[f"diffusion_model.layers.{i}.proj.{part}.lora_B.weight"] = torch.randn(DIM, 4) * 0.5
    outputs = []
    for model in (dit, loaded):
        stack = lora.LoraStack(model, torch.device("cpu"), torch.float32)
        patches, unknown = lora.parse(sd, stack.modules)
        assert not unknown and len(patches) == 2 * LAYERS
        for name, parts in patches.items():
            stack.active[name] = [(part, 0) for part in parts]
            stack._hook(name)
        stack.weights = [1.0]
        outputs.append(forward(model))
        stack.clear()
    # what the LoRA changes is the same on the quantized model as on the full one
    change_full, change_quantized = outputs[0] - a, outputs[1] - b
    assert change_full.norm() > 0.02 * a.norm()
    assert torch.nn.functional.cosine_similarity(change_full.flatten(), change_quantized.flatten(), dim=0) > 0.98


def test_gguf_models_are_found_and_load_in_either_memory_mode(dit, tmp_path, monkeypatch):
    monkeypatch.setattr(config.args, "models_dir", str(tmp_path / "moss"))
    monkeypatch.setattr(config.args, "data_dir", str(tmp_path))
    monkeypatch.setattr(config.args, "cpu", True)
    top = config.models_path()
    dit_gguf(os.path.join(top, "Tiny-Q8_0.gguf"), dit)
    write_gguf(os.path.join(top, "other-model.gguf"), {"blk.0.weight": stored(torch.zeros(2, 32), "Q8_0")}, {"general.architecture": "llama"})
    write_gguf(os.path.join(top, "Tiny-IQ4_XS.gguf"), {"video_embedder.weight": ("IQ4_XS", (DIM, 128), b"")}, {"general.architecture": "moss_image"})
    with open(os.path.join(top, "broken.gguf"), "wb") as f:
        f.write(b"GGUF")
    found = models.scan()
    assert [m.name for m in found.models] == ["Tiny-Q8_0"] and found.models[0].size > 0
    eng = engine_module.Engine()
    for mode in ("full", "low"):
        monkeypatch.setattr(config.args, "memory", mode)
        loaded = eng.load_model("Tiny-Q8_0")
        assert loaded.layers[0].proj["o"].weight is None  # stays as quantized as its file
        assert torch.nn.functional.cosine_similarity(forward(dit).flatten(), forward(loaded).float().flatten(), dim=0) > 0.999
    assert eng.info()["quantized"] is True


# ---------------------------------------------------------------- text encoder


def llama_names(cfg) -> dict[str, str]:
    out = {"embed_tokens.weight": "token_embd.weight", "norm.weight": "output_norm.weight"}
    for i in range(cfg.num_hidden_layers):
        for theirs, ours in engine_module.TE_LAYER_KEYS.items():
            out[f"layers.{i}.{ours}"] = f"blk.{i}.{theirs}"
    return out


def as_llama_cpp_converts(name: str, value: torch.Tensor, cfg) -> torch.Tensor:
    """What llama.cpp's convert_hf_to_gguf does to Qwen3.5's tensors (conversion/qwen.py: Qwen3NextModel and
    _LinearAttentionVReorderBase.modify_tensors), written from that code."""
    key_heads, value_heads = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    key_dim, value_dim = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    per_key = value_heads // key_heads

    def reorder(t, dim, head_dim):
        shape = list(t.shape)
        t = t.reshape(*shape[:dim], key_heads, per_key, head_dim, *shape[dim + 1 :])
        return t.transpose(dim, dim + 1).contiguous().reshape(shape)

    if name.endswith(".A_log"):
        value = -torch.exp(value)
    elif "conv1d" in name:
        value = value.squeeze()
    elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
        value = value + 1
    if "linear_attn." in name:
        qk = 2 * key_heads * key_dim
        if ".in_proj_qkv." in name:
            value = torch.cat([value[:qk], reorder(value[qk:], 0, value_dim)])
        elif ".in_proj_z." in name:
            value = reorder(value, 0, value_dim)
        elif ".in_proj_a." in name or ".in_proj_b." in name:
            value = reorder(value, 0, 1)
        elif ".A_log" in name or ".dt_bias" in name:
            value = reorder(value.unsqueeze(-1), 0, 1).squeeze(-1)
        elif ".conv1d" in name:
            value = torch.cat([value[:qk], reorder(value[qk:], 0, value_dim)])
        elif ".out_proj." in name:
            value = reorder(value, 1, value_dim)
    return value


def test_text_encoder_from_a_llama_cpp_gguf(tmp_path):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3_5TextModel"):
        pytest.skip("needs transformers 5 (Venno MOSS's own environment, venv-moss)")
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    cfg = Qwen3_5TextConfig(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"], full_attention_interval=4)
    torch.manual_seed(2)
    model = Qwen3_5TextModel(cfg).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.08)
    names = llama_names(cfg)
    tensors = {"output.weight": stored(torch.zeros(96, 64), "Q8_0")}  # the language head: ignored
    for name, value in model.state_dict().items():
        if name not in names:
            assert name.startswith("rotary_emb."), name
            continue
        value = as_llama_cpp_converts(name, value, cfg)
        tensors[names[name]] = stored(value, "Q8_0" if value.ndim == 2 and "conv1d" not in name else "F32")
    tensors["blk.4.ffn_up.weight"] = stored(torch.zeros(128, 64), "Q8_0")  # a next-token head's extra layer: ignored
    path = str(tmp_path / "Tiny-Qwen-Q8_0.gguf")
    write_gguf(path, tensors, {"general.architecture": "qwen35"})

    loaded = engine_module.gguf_text_encoder(path, cfg, torch.float32)
    rotary = loaded.rotary_emb
    if any(b.is_meta for b in rotary.buffers()):
        loaded.rotary_emb = type(rotary)(config=cfg)
    engine_module.assert_materialized(loaded, "text encoder")
    loaded = loaded.eval()
    assert loaded.embed_tokens.weight is None and loaded.layers[0].linear_attn.out_proj.gguf_input_index is not None
    ids = torch.tensor([[3, 17, 42, 9, 77, 5, 60, 31, 8, 90, 11, 2]])
    with torch.no_grad():
        expected = model(inputs_embeds=model.embed_tokens(ids), use_cache=False).last_hidden_state
        embeds = loaded.embed_tokens(ids)
        got = loaded(inputs_embeds=embeds, use_cache=False).last_hidden_state
    assert torch.nn.functional.cosine_similarity(model.embed_tokens(ids).flatten(), embeds.flatten(), dim=0) > 0.9999
    assert torch.nn.functional.cosine_similarity(expected.flatten(), got.flatten(), dim=0) > 0.999

    # the same file without undoing llama.cpp's value-head order gives something else: the test would notice
    plain = {name: as_llama_cpp_converts(name, value, cfg) for name, value in model.state_dict().items() if "linear_attn" in name and ("in_proj_z" in name or "out_proj" in name)}
    wrong = Qwen3_5TextModel(cfg).eval()
    wrong.load_state_dict({**model.state_dict(), **plain})
    with torch.no_grad():
        other = wrong(inputs_embeds=model.embed_tokens(ids), use_cache=False).last_hidden_state
    assert torch.nn.functional.cosine_similarity(expected.flatten(), other.flatten(), dim=0) < 0.99

    with pytest.raises(ValueError, match="not a Qwen3.5"):
        write_gguf(str(tmp_path / "llama.gguf"), {"token_embd.weight": stored(torch.zeros(96, 64), "Q8_0")}, {"general.architecture": "llama"})
        engine_module.gguf_text_encoder(str(tmp_path / "llama.gguf"), cfg, torch.float32)


def test_text_encoders_found_and_chosen(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    monkeypatch.setattr(config.args, "models_dir", str(tmp_path / "moss"))
    monkeypatch.setattr(config.args, "data_dir", str(tmp_path))
    top = config.models_path()
    small = os.path.join(top, "shared", "text_encoder")
    os.makedirs(small)
    with open(os.path.join(small, "config.json"), "w") as f:
        f.write("{}")
    for name, rows in (("Qwen3.5-9B-Q4_K_M.gguf", 2), ("Qwen3.5-9B-Q6_K.gguf", 4)):
        write_gguf(os.path.join(small, name), {"token_embd.weight": stored(torch.zeros(rows, 32), "Q8_0")}, {"general.architecture": "qwen35"})
    found = models.scan()
    assert [t.name for t in found.text_encoders] == ["Qwen3.5-9B-Q6_K", "Qwen3.5-9B-Q4_K_M"]  # the larger GGUF first
    assert found.text_encoder.endswith("Qwen3.5-9B-Q6_K.gguf") and found.text_encoder_name == "Qwen3.5-9B-Q6_K"
    full = os.path.join(top, "MOSS-Image-Base", "text_encoder")
    os.makedirs(full)
    save_file({"w": torch.zeros(2)}, os.path.join(full, "model.safetensors"))
    found = models.scan()
    assert found.text_encoder == full and [t.name for t in found.text_encoders][0] == models.FULL_TEXT_ENCODER  # the full one by default
    config.save_settings(text_encoder="Qwen3.5-9B-Q4_K_M")
    assert models.scan().text_encoder_name == "Qwen3.5-9B-Q4_K_M"
    config.save_settings(text_encoder="gone")
    assert models.scan().text_encoder == full


# ---------------------------------------------------------------- the small downloads


def test_small_downloads(tmp_path, monkeypatch):
    import io
    import json

    from safetensors.torch import save_file

    from venno_moss import downloads

    monkeypatch.setattr(config.args, "models_dir", str(tmp_path / "moss"))
    monkeypatch.setattr(config.args, "data_dir", str(tmp_path))
    write_gguf(tmp_path / "te.gguf", {"token_embd.weight": stored(torch.zeros(4, 32), "Q8_0")}, {"general.architecture": "qwen35"})
    write_gguf(tmp_path / "dit.gguf", {"video_embedder.weight": stored(torch.zeros(64, 128), "Q8_0")}, {"general.architecture": "moss_image"})
    save_file({"w": torch.zeros(2)}, str(tmp_path / "vae.safetensors"))
    te, dit, vae = ((tmp_path / n).read_bytes() for n in ("te.gguf", "dit.gguf", "vae.safetensors"))
    hub = {
        "unsloth/MOSS-Image-SFT-GGUF": {"MOSS-Image-SFT-Q4_K_M.gguf": dit, "MOSS-Image-SFT-Q8_0.gguf": dit + b"\0" * 64, "README.md": b"x"},
        "molbal/Kromoss-GGUF": {"kromoss-Q8_0.gguf": dit, "kromoss-Q4_K_M.gguf": dit},
        "unsloth/Qwen3.5-9B-GGUF": {"Qwen3.5-9B-Q4_K_M.gguf": te, "Qwen3.5-9B-Q6_K.gguf": te + b"\0" * 64, "mmproj-F16.gguf": b"x"},
        "OpenMOSS-Team/MOSS-Image-Base": {"text_encoder/config.json": b"{}", "text_encoder/model-00001-of-00004.safetensors": b"big", "tokenizer/tokenizer.json": b"{}",
                                          "vae/config.json": b"{}", "vae/diffusion_pytorch_model.safetensors": vae, "transformer/a.safetensors": b"big"},
    }
    fetched = []

    class Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=None):  # stands in for downloads._open
        url = req.full_url
        if "/api/models/" in url:
            repo = url.split("/api/models/", 1)[1].split("/tree/", 1)[0]
            return Resp(json.dumps([{"type": "file", "path": p, "size": len(b)} for p, b in hub[repo].items()]).encode())
        repo, path = url.split("huggingface.co/", 1)[1].split("/resolve/main/", 1)
        fetched.append(f"{repo}:{path}")
        return Resp(hub[repo][path])

    monkeypatch.setattr(downloads, "_open", urlopen)
    sizes = {x["variant"]: x for entry in downloads.catalog() for x in entry["sizes"]}
    assert sizes["sft-q4"]["size_gb"] == 9.2 and sizes["sft-q8"]["size_gb"] == 13.3 and sizes["sft"]["size_gb"] == 27.2 and not sizes["sft-q4"]["installed"]

    # the first small model brings the text encoder (a GGUF with its config), the tokenizer and the VAE
    job = downloads.Download("sft-q4")
    job.run()
    assert job.status == "done" and job.to_dict()["shared"], job.error
    top = tmp_path / "moss"
    assert (top / "MOSS-Image-SFT-Q4_K_M.gguf").read_bytes() == dit
    assert sorted(os.listdir(top / "shared" / "text_encoder")) == ["Qwen3.5-9B-Q4_K_M.gguf", "config.json"]
    assert (top / "shared" / "tokenizer" / "tokenizer.json").exists() and (top / "shared" / "vae" / "diffusion_pytorch_model.safetensors").exists()
    assert not any("model-00001" in f or "transformer/" in f or "mmproj" in f for f in fetched)
    found = models.scan()
    assert found.complete and [m.name for m in found.models] == ["MOSS-Image-SFT-Q4_K_M"] and found.text_encoder_name == "Qwen3.5-9B-Q4_K_M"

    # the next one only its own file, under the name the model list shows
    fetched.clear()
    job = downloads.Download("kromoss-q8")
    job.run()
    assert job.status == "done" and not job.to_dict()["shared"] and fetched == ["molbal/Kromoss-GGUF:kromoss-Q8_0.gguf"]
    assert (top / "Kromoss-Q8_0.gguf").exists()

    # another text encoder size on its own; the larger one is then the default
    downloads.Download("te-q6").run()
    assert [t.name for t in models.scan().text_encoders] == ["Qwen3.5-9B-Q6_K", "Qwen3.5-9B-Q4_K_M"]
    sizes = {x["variant"]: x for entry in downloads.catalog() for x in entry["sizes"]}
    assert sizes["sft-q4"]["installed"] and sizes["kromoss-q8"]["installed"] and sizes["te-q6"]["installed"] and sizes["te-q4"]["installed"] and not sizes["te-full"]["installed"]
    assert sizes["sft-q8"]["size_gb"] == 5.5 and sizes["base"]["size_gb"] == 10.2  # the shared parts are here now

    with pytest.raises(ValueError, match="Unknown download"):
        downloads.start("sft-q2")
