"""Anima 3.8B (lylogummy/Anima-3.8B): the 52-block Anima conditioned by Qwen3 0.6B *and* Qwen3.5 4B.

The 52-block diffusion model loads like any Anima. What differs is the text side:
- the prompt goes through the native Qwen3 0.6B encoder as usual (its features, plus T5 token ids/weights) and
  through Qwen3.5 4B, from which four hidden layers are kept;
- a connector replaces Anima's native LLM adapter and turns both into the model's context:
  v1.1 ("Semantic Connector v2", bundled in the checkpoint, found by its safetensors metadata) depends on
  the diffusion timestep, so the context is rebuilt at every step inside the model's forward
  (`Anima.venno_context_expander`), from a placeholder context plus a run id carried in the `y` condition;
  v1 (a separate `…expanded_adapter.safetensors` next to the text encoders) is static and computed once.
Negative prompts stay on the native path (the reference workflows do the same).
Both text encoders are released after encoding; only the connector stays with the model while sampling.

Ported from GumGum10/forge-anima-3.8B (MIT, see LICENSE in this folder) to Venno's engine.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import os
import threading

import torch
import torch.nn.functional as F
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file

from backend import memory_management

from .connector import LAYERS, CrossAdapter, SamplingUnit, SemanticConnectorV2

BUNDLE_ARCHITECTURE = "anima_3_8b_semantic_connector_v2_bundle"
V2_ARCHITECTURE = "anima_qwen35_quality_anchored_semantic_connector_v2"
V1_ARCHITECTURE = "anima_progressive_qwen35_cross_adapter_v1"
CONNECTOR_PREFIX = "net.anima_v2_connector."
QWEN35_MARKERS = ("qwen35_4b", "qwen3.5-4b", "qwen3_5_4b")
CONTEXT_TOKENS = 512
# Qwen3.5's tokenizer, fetched once from the official repo (pinned, verified) into the text encoder folder
TOKENIZER_REPO = "Qwen/Qwen3.5-4B"
TOKENIZER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
PAD_TOKEN = 151643
MAX_TOKENS = 1024


# ---------------------------------------------------------------- files


def _metadata(path: str) -> dict:
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.metadata() or {}
    except (OSError, ValueError, SafetensorError):
        return {}


def bundle_metadata(path: str):
    """The v1.1 bundle's metadata, or None for any other checkpoint."""
    if not str(path).lower().endswith((".safetensors", ".sft")):
        return None
    meta = _metadata(path)
    return meta if meta.get("architecture") == BUNDLE_ARCHITECTURE and meta.get("anima_v2_bundle_format") == "1" else None


def is_qwen35(name: str) -> bool:
    return any(m in os.path.basename(name).lower() for m in QWEN35_MARKERS)


def _text_encoder_files() -> list[str]:
    from venno.engine import models

    return [p for name, p in models.modules().items() if name.lower().endswith((".safetensors", ".sft"))]


def qwen35_file():
    found = sorted((p for p in _text_encoder_files() if is_qwen35(p)), key=lambda p: (os.path.basename(p) != "qwen35_4b.safetensors", p))
    return found[0] if found else None


def v1_adapter_file():
    for path in _text_encoder_files():
        if "adapter" in os.path.basename(path).lower() and _metadata(path).get("architecture") == V1_ARCHITECTURE:
            return path
    return None


def tokenizer_path() -> str:
    from venno import config

    folder = os.path.join(config.model_folder("text_encoder"), "qwen35_tokenizer")
    path = os.path.join(folder, "tokenizer.json")
    if not os.path.isfile(path):
        from huggingface_hub import hf_hub_download

        print(f"[Venno] downloading the Qwen3.5 tokenizer from {TOKENIZER_REPO}")
        hf_hub_download(TOKENIZER_REPO, "tokenizer.json", revision=TOKENIZER_REVISION, local_dir=folder)
    with open(path, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != TOKENIZER_SHA256:
            raise RuntimeError(f"{path} is not the expected Qwen3.5 tokenizer; delete it to download it again")
    return path


# ---------------------------------------------------------------- Qwen3.5 4B (shared, loaded on demand)

_qwen = {"path": None, "model": None, "patcher": None, "tokenizer": None}
_qwen_lock = threading.Lock()


def _load_qwen35():
    from backend.operations import ForgeOperations, using_forge_operations
    from backend.patcher.base import ModelPatcher
    from tokenizers import Tokenizer

    from .qwen35 import Qwen35

    path = qwen35_file()
    if path is None:
        raise FileNotFoundError("Anima 3.8B needs qwen35_4b.safetensors (Qwen3.5 4B) in models/text_encoder — get it in the Models tab")
    with _qwen_lock:
        if _qwen["path"] == path:
            return _qwen
        print(f"[Venno] loading Qwen3.5 4B from {path}")
        state = load_file(path, device="cpu")
        dtype = next((state[k].dtype for k in ("layers.0.input_layernorm.weight", "embed_tokens.weight") if k in state), torch.bfloat16)
        with torch.device("meta"), using_forge_operations(device=torch.device("meta"), dtype=dtype, manual_cast_enabled=True):
            model = Qwen35(ops=ForgeOperations, dtype=dtype)
        result = model.load_state_dict(state, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Not a Qwen3.5 4B checkpoint: missing {result.missing_keys[:5]}, unexpected {result.unexpected_keys[:5]}")
        del state
        model.eval().requires_grad_(False)
        patcher = ModelPatcher(model, load_device=memory_management.text_encoder_device(), offload_device=memory_management.text_encoder_offload_device())
        _qwen.update(path=path, model=model, patcher=patcher, tokenizer=Tokenizer.from_file(tokenizer_path()))
        return _qwen


def _unload(*patchers):
    """Release text encoders right after use, so the diffusion model gets the memory back."""
    loaded = memory_management.current_loaded_models
    for i in range(len(loaded) - 1, -1, -1):
        p = loaded[i].model
        if any(p is t or p.is_clone(t) for t in patchers if t is not None):
            loaded[i].model_unload()
            loaded.pop(i)
    memory_management.soft_empty_cache()


# ---------------------------------------------------------------- runtime


class Runtime:
    """Attached to a loaded 3.8B engine as `engine.venno_anima38`."""

    def __init__(self, engine, checkpoint: str, bundle: dict | None):
        self.engine = engine
        self.version = "v1.1" if bundle is not None else "v1"
        clip = engine.forge_objects_original.clip
        self.native = clip.cond_stage_model.qwen3_06b.llm_adapter
        self.dtype = self.native.embed.weight.dtype
        self.runs: dict[int, dict] = {}
        self._ids = itertools.count(1)
        self.connector = self.adapter = None
        if bundle is not None:
            self.connector = self._load_v2(checkpoint, bundle)
            unet = engine.forge_objects_original.unet
            # a private copy of the native adapter: the text encoder's own copy moves with the text encoder
            unit = SamplingUnit(copy.deepcopy(self.native), self.connector)
            self.connector.quality_anchor.__dict__["native"] = unit.native
            unet.add_extra_torch_module_during_sampling(unit, cast_to_unet_dtype=False)
            unet.model.diffusion_model.venno_context_expander = self
        else:
            path = v1_adapter_file()
            if path is None:
                raise FileNotFoundError("This Anima 3.8B (v1) checkpoint needs Anima-3.8B-expanded_adapter.safetensors in models/text_encoder (or use the v1.1 bundle)")
            self.adapter = CrossAdapter(self.native)
            result = self.adapter.load_state_dict(load_file(path, device="cpu"), strict=True)
            if result.missing_keys or result.unexpected_keys:
                raise RuntimeError(f"{os.path.basename(path)} does not match the Anima 3.8B v1 adapter")
            self.adapter.to(self.dtype).eval().requires_grad_(False)
        self.info = {"Anima 3.8B": self.version}

    def _load_v2(self, path: str, meta: dict) -> SemanticConnectorV2:
        conf = {k[len("anima_v2_adapter_"):]: v for k, v in meta.items() if k.startswith("anima_v2_adapter_")}
        if conf.get("architecture") != V2_ARCHITECTURE:
            raise RuntimeError("The bundled connector is not Semantic Connector v2")
        prefix = meta.get("anima_v2_connector_prefix", CONNECTOR_PREFIX)
        with safe_open(path, framework="pt", device="cpu") as f:
            state = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
        if not state:
            raise RuntimeError("The Anima 3.8B bundle contains no connector weights")
        with torch.device("meta"):
            connector = SemanticConnectorV2(
                self.native,
                queries=int(conf.get("semantic_query_tokens", 64)),
                blocks=int(conf.get("semantic_resampler_blocks", 6)),
                dim=int(conf.get("semantic_resampler_dim", 2048)),
                heads=int(conf.get("semantic_resampler_heads", 16)),
                mlp_dim=int(conf.get("semantic_resampler_mlp_hidden_dim", 5632)),
            )
        result = connector.load_state_dict(state, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Bundled connector mismatch: missing {result.missing_keys[:5]}, unexpected {result.unexpected_keys[:5]}")
        return connector.to(self.dtype).eval().requires_grad_(False)

    def reset(self):
        """A new job: forget the previous job's prompts."""
        self.runs.clear()

    # -- encoding --
    def _features(self, text: str):
        """Native features (source, T5 ids, T5 weights) and the four Qwen3.5 layers, stored on the offload device."""
        clip = self.engine.forge_objects.clip
        store = clip.patcher.offload_device
        memory_management.load_model_gpu(clip.patcher)
        try:
            source, ids, weights = self.engine.text_processing_engine_qwen.features(text)
            source, ids, weights = source.to(store, self.dtype), ids[:, :CONTEXT_TOKENS].to(store), weights[:, :CONTEXT_TOKENS].to(store, self.dtype)
        finally:
            _unload(clip.patcher)
        qwen = _load_qwen35()
        memory_management.load_model_gpu(qwen["patcher"])
        try:
            device = qwen["patcher"].load_device
            tokens = qwen["tokenizer"].encode(text, add_special_tokens=False).ids[:MAX_TOKENS] or [PAD_TOKEN]
            layers = qwen["model"](torch.tensor([tokens], device=device), LAYERS)
            semantic = [h.to(store, self.dtype) for h in layers]
        finally:
            _unload(qwen["patcher"])
        mask = torch.ones(1, len(tokens), dtype=torch.long, device=store)
        return dict(source=source, ids=ids, weights=weights, semantic=semantic, mask=mask)

    @torch.inference_mode()
    def encode(self, text: str):
        """Positive prompt → (1, 512, D) context (v1), or {"crossattn": placeholder, "vector": run id} (v1.1)."""
        run = self._features(text)
        if self.connector is not None:
            run_id = next(self._ids)
            self.runs[run_id] = run
            placeholder = F.pad(run["source"][:, :CONTEXT_TOKENS], (0, 0, 0, max(0, CONTEXT_TOKENS - run["source"].shape[1])))
            return {"crossattn": placeholder, "vector": torch.tensor([[run_id]], dtype=torch.long)}
        device = memory_management.text_encoder_device()
        clip = self.engine.forge_objects.clip
        memory_management.load_model_gpu(clip.patcher)  # the native adapter runs inside the v1 adapter
        try:
            self.adapter.to(device)
            out = self.adapter(run["source"].to(device), run["ids"].to(device), [s.to(device) for s in run["semantic"]], run["mask"].to(device))
            out = out * run["weights"].to(device)[:, : out.shape[1]]
            return F.pad(out, (0, 0, 0, max(0, CONTEXT_TOKENS - out.shape[1]))).to(clip.patcher.offload_device)
        finally:
            self.adapter.to(clip.patcher.offload_device)
            _unload(clip.patcher)

    # -- sampling (v1.1) --
    def _context(self, run_id: int, timesteps: torch.Tensor, device, dtype) -> torch.Tensor:
        run = self.runs.get(run_id)
        if run is None:
            raise RuntimeError(f"Anima 3.8B prompt {run_id} is no longer available; generate again")
        n = timesteps.shape[0]
        to = lambda t: t.to(device, self.dtype)
        out = self.connector(
            to(run["source"]).expand(n, -1, -1), run["ids"].to(device).expand(n, -1), [to(s).expand(n, -1, -1) for s in run["semantic"]],
            timesteps.to(device), run["mask"].to(device).expand(n, -1),
        )
        out = out * to(run["weights"])[:, : out.shape[1]]
        return F.pad(out, (0, 0, 0, max(0, CONTEXT_TOKENS - out.shape[1]))).to(dtype)

    def expand(self, context: torch.Tensor, timesteps: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Called by the diffusion model: placeholder contexts → connector output at these timesteps (row by row)."""
        ids = y.reshape(-1).long()
        t = timesteps.reshape(-1)
        if t.numel() != context.shape[0]:
            t = t.repeat((context.shape[0] + t.numel() - 1) // t.numel())[: context.shape[0]]
        out = context.clone()
        for run_id in ids.unique().tolist():
            rows = (ids == run_id).nonzero().reshape(-1)
            out[rows] = self._context(int(run_id), t[rows], context.device, context.dtype)
        return out

    def expand_one(self, cond: dict, timesteps: torch.Tensor, device, dtype) -> torch.Tensor:
        """A region prompt's context at the current step (regions carry the same placeholder + run id)."""
        return self._context(int(cond["vector"].reshape(-1)[0]), timesteps.reshape(-1)[:1], device, dtype)


def attach(engine, checkpoint: str):
    """Set up 3.8B conditioning when `checkpoint` is an Anima 3.8B (bundle metadata or 52 blocks); returns the runtime or None."""
    bundle = bundle_metadata(checkpoint)
    blocks = len(engine.forge_objects_original.unet.model.diffusion_model.blocks)
    if bundle is None and blocks != 52:
        return None
    runtime = Runtime(engine, checkpoint, bundle)
    engine.venno_anima38 = runtime
    print(f"[Venno] Anima 3.8B {runtime.version}: Qwen3.5 4B conditioning enabled")
    return runtime
