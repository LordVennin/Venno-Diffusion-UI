import importlib
import logging
import os.path
from functools import partial
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from backend.diffusion_engine.base import ForgeDiffusionEngine

import torch
import yaml
from transformers.modeling_utils import no_init_weights

import backend.args
from backend import memory_management, utils
from backend.diffusion_engine.anima import Anima
from backend.logging import setup_logger
from backend.operations import using_forge_operations
from backend.state_dict import (
    convert_quantization,
    detect_quantization,
    load_state_dict,
    state_dict_prefix_replace,
    try_filter_state_dict,
)
from backend.utils import (
    beautiful_print_gguf_state_dict_statics,
    load_torch_file,
    read_arbitrary_config,
)
from comfy.utils import convert_diffusers_mmdit

# Venno: Anima is the only supported architecture
possible_models: tuple["ForgeDiffusionEngine"] = (Anima,)

logger = logging.getLogger("loader")
setup_logger(logger)

HF = os.path.join(os.path.dirname(__file__), "huggingface")


def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
    config_path = os.path.join(repo_path, component_name)

    if component_name in ["feature_extractor", "safety_checker"]:
        return None

    if lib_name in ["transformers", "diffusers"]:
        if component_name == "scheduler":
            cls = getattr(importlib.import_module(lib_name), cls_name)
            return cls.from_pretrained(os.path.join(repo_path, component_name))
        if component_name.startswith("tokenizer"):
            cls = getattr(importlib.import_module(lib_name), cls_name)
            comp = cls.from_pretrained(os.path.join(repo_path, component_name))
            comp._eventual_warn_about_too_long_sequence = lambda *args, **kwargs: None
            return comp

        # region dtype

        state_dict_dtype = utils.weight_dtype(state_dict) if isinstance(state_dict, dict) else None

        def _detect_dtype(key: str, storage_dtype: torch.dtype, quant_config: dict = None) -> torch.dtype:
            if quant_config is not None:
                logger.info(f"Using MixedPrecision for {key}")
                return state_dict_dtype
            elif state_dict_dtype in (torch.float8_e4m3fn, torch.float8_e5m2, "gguf"):
                _log = f"{state_dict_dtype}" + (" (pre-quant)" if state_dict_dtype == "gguf" else "")
                logger.info(f"Using Detected {key} Data Type: {_log}")
                if state_dict_dtype == "gguf":
                    beautiful_print_gguf_state_dict_statics(state_dict)
                return state_dict_dtype
            else:
                logger.info(f"Using Default {key} Data Type: {storage_dtype}")
                return storage_dtype

        # region VAE

        if cls_name in ["AutoencoderKLWan", "AutoencoderKLQwenImage"]:
            assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have VAE state dict!"

            if "post_quant_conv.weight" in state_dict:  # 2D
                from backend.nn.wan_vae_2d import Qwen2DVAE as WanVAE

                config = {}
                dtype = memory_management.vae_dtype()

            else:
                from backend.nn.wan_vae import WanVAE

                config = WanVAE.load_config(config_path)
                dtype = memory_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float16, torch.float32])

            with no_init_weights():
                with using_forge_operations(device=memory_management.cpu, dtype=dtype, extra_dtype="vae"):
                    model = WanVAE.from_config(config)

            load_state_dict(model, state_dict)
            return model

        # region Text Encoder

        if cls_name in ["Qwen3Model", "Qwen3ForCausalLM", "Qwen3VLModel", "Qwen3VLForConditionalGeneration"]:
            assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have Qwen3 state dict!"

            config = read_arbitrary_config(config_path)

            if cls_name == "Qwen3VLModel":
                from backend.nn.llm.llama import Qwen3VL

                QTE = partial(Qwen3VL, model_type="qwen3vl_4b")
            elif cls_name == "Qwen3VLForConditionalGeneration":
                from backend.nn.llm.llama import Qwen3VL

                QTE = partial(Qwen3VL, model_type="qwen3vl_8b")
            elif config["hidden_size"] == 4096:
                from backend.nn.llm.llama import Qwen3_8B as QTE
            elif config["hidden_size"] == 2560:
                from backend.nn.llm.llama import Qwen3_4B as QTE
            else:
                from backend.nn.llm.llama import Qwen3_06B as QTE

            quant_config = detect_quantization(state_dict)
            storage_dtype = _detect_dtype("Qwen3", memory_management.text_encoder_dtype(), quant_config)

            if storage_dtype == "gguf":
                with no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=memory_management.text_encoder_dtype(), manual_cast_enabled=False, extra_dtype=storage_dtype):
                        model = QTE(config)
            else:
                with no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=storage_dtype, manual_cast_enabled=True, sd_dtype=state_dict_dtype, extra_dtype=quant_config):
                        model = QTE(config)

            if cls_name.startswith("Qwen3VL"):
                state_dict = state_dict_prefix_replace(
                    state_dict,
                    {
                        "model.language_model.": "model.",
                        "model.visual.": "visual.",
                        "lm_head.": "model.lm_head.",
                    },
                )

            load_state_dict(model, state_dict, log_name=cls_name, ignore_start="lm_head.")
            return model
        # region UNet / DiT

        if cls_name == "CosmosTransformer3DModel":
            assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have model state dict!"
            from backend.nn.anima import Anima

            model_loader = lambda c: Anima(**c)

            load_device = memory_management.get_torch_device()
            offload_device = memory_management.unet_offload_device()

            unet_config = guess.unet_config.copy()
            state_dict_parameters = utils.calculate_parameters(state_dict)
            quant_config = detect_quantization(state_dict, is_unet=True)

            override_dtype = backend.args.dynamic_args.forge_unet_storage_dtype

            if quant_config is not None:
                storage_dtype = torch.bfloat16
                logger.info("Using MixedPrecision for Model")
            elif state_dict_dtype in (torch.float8_e4m3fn, torch.float8_e5m2, "gguf"):
                storage_dtype = state_dict_dtype
                _log = f"{state_dict_dtype}" + (" (pre-quant)" if state_dict_dtype == "gguf" else "")
                logger.info(f"Using Detected Model Data Type: {_log}")
                if state_dict_dtype == "gguf":
                    beautiful_print_gguf_state_dict_statics(state_dict)
            else:
                if override_dtype is not None:
                    storage_dtype = override_dtype
                else:
                    storage_dtype = memory_management.unet_dtype(device=load_device, model_params=state_dict_parameters, supported_dtypes=guess.supported_inference_dtypes, weight_dtype=state_dict_dtype)
                if storage_dtype == state_dict_dtype:
                    logger.info(f"Using Default Model Data Type: {storage_dtype}")
                else:
                    logger.info(f"Using Override Model Data Type: {storage_dtype}")

            computation_dtype = memory_management.inference_cast(weight_dtype=storage_dtype, inference_device=load_device, supported_dtypes=guess.supported_inference_dtypes)

            backend.args.dynamic_args.ops = None

            if storage_dtype == "gguf":
                initial_device = memory_management.unet_initial_load_device(parameters=state_dict_parameters, dtype=computation_dtype)
                need_manual_cast = False
                to_args = dict(device=initial_device, dtype=computation_dtype)

                with no_init_weights():
                    with using_forge_operations(**to_args, manual_cast_enabled=need_manual_cast, extra_dtype=storage_dtype):
                        model = model_loader(unet_config)
            else:
                initial_device = memory_management.unet_initial_load_device(parameters=state_dict_parameters, dtype=storage_dtype)
                need_manual_cast = storage_dtype != computation_dtype
                to_args = dict(device=initial_device, dtype=storage_dtype)

                with no_init_weights():
                    with using_forge_operations(**to_args, manual_cast_enabled=need_manual_cast, sd_dtype=state_dict_dtype, extra_dtype=quant_config):
                        model = model_loader(unet_config)

            load_state_dict(model, state_dict)

            model.config = unet_config
            model.storage_dtype = storage_dtype
            model.computation_dtype = computation_dtype
            model.load_device = load_device
            model.initial_device = initial_device
            model.offload_device = offload_device

            return model

        if component_name in ("transformer", "unet", "vae") or component_name.startswith("text_encoder"):
            raise ModuleNotFoundError(f'Unsupported {component_name} "{cls_name}": Venno Diffusion UI only supports Anima (Qwen3 text encoder, Qwen-Image VAE)')

    logger.warning(f'Skipping "{component_name}" ({lib_name}.{cls_name})')
    return None


def replace_state_dict(sd: dict[str, torch.Tensor], asd: dict[str, torch.Tensor], guess, path: os.PathLike):
    vae_key_prefix = guess.vae_key_prefix[0]
    text_encoder_key_prefix = guess.text_encoder_key_prefix[0]

    if path.endswith("gguf"):
        from backend.loader_gguf import gguf_remapping

        asd = gguf_remapping(asd)

    #   flux 2
    if "decoder.post_quant_conv.weight" in asd:
        asd = state_dict_prefix_replace(asd, {"decoder.post_quant_conv.": "post_quant_conv.", "encoder.quant_conv.": "quant_conv."})

    #   diffusers format
    if "decoder.up_blocks.0.resnets.0.norm1.weight" in asd:
        from huggingface_guess.diffusers_convert import (
            convert_vae_state_dict,
        )

        asd = convert_vae_state_dict(asd)

    #   sd / sdxl / wan                  # wan
    if "decoder.conv_in.weight" in asd or "decoder.middle.0.residual.0.gamma" in asd:
        keys_to_delete = [k for k in sd if k.startswith(vae_key_prefix)]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            sd[vae_key_prefix + k] = v

    ##  identify model type
    flux_test_key = "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale"
    flux_test_key_weight = "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.weight"
    svdq_test_key = "model.diffusion_model.single_transformer_blocks.0.mlp_fc1.qweight"
    legacy_test_key = "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"

    model_type = "-"
    if legacy_test_key in sd:
        match sd[legacy_test_key].shape[1]:
            case 768:
                model_type = "sd1"
            case 1280:
                model_type = "xlrf"  # sdxl refiner model
            case 2048:
                model_type = "sdxl"
    elif flux_test_key in sd or flux_test_key_weight in sd or svdq_test_key in sd:
        model_type = "flux"

    ##  prefixes used by various model types for CLIP-L
    prefix_L = {
        "-": None,
        "sd1": "cond_stage_model.transformer.",
        "xlrf": None,
        "sdxl": "conditioner.embedders.0.transformer.",
        "flux": "text_encoders.clip_l.transformer.",
    }
    ##  prefixes used by various model types for CLIP-G
    prefix_G = {
        "-": None,
        "sd1": None,
        "xlrf": "conditioner.embedders.0.model.transformer.",
        "sdxl": "conditioner.embedders.1.model.transformer.",
        "flux": None,
    }

    ##  VAE format 0 (extracted from model, could be sd1/sdxl)
    if "first_stage_model.decoder.conv_in.weight" in asd:
        if model_type in ("sd1", "xlrf", "sdxl"):
            assert asd["first_stage_model.decoder.conv_in.weight"].shape[1] == 4
            for k, v in asd.items():
                sd[k] = v

    ##  CLIP-G
    CLIP_G = {"conditioner.embedders.1.model.transformer.resblocks.0.ln_1.bias": "conditioner.embedders.1.model.transformer.", "text_encoders.clip_g.transformer.text_model.encoder.layers.0.layer_norm1.bias": "text_encoders.clip_g.transformer.", "text_model.encoder.layers.0.layer_norm1.bias": "", "transformer.resblocks.0.ln_1.bias": "transformer."}  #   key to identify source model                                                old_prefix
    for CLIP_key in CLIP_G.keys():
        if CLIP_key in asd and asd[CLIP_key].shape[0] == 1280:
            new_prefix = prefix_G[model_type]
            old_prefix = CLIP_G[CLIP_key]

            if new_prefix is not None:
                if "resblocks" not in CLIP_key:  # need to convert

                    def convert_transformers(statedict, prefix_from, prefix_to, number):
                        keys_to_replace = {
                            "{}text_model.embeddings.position_embedding.weight": "{}positional_embedding",
                            "{}text_model.embeddings.token_embedding.weight": "{}token_embedding.weight",
                            "{}text_model.final_layer_norm.weight": "{}ln_final.weight",
                            "{}text_model.final_layer_norm.bias": "{}ln_final.bias",
                            "text_projection.weight": "{}text_projection",
                        }
                        resblock_to_replace = {
                            "layer_norm1": "ln_1",
                            "layer_norm2": "ln_2",
                            "mlp.fc1": "mlp.c_fc",
                            "mlp.fc2": "mlp.c_proj",
                            "self_attn.out_proj": "attn.out_proj",
                        }

                        for x in keys_to_replace:  #   remove trailing 'transformer.' from new prefix
                            k = x.format(prefix_from)
                            statedict[keys_to_replace[x].format(prefix_to[:-12])] = statedict.pop(k)

                        for resblock in range(number):
                            for y in ["weight", "bias"]:
                                for x in resblock_to_replace:
                                    k = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_from, resblock, x, y)
                                    k_to = "{}resblocks.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                                    statedict[k_to] = statedict.pop(k)

                                k_from = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_from, resblock, "self_attn.q_proj", y)
                                weightsQ = statedict.pop(k_from)
                                k_from = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_from, resblock, "self_attn.k_proj", y)
                                weightsK = statedict.pop(k_from)
                                k_from = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_from, resblock, "self_attn.v_proj", y)
                                weightsV = statedict.pop(k_from)

                                k_to = "{}resblocks.{}.attn.in_proj_{}".format(prefix_to, resblock, y)

                                statedict[k_to] = torch.cat((weightsQ, weightsK, weightsV))
                        return statedict

                    asd = convert_transformers(asd, old_prefix, new_prefix, 32)
                    for k, v in asd.items():
                        sd[k] = v

                elif old_prefix == "":
                    for k, v in asd.items():
                        new_k = new_prefix + k
                        sd[new_k] = v
                else:
                    for k, v in asd.items():
                        new_k = k.replace(old_prefix, new_prefix)
                        sd[new_k] = v

    ##  CLIP-L
    CLIP_L = {"cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.bias": "cond_stage_model.transformer.", "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.bias": "conditioner.embedders.0.transformer.", "text_encoders.clip_l.transformer.text_model.encoder.layers.0.layer_norm1.bias": "text_encoders.clip_l.transformer.", "text_model.encoder.layers.0.layer_norm1.bias": "", "transformer.resblocks.0.ln_1.bias": "transformer."}  #   key to identify source model                                                    old_prefix

    for CLIP_key in CLIP_L.keys():
        if CLIP_key in asd and asd[CLIP_key].shape[0] == 768:
            new_prefix = prefix_L[model_type]
            old_prefix = CLIP_L[CLIP_key]

            if new_prefix is not None:
                if "resblocks" in CLIP_key:  # need to convert

                    def transformers_convert(statedict, prefix_from, prefix_to, number):
                        keys_to_replace = {
                            "positional_embedding": "{}text_model.embeddings.position_embedding.weight",
                            "token_embedding.weight": "{}text_model.embeddings.token_embedding.weight",
                            "ln_final.weight": "{}text_model.final_layer_norm.weight",
                            "ln_final.bias": "{}text_model.final_layer_norm.bias",
                            "text_projection": "text_projection.weight",
                        }
                        resblock_to_replace = {
                            "ln_1": "layer_norm1",
                            "ln_2": "layer_norm2",
                            "mlp.c_fc": "mlp.fc1",
                            "mlp.c_proj": "mlp.fc2",
                            "attn.out_proj": "self_attn.out_proj",
                        }

                        for k in keys_to_replace:
                            statedict[keys_to_replace[k].format(prefix_to)] = statedict.pop(k)

                        for resblock in range(number):
                            for y in ["weight", "bias"]:
                                for x in resblock_to_replace:
                                    k = "{}resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                                    k_to = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                                    statedict[k_to] = statedict.pop(k)

                                k_from = "{}resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
                                weights = statedict.pop(k_from)
                                shape_from = weights.shape[0] // 3
                                for x in range(3):
                                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                                    k_to = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                                    statedict[k_to] = weights[shape_from * x : shape_from * (x + 1)]
                        return statedict

                    asd = transformers_convert(asd, old_prefix, new_prefix, 12)
                    for k, v in asd.items():
                        sd[k] = v

                elif old_prefix == "":
                    for k, v in asd.items():
                        new_k = new_prefix + k
                        sd[new_k] = v
                else:
                    for k, v in asd.items():
                        new_k = k.replace(old_prefix, new_prefix)
                        sd[new_k] = v

    if "encoder.block.0.layer.0.SelfAttention.k.weight" in asd:
        _key = "umt5xxl" if asd["shared.weight"].size(0) == 256384 else "t5xxl"
        keys_to_delete = [k for k in sd if k.startswith(f"{text_encoder_key_prefix}{_key}.")]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            if k == "spiece_model":
                continue
            sd[f"{text_encoder_key_prefix}{_key}.transformer.{k}"] = v

    elif "encoder.block.0.layer.0.SelfAttention.k.qweight" in asd:
        keys_to_delete = [k for k in sd if k.startswith(f"{text_encoder_key_prefix}t5xxl.")]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            sd[f"{text_encoder_key_prefix}t5xxl.transformer.{k}"] = True
        sd[f"{text_encoder_key_prefix}t5xxl.transformer.filename"] = str(path)

    if "model.layers.0.post_feedforward_layernorm.weight" in asd:
        assert "model.layers.0.self_attn.q_norm.weight" not in asd
        for k, v in asd.items():
            if k == "spiece_model":
                continue
            sd[f"{text_encoder_key_prefix}gemma2_2b.{k}"] = v

    elif "model.visual.deepstack_merger_list.0.norm.weight" in asd:
        size: int = 4 if asd["model.visual.merger.linear_fc2.weight"].shape[0] == 2560 else 8
        for k, v in asd.items():
            sd[f"{text_encoder_key_prefix}qwen3vl_{size}b.transformer.{k}"] = v

    elif "model.layers.0.self_attn.k_proj.bias" in asd:
        weight = asd["model.layers.0.self_attn.k_proj.bias"]
        assert weight.shape[0] == 512
        for k, v in asd.items():
            sd[f"{text_encoder_key_prefix}qwen25_7b.{k}"] = v

    elif "model.layers.0.post_attention_layernorm.weight" in asd and "model.layers.0.self_attn.q_norm.weight" in asd:
        weight: torch.Tensor = asd["model.layers.0.post_attention_layernorm.weight"]
        size: str = "06b" if weight.shape[0] == 1024 else ("4b" if weight.shape[0] == 2560 else "8b")
        for k, v in asd.items():
            sd[f"{text_encoder_key_prefix}qwen3_{size}.transformer.{k}"] = v

    elif "model.layers.0.post_attention_layernorm.weight" in asd:
        weight: torch.Tensor = asd["model.layers.0.post_attention_layernorm.weight"]
        assert weight.shape[0] == 3072

        for k, v in asd.items():
            if not k.startswith("model"):
                continue
            sd[f"{text_encoder_key_prefix}ministral3_3b.transformer.{k}"] = v

    if "visual.blocks.0.attn.proj.weight" in asd:
        for k, v in asd.items():
            sd[f"{text_encoder_key_prefix}qwen25_7b.{k}"] = v

    return sd


def preprocess_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(k.startswith(("model.diffusion_model.", "net.")) for k in sd.keys()):
        sd = {f"model.diffusion_model.{k}": v for k, v in sd.items()}

    return sd


def process_anima(dit: dict[str, torch.Tensor], enc: dict[str, torch.Tensor]):
    # move LLMAdapter from transformer to text_encoder

    keys = list(dit.keys())
    for k in keys:
        if k.startswith("llm_adapter"):
            enc[k] = dit.pop(k)


def _load_unet(path: os.PathLike):
    import huggingface_guess

    sd, metadata = load_torch_file(path, return_metadata=True)
    sd, metadata = convert_quantization(sd, metadata)
    sd = preprocess_state_dict(sd)
    guess = huggingface_guess.guess(sd)

    return sd, metadata, guess


def _load_diffuser(path: os.PathLike):
    import huggingface_guess

    sd, metadata = load_torch_file(path, return_metadata=True)
    sd, metadata = convert_quantization(sd, metadata)
    if (sd := convert_diffusers_mmdit(sd, "")) is None:
        raise ModuleNotFoundError("Failed to recognize model... (only Anima is supported)")
    sd = preprocess_state_dict(sd)
    guess = huggingface_guess.guess(sd)

    return sd, metadata, guess


def split_state_dict(path: os.PathLike, additional_state_dicts: list[os.PathLike] = None):
    try:
        sd, metadata, guess = _load_unet(path)
    except ModuleNotFoundError:
        sd, metadata, guess = _load_diffuser(path)
    finally:
        memory_management.soft_empty_cache()

    if isinstance(additional_state_dicts, list):
        for asd in additional_state_dicts:
            _asd, _meta = load_torch_file(asd, return_metadata=True)
            _asd, _ = convert_quantization(_asd, _meta)
            sd = replace_state_dict(sd, _asd, guess, asd)
            del _asd

    guess.clip_target = guess.clip_target(sd)
    guess.model_type = guess.model_type(sd)
    guess.ztsnr = "ztsnr" in sd

    sd = guess.process_vae_state_dict(sd)

    state_dict = {guess.unet_target: try_filter_state_dict(sd, guess.unet_key_prefix), guess.vae_target: try_filter_state_dict(sd, guess.vae_key_prefix)}

    sd = guess.process_clip_state_dict(sd)

    for k, v in guess.clip_target.items():
        state_dict[v] = try_filter_state_dict(sd, [k + "."])

    state_dict["ignore"] = sd

    if "Anima" in guess.huggingface_repo:
        process_anima(state_dict["transformer"], state_dict["text_encoder"])

    print_dict = {k: len(v) for k, v in state_dict.items()}
    logger.debug(f"StateDict Keys: {print_dict}")

    del state_dict["ignore"]

    return state_dict, guess


@torch.inference_mode()
def forge_loader(sd: os.PathLike, additional_state_dicts: list[os.PathLike] = None) -> "ForgeDiffusionEngine":
    state_dicts, estimated_config = split_state_dict(sd, additional_state_dicts=additional_state_dicts)
    repo_name: str = estimated_config.huggingface_repo

    if not any(type(estimated_config) is x for M in possible_models for x in M.matched_guesses):
        raise ModuleNotFoundError(f"Venno Diffusion UI only supports Anima models (detected: {repo_name})")

    backend.args.dynamic_args.reset()
    backend.args.dynamic_args.anima = True  # other architecture flags (wan, pid, klein, ...) stay at their reset defaults

    from diffusers import DiffusionPipeline

    local_path = os.path.join(HF, repo_name)
    config: dict = DiffusionPipeline.load_config(local_path)

    huggingface_components = {}
    for component_name, v in config.items():
        if isinstance(v, list) and len(v) == 2:
            lib_name, cls_name = v
            component_sd = state_dicts.pop(component_name, None)
            component = load_huggingface_component(estimated_config, component_name, lib_name, cls_name, local_path, component_sd)
            if component_sd is not None:
                del component_sd
            if component is not None:
                huggingface_components[component_name] = component

    del state_dicts

    config_filename = os.path.splitext(sd)[0] + ".yaml"
    yaml_pred_type = None

    if os.path.isfile(config_filename):
        with open(config_filename, "r") as stream:
            yaml_config: dict[str, dict] = yaml.safe_load(stream)

        _params: dict[str, dict] = yaml_config.get("model", {}).get("params", {})
        _pred_type: str = _params.get("parameterization", "") or _params.get("denoiser_config", {}).get("params", {}).get("scaling_config", {}).get("target", "")

        if _pred_type == "v" or _pred_type.endswith(".VScaling"):
            yaml_pred_type = "v_prediction"
        else:
            yaml_pred_type = None

    PRED_TYPES = {
        "EPS": "epsilon",
        "V_PREDICTION": "v_prediction",
        "FLUX": "const",
        "FLOW": "const",
    }

    if "prediction_type" in getattr(huggingface_components.get("scheduler", None), "config", {}):
        if yaml_pred_type:
            huggingface_components["scheduler"].config.prediction_type = yaml_pred_type
        elif estimated_config.model_type.name in PRED_TYPES:
            huggingface_components["scheduler"].config.prediction_type = PRED_TYPES[estimated_config.model_type.name]

    for M in possible_models:
        if any(type(estimated_config) is x for x in M.matched_guesses):
            return M(estimated_config=estimated_config, huggingface_components=huggingface_components)

    raise ModuleNotFoundError("Failed to recognize model... (only Anima is supported)")
