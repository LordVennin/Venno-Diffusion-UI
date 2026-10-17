"""python -m venno [options]: start the Venno server (Studio at http://127.0.0.1:7860/)."""

import os
import sys
import threading
import time


def _print_help():
    from backend.args import parser as engine_parser
    from venno import config

    config.parser.print_help()
    print("\nengine options:")
    for action in engine_parser._actions:
        if action.option_strings:
            print(f"  {', '.join(action.option_strings):<34} {action.help or ''}")


def main():
    t0 = time.time()
    from venno import config

    if config.args.help:
        _print_help()
        return

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    if os.name == "nt":
        os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
    hf_cache = os.path.join(config.models_path(), "diffusers")  # Hugging Face downloads stay inside the models folder
    for var in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "HUGGINGFACE_HUB_CACHE", "HUGGINGFACE_ASSETS_CACHE"):
        os.environ.setdefault(var, hf_cache)

    from backend.args import args as engine_args

    if engine_args.gpu_device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(engine_args.gpu_device_id)
    from backend import cuda_malloc

    if "rocm" in cuda_malloc.get_torch_version():
        os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
        os.environ.setdefault("OCL_SET_SVM_SIZE", "262144")
    if engine_args.cuda_malloc:
        cuda_malloc.try_cuda_malloc()
    if engine_args.expandable_segments:
        cuda_malloc.try_expandable_segments()

    import torch

    from backend import memory_management
    from backend.huggingface import process as unpack_tokenizers

    device = memory_management.get_torch_device()
    torch.zeros((1, 1)).to(device, torch.float32)  # warm up
    memory_management.soft_empty_cache()
    unpack_tokenizers()

    import uvicorn

    from venno.server.app import create_app

    app = create_app()
    host, port = config.listen_host(), config.args.port
    url = f"http://127.0.0.1:{port}/"
    print(f"[Venno] device: {device}, ready in {time.time() - t0:.1f}s")
    print(f"[Venno] Studio running on: {url}" + (f" (also on the network, {host}:{port})" if config.is_shared() else ""))
    if config.args.autolaunch:
        import webbrowser

        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)


if __name__ == "__main__":
    sys.exit(main())
