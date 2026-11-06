"""python -m venno_moss [options]: start Venno MOSS (Studio at http://127.0.0.1:7870/). moss.sh / moss.bat run this
inside venv-moss after installing what it needs."""

import os
import sys
import threading
import time
import warnings


def main():
    t0 = time.time()
    from venno_moss import VERSION, config

    args = config.parse(sys.argv[1:])
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # fewer fragmentation OOMs on long runs
    warnings.filterwarnings("ignore", message=r".*falling back to its reference PyTorch implementation.*")
    import uvicorn

    from venno_moss import engine, server

    eng = engine.get()
    found = eng.found
    print(f"[Venno MOSS] {VERSION} · device: {eng.device.type}{' (' + eng.info()['gpu'] + ', ' + str(eng.info()['vram_gb']) + ' GB)' if eng.device.type == 'cuda' else ''} · memory: {eng.resolved_memory()}")
    print(f"[Venno MOSS] models: {', '.join(m.name for m in found.models) or 'none yet (download one in Settings → Models)'}")
    host = "0.0.0.0" if args.listen else "127.0.0.1"
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[Venno MOSS] ready in {time.time() - t0:.1f}s: {url}", flush=True)
    if args.autolaunch:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(server.create_app(), host=host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
