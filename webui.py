if __name__ == "__main__":
    raise SystemError("Call launch.py instead")


from threading import Thread

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from modules import initialize, initialize_util, timer
from modules_forge.initialization import initialize_forge

startup_timer = timer.startup_timer
startup_timer.record("launcher")

initialize.shush()

with startup_timer.subcategory("forge init"):
    initialize_forge()

initialize.imports()

initialize.check_versions()

initialize.initialize()


def _handle_exception(request: Request, e: Exception):
    error_information = vars(e)
    content = {
        "error": type(e).__name__,
        "detail": error_information.get("detail", ""),
        "body": error_information.get("body", ""),
        "message": str(e),
    }
    return JSONResponse(status_code=int(error_information.get("status_code", 500)), content=jsonable_encoder(content))


def create_api(app):
    from modules.api.api import Api
    from modules.call_queue import queue_lock

    api = Api(app, queue_lock)
    return api


def api_only_worker():
    """Venno Studio: FastAPI server with the generation API and the Studio front-end (no Gradio page)"""
    from fastapi import FastAPI

    from modules.shared_cmd_options import cmd_opts

    app = FastAPI(exception_handlers={Exception: _handle_exception})
    initialize_util.setup_middleware(app)
    api = create_api(app)

    from modules import script_callbacks

    script_callbacks.before_ui_callback()
    script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    api.launch(server_name=initialize_util.gradio_server_name(), port=cmd_opts.port if cmd_opts.port else 7860, root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else "")


def api_only():
    Thread(target=api_only_worker, daemon=True).start()
