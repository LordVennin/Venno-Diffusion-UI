#!/usr/bin/env bash
# Venno Diffusion UI (Linux / macOS). First start: creates ./venv and installs everything.
# Options go straight to the server, e.g.  ./venno.sh --listen --lowvram   (./venno.sh --help lists them)
# Settings you always want can go in VENNO_ARGS, e.g.  export VENNO_ARGS="--models-dir /data/models"
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python3}"
if [ -z "$VIRTUAL_ENV" ] && [ -z "$VENNO_NO_VENV" ]; then
    if [ ! -x venv/bin/python ]; then
        echo "[Venno] creating venv…"
        "$PYTHON" -m venv venv || exit 1
    fi
    PYTHON="venv/bin/python"
fi
# shellcheck disable=SC2086
"$PYTHON" -m venno.install $VENNO_ARGS "$@" || exit 1
# shellcheck disable=SC2086
exec "$PYTHON" -m venno $VENNO_ARGS "$@"
