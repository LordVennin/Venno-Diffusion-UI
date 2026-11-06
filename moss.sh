#!/usr/bin/env bash
# Venno MOSS: test MOSS-Image LoRAs (Linux / macOS). First start: creates ./venv-moss and installs everything.
# It is separate from Venno's ./venv because MOSS-Image needs newer libraries; both can run at the same time.
# Options go straight to the server, e.g.  ./moss.sh --memory low --lora-dir ~/venno-trainer/output
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python3}"
if [ ! -x venv-moss/bin/python ]; then
    echo "[Venno MOSS] creating venv-moss…"
    "$PYTHON" -m venv venv-moss || exit 1
fi
venv-moss/bin/python -m venno_moss.install $MOSS_ARGS "$@" || exit 1
# shellcheck disable=SC2086
exec venv-moss/bin/python -m venno_moss $MOSS_ARGS "$@"
