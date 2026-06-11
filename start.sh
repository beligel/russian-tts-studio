#!/usr/bin/env bash
# One-click launcher for XTTS Russian TTS Studio.
# Double-click this file in your file manager, or run:  ./start.sh
#
# What it does:
#   1. cd into the project root (no matter where the script lives)
#   2. Auto-pick native window or browser
#   3. Open the UI

set -e

# Resolve project root (parent of the directory holding this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

cd "$PROJECT_ROOT"

# Upstream CosyVoice (the upstream Chinese TTS model) is a source-only
# dependency (no setup.py / pyproject.toml). This project no longer
# requires it (the only TTS engine shipped is XTTS-v2), but we still
# set PYTHONPATH when a sibling checkout is present, so a user who
# keeps CosyVoice/ around for their own experiments won't be surprised
# by missing imports.
COSYVOICE_SRC=""
for candidate in \
    "$PROJECT_ROOT/../CosyVoice" \
    "$HOME/CosyVoice" \
    "/opt/CosyVoice"; do
    if [ -d "$candidate/cosyvoice" ]; then
        COSYVOICE_SRC="$candidate"
        break
    fi
done
if [ -n "$COSYVOICE_SRC" ]; then
    export PYTHONPATH="$COSYVOICE_SRC${PYTHONPATH:+:$PYTHONPATH}"
fi

# Coqui TTS (XTTS-v2) asks the user to confirm the non-commercial CPML
# license on first model download — without this env var the call blocks
# on stdin and the web server never starts. The user accepts the
# non-commercial terms via the engine picker in the UI.
export COQUI_TOS_AGREED=1

# Default port for the Web UI. Override with `PORT=9000 ./start.sh --port 9000`
# or just `./start.sh --port 9000`. Keep this in sync with web/run.py and
# web/start.py defaults (8129) so that `./start.sh` and `make start` open the
# same URL.
export PORT="${PORT:-8129}"

# Pick a Python — prefer the local venv if it exists, then `python3`, then `python`
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PY="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "ERROR: neither 'python3' nor 'python' is in PATH" >&2
    echo "Install Python 3.10+ and try again." >&2
    exit 1
fi

echo "=========================================="
echo " Russian TTS Studio"
echo " Project:      $PROJECT_ROOT"
echo " CosyVoice:    ${COSYVOICE_SRC:-not on PYTHONPATH (not required)}"
echo " Python:       $($PY --version)"
echo " Web UI port:  $PORT  (override: PORT=9000 ./start.sh)"
echo " Log file:     $PROJECT_ROOT/output/logs/web-$(date +%Y%m%d).log"
echo "=========================================="
echo "If something fails, attach that log file to your bug report."
echo "Once started, open http://127.0.0.1:$PORT in your browser."
echo ""

exec "$PY" -m web.start "$@"
