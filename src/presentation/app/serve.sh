#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8080}"

echo "DP Synthetic Data demo"
echo "Serving from: $SCRIPT_DIR"
echo "Port: $PORT"
echo ""
echo "On your laptop, run:"
echo "  ssh -L ${PORT}:localhost:${PORT} <your-ssh-alias>"
echo "Then open: http://localhost:${PORT}"
echo ""
echo "Press Ctrl+C to stop."
echo "----------------------------------------"

# Find python3 — prefer the project venv, fall back to system
PYTHON="${PYTHON:-$(command -v python3 2>/dev/null || echo python3)}"
if [ -x /opt/venv/bin/python3 ]; then PYTHON=/opt/venv/bin/python3; fi

# Re-exec under sudo if not root (files are owned by root)
if [ "$(id -u)" -ne 0 ]; then
  echo "Not running as root — re-launching with sudo..."
  exec sudo "$PYTHON" -m http.server "$PORT" --directory "$SCRIPT_DIR"
fi

exec "$PYTHON" -m http.server "$PORT" --directory "$SCRIPT_DIR"
