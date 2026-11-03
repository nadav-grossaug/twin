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

cd "$SCRIPT_DIR"
# The app files are owned by root; re-exec under sudo if needed
if [ "$(id -u)" -ne 0 ]; then
  echo "Not running as root — re-launching with sudo..."
  exec sudo python3 -m http.server "$PORT" --directory "$SCRIPT_DIR"
fi
python3 -m http.server "$PORT"
