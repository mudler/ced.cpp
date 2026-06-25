#!/usr/bin/env bash
# Render the ced.cpp CAPABILITY REEL (mp4 + gif, 16:9 hero + 1:1 square).
#
# Unlike a screen recorder, this composes every frame with Pillow and stitches
# them with ffmpeg, so the content is real: the three Public-Domain clips in
# clips/ (rooster, thunderstorm, acoustic guitar), each shown as a mel-spectrogram
# wiping in under a playhead, then the REAL top-5 AudioSet tags from spec.json
# snapping in with confidence bars, ending on the LocalAI card.
#
# spec.json holds the real measured numbers. Regenerate them from a built
# checkout (binary at build/examples/cli/ced-cli, model in models/) with:
#   ced-cli classify models/ced-base-f16.gguf clips/<x>.wav --top-k 5   # tags
#   ced-cli bench    models/ced-base-f16.gguf clips/<x>.wav --threads 4 # latency
# then paste the values into spec.json. See clips/SOURCES.md for clip provenance.
#
#   ./make.sh                 # writes out/ced_reel.{mp4,gif} + out/ced_reel_square.{mp4,gif}
#   ./make.sh --beat 3.4      # extra renderer flags pass through
#
# env:
#   PYTHON  python with Pillow + numpy (default: ~/recon-demos/venv/bin/python, then python3)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x "$HOME/recon-demos/venv/bin/python" ]; then
  PY="$HOME/recon-demos/venv/bin/python"
else
  PY="python3"
fi

"$PY" "$HERE/ced_reel.py" --spec "$HERE/spec.json" --clips "$HERE" \
  --out "$HERE/out" "$@"

echo "-> $HERE/out/ced_reel.mp4 (+ .gif)"
echo "-> $HERE/out/ced_reel_square.mp4 (+ .gif)"
