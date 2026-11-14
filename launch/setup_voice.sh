#!/usr/bin/env bash
# One-time optional setup on Jetson Ubuntu 22.04 / Python 3.10 / aarch64.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICE_PYTHON="${EYEREACH_PYTHON:-$HOME/eyereach_venv/bin/python}"
VOICE_MODEL_ROOT="$HOME/.eyereach/models"
VOICE_MODEL_NAME=vosk-model-small-en-us-0.15
if [ ! -x "$VOICE_PYTHON" ]; then
  echo "Set up ~/eyereach_venv first (README), or set EYEREACH_PYTHON to its Python executable."
  exit 1
fi
sudo apt-get update
sudo apt-get install -y libportaudio2 portaudio19-dev alsa-utils curl unzip
"$VOICE_PYTHON" -m pip install -r "$HERE/../requirements-voice.txt"
mkdir -p "$VOICE_MODEL_ROOT"
if [ ! -d "$VOICE_MODEL_ROOT/$VOICE_MODEL_NAME" ]; then
  VOICE_DOWNLOAD="$(mktemp -d)"
  trap 'rm -rf "$VOICE_DOWNLOAD"' EXIT
  curl --fail --location --retry 3 \
    "https://alphacephei.com/vosk/models/$VOICE_MODEL_NAME.zip" -o "$VOICE_DOWNLOAD/model.zip"
  unzip -q "$VOICE_DOWNLOAD/model.zip" -d "$VOICE_DOWNLOAD"
  test -f "$VOICE_DOWNLOAD/$VOICE_MODEL_NAME/am/final.mdl"
  mv "$VOICE_DOWNLOAD/$VOICE_MODEL_NAME" "$VOICE_MODEL_ROOT/"
fi
cd "$HERE/.."
"$VOICE_PYTHON" -m eyereach.voice_commands --list-devices
echo "Ready. Test recognition without a robot:"
echo "  $VOICE_PYTHON -m eyereach.voice_commands"
echo "Then run: launch/eyereach.sh --no-arm --voice"
