#!/usr/bin/env bash
# Talk to EyeReach with Amazon Nova 2 Sonic: speak into the microphone, it answers through the speaker.
#
#   launch/talk.sh                      talk (Ctrl+C ends)
#   launch/talk.sh --text "Hello"       one typed sentence; the spoken reply goes to /tmp/eyereach_sonic_reply.wav
#   launch/talk.sh --speaker 25         pick the speaker (launch/talk.sh --list shows the devices)
#
# Needs AWS keys in the repo's .env and ~/eyereach_agent_venv (Python 3.12, made by launch/setup_jetson.sh,
# plus: uv pip install --python ~/eyereach_agent_venv/bin/python "aws_sdk_bedrock_runtime>=0.1.0,<0.2.0" smithy-aws-core sounddevice).
# It only talks: it never moves the arm.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PYTHON="$HOME/eyereach_agent_venv/bin/python"
[ -x "$PYTHON" ] || { echo "Missing $PYTHON: run bash launch/setup_jetson.sh first."; exit 1; }
if [ -f "$REPO/.env" ]; then
  set -a; . "$REPO/.env"; set +a
fi
cd "$REPO"
exec "$PYTHON" -u -m agent.voice_sonic "$@"
