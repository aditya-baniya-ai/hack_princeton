"""Amazon Nova finds the objects on the table and says which one a request means.

    python3 -m eyereach.vision_nova photo.jpg                 prints every object Nova finds
    python3 -m eyereach.vision_nova photo.jpg "the pills"     also asks which object is the pills

One image per call and no retries, because there are no AWS credits. If a call fails or times out it is
logged and the answer is empty, so the app keeps running.

Environment variables:
    AWS_BEARER_TOKEN_BEDROCK   a Bedrock API key, or the usual AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
    AWS_REGION                 default us-east-1
    NOVA_MODEL_ID              default us.amazon.nova-2-lite-v1:0
"""

import json
import logging
import os
import re
import sys
from dataclasses import dataclass

import cv2

log = logging.getLogger(__name__)

MODEL = os.environ.get("NOVA_MODEL_ID", "us.amazon.nova-2-lite-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")
TIMEOUT = 20                                   # seconds to wait for Nova's answer

from eyereach.project_context import VISION_CONTEXT  # noqa: E402  (what the project is, short for vision)

DETECT_PROMPT = VISION_CONTEXT + """

Find the objects on this table{only}.
Answer with only a JSON list, one entry per object:
[{{"name": "cup", "box": [x1, y1, x2, y2]}}]
Box coordinates are scaled 0 to 1000 to the image width and height. Use short everyday names, as specific
as you can tell, with the colour when it helps ("orange", "red apple", "white bottle"); look closely at
objects partly hidden by the gripper (an orange has a dimpled orange peel; an apple is smooth and shiny).
Only list things a robot arm could pick up: not the table, the wall or the background.
If there are no objects, answer []."""

PICK_PROMPT = VISION_CONTEXT + """

The person asked for: "{request}".
Which single object in this image is it? Answer with only a JSON object:
{{"name": "pills", "box": [x1, y1, x2, y2], "reason": "the white bottle labelled pills"}}
Box coordinates are scaled 0 to 1000 to the image width and height.
If no object clearly matches, or two could, answer {{"name": null, "box": null, "reason": "<why, in a few words>"}}."""


@dataclass
class Detection:
    name: str
    box: tuple                                 # (x0, y0, x1, y1) in pixels
    confidence: float = None                   # Nova does not give one; kept for other detectors


def _client():
    import boto3
    from botocore.config import Config
    config = Config(connect_timeout=5, read_timeout=TIMEOUT, retries={"max_attempts": 1})
    return boto3.client("bedrock-runtime", region_name=REGION, config=config)


def _ask(image_bgr, prompt):
    """One image and one question to Nova -> its text answer, or None if the call failed."""
    ok, jpg = cv2.imencode(".jpg", image_bgr)
    if not ok:
        log.warning("Nova: could not encode the image")
        return None
    try:
        response = _client().converse(
            modelId=MODEL,
            messages=[{"role": "user", "content": [
                {"image": {"format": "jpeg", "source": {"bytes": jpg.tobytes()}}},
                {"text": prompt}]}],
            inferenceConfig={"temperature": 0, "maxTokens": 1000})
    except Exception as e:                     # no network, no keys, timeout: the app must keep going
        log.warning("Nova call failed: %s", e)
        return None
    return response_text(response)


def response_text(response):
    """The text of a Bedrock Converse response."""
    return "".join(part.get("text", "") for part in response["output"]["message"]["content"])


def _json(text):
    """The first JSON list or object in Nova's text (it sometimes wraps it in ```json fences)."""
    match = re.search(r"[\[{].*[\]}]", text or "", re.S)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def to_pixels(box, width, height):
    """A Nova box on the 0-1000 scale -> (x0, y0, x1, y1) in pixels, inside the image."""
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    clamp = lambda v, size: int(round(min(max(v, 0), 1000) * size / 1000))  # noqa: E731
    return clamp(x0, width), clamp(y0, height), clamp(x1, width), clamp(y1, height)


def _detection(item, width, height):
    """One {"name", "box"} entry -> a Detection, or None if it is malformed."""
    try:
        name, box = str(item["name"]).strip(), item["box"]
        if not name or len(box) != 4:
            return None
        return Detection(name, to_pixels(box, width, height), item.get("confidence"))
    except (KeyError, TypeError, ValueError):
        return None


def parse_detections(text, width, height):
    """Nova's answer to DETECT_PROMPT -> a list of Detections in pixels (bad entries skipped)."""
    items = _json(text)
    if not isinstance(items, list):
        return []
    found = (_detection(item, width, height) for item in items if isinstance(item, dict))
    return [d for d in found if d]


def parse_match(text, width, height):
    """Nova's answer to PICK_PROMPT -> (Detection or None, a short reason)."""
    answer = _json(text)
    if not isinstance(answer, dict):
        return None, "could not read Nova's answer"
    reason = str(answer.get("reason") or "")
    if not answer.get("name") or not answer.get("box"):
        return None, reason or "no object clearly matches"
    detection = _detection(answer, width, height)
    return (detection, reason) if detection else (None, "Nova's box was malformed")


def detect_objects(image_bgr, names=None):
    """Every object Nova sees in a BGR image (only the given names, if any) -> list of Detections."""
    only = f", only these: {', '.join(names)}" if names else ""
    text = _ask(image_bgr, DETECT_PROMPT.format(only=only))
    height, width = image_bgr.shape[:2]
    return parse_detections(text, width, height) if text else []


def pick_match(image_bgr, request):
    """Which object is the one asked for ("the pills") -> (Detection, reason), or (None, reason) if unsure."""
    text = _ask(image_bgr, PICK_PROMPT.format(request=request))
    if text is None:
        return None, "Nova could not be reached"
    height, width = image_bgr.shape[:2]
    return parse_match(text, width, height)


def main():
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        sys.exit("usage: python3 -m eyereach.vision_nova photo.jpg [\"request\"]")
    image = cv2.imread(sys.argv[1])
    if image is None:
        sys.exit(f"could not read {sys.argv[1]}")
    found = detect_objects(image)
    print(f"{len(found)} objects ({MODEL}):")
    for d in found:
        print(f"  {d.name:20s} box {d.box}")
    if len(sys.argv) > 2:
        match, reason = pick_match(image, sys.argv[2])
        print(f'"{sys.argv[2]}" -> {match.name + " " + str(match.box) if match else "not sure"} ({reason})')


if __name__ == "__main__":
    main()
