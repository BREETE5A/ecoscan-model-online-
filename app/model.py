import os
import io
import json
import base64
import anthropic
from PIL import Image

CLAUDE_MODEL = "claude-haiku-4-5"

_client: anthropic.Anthropic | None = None

VALID_LABELS = {"plastique", "verre", "metal", "cardboard", "paper", "organique", "electronique", "trash"}

PROMPT = """Analyze this image and classify the waste item visible.

Respond with ONLY a valid JSON object, no explanation, no markdown:
{"label": "plastique", "confidence": 0.92}

The label must be exactly one of:
- "plastique"    -> plastic bottles, bags, containers, cups, straws, foam
- "verre"        -> glass bottles, jars, broken glass
- "metal"        -> cans, aluminum foil, bottle caps, scrap metal
- "cardboard"    -> cardboard boxes, egg cartons, drink cartons
- "paper"        -> newspapers, magazines, tissues, paper bags
- "organique"    -> food waste, organic matter
- "electronique" -> batteries, cables, electronics
- "trash"        -> non-recyclable or unidentifiable waste

Confidence must be a float between 0.0 and 1.0."""


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    return _client


def run_inference(image: Image.Image) -> dict:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    image_b64 = base64.b64encode(buf.read()).decode("utf-8")

    response = _get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": PROMPT},
            ],
        }],
    )

    text = response.content[0].text.strip()
    data = json.loads(text)

    label = data.get("label", "trash")
    if label not in VALID_LABELS:
        label = "trash"
    confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))

    return {
        "detections": [{"label": label, "confidence": round(confidence, 3)}],
        "count": 1,
        "top_label": label,
        "top_confidence": round(confidence, 3),
        "source": "claude",
    }
