import os
import io
import json
import base64
import google.generativeai as genai
from PIL import Image

GEMINI_MODEL = "gemini-2.0-flash"

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

_model = None


def _get_model():
    global _model
    if _model is None:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
        _model = genai.GenerativeModel(GEMINI_MODEL)
    return _model


def run_inference(image: Image.Image) -> dict:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    buf.seek(0)

    response = _get_model().generate_content([PROMPT, Image.open(buf)])

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

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
        "source": "gemini",
    }
