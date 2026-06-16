import os
import io
import json
import base64
import requests
from PIL import Image

GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

VALID_LABELS = {"plastique", "verre", "metal", "cardboard", "paper", "organique", "electronique", "textile", "dangereux", "bois", "trash"}

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
- "electronique" -> batteries, cables, electronic devices, chargers
- "textile"      -> clothing, shoes, fabric, bags, curtains, bedding
- "dangereux"    -> paint, chemicals, motor oil, medications, syringes, pesticides
- "bois"         -> wood furniture, planks, pallets, branches
- "trash"        -> non-recyclable or unidentifiable waste

Confidence must be a float between 0.0 and 1.0."""


def run_inference(image: Image.Image) -> dict:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    payload = {
        "contents": [{
            "parts": [
                {"text": PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
            ]
        }]
    }

    response = requests.post(
        GEMINI_URL,
        params={"key": api_key},
        json=payload,
        timeout=30,
    )
    response.raise_for_status()

    text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
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
