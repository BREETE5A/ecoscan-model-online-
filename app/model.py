import os
import io
from inference_sdk import InferenceHTTPClient
from PIL import Image

ROBOFLOW_API_KEY  = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_API_URL  = "https://serverless.roboflow.com"
ROBOFLOW_MODEL_ID = "yolov8-trash-detections-kgnug/11"
CONFIDENCE        = 0.25

_client = InferenceHTTPClient(
    api_url=ROBOFLOW_API_URL,
    api_key=ROBOFLOW_API_KEY,
)


def run_inference(image: Image.Image) -> dict:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    buf.seek(0)

    result = _client.infer(buf, model_id=ROBOFLOW_MODEL_ID)

    detections = []
    for p in result.get("predictions", []):
        conf = float(p["confidence"])
        if conf < CONFIDENCE:
            continue
        w, h, cx, cy = p["width"], p["height"], p["x"], p["y"]
        detections.append({
            "label":      p["class"],
            "confidence": round(conf, 3),
            "box": {
                "x1": round(cx - w / 2),
                "y1": round(cy - h / 2),
                "x2": round(cx + w / 2),
                "y2": round(cy + h / 2),
            },
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return {
        "detections":     detections,
        "count":          len(detections),
        "top_label":      detections[0]["label"]      if detections else None,
        "top_confidence": detections[0]["confidence"] if detections else None,
        "source":         "roboflow",
    }
