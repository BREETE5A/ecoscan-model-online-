import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image

_model = None
MODEL_REPO = "turhancan97/yolov8-segment-trash-detection"
MODEL_FILE = "yolov8m-seg.pt"


def get_model() -> YOLO:
    global _model
    if _model is None:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
        )
        _model = YOLO(model_path)
    return _model


def run_inference(image: Image.Image) -> dict:
    model = get_model()
    results = model(image, conf=0.25, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            xyxy = box.xyxy[0].tolist()
            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "box": {
                    "x1": round(xyxy[0]),
                    "y1": round(xyxy[1]),
                    "x2": round(xyxy[2]),
                    "y2": round(xyxy[3]),
                }
            })

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return {
        "detections": detections,
        "count": len(detections),
        "top_label": detections[0]["label"] if detections else None,
        "top_confidence": detections[0]["confidence"] if detections else None,
    }
