import os
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from app.model import get_model, run_inference

app = FastAPI(
    title="EcoScan Scan API",
    description="API de détection de déchets YOLOv8",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    get_model()


@app.get("/")
def root():
    return {"message": "EcoScan Scan API fonctionne correctement."}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = run_inference(image)
    return JSONResponse(content=results)
