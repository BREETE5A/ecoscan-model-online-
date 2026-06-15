import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from app.model import run_inference, CLAUDE_MODEL

app = FastAPI(
    title="EcoScan Scan API",
    description="API de détection de déchets — Claude Vision",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "EcoScan Scan API — Claude Vision", "model": CLAUDE_MODEL}


@app.get("/health")
def health():
    return {"status": "healthy", "model": CLAUDE_MODEL}


@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = run_inference(image)
    return JSONResponse(content=results)
