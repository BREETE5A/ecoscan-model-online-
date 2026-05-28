FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pré-télécharger le modèle au build pour éviter le cold start
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='turhancan97/yolov8-segment-trash-detection', filename='best.pt')"

COPY app/ ./app/

EXPOSE 10000

COPY start.sh .
RUN chmod +x start.sh
CMD ["sh", "start.sh"]
