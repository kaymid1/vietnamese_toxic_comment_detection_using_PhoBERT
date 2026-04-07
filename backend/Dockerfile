# =============================================================================
# Backend Dockerfile — CPU-only PyTorch inference
#
# Runs on any machine: laptop, AMD64 VM, or Apple Silicon (ARM64).
# No GPU required.
#
# TODO(gpu-upgrade): when you get a GPU VM, swap this entire file with:
#
#   FROM pytorch/pytorch:2.9.1-cuda12.1-cudnn8-runtime
#   WORKDIR /app
#   RUN apt-get update && apt-get install -y --no-install-recommends curl \
#       && rm -rf /var/lib/apt/lists/*
#   COPY requirements-ml.txt requirements-base.txt ./
#   RUN pip install --no-cache-dir -r requirements-ml.txt \
#    && pip install --no-cache-dir -r requirements-base.txt
#   COPY . .
#   EXPOSE 8000
#   CMD ["gunicorn", "--worker-class", "uvicorn.workers.UvicornWorker", \
#        "--bind", "0.0.0.0:8000", "--workers", "1", "backend.app:app"]
#
# =============================================================================

FROM python:3.12-slim
WORKDIR /app

# libgomp1: required by PyTorch (OpenMP threading support)
# curl: used by healthcheck in docker-compose
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgomp1 \
       curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# Layer 1: torch CPU-only wheel (~800 MB uncompressed)
#
# Installed separately from its own wheel index so:
#   a) the CUDA wheel (~2.5 GB) is never accidentally pulled, and
#   b) this massive layer is cached independently — it only rebuilds
#      when torch version changes.
#
# TODO(gpu-upgrade): change --index-url to:
#   https://download.pytorch.org/whl/cu121
# -------------------------------------------------------------------
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.9.1

# -------------------------------------------------------------------
# Layer 2: ML ecosystem (transformers, datasets, sklearn, etc.)
# Separate layer from app code — rebuilds only when model deps change.
# -------------------------------------------------------------------
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# -------------------------------------------------------------------
# Layer 3: API / utility deps (fastapi, gunicorn, crawling libs…)
# Separate layer from torch — avoids re-downloading torch on API changes.
# -------------------------------------------------------------------
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# -------------------------------------------------------------------
# Layer 4: Application source (changes most often — always last)
# Context is the repo root because backend/app.py imports root-level
# modules: domain_classifier, setup_and_crawl, infer_crawled_local
# -------------------------------------------------------------------
COPY . .

# Model weights (PhoBERT) and VnCoreNLP JARs are NOT baked into the image.
# Mount them at runtime via docker-compose volumes:
#   ./models    → /app/models    (PhoBERT weights)
#   ./vncorenlp → /app/vncorenlp (VnCoreNLP JARs)
EXPOSE 8000

CMD ["gunicorn", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "backend.app:app"]
