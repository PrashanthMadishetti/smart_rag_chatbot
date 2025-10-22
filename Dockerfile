# ========= Builder =========
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (OpenMP for faiss/torch + build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a venv so runtime stays slim
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install deps (torch from CPU index first, then the rest)
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install -r requirements.txt

# Pre-download a small SentenceTransformers model at build time (avoid cold-start download)
# (Make sure 'sentence-transformers' is listed in requirements.txt)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ========= Runtime =========
FROM python:3.11-slim

# OpenMP runtime for faiss/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy Python venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd -m appuser

WORKDIR /app
COPY . /app

# Copy the Hugging Face cache that builder created (so appuser can access the preloaded model)
RUN mkdir -p /home/appuser/.cache/huggingface
COPY --from=builder /root/.cache/huggingface /home/appuser/.cache/huggingface

# Ownership
RUN chown -R appuser:appuser /app /home/appuser/.cache
USER appuser

# Cloud Run expects port 8080
EXPOSE 8080
CMD ["/bin/sh","-c","uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]