# ========= Builder =========
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (slim but covers faiss/torch openmp runtime + basic build needs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps into a venv for smaller runtime image & better caching
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# If using requirements.txt:
COPY requirements.txt .
# RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install -r requirements.txt

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
RUN chown -R appuser:appuser /app
USER appuser

# Container paths you’ll use
ENV INDEX_DIR=/data/indexes/faiss \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# Default command (no reload in production image)
# Dockerfile (bottom)
EXPOSE 8080
CMD ["/bin/sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]