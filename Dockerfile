# Multi-stage build for minimal image size
# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download Wav2Vec2 model to cache (reduces cold start time)
# Set HF_HUB_CACHE so builder and runtime use the same cache path
ENV HF_HUB_CACHE=/root/.cache/huggingface/hub
RUN python -c "from transformers import Wav2Vec2Model, Wav2Vec2Processor; Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h'); Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')"

# Stage 2: Runtime (minimal)
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies (FFmpeg for pydub)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
ENV PATH=/root/.local/bin:$PATH

# Copy only necessary application files
COPY app.py .
COPY audio/ ./audio/
COPY model/ ./model/
COPY utils/ ./utils/

# Environment
ENV PYTHONUNBUFFERED=1
# Use HF_HUB_CACHE (not deprecated TRANSFORMERS_CACHE) and match builder path
ENV HF_HUB_CACHE=/root/.cache/huggingface/hub
ENV PRODUCTION=true

# Memory optimization for 1GB RAM environments
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MALLOC_TRIM_THRESHOLD_=65536
ENV PYTHONMALLOC=malloc
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 8000

# Start server - single worker, no reload, memory-optimized
# limit-concurrency raised to 10 so platform health probes don't block real requests
CMD sh -c "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 120 --workers 1 --limit-concurrency 10"
