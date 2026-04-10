# NS-TEA: Neuro-Symbolic Temporal EHR Agent
# Production Docker image — CPU-only, minimal footprint

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install CPU-only PyTorch first (saves ~600 MB vs full torch) ──
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# ── Install project dependencies ──
COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

# ── Copy application data ──
COPY data/ ./data/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/

# ── Runtime configuration ──
# PORT is set by Railway automatically; fallback to 8001
ENV PORT=8001

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# Expose port
EXPOSE ${PORT}

# Start FastAPI via uvicorn
# - reads PORT from env (Railway sets this)
# - 0.0.0.0 required for container networking
# - no --reload in production
CMD uvicorn nstea.api:app --host 0.0.0.0 --port ${PORT}
