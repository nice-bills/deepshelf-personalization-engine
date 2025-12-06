# Build Stage
FROM python:3.10-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

COPY requirements.txt .

# Create virtual environment and install dependencies
ENV UV_HTTP_TIMEOUT=300
RUN uv venv .venv && \
    uv pip install --no-cache -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# --- Runtime Stage ---
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv:$PATH"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install runtime dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy Scripts
COPY scripts/ ./scripts/

# Data & Model Baking
ENV HF_HOME=/app/data/model_cache

# Download Model
RUN /app/.venv/bin/python scripts/download_model.py

# Download Data
# Ensure data directory exists
RUN mkdir -p data/catalog data/index
RUN /app/.venv/bin/python scripts/download_artifacts.py

# Copy Code (Last to maximize layer caching)
COPY src/ ./src/

# Create directories and permissions
# RUN addgroup --system app && adduser --system --group app && \
#     chown -R app:app /app

# USER app

# Expose port
EXPOSE 7860

# Run Command
CMD ["uvicorn", "src.personalization.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
