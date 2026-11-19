# Multi-stage build: dependencies are installed in a throw-away "builder" stage, and only
# the ready virtualenv + the app code are copied into a small runtime image.

# ---- Stage 1: build the virtualenv ----
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- Stage 2: runtime ----
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser
COPY --from=builder /opt/venv /opt/venv
COPY app ./app

# The ONNX model is mounted at runtime (see docker-compose.yml: ./models -> /app/models)
ENV MODEL_PATH=/app/models/best.onnx \
    LABELS_PATH=/app/models/labels.json \
    METADATA_PATH=/app/models/metadata.json
USER appuser
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
