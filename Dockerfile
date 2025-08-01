# api/Dockerfile
# 1) Build dependencies only in builder
FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    libglib2.0-0 libsm6 libxext6 libgl1-mesa-glx libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install Python dependencies with CPU-optimized wheels for faster builds
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# 2) Final runtime with only runtime libs
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx libsm6 libxext6 libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

# העתקת כל ה-Python packages מהמכולה של ה-builder
COPY --from=builder /install /usr/local

# העתקת קוד ה-API שלך
COPY . .

# יצירת משתמש לא שורש
RUN useradd -m appuser && chown -R appuser:appuser /app

USER appuser

ENV PYTHONPATH=/app
ENV MODEL_DIR=/app
ENV API_PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${API_PORT}/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
