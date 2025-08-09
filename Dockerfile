# Hazard Detection API - Multi-stage Docker build for production
FROM python:3.11.9-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libgomp1 \
    libfontconfig1 \
    libxrender1 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install Python dependencies with pinned versions
RUN pip install --no-cache-dir --upgrade pip==23.3.1 setuptools==69.0.2 wheel==0.42.0 && \
    pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Production runtime stage
FROM python:3.11.9-slim AS runtime

# Set working directory
WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libgomp1 \
    libfontconfig1 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder stage
COPY --from=builder /install /usr/local

# Create non-root user for security (before copying files)
RUN useradd -m -u 1000 appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Set proper permissions
RUN chmod -R 755 /app && \
    find /app -name "*.py" -exec chmod 644 {} \;

# Switch to non-root user
USER appuser

# Set Python environment variables for production
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app \
    MODEL_BACKEND=openvino \
    PORT=8080 \
    HOST=0.0.0.0 \
    LOG_LEVEL=INFO

# Expose application port
EXPOSE 8080

# Health check configuration
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start the application using uvicorn directly for better production control
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
