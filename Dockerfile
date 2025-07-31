# Dockerfile for Hazard Detection API Service
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ cmake pkg-config wget curl \
    libhdf5-dev libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev libgomp1 \
    libusb-1.0-0-dev libgtk-3-dev libavcodec-dev \
    libavformat-dev libswscale-dev libv4l-dev \
    cpuid util-linux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user
RUN useradd -m -s /bin/bash appuser && \
    mkdir -p /home/appuser/.local/bin && \
    chown -R appuser:appuser /home/appuser

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application files
COPY . .

# Create model directories
RUN mkdir -p /app/models/openvino /app/models/pytorch \
    && chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models
ENV API_PORT=8000
ENV PATH=/home/appuser/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER appuser

# Start command
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]