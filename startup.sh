#!/bin/bash
set -e

echo "🚀 Starting Hazard Detection API..."

# Check Python and dependencies
echo "🔍 Python version: $(python --version)"
echo "🔍 Python path: $PYTHONPATH"

# Check if critical modules can be imported
echo "🔍 Testing imports..."
python -c "
import sys
try:
    import fastapi
    print('✅ FastAPI OK')
    import uvicorn 
    print('✅ Uvicorn OK')
    import openvino
    print('✅ OpenVINO OK')
    import aioredis
    print('✅ aioredis OK')
    from app.core.config import settings
    print('✅ Config OK')
    from app.services.model_service import model_service
    print('✅ Model service OK')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
"

# Check if model files exist
echo "🔍 Checking model files..."
ls -la /app/best* 2>/dev/null || echo "⚠️ No best* model files found"
ls -la /app/best*/* 2>/dev/null || echo "⚠️ No model files in subdirectories"

# Check environment
echo "🔍 Environment variables:"
echo "MODEL_DIR: $MODEL_DIR"
echo "MODEL_BACKEND: $MODEL_BACKEND" 
echo "PORT: $PORT"
echo "LOG_LEVEL: $LOG_LEVEL"

# Start the application with better error handling
echo "🚀 Starting uvicorn server..."
exec python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 1 \
    --log-level info \
    --access-log \
    --use-colors