#!/bin/bash
set -e

echo "🚀 Starting Hazard Detection API..."

# Check basic system info
echo "🔍 System info:"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "Python path: $PYTHONPATH"

# Test basic imports first
echo "🔍 Testing basic imports..."
python -c "
import sys
print(f'Python executable: {sys.executable}')
print(f'Python path: {sys.path[:3]}...')

# Test core dependencies
try:
    import fastapi; print('✅ FastAPI OK')
    import uvicorn; print('✅ Uvicorn OK')  
    import pydantic; print('✅ Pydantic OK')
except ImportError as e:
    print(f'❌ Core import failed: {e}')
    print('🚨 Starting in minimal mode...')
    sys.exit(42)  # Special code for minimal mode
"

# Check import result
IMPORT_RESULT=$?
if [ $IMPORT_RESULT -eq 42 ]; then
    echo "🚨 Core imports failed, starting minimal app..."
    exec python -m uvicorn app.main_minimal:app --host 0.0.0.0 --port 8080 --log-level info
fi

# Test advanced dependencies
echo "🔍 Testing advanced imports..."
python -c "
import sys
try:
    import openvino; print('✅ OpenVINO OK')
except ImportError as e:
    print(f'⚠️ OpenVINO not available: {e}')

try:
    import aioredis; print('✅ aioredis OK') 
except ImportError as e:
    print(f'⚠️ aioredis not available: {e}')

try:
    from app.core.config import settings; print('✅ Config OK')
except ImportError as e:
    print(f'⚠️ Config import failed: {e}')
"

# Check file system
echo "🔍 Checking file system..."
echo "App directory contents:"
ls -la /app/ | head -10

echo "Model files:"
find /app -name "*.onnx" -o -name "*.xml" -o -name "*.pt" 2>/dev/null | head -5 || echo "No model files found"

# Environment variables
echo "🔍 Key environment variables:"
echo "PORT: ${PORT:-not set}"
echo "MODEL_DIR: ${MODEL_DIR:-not set}" 
echo "RAILWAY_ENVIRONMENT_NAME: ${RAILWAY_ENVIRONMENT_NAME:-not set}"

# Start with timeout to prevent hanging
echo "🚀 Starting main application..."
timeout 60s python -c "
import uvicorn
print('Starting uvicorn...')
uvicorn.run(
    'app.main:app', 
    host='0.0.0.0', 
    port=8080,
    log_level='info',
    access_log=True
)
" || {
    echo "❌ Main app startup timed out or failed"
    echo "🚨 Falling back to minimal mode..."
    exec python -m uvicorn app.main_minimal:app --host 0.0.0.0 --port 8080 --log-level info
}