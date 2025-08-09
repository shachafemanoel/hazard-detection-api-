#!/bin/bash

# Don't exit on errors - we want to try multiple fallback modes
set +e

echo "🚀 RAILWAY STARTUP DEBUG - $(date)"
echo "🔍 Container info:"
echo "  Python: $(python --version 2>&1)"
echo "  Working dir: $(pwd)"
echo "  User: $(whoami)"
echo "  Port from env: ${PORT:-NOT_SET}"
echo "  Home: ${HOME:-NOT_SET}"

# Check if we can even run Python
if ! python --version >/dev/null 2>&1; then
    echo "❌ Python not available, starting emergency server"
    exec python3 /app/emergency.py
fi

# Check basic Python functionality
echo "🔍 Testing Python basics..."
if ! python -c "print('Python OK')" 2>/dev/null; then
    echo "❌ Python execution failed, starting emergency server"
    exec python3 /app/emergency.py
fi

# Test core imports with detailed error reporting
echo "🔍 Testing imports..."
python -c "
import sys
import os
print(f'Python executable: {sys.executable}')
print(f'Python version: {sys.version}')
print(f'Current directory: {os.getcwd()}')
print(f'App directory exists: {os.path.exists(\"/app\")}')

# Test FastAPI import
try:
    import fastapi
    print('✅ FastAPI available')
    FASTAPI_OK = True
except Exception as e:
    print(f'❌ FastAPI failed: {e}')
    FASTAPI_OK = False

# Test our app import
try:
    from app.main_minimal import app
    print('✅ Minimal app import OK')
    MINIMAL_OK = True
except Exception as e:
    print(f'❌ Minimal app import failed: {e}')
    MINIMAL_OK = False

# Exit codes: 0=all good, 1=use minimal, 2=use emergency
if not FASTAPI_OK or not MINIMAL_OK:
    sys.exit(2)  # Emergency mode
sys.exit(0)  # Try main app
" 2>&1

IMPORT_RESULT=$?
echo "Import test result: $IMPORT_RESULT"

if [ $IMPORT_RESULT -eq 2 ]; then
    echo "🚨 Core imports failed - EMERGENCY MODE"
    echo "Starting emergency server with zero dependencies..."
    exec python3 /app/emergency.py
fi

# Try minimal app first (most likely to work)
echo "🔍 Attempting minimal app startup..."
timeout 30s python -m uvicorn app.main_minimal:app \
    --host 0.0.0.0 \
    --port 8080 \
    --log-level info \
    --access-log 2>&1 &

MINIMAL_PID=$!
sleep 5

# Check if minimal app is running
if kill -0 $MINIMAL_PID 2>/dev/null; then
    echo "✅ Minimal app started successfully (PID: $MINIMAL_PID)"
    wait $MINIMAL_PID
else
    echo "❌ Minimal app failed, trying emergency mode"
    kill $MINIMAL_PID 2>/dev/null
    exec python3 /app/emergency.py
fi