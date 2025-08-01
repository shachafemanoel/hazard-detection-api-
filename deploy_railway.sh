#!/bin/bash
# Railway Deployment Script for Hazard Detection API

set -e

echo "🚀 Railway Deployment Setup for Hazard Detection API"
echo "=================================================="

# Check if railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Check if user is logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Please login to Railway..."
    railway login
fi

# Initialize project if needed
if [ ! -f "railway.toml" ]; then
    echo "⚠️  railway.toml not found. This script assumes it already exists."
    exit 1
fi

echo "📁 Current project structure:"
ls -la

echo ""
echo "🔧 Validating required files..."

# Check for required files
REQUIRED_FILES=("app.py" "requirements.txt" "railway.toml")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    else
        echo "✅ $file"
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo "❌ Missing required files:"
    printf '   %s\n' "${MISSING_FILES[@]}"
    exit 1
fi

# Check for model files
echo ""
echo "🧠 Checking model files..."
if [ -f "best.pt" ]; then
    echo "✅ PyTorch model (best.pt) found"
    MODEL_SIZE=$(du -h "best.pt" | cut -f1)
    echo "   Size: $MODEL_SIZE"
fi

if [ -d "best_openvino_model" ] && [ -f "best_openvino_model/best.xml" ]; then
    echo "✅ OpenVINO model found"
    if [ -f "best_openvino_model/best.bin" ]; then
        echo "✅ OpenVINO weights found"
        OPENVINO_SIZE=$(du -sh "best_openvino_model" | cut -f1)
        echo "   Size: $OPENVINO_SIZE"
    else
        echo "⚠️  OpenVINO weights (best.bin) missing"
    fi
else
    echo "⚠️  OpenVINO model not found"
fi

if [ ! -f "best.pt" ] && [ ! -f "best_openvino_model/best.xml" ]; then
    echo "❌ No model files found! Please add either:"
    echo "   - best.pt (PyTorch model)"
    echo "   - best_openvino_model/best.xml + best.bin (OpenVINO model)"
    echo ""
    echo "Continuing deployment anyway (API will run but detection will fail)"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "🌐 Deploying to Railway..."

# Deploy to Railway
railway up

echo ""
echo "⏳ Waiting for deployment to complete..."
sleep 5

# Get the deployed URL
RAILWAY_URL=$(railway domain 2>/dev/null || echo "")

if [ -n "$RAILWAY_URL" ]; then
    echo "✅ Deployment complete!"
    echo "🌐 Your API is available at: https://$RAILWAY_URL"
    echo ""
    echo "🧪 Testing deployment..."
    
    # Test the health endpoint
    if curl -s -f "https://$RAILWAY_URL/health" > /dev/null; then
        echo "✅ Health check passed"
        echo ""
        echo "📋 API Endpoints:"
        echo "   Health: https://$RAILWAY_URL/health"
        echo "   Root: https://$RAILWAY_URL/"
        echo "   Docs: https://$RAILWAY_URL/docs"
        echo "   Start Session: https://$RAILWAY_URL/session/start"
        echo "   Detect: https://$RAILWAY_URL/detect/{session_id}"
    else
        echo "⚠️  Health check failed - deployment may still be starting"
        echo "   Check status at: https://$RAILWAY_URL/health"
    fi
    
    echo ""
    echo "🔧 Environment Variables:"
    echo "   Set these in Railway dashboard if needed:"
    echo "   - GOOGLE_MAPS_API_KEY (for geocoding)"
    echo "   - REDIS_URL (for caching)"
    echo "   - CLOUDINARY_* (for image storage)"
    
    echo ""
    echo "📝 Integration Example:"
    echo "   export HAZARD_API_URL=https://$RAILWAY_URL"
    echo "   python integration_example.py"
    
else
    echo "⚠️  Could not determine Railway URL"
    echo "   Check your deployment status with: railway status"
fi

echo ""
echo "🎉 Deployment script completed!"
echo ""
echo "📚 Next steps:"
echo "1. Test your API: curl https://your-domain.up.railway.app/health"
echo "2. Update environment variables in Railway dashboard"
echo "3. Use integration_example.py to test from other services"
echo "4. Check the logs: railway logs"