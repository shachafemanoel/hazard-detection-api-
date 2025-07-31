#!/bin/bash
# Build script for Hazard Detection API service

set -e

echo "🐳 Building Hazard Detection API Docker image..."

# Navigate to API directory
cd "$(dirname "$0")"

# Build the Docker image
docker build -t hazard-detection-api:latest .

echo "✅ API Docker image built successfully!"
echo "🚀 To run locally:"
echo "   docker run -p 8000:8000 hazard-detection-api:latest"
echo ""
echo "🌐 To deploy to Railway:"
echo "   1. Push this folder to a separate Git repository"
echo "   2. Connect Railway to the repository"
echo "   3. Railway will automatically deploy using railway.toml"