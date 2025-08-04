# Railway Deployment Guide

This guide covers deploying the Hazard Detection API to Railway and integrating it with other services.

## 🚀 Quick Railway Deployment

### 1. Prerequisites
- Railway account ([railway.app](https://railway.app))
- GitHub repository with your code
- Model files (`best.pt` and/or `best_openvino_model/`)

### 2. Deploy to Railway

#### Option A: Deploy from GitHub (Recommended)
1. **Connect Repository**
   ```bash
   # From Railway dashboard
   1. Click "New Project"
   2. Select "Deploy from GitHub repo"
   3. Choose your hazard-detection-api repository
   4. Click "Deploy Now"
   ```

#### Option B: Deploy via Railway CLI
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Deploy
railway up
```

Once the service is deployed, open the service **Settings** in Railway and disable **Public Networking** while keeping **Private Networking** enabled. This ensures the OpenVINO API is reachable only within Railway's internal network.

### 3. Configure Environment Variables

Set these in Railway Dashboard → Your Project → Variables:

#### Required Variables
```bash
RAILWAY_ENVIRONMENT_NAME=production
MODEL_DIR=/app
MODEL_BACKEND=auto
PYTHONPATH=/app
```
Railway automatically provides the `PORT` environment variable, so you don't need to set it manually.

#### Optional API Service Variables
```bash
# Google Maps API (for geocoding)
GOOGLE_MAPS_API_KEY=your_google_maps_api_key

# Redis (for caching)
REDIS_URL=redis://redis:6379

# Cloudinary (for image storage)
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_key
CLOUDINARY_API_SECRET=your_cloudinary_secret

# Custom domain (if using)
FRONTEND_URL=https://your-frontend-domain.com

# Railway API (for service discovery)
RAILWAY_TOKEN=your_railway_token
```

### 4. Verify Deployment

Once deployed, your API will be available at:
```
https://your-project-name.up.railway.app
```

Test the deployment:
```bash
curl https://your-project-name.up.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_status": "loaded_openvino",
  "backend_inference": true
}
```

## 🔧 Railway Configuration Details

### railway.toml Explanation

```toml
[build]
builder = "nixpacks"  # Uses Railway's Nixpacks for automatic builds

[deploy]
startCommand = "uvicorn app:app --host 0.0.0.0 --port $PORT"  # Starts the FastAPI server
healthcheckPath = "/health"          # Railway monitors this endpoint
healthcheckTimeout = 300             # 5 minutes timeout for health checks
restartPolicyType = "ON_FAILURE"     # Restart only on failures
restartPolicyMaxRetries = 3          # Maximum restart attempts

[environments.production.variables]
MODEL_DIR = "/app"                   # Model files location
MODEL_BACKEND = "auto"               # Auto-select best backend
PYTHONPATH = "/app"                  # Python path configuration
RAILWAY_ENVIRONMENT_NAME = "production"  # Environment identifier
```

### Build Process

Railway automatically:
1. Detects Python project
2. Installs dependencies from `requirements.txt`
3. Sets up the environment
4. Starts the application using the start command

## 🌐 Integration Guide: Using Railway API from Other Services

### Service-to-Service Communication

#### 1. Environment-based Configuration

In your client service, set the Railway API URL:

```bash
# In your client service environment variables
HAZARD_API_URL=https://your-hazard-detection-api.up.railway.app
```

#### 2. Python Client Integration

```python
import os
import requests
import asyncio
import aiohttp
from typing import List, Dict, Optional

class HazardDetectionClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv('HAZARD_API_URL', 'http://localhost:8000')
        self.session_id = None
    
    async def start_session(self) -> str:
        """Start a new detection session"""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/session/start") as response:
                if response.status == 200:
                    data = await response.json()
                    self.session_id = data['session_id']
                    return self.session_id
                else:
                    raise Exception(f"Failed to start session: {response.status}")
    
    async def detect_hazards(self, image_data: bytes, filename: str = "image.jpg") -> Dict:
        """Detect hazards in an image"""
        if not self.session_id:
            await self.start_session()
        
        form_data = aiohttp.FormData()
        form_data.add_field('file', image_data, filename=filename, content_type='image/jpeg')
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/detect/{self.session_id}", 
                data=form_data
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Detection failed: {response.status}")
    
    async def end_session(self) -> Dict:
        """End the current session"""
        if not self.session_id:
            return {"message": "No active session"}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/session/{self.session_id}/end") as response:
                data = await response.json()
                self.session_id = None
                return data

# Usage example
async def process_road_images(image_paths: List[str]):
    client = HazardDetectionClient()
    
    try:
        await client.start_session()
        
        for image_path in image_paths:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            result = await client.detect_hazards(image_data, image_path)
            
            print(f"Image: {image_path}")
            print(f"Hazards found: {len(result['detections'])}")
            
            for detection in result['detections']:
                if detection['is_new']:
                    print(f"  NEW: {detection['class_name']} ({detection['confidence']:.2f})")
        
        summary = await client.end_session()
        print(f"Session summary: {summary}")
        
    except Exception as e:
        print(f"Error: {e}")

# Run the example
# asyncio.run(process_road_images(['road1.jpg', 'road2.jpg']))
```

#### 3. Node.js Integration

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class HazardDetectionClient {
    constructor(baseUrl = process.env.HAZARD_API_URL || 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.sessionId = null;
    }
    
    async startSession() {
        try {
            const response = await axios.post(`${this.baseUrl}/session/start`);
            this.sessionId = response.data.session_id;
            return this.sessionId;
        } catch (error) {
            throw new Error(`Failed to start session: ${error.response?.status}`);
        }
    }
    
    async detectHazards(imagePath) {
        if (!this.sessionId) {
            await this.startSession();
        }
        
        const form = new FormData();
        form.append('file', fs.createReadStream(imagePath));
        
        try {
            const response = await axios.post(
                `${this.baseUrl}/detect/${this.sessionId}`,
                form,
                {
                    headers: form.getHeaders()
                }
            );
            return response.data;
        } catch (error) {
            throw new Error(`Detection failed: ${error.response?.status}`);
        }
    }
    
    async endSession() {
        if (!this.sessionId) {
            return { message: "No active session" };
        }
        
        try {
            const response = await axios.post(`${this.baseUrl}/session/${this.sessionId}/end`);
            this.sessionId = null;
            return response.data;
        } catch (error) {
            throw new Error(`Failed to end session: ${error.response?.status}`);
        }
    }
}

// Usage example
async function processRoadImages(imagePaths) {
    const client = new HazardDetectionClient();
    
    try {
        await client.startSession();
        
        for (const imagePath of imagePaths) {
            const result = await client.detectHazards(imagePath);
            
            console.log(`Image: ${imagePath}`);
            console.log(`Hazards found: ${result.detections.length}`);
            
            result.detections.forEach(detection => {
                if (detection.is_new) {
                    console.log(`  NEW: ${detection.class_name} (${detection.confidence.toFixed(2)})`);
                }
            });
        }
        
        const summary = await client.endSession();
        console.log('Session summary:', summary);
        
    } catch (error) {
        console.error('Error:', error.message);
    }
}

// Run the example
// processRoadImages(['road1.jpg', 'road2.jpg']);
```

#### 4. React/Frontend Integration

```javascript
// React hook for hazard detection
import { useState, useCallback } from 'react';

const useHazardDetection = (apiUrl = process.env.REACT_APP_HAZARD_API_URL) => {
    const [sessionId, setSessionId] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    
    const startSession = useCallback(async () => {
        try {
            setLoading(true);
            const response = await fetch(`${apiUrl}/session/start`, {
                method: 'POST'
            });
            const data = await response.json();
            setSessionId(data.session_id);
            return data.session_id;
        } catch (err) {
            setError(err.message);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [apiUrl]);
    
    const detectHazards = useCallback(async (imageFile) => {
        try {
            setLoading(true);
            setError(null);
            
            if (!sessionId) {
                await startSession();
            }
            
            const formData = new FormData();
            formData.append('file', imageFile);
            
            const response = await fetch(`${apiUrl}/detect/${sessionId}`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            return result;
        } catch (err) {
            setError(err.message);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [apiUrl, sessionId, startSession]);
    
    const endSession = useCallback(async () => {
        if (!sessionId) return;
        
        try {
            const response = await fetch(`${apiUrl}/session/${sessionId}/end`, {
                method: 'POST'
            });
            const data = await response.json();
            setSessionId(null);
            return data;
        } catch (err) {
            setError(err.message);
            throw err;
        }
    }, [apiUrl, sessionId]);
    
    return {
        sessionId,
        loading,
        error,
        startSession,
        detectHazards,
        endSession
    };
};

// Usage in component
function HazardDetectionComponent() {
    const { detectHazards, loading, error } = useHazardDetection();
    const [results, setResults] = useState(null);
    
    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;
        
        try {
            const result = await detectHazards(file);
            setResults(result);
        } catch (err) {
            console.error('Detection failed:', err);
        }
    };
    
    return (
        <div>
            <input type="file" accept="image/*" onChange={handleFileUpload} />
            {loading && <p>Detecting hazards...</p>}
            {error && <p>Error: {error}</p>}
            {results && (
                <div>
                    <h3>Detection Results:</h3>
                    <p>Found {results.detections.length} hazards</p>
                    {results.detections.map((detection, index) => (
                        <div key={index}>
                            {detection.class_name}: {(detection.confidence * 100).toFixed(1)}%
                            {detection.is_new && <span> (NEW!)</span>}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
```

## 🔗 Service Discovery & Communication

### Using Railway's Internal Network

If both services are on Railway, you can use internal networking:

```python
# Use Railway's internal URL for faster communication
INTERNAL_API_URL = f"http://{os.getenv('RAILWAY_SERVICE_NAME', 'hazard-detection-api')}:8000"
```

### Load Balancing & Scaling

```python
import random
from typing import List

class LoadBalancedHazardClient:
    def __init__(self, api_urls: List[str]):
        self.api_urls = api_urls
        self.clients = [HazardDetectionClient(url) for url in api_urls]
    
    def get_client(self):
        """Get a random client for load balancing"""
        return random.choice(self.clients)
    
    async def detect_with_fallback(self, image_data: bytes):
        """Try detection with fallback to other instances"""
        for client in self.clients:
            try:
                return await client.detect_hazards(image_data)
            except Exception as e:
                print(f"Client {client.base_url} failed: {e}")
                continue
        raise Exception("All API instances failed")
```

## 🛠️ Monitoring & Debugging

### Health Check Integration

```python
async def check_api_health():
    """Check if the hazard detection API is healthy"""
    try:
        response = await aiohttp.ClientSession().get(f"{API_URL}/health")
        if response.status == 200:
            data = await response.json()
            return {
                'healthy': data.get('status') == 'healthy',
                'model_loaded': data.get('backend_inference', False),
                'backend': data.get('backend_type', 'unknown')
            }
    except Exception as e:
        return {'healthy': False, 'error': str(e)}
```

### Error Handling Best Practices

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustHazardClient(HazardDetectionClient):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def detect_hazards_with_retry(self, image_data: bytes):
        """Detect hazards with automatic retry"""
        return await self.detect_hazards(image_data)
```

## 📊 Performance Optimization

### Batch Processing

```python
async def process_images_batch(image_paths: List[str], batch_size: int = 5):
    """Process images in batches for better throughput"""
    client = HazardDetectionClient()
    
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        
        # Process batch concurrently
        tasks = []
        for image_path in batch:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            tasks.append(client.detect_hazards(image_data))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for j, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Failed to process {batch[j]}: {result}")
            else:
                print(f"Processed {batch[j]}: {len(result['detections'])} hazards")
```

This comprehensive guide covers Railway deployment and integration patterns for using your hazard detection API from other services and applications.