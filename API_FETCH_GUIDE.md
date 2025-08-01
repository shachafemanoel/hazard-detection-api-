# Hazard Detection API - Complete Fetch Guide

This comprehensive guide shows you how to interact with the Hazard Detection API using JavaScript `fetch()`, `curl`, and other HTTP clients.

## 🌍 Base URLs

```javascript
// Production (Railway)
const API_BASE_URL = "https://hazard-api-production-production.up.railway.app/";

// Local Development
const API_BASE_URL = "http://localhost:8080";

// Internal Railway Service (if calling from another Railway service)
const INTERNAL_API_URL = `http://${process.env.RAILWAY_SERVICE_NAME || 'hazard-detection-api'}:8080`;
```

## 📋 Available Endpoints

### 1. Health Check
**GET `/health`** - Check API and model status

```javascript
// Fetch Example
const checkHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const data = await response.json();
    console.log('API Status:', data);
    return data;
  } catch (error) {
    console.error('Health check failed:', error);
  }
};
```

```bash
# cURL Example
curl -X GET "https://hazard-api-production-production.up.railway.app//health"
```

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded_openvino",
  "backend_inference": true,
  "backend_type": "openvino",
  "active_sessions": 0,
  "endpoints": {
    "session_start": "/session/start",
    "session_detect": "/detect/{session_id}",
    "legacy_detect": "/detect",
    "batch_detect": "/detect-batch"
  }
}
```

### 2. Session Management

#### Start Session
**POST `/session/start`** - Start a new detection session

```javascript
const startSession = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/session/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    const data = await response.json();
    return data.session_id;
  } catch (error) {
    console.error('Failed to start session:', error);
  }
};
```

```bash
curl -X POST "https://hazard-api-production-production.up.railway.app/session/start" \
  -H "Content-Type: application/json"
```

#### End Session
**POST `/session/{session_id}/end`** - End session and get summary

```javascript
const endSession = async (sessionId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/session/${sessionId}/end`, {
      method: 'POST'
    });
    return await response.json();
  } catch (error) {
    console.error('Failed to end session:', error);
  }
};
```

### 3. Hazard Detection

#### Session-based Detection (Recommended)
**POST `/detect/{session_id}`** - Detect hazards with tracking

```javascript
const detectHazards = async (sessionId, imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);

  try {
    const response = await fetch(`${API_BASE_URL}/detect/${sessionId}`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Detection failed:', error);
    throw error;
  }
};
```

```bash
curl -X POST "https://hazard-api-production-production.up.railway.app/" \
  -F "file=@road_image.jpg"
```

#### Legacy Detection
**POST `/detect`** - Simple detection without session tracking

```javascript
const detectHazardsLegacy = async (imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);

  try {
    const response = await fetch(`${API_BASE_URL}/detect`, {
      method: 'POST',
      body: formData
    });
    return await response.json();
  } catch (error) {
    console.error('Detection failed:', error);
  }
};
```

#### Batch Detection
**POST `/detect-batch`** - Process multiple images at once

```javascript
const detectBatch = async (imageFiles) => {
  const formData = new FormData();
  
  // Add multiple files
  imageFiles.forEach((file, index) => {
    formData.append('files', file);
  });

  try {
    const response = await fetch(`${API_BASE_URL}/detect-batch`, {
      method: 'POST',
      body: formData
    });
    return await response.json();
  } catch (error) {
    console.error('Batch detection failed:', error);
  }
};
```

## 🔧 Complete Usage Examples

### React Hook for Hazard Detection

```javascript
import { useState, useCallback } from 'react';

const useHazardDetection = (apiUrl = process.env.REACT_APP_API_URL) => {
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

      if (!response.ok) {
        throw new Error(`Detection failed: ${response.status}`);
      }

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
              <strong>{detection.class_name}</strong>: {(detection.confidence * 100).toFixed(1)}%
              {detection.is_new && <span style={{color: 'red'}}> (NEW!)</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

### Node.js Client Class

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class HazardDetectionClient {
  constructor(baseUrl = process.env.API_URL || 'http://localhost:8080') {
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
        { headers: form.getHeaders() }
      );
      return response.data;
    } catch (error) {
      throw new Error(`Detection failed: ${error.response?.status}`);
    }
  }

  async detectHazardsFromBuffer(imageBuffer, filename = 'image.jpg') {
    if (!this.sessionId) {
      await this.startSession();
    }

    const form = new FormData();
    form.append('file', imageBuffer, { filename });

    try {
      const response = await axios.post(
        `${this.baseUrl}/detect/${this.sessionId}`,
        form,
        { headers: form.getHeaders() }
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
      const data = response.data;
      this.sessionId = null;
      return data;
    } catch (error) {
      throw new Error(`Failed to end session: ${error.response?.status}`);
    }
  }

  async checkHealth() {
    try {
      const response = await axios.get(`${this.baseUrl}/health`);
      return response.data;
    } catch (error) {
      throw new Error(`Health check failed: ${error.response?.status}`);
    }
  }
}

// Usage example
async function processImages() {
  const client = new HazardDetectionClient();
  
  try {
    // Check if API is healthy
    const health = await client.checkHealth();
    console.log('API Status:', health.status);
    
    // Start session
    await client.startSession();
    console.log('Session started:', client.sessionId);
    
    // Process images
    const imagePaths = ['road1.jpg', 'road2.jpg', 'road3.jpg'];
    
    for (const imagePath of imagePaths) {
      const result = await client.detectHazards(imagePath);
      console.log(`\n${imagePath}:`);
      console.log(`- Found ${result.detections.length} hazards`);
      
      result.detections.forEach(detection => {
        console.log(`  - ${detection.class_name}: ${(detection.confidence * 100).toFixed(1)}%${detection.is_new ? ' (NEW!)' : ''}`);
      });
    }
    
    // End session and get summary
    const summary = await client.endSession();
    console.log('\nSession Summary:', summary);
    
  } catch (error) {
    console.error('Error:', error.message);
  }
}

module.exports = { HazardDetectionClient, processImages };
```

### Python Client

```python
import requests
import asyncio
import aiohttp
from typing import List, Dict, Optional

class HazardDetectionClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or "http://localhost:8080"
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
    
    async def detect_hazards_from_file(self, file_path: str) -> Dict:
        """Detect hazards from a file path"""
        with open(file_path, 'rb') as f:
            image_data = f.read()
        return await self.detect_hazards(image_data, file_path)
    
    async def end_session(self) -> Dict:
        """End the current session"""
        if not self.session_id:
            return {"message": "No active session"}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/session/{self.session_id}/end") as response:
                data = await response.json()
                self.session_id = None
                return data
    
    async def check_health(self) -> Dict:
        """Check API health"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                return await response.json()

# Usage example
async def process_road_images(image_paths: List[str]):
    client = HazardDetectionClient("https://hazard-api-production-production.up.railway.app/")
    
    try:
        # Check health
        health = await client.check_health()
        print(f"API Status: {health['status']}")
        
        # Start session
        await client.start_session()
        print(f"Session started: {client.session_id}")
        
        # Process images
        for image_path in image_paths:
            result = await client.detect_hazards_from_file(image_path)
            
            print(f"\n{image_path}:")
            print(f"- Found {len(result['detections'])} hazards")
            
            for detection in result['detections']:
                new_marker = " (NEW!)" if detection.get('is_new') else ""
                print(f"  - {detection['class_name']}: {detection['confidence']:.2f}{new_marker}")
        
        # End session
        summary = await client.end_session()
        print(f"\nSession summary: {summary}")
        
    except Exception as e:
        print(f"Error: {e}")

# Run example
# asyncio.run(process_road_images(['road1.jpg', 'road2.jpg']))
```

## 🚀 Advanced Features

### Error Handling with Retry Logic

```javascript
const detectWithRetry = async (sessionId, imageFile, maxRetries = 3) => {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const result = await detectHazards(sessionId, imageFile);
      return result;
    } catch (error) {
      console.warn(`Attempt ${attempt} failed:`, error.message);
      
      if (attempt === maxRetries) {
        throw new Error(`Detection failed after ${maxRetries} attempts: ${error.message}`);
      }
      
      // Wait before retry (exponential backoff)
      await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
    }
  }
};
```

### Batch Processing with Progress Tracking

```javascript
const processBatchWithProgress = async (imageFiles, onProgress) => {
  const results = [];
  const total = imageFiles.length;
  
  for (let i = 0; i < imageFiles.length; i++) {
    try {
      const result = await detectHazardsLegacy(imageFiles[i]);
      results.push({ success: true, result, file: imageFiles[i].name });
      
      // Report progress
      onProgress?.({
        completed: i + 1,
        total,
        percentage: Math.round(((i + 1) / total) * 100)
      });
    } catch (error) {
      results.push({ success: false, error: error.message, file: imageFiles[i].name });
    }
  }
  
  return results;
};

// Usage
processBatchWithProgress(imageFiles, (progress) => {
  console.log(`Progress: ${progress.percentage}% (${progress.completed}/${progress.total})`);
});
```

## 🔍 Response Format Examples

### Detection Response
```json
{
  "success": true,
  "detections": [
    {
      "bbox": [123.45, 67.89, 234.56, 178.91],
      "confidence": 0.87,
      "class_id": 0,
      "class_name": "pothole",
      "center_x": 179.005,
      "center_y": 123.4,
      "width": 111.11,
      "height": 111.02,
      "area": 12345.67,
      "is_new": true,
      "report_id": "uuid-string-here"
    }
  ],
  "new_reports": [...],
  "session_stats": {
    "total_detections": 5,
    "unique_hazards": 2,
    "pending_reports": 2
  },
  "processing_time_ms": 1250.45,
  "model_info": {
    "backend": "openvino",
    "classes": ["crack", "knocked", "pothole", "surface_damage"]
  }
}
```

### Error Response
```json
{
  "detail": "Session not found. Start a session first.",
  "status_code": 404
}
```

This comprehensive guide covers all the ways to interact with your Hazard Detection API using modern web technologies and HTTP clients.