# Node.js Integration Guide for OpenVINO Hazard Detection API

## 🚀 Quick Start

This guide shows how external Node.js services can integrate with your OpenVINO-powered hazard detection API running at `https://hazard-api-production-production.up.railway.app`.

## 📋 Table of Contents

1. [API Overview](#api-overview)
2. [Setup](#setup)
3. [Basic Usage](#basic-usage)
4. [Session Management](#session-management)
5. [Detection Workflows](#detection-workflows)
6. [Error Handling](#error-handling)
7. [Performance Optimization](#performance-optimization)
8. [Security Considerations](#security-considerations)

## 🔍 API Overview

### Base URL
```
https://hazard-api-production-production.up.railway.app
```

### Key Features
- **OpenVINO 2024 Optimized**: Latest performance improvements
- **Session-based Detection**: Track hazards across multiple images
- **Real-time Processing**: Fast inference with async pipeline
- **Auto Device Selection**: Intelligent CPU/GPU utilization
- **Batch Processing**: Handle multiple images efficiently

### Core Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` | Service health and model status |
| `POST` | `/session/start` | Start detection session |
| `POST` | `/detect/{session_id}` | Detect hazards in image |
| `POST` | `/detect` | Legacy single detection |
| `POST` | `/detect-batch` | Batch detection |
| `GET` | `/session/{session_id}/summary` | Get session results |

## ⚙️ Setup

### 1. Install Dependencies

```bash
npm install axios form-data fs path
```

### 2. Basic Configuration

```javascript
// config.js
const API_CONFIG = {
  baseURL: 'https://hazard-api-production-production.up.railway.app',
  timeout: 30000, // 30 seconds for image processing
  headers: {
    'Accept': 'application/json',
    'User-Agent': 'NodeJS-Client/1.0'
  }
};

module.exports = API_CONFIG;
```

### 3. API Client Setup

```javascript
// hazardDetectionClient.js
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const API_CONFIG = require('./config');

class HazardDetectionClient {
  constructor(config = API_CONFIG) {
    this.client = axios.create(config);
    this.setupInterceptors();
  }

  setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(`🚀 ${config.method.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        console.log(`✅ ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error(`❌ ${error.response?.status || 'Network Error'} ${error.config?.url}`);
        return Promise.reject(error);
      }
    );
  }

  // Health check
  async checkHealth() {
    try {
      const response = await this.client.get('/health');
      return response.data;
    } catch (error) {
      throw new Error(`Health check failed: ${error.message}`);
    }
  }

  // Start session
  async startSession() {
    try {
      const response = await this.client.post('/session/start');
      return response.data.session_id;
    } catch (error) {
      throw new Error(`Failed to start session: ${error.message}`);
    }
  }

  // Detect hazards with session
  async detectHazards(sessionId, imagePath) {
    try {
      const formData = new FormData();
      formData.append('file', fs.createReadStream(imagePath));

      const response = await this.client.post(`/detect/${sessionId}`, formData, {
        headers: {
          ...formData.getHeaders(),
          'Content-Type': 'multipart/form-data'
        }
      });

      return response.data;
    } catch (error) {
      throw new Error(`Detection failed: ${error.message}`);
    }
  }

  // Legacy single detection
  async detectSingle(imagePath) {
    try {
      const formData = new FormData();
      formData.append('file', fs.createReadStream(imagePath));

      const response = await this.client.post('/detect', formData, {
        headers: {
          ...formData.getHeaders(),
          'Content-Type': 'multipart/form-data'
        }
      });

      return response.data;
    } catch (error) {
      throw new Error(`Single detection failed: ${error.message}`);
    }
  }

  // Batch detection
  async detectBatch(imagePaths) {
    try {
      const formData = new FormData();
      
      imagePaths.forEach((imagePath, index) => {
        formData.append('files', fs.createReadStream(imagePath));
      });

      const response = await this.client.post('/detect-batch', formData, {
        headers: {
          ...formData.getHeaders(),
          'Content-Type': 'multipart/form-data'
        }
      });

      return response.data;
    } catch (error) {
      throw new Error(`Batch detection failed: ${error.message}`);
    }
  }

  // Get session summary
  async getSessionSummary(sessionId) {
    try {
      const response = await this.client.get(`/session/${sessionId}/summary`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to get session summary: ${error.message}`);
    }
  }

  // End session
  async endSession(sessionId) {
    try {
      const response = await this.client.post(`/session/${sessionId}/end`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to end session: ${error.message}`);
    }
  }

  // Confirm report
  async confirmReport(sessionId, reportId) {
    try {
      const response = await this.client.post(`/session/${sessionId}/report/${reportId}/confirm`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to confirm report: ${error.message}`);
    }
  }

  // Dismiss report
  async dismissReport(sessionId, reportId) {
    try {
      const response = await this.client.post(`/session/${sessionId}/report/${reportId}/dismiss`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to dismiss report: ${error.message}`);
    }
  }
}

module.exports = HazardDetectionClient;
```

## 🚀 Basic Usage

### Quick Start Example

```javascript
// quickStart.js
const HazardDetectionClient = require('./hazardDetectionClient');

async function quickStart() {
  const client = new HazardDetectionClient();
  
  try {
    // 1. Check service health
    console.log('🔍 Checking service health...');
    const health = await client.checkHealth();
    console.log('Service Status:', health.status);
    console.log('Model Backend:', health.model_info?.backend);
    console.log('Performance Mode:', health.device_info?.performance_mode);

    // 2. Single image detection (simple)
    console.log('\n📸 Single image detection...');
    const singleResult = await client.detectSingle('./test-images/road-damage.jpg');
    console.log('Detections found:', singleResult.detections.length);
    
    // Display detections
    singleResult.detections.forEach((detection, index) => {
      console.log(`Detection ${index + 1}:`, {
        type: detection.class_name,
        confidence: (detection.confidence * 100).toFixed(1) + '%',
        bbox: detection.bbox
      });
    });

  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

quickStart();
```

## 🎯 Session Management

### Session-based Workflow

```javascript
// sessionExample.js
const HazardDetectionClient = require('./hazardDetectionClient');

async function sessionWorkflow() {
  const client = new HazardDetectionClient();
  
  try {
    // Start session
    console.log('🔄 Starting detection session...');
    const sessionId = await client.startSession();
    console.log('Session ID:', sessionId);

    // Process multiple images
    const imagePaths = [
      './images/pothole1.jpg',
      './images/crack1.jpg',
      './images/patch1.jpg'
    ];

    let allDetections = [];

    for (const imagePath of imagePaths) {
      console.log(`\n📸 Processing: ${imagePath}`);
      const result = await client.detectHazards(sessionId, imagePath);
      
      console.log(`Found ${result.detections.length} detections`);
      console.log(`New reports: ${result.new_reports.length}`);
      
      allDetections.push(...result.detections);

      // Show new reports that need review
      result.new_reports.forEach(report => {
        console.log(`🚨 New Report: ${report.detection.class_name} (${(report.detection.confidence * 100).toFixed(1)}%)`);
      });
    }

    // Get session summary
    console.log('\n📊 Session Summary...');
    const summary = await client.getSessionSummary(sessionId);
    console.log('Total detections:', summary.detection_count);
    console.log('Unique hazards:', summary.unique_hazards);
    console.log('Pending reports:', summary.reports.filter(r => r.status === 'pending').length);

    // Review and confirm reports
    for (const report of summary.reports) {
      if (report.status === 'pending' && report.detection.confidence > 0.8) {
        console.log(`✅ Auto-confirming high-confidence report: ${report.detection.class_name}`);
        await client.confirmReport(sessionId, report.report_id);
      }
    }

    // End session
    console.log('\n🏁 Ending session...');
    const endResult = await client.endSession(sessionId);
    console.log('Session ended successfully');

  } catch (error) {
    console.error('❌ Session error:', error.message);
  }
}

sessionWorkflow();
```

## 📦 Batch Processing

### Efficient Batch Detection

```javascript
// batchExample.js
const HazardDetectionClient = require('./hazardDetectionClient');
const path = require('path');
const fs = require('fs');

async function batchProcessing() {
  const client = new HazardDetectionClient();
  
  try {
    // Get all images from directory
    const imageDir = './batch-images';
    const imageFiles = fs.readdirSync(imageDir)
      .filter(file => /\.(jpg|jpeg|png)$/i.test(file))
      .map(file => path.join(imageDir, file));

    console.log(`📦 Processing ${imageFiles.length} images in batch...`);

    // Process in batches of 5 for optimal performance
    const batchSize = 5;
    const results = [];

    for (let i = 0; i < imageFiles.length; i += batchSize) {
      const batch = imageFiles.slice(i, i + batchSize);
      console.log(`\n🔄 Processing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(imageFiles.length/batchSize)}`);

      const batchResult = await client.detectBatch(batch);
      results.push(...batchResult.results);

      // Show progress
      batchResult.results.forEach((result, index) => {
        if (result.success) {
          console.log(`✅ ${result.filename}: ${result.detections.length} detections`);
        } else {
          console.log(`❌ ${result.filename}: ${result.error}`);
        }
      });
    }

    // Aggregate results
    const totalDetections = results.reduce((sum, result) => 
      sum + (result.detections?.length || 0), 0
    );
    const successfulProcessing = results.filter(r => r.success).length;

    console.log('\n📊 Batch Summary:');
    console.log(`Processed: ${successfulProcessing}/${results.length} images`);
    console.log(`Total detections: ${totalDetections}`);

  } catch (error) {
    console.error('❌ Batch processing error:', error.message);
  }
}

batchProcessing();
```

## 🛡️ Error Handling

### Robust Error Handling

```javascript
// errorHandling.js
const HazardDetectionClient = require('./hazardDetectionClient');

class RobustHazardClient extends HazardDetectionClient {
  constructor(config) {
    super(config);
    this.retryAttempts = 3;
    this.retryDelay = 1000; // 1 second
  }

  async withRetry(operation, maxRetries = this.retryAttempts) {
    let lastError;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;
        
        // Don't retry on client errors (4xx)
        if (error.response?.status >= 400 && error.response?.status < 500) {
          throw error;
        }

        if (attempt === maxRetries) {
          break;
        }

        console.log(`⚠️ Attempt ${attempt} failed, retrying in ${this.retryDelay}ms...`);
        await new Promise(resolve => setTimeout(resolve, this.retryDelay * attempt));
      }
    }
    
    throw lastError;
  }

  async detectHazardsWithRetry(sessionId, imagePath) {
    return this.withRetry(() => this.detectHazards(sessionId, imagePath));
  }

  async detectSingleWithRetry(imagePath) {
    return this.withRetry(() => this.detectSingle(imagePath));
  }

  async safeDetection(imagePath, useSession = true) {
    try {
      // Validate file exists
      if (!require('fs').existsSync(imagePath)) {
        throw new Error(`Image file not found: ${imagePath}`);
      }

      // Check service health first
      const health = await this.checkHealth();
      if (health.status !== 'healthy') {
        throw new Error(`Service not ready: ${health.status}`);
      }

      if (useSession) {
        const sessionId = await this.startSession();
        const result = await this.detectHazardsWithRetry(sessionId, imagePath);
        await this.endSession(sessionId);
        return result;
      } else {
        return await this.detectSingleWithRetry(imagePath);
      }

    } catch (error) {
      console.error('🚨 Detection failed:', {
        error: error.message,
        image: imagePath,
        timestamp: new Date().toISOString()
      });
      
      // Return error in consistent format
      return {
        success: false,
        error: error.message,
        image: imagePath
      };
    }
  }
}

// Usage example
async function robustExample() {
  const client = new RobustHazardClient();
  
  const result = await client.safeDetection('./test-image.jpg');
  
  if (result.success !== false) {
    console.log('✅ Detection successful:', result.detections.length, 'hazards found');
  } else {
    console.log('❌ Detection failed:', result.error);
  }
}

module.exports = RobustHazardClient;
```

## ⚡ Performance Optimization

### Performance Best Practices

```javascript
// performanceOptimized.js
const HazardDetectionClient = require('./hazardDetectionClient');

class PerformanceOptimizedClient extends HazardDetectionClient {
  constructor(config) {
    super(config);
    this.sessionPool = new Map(); // Reuse sessions
    this.maxSessionDuration = 5 * 60 * 1000; // 5 minutes
  }

  async getOrCreateSession() {
    const now = Date.now();
    
    // Clean up expired sessions
    for (const [sessionId, sessionData] of this.sessionPool.entries()) {
      if (now - sessionData.created > this.maxSessionDuration) {
        await this.endSession(sessionId).catch(console.error);
        this.sessionPool.delete(sessionId);
      }
    }

    // Reuse existing session if available
    for (const [sessionId, sessionData] of this.sessionPool.entries()) {
      if (now - sessionData.lastUsed < 30000) { // 30 seconds
        sessionData.lastUsed = now;
        return sessionId;
      }
    }

    // Create new session
    const sessionId = await this.startSession();
    this.sessionPool.set(sessionId, {
      created: now,
      lastUsed: now
    });

    return sessionId;
  }

  async optimizedDetection(imagePath) {
    const sessionId = await this.getOrCreateSession();
    return await this.detectHazards(sessionId, imagePath);
  }

  async cleanup() {
    // Clean up all sessions
    for (const sessionId of this.sessionPool.keys()) {
      await this.endSession(sessionId).catch(console.error);
    }
    this.sessionPool.clear();
  }
}

// Usage with performance monitoring
async function performanceExample() {
  const client = new PerformanceOptimizedClient();
  
  try {
    const startTime = Date.now();
    
    // Process multiple images efficiently
    const images = ['img1.jpg', 'img2.jpg', 'img3.jpg'];
    const results = [];

    for (const image of images) {
      const imageStart = Date.now();
      const result = await client.optimizedDetection(image);
      const imageTime = Date.now() - imageStart;
      
      console.log(`📸 ${image}: ${result.detections.length} detections in ${imageTime}ms`);
      results.push(result);
    }

    const totalTime = Date.now() - startTime;
    console.log(`⚡ Total processing: ${totalTime}ms (avg: ${Math.round(totalTime/images.length)}ms per image)`);

  } finally {
    await client.cleanup();
  }
}

module.exports = PerformanceOptimizedClient;
```

## 🔒 Security Considerations

### Security Best Practices

```javascript
// secureClient.js
const crypto = require('crypto');
const HazardDetectionClient = require('./hazardDetectionClient');

class SecureHazardClient extends HazardDetectionClient {
  constructor(config) {
    super({
      ...config,
      httpsAgent: new (require('https').Agent)({
        rejectUnauthorized: true, // Verify SSL certificates
      })
    });
  }

  // Validate image before upload
  validateImage(imagePath) {
    const fs = require('fs');
    const path = require('path');
    
    // Check file exists
    if (!fs.existsSync(imagePath)) {
      throw new Error('Image file not found');
    }

    // Check file size (max 10MB)
    const stats = fs.statSync(imagePath);
    if (stats.size > 10 * 1024 * 1024) {
      throw new Error('Image file too large (max 10MB)');
    }

    // Check file extension
    const ext = path.extname(imagePath).toLowerCase();
    if (!['.jpg', '.jpeg', '.png'].includes(ext)) {
      throw new Error('Invalid image format. Use JPG or PNG');
    }

    return true;
  }

  // Generate request signature for audit trail
  generateRequestSignature(method, endpoint, timestamp) {
    const data = `${method}:${endpoint}:${timestamp}`;
    return crypto.createHash('sha256').update(data).digest('hex');
  }

  async secureDetection(imagePath, options = {}) {
    try {
      // Validate input
      this.validateImage(imagePath);

      // Add audit trail
      const timestamp = Date.now();
      const signature = this.generateRequestSignature('POST', '/detect', timestamp);
      
      console.log('🔐 Secure detection initiated:', {
        timestamp,
        signature,
        image: require('path').basename(imagePath)
      });

      // Perform detection
      const result = await this.detectSingle(imagePath);

      // Log success
      console.log('✅ Secure detection completed:', {
        detections: result.detections.length,
        processingTime: result.processing_time_ms,
        signature
      });

      return result;

    } catch (error) {
      console.error('🚨 Secure detection failed:', {
        error: error.message,
        image: require('path').basename(imagePath),
        timestamp: Date.now()
      });
      throw error;
    }
  }
}

module.exports = SecureHazardClient;
```

## 📊 Response Formats

### API Response Examples

```javascript
// Response format examples

// Health Check Response
const healthResponse = {
  "status": "healthy",
  "model_status": "loaded_openvino",
  "backend_inference": true,
  "backend_type": "openvino",
  "device_info": {
    "device": "AUTO",
    "performance_mode": "LATENCY",
    "async_inference": true,
    "openvino_version": "2024_optimized"
  }
};

// Detection Response
const detectionResponse = {
  "success": true,
  "detections": [
    {
      "bbox": [100.5, 200.3, 150.8, 250.9],
      "confidence": 0.87,
      "class_id": 7,
      "class_name": "Pothole",
      "center_x": 125.65,
      "center_y": 225.6,
      "width": 50.3,
      "height": 50.6,
      "area": 2545.18,
      "is_new": true,
      "report_id": "uuid-string"
    }
  ],
  "new_reports": [
    {
      "report_id": "uuid-string",
      "session_id": "session-uuid",
      "detection": { /* detection object */ },
      "timestamp": "2024-01-15T10:30:00",
      "status": "pending"
    }
  ],
  "processing_time_ms": 245.67,
  "model_info": {
    "backend": "openvino",
    "classes": ["Alligator Crack", "Block Crack", /* ... */]
  }
};
```

## 🧪 Testing

### Test Suite Example

```javascript
// test.js
const HazardDetectionClient = require('./hazardDetectionClient');

async function runTests() {
  const client = new HazardDetectionClient();
  
  console.log('🧪 Running API Tests...\n');

  try {
    // Test 1: Health Check
    console.log('1️⃣ Testing health check...');
    const health = await client.checkHealth();
    console.log('✅ Health check passed:', health.status);

    // Test 2: Session lifecycle
    console.log('\n2️⃣ Testing session lifecycle...');
    const sessionId = await client.startSession();
    console.log('✅ Session started:', sessionId);
    
    const summary = await client.getSessionSummary(sessionId);
    console.log('✅ Session summary retrieved');
    
    await client.endSession(sessionId);
    console.log('✅ Session ended');

    // Test 3: Error handling
    console.log('\n3️⃣ Testing error handling...');
    try {
      await client.getSessionSummary('invalid-session');
    } catch (error) {
      console.log('✅ Error handling works:', error.message.includes('404'));
    }

    console.log('\n🎉 All tests passed!');

  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
}

runTests();
```

## 📚 Additional Resources

### Package.json Example

```json
{
  "name": "hazard-detection-client",
  "version": "1.0.0",
  "description": "Node.js client for OpenVINO Hazard Detection API",
  "main": "hazardDetectionClient.js",
  "scripts": {
    "test": "node test.js",
    "example": "node quickStart.js"
  },
  "dependencies": {
    "axios": "^1.6.0",
    "form-data": "^4.0.0"
  },
  "keywords": ["openvino", "hazard-detection", "computer-vision", "api-client"]
}
```

### Environment Variables

```bash
# .env
HAZARD_API_BASE_URL=https://hazard-api-production-production.up.railway.app
HAZARD_API_TIMEOUT=30000
LOG_LEVEL=info
```

## 🚀 Getting Started Checklist

- [ ] Install Node.js dependencies
- [ ] Copy the `HazardDetectionClient` class
- [ ] Test health endpoint connectivity
- [ ] Try single image detection
- [ ] Implement session-based workflow
- [ ] Add error handling and retries
- [ ] Optimize for your use case
- [ ] Set up monitoring and logging

## 💡 Tips for Production

1. **Connection Pooling**: Reuse HTTP connections
2. **Session Management**: Pool sessions for better performance
3. **Image Optimization**: Resize images before upload if possible
4. **Error Monitoring**: Log all API interactions
5. **Rate Limiting**: Respect API rate limits
6. **Caching**: Cache results when appropriate
7. **Security**: Validate all inputs and use HTTPS
8. **Monitoring**: Track API response times and error rates

## 🆘 Troubleshooting

### Common Issues

1. **Connection Timeout**: Increase timeout for large images
2. **File Size Error**: Ensure images are under 10MB
3. **Invalid Format**: Use JPG or PNG only
4. **503 Service Unavailable**: Model is still loading, wait and retry
5. **Session Not Found**: Check session ID validity

### Support

For issues or questions:
- Check the API health endpoint first
- Review error messages for specific guidance
- Monitor Railway logs for server-side issues
- Implement retry logic for transient failures