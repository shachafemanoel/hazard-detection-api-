# Node.js Examples for OpenVINO Hazard Detection API

This directory contains practical examples showing how to integrate with the OpenVINO-powered hazard detection API running at `https://hazard-api-production-production.up.railway.app`.
For a full walkthrough of client workflows, see the [Client Guide](../CLIENT_GUIDE.md).

## 🚀 Quick Setup

```bash
# Install dependencies
npm install

# Or install manually
npm install axios form-data chokidar
```

## 📁 Examples Overview

### 1. Quick Start (`quickStart.js`)
Simple single-image detection example that logs bounding box coordinates, center, and size.

```bash
npm run quick-start
```

**What it does:**
- Checks API health
- Detects hazards in a single image
- Shows basic response handling

### 2. Session Management (`sessionExample.js`)
Advanced session-based tracking with report management.

```bash
npm run session-example
```

**Features:**
- Session lifecycle management
- Multi-image processing with tracking
- Automatic report review and confirmation
- Detailed statistics

### 3. Batch Processing (`batchProcessing.js`)
Efficient processing of multiple images with retry logic.

```bash
npm run batch-processing
```

**Capabilities:**
- Directory-based batch processing
- Automatic retry on failures
- Performance monitoring
- Detailed reporting

### 4. Real-Time Monitoring (`realTimeMonitoring.js`)
File system watcher for real-time hazard detection.

```bash
npm run real-time-monitor
# Or test with simulation mode
npm run simulate-monitor
```

**Features:**
- Real-time file system monitoring
- Session management with auto-renewal
- Alert system for high-confidence detections
- Statistics and reporting

## 🏃‍♂️ Getting Started

### Step 1: Test Connectivity
```bash
node quickStart.js
```

### Step 2: Prepare Test Images
Create test images in these locations:
- `./test-image.jpg` (for quick start)
- `./images/` directory (for session example)
- `./test-images/` directory (for batch processing)
- `./incoming-images/` directory (for real-time monitoring)

### Step 3: Run Examples
Choose the example that fits your use case and run it!

## 📊 API Endpoints Used

| Endpoint | Example Usage | Purpose |
|----------|---------------|---------|
| `GET /health` | All examples | Check service status |
| `POST /session/start` | Session, Real-time | Start tracking session |
| `POST /detect/{session_id}` | Session, Real-time | Session-based detection |
| `POST /detect` | Quick start | Single image detection |
| `POST /detect-batch` | Batch processing | Multiple image detection |
| `GET /session/{id}/summary` | Session | Get session results |
| `POST /session/{id}/end` | Session, Real-time | End session |

## 🔧 Configuration Options

### API Configuration
```javascript
const API_CONFIG = {
  baseURL: 'https://hazard-api-production-production.up.railway.app',
  timeout: 30000, // 30 seconds
  headers: {
    'Accept': 'application/json',
    'User-Agent': 'NodeJS-Client/1.0'
  }
};
```

### Batch Processing Options
```javascript
const processor = new BatchProcessor({
  batchSize: 5,        // Images per batch
  maxRetries: 3,       // Retry attempts
  retryDelay: 1000     // Delay between retries (ms)
});
```

### Real-Time Monitoring Options
```javascript
const monitor = new RealTimeHazardMonitor({
  watchDirectory: './incoming-images',     // Directory to watch
  outputDirectory: './processed-results',  // Results output
  alertThreshold: 0.8,                    // Confidence threshold for alerts
  sessionDuration: 10 * 60 * 1000         // Session duration (ms)
});
```

## 🛡️ Error Handling

All examples include comprehensive error handling:

- **Network Errors**: Automatic retry with exponential backoff
- **File Errors**: Validation and graceful handling
- **API Errors**: Detailed error messages and status codes
- **Session Errors**: Automatic session recovery

## 📈 Performance Tips

1. **Use Sessions**: For multiple images, use sessions to track relationships
2. **Batch Processing**: Process multiple images together when possible
3. **Image Optimization**: Resize large images before uploading
4. **Connection Pooling**: Reuse HTTP connections for better performance
5. **Async Processing**: Use real-time monitoring for continuous workflows

## 🔍 Response Format

All detection responses include:

```javascript
{
  "success": true,
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.87,
      "class_name": "Pothole",
      "center_x": 125.65,
      "center_y": 225.6,
      "width": 50.3,
      "height": 50.6
    }
  ],
  "processing_time_ms": 245.67,
  "model_info": {
    "backend": "openvino",
    "performance_mode": "LATENCY"
  }
}
```

## 🚨 Alert System

The real-time monitoring example includes an alert system:

```javascript
// High-confidence detections trigger alerts
if (detection.confidence >= alertThreshold) {
  await onHazardDetected(detection, filename, imagePath);
}
```

## 📝 Logging

All examples include structured logging:
- Processing progress
- Performance metrics
- Error details
- Session information

## 🔒 Security Notes

- Always validate file paths and types
- Use HTTPS for all API calls
- Implement rate limiting in production
- Monitor API usage and costs
- Validate image file sizes and formats

## 🆘 Troubleshooting

### Common Issues

1. **"Connection refused"**
   - Check API URL is correct
   - Verify internet connectivity
   - Check if service is running

2. **"File not found"**
   - Ensure test images exist
   - Check file paths are correct
   - Verify file permissions

3. **"Request timeout"**
   - Increase timeout for large images
   - Check image file size (max 10MB)
   - Verify network stability

4. **"Session not found"**
   - Sessions expire after inactivity
   - Start new session if needed
   - Check session ID validity

### Debug Mode

Enable debug logging by setting environment variable:
```bash
DEBUG=true node quickStart.js
```

## 📚 Additional Resources

- [Main Integration Guide](../NODEJS_INTEGRATION_GUIDE.md)
- [API Documentation](../API_FETCH_GUIDE.md)
- [OpenVINO Performance Guide](../OPENVINO_VALIDATION_REPORT.md)

## 🤝 Contributing

Feel free to add your own examples or improvements:

1. Follow the existing code style
2. Include error handling
3. Add documentation
4. Test with the live API

## 📞 Support

For issues or questions:
- Check the API health endpoint first
- Review error messages for guidance
- Monitor Railway logs for server issues
- Implement retry logic for transient failures