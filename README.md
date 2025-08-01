# Hazard Detection API

A high-performance FastAPI-based object detection service for identifying road hazards using computer vision. The API supports both OpenVINO and PyTorch backends for flexible deployment scenarios.

## 🚀 Features

- **Real-time Object Detection**: Detect road hazards including cracks, potholes, knocked surfaces, and surface damage
- **Dual Backend Support**: OpenVINO for optimized CPU inference and PyTorch/YOLO for flexibility
- **Session Management**: Track detections across multiple images with intelligent duplicate filtering
- **External Service Integration**: Support for geocoding, caching, and cloud storage
- **Production Ready**: CORS configured, deployment-ready for Railway, Render, and other platforms
- **Mobile Friendly**: Optimized for mobile device integration

## 📋 Supported Hazard Types

- `crack` - Road surface cracks
- `knocked` - Knocked or damaged surfaces  
- `pothole` - Potholes and road holes
- `surface_damage` - General surface damage

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hazard-detection-api
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add model files**
   - Place your YOLO model (`best.pt`) in the root directory
   - For OpenVINO, place model files in `best_openvino_model/` directory:
     - `best.xml` (model architecture)
     - `best.bin` (model weights)

5. **Configure environment variables (optional)**
   ```bash
   # External API services (optional)
   export GOOGLE_MAPS_API_KEY="your_google_maps_key"
   export REDIS_URL="redis://localhost:6379"
   export CLOUDINARY_CLOUD_NAME="your_cloud_name"
   export CLOUDINARY_API_KEY="your_api_key"
   export CLOUDINARY_API_SECRET="your_api_secret"
   export RENDER_API_KEY="your_render_key"
   
   # Model configuration
   export MODEL_BACKEND="auto"  # auto, openvino, pytorch
   export MODEL_DIR="/app/models"
   ```

## 🚀 Running the API

### Development
```bash
python app.py
# or
uvicorn app:app --reload --port 8000
```

### Production
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## 📖 API Documentation

### Base URL
- Development: `http://localhost:8000`
- Production: Your deployed URL

### Authentication
No authentication required for basic detection endpoints.

## 🔗 API Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded_openvino",
  "backend_inference": true,
  "backend_type": "openvino",
  "active_sessions": 0,
  "device_info": {
    "device": "CPU",
    "input_shape": [1, 3, 640, 640],
    "output_shape": [1, 25200, 85],
    "backend": "openvino"
  }
}
```

### Root Endpoint
```http
GET /
```

### Session Management

#### Start Session
```http
POST /session/start
```

**Response:**
```json
{
  "session_id": "uuid-string",
  "message": "Detection session started"
}
```

#### Get Session Summary
```http
GET /session/{session_id}/summary
```

#### End Session
```http
POST /session/{session_id}/end
```

### Detection Endpoints

#### Session-based Detection (Recommended)
```http
POST /detect/{session_id}
Content-Type: multipart/form-data

file: [image file]
```

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.85,
      "class_id": 0,
      "class_name": "pothole",
      "center_x": 320.5,
      "center_y": 240.0,
      "width": 150.0,
      "height": 80.0,
      "area": 12000.0,
      "is_new": true,
      "report_id": "uuid-string"
    }
  ],
  "new_reports": [...],
  "session_stats": {
    "total_detections": 1,
    "unique_hazards": 1,
    "pending_reports": 1
  },
  "processing_time_ms": 45.2,
  "model_info": {
    "backend": "openvino",
    "classes": ["crack", "knocked", "pothole", "surface_damage"]
  }
}
```

#### Legacy Detection
```http
POST /detect
Content-Type: multipart/form-data

file: [image file]
```

#### Batch Detection
```http
POST /detect-batch
Content-Type: multipart/form-data

files: [multiple image files]
```

### Report Management

#### Confirm Report
```http
POST /session/{session_id}/report/{report_id}/confirm
```

#### Dismiss Report
```http
POST /session/{session_id}/report/{report_id}/dismiss
```

### External API Endpoints

#### API Health Check
```http
GET /api/health
```

#### Geocoding
```http
POST /api/geocode?address=New York, NY
```

#### Reverse Geocoding
```http
POST /api/reverse-geocode?lat=40.7128&lng=-74.0060
```

## 💻 Usage Examples

### Python Client Example

```python
import requests
import json

# Base URL
API_URL = "http://localhost:8000"

def detect_hazards(image_path):
    # Start a session
    session_response = requests.post(f"{API_URL}/session/start")
    session_id = session_response.json()["session_id"]
    
    # Upload image for detection
    with open(image_path, 'rb') as image_file:
        files = {'file': image_file}
        response = requests.post(f"{API_URL}/detect/{session_id}", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Found {len(result['detections'])} hazards:")
        
        for detection in result['detections']:
            print(f"- {detection['class_name']}: {detection['confidence']:.2f} confidence")
            print(f"  Location: ({detection['center_x']:.1f}, {detection['center_y']:.1f})")
            if detection['is_new']:
                print(f"  New hazard! Report ID: {detection['report_id']}")
    
    # End session
    requests.post(f"{API_URL}/session/{session_id}/end")

# Usage
detect_hazards("road_image.jpg")
```

### JavaScript/Fetch Example

```javascript
async function detectHazards(imageFile) {
    try {
        // Start session
        const sessionResponse = await fetch('http://localhost:8000/session/start', {
            method: 'POST'
        });
        const sessionData = await sessionResponse.json();
        const sessionId = sessionData.session_id;
        
        // Prepare form data
        const formData = new FormData();
        formData.append('file', imageFile);
        
        // Detect hazards
        const detectResponse = await fetch(`http://localhost:8000/detect/${sessionId}`, {
            method: 'POST',
            body: formData
        });
        
        const result = await detectResponse.json();
        
        if (result.success) {
            console.log(`Found ${result.detections.length} hazards:`);
            result.detections.forEach(detection => {
                console.log(`- ${detection.class_name}: ${(detection.confidence * 100).toFixed(1)}%`);
            });
        }
        
        // End session
        await fetch(`http://localhost:8000/session/${sessionId}/end`, {
            method: 'POST'
        });
        
        return result;
    } catch (error) {
        console.error('Detection failed:', error);
    }
}

// Usage with file input
document.getElementById('imageInput').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        await detectHazards(file);
    }
});
```

### cURL Examples

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Start session
curl -X POST "http://localhost:8000/session/start"

# Detect hazards (replace SESSION_ID with actual session ID)
curl -X POST "http://localhost:8000/detect/SESSION_ID" \
  -F "file=@road_image.jpg"

# Legacy detection
curl -X POST "http://localhost:8000/detect" \
  -F "file=@road_image.jpg"

# Batch detection
curl -X POST "http://localhost:8000/detect-batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

## 🧪 Testing

Run the included test suite:

```bash
# Make sure the API is running on localhost:8000
python test_api.py
```

The test suite checks:
- Health endpoints
- Session management
- Detection endpoints
- API connector functionality

## 🚀 Deployment

### Railway (Recommended)

#### Quick Deployment
```bash
# Run the automated deployment script
./deploy_railway.sh
```

#### Manual Deployment
1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. **Deploy from GitHub**
   - Connect repository to Railway dashboard
   - Railway will automatically detect Python project
   - Uses the included `railway.toml` configuration

3. **Set Environment Variables** (in Railway dashboard)
   ```bash
   MODEL_BACKEND=auto
   MODEL_DIR=/app
   RAILWAY_ENVIRONMENT_NAME=production
   ```
   Railway automatically sets the `PORT` value for you.

4. **Your API will be available at:**
   ```
   https://your-project-name.up.railway.app
   ```

#### Integration from Other Services
```python
# Set environment variable in your client service
HAZARD_API_URL=https://your-project-name.up.railway.app

# Use the integration example
python integration_example.py
```

**📋 See [RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md) for complete Railway deployment and integration guide.**

### Render

1. Connect repository to Render
2. Set environment variables
3. Use the build command: `pip install -r requirements.txt`
4. Use the start command: `python app.py`

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "app.py"]
```

## ⚙️ Configuration

### Model Backend Selection

The API automatically selects the best available backend:

1. **Auto mode (default)**: Tries OpenVINO first, falls back to PyTorch
2. **OpenVINO mode**: CPU-optimized inference with Intel OpenVINO
3. **PyTorch mode**: Uses Ultralytics YOLO implementation

Set via environment variable:
```bash
export MODEL_BACKEND="auto"  # auto, openvino, pytorch
```

### External Services

Configure optional external services via environment variables:

```bash
# Google Maps (for geocoding)
GOOGLE_MAPS_API_KEY=your_key

# Redis (for caching)
REDIS_URL=redis://localhost:6379

# Cloudinary (for image storage)
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_key
CLOUDINARY_API_SECRET=your_secret

# Render (for deployment management)
RENDER_API_KEY=your_key
```

## 🔧 Model Training

To train your own model:

1. Prepare dataset in YOLO format
2. Train using Ultralytics:
   ```python
   from ultralytics import YOLO
   
   model = YOLO('yolo11n.pt')
   results = model.train(data='dataset.yaml', epochs=100)
   ```
3. Export to OpenVINO format:
   ```python
   model.export(format='openvino')
   ```

## 📊 Performance

### Typical Performance Metrics

- **OpenVINO CPU**: ~50-100ms per image (640x640)
- **PyTorch CPU**: ~100-200ms per image (640x640)
- **Memory Usage**: ~500MB-1GB depending on backend
- **Accuracy**: 85-95% mAP on road hazard datasets

### Optimization Tips

1. Use OpenVINO for CPU deployment
2. Enable model caching
3. Resize images to 640x640 for best speed/accuracy balance
4. Use batch detection for multiple images

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

[Your License Here]

## 🆘 Support

For issues and questions:

1. Check the health endpoint for system status
2. Review logs for error details
3. Ensure model files are properly placed
4. Verify dependencies are installed correctly

## 🔄 Changelog

### v1.0.0
- Initial release
- OpenVINO and PyTorch backend support
- Session-based detection with duplicate filtering
- External API integrations
- Production deployment ready