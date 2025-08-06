# 🛡️ Hazard Detection API

[![Railway Deploy](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/your-username/hazard-detection-api)

A production-ready **FastAPI service** for real-time road hazard detection using advanced computer vision models. Built with enterprise-grade architecture, comprehensive testing, and seamless deployment capabilities.

## 🏗️ Architecture Overview

This API features a **modular, service-oriented architecture** with clear separation of concerns:

```
app/
├── main.py              # 🚀 Application entry point & configuration
├── api/                 # 🌐 API route definitions
│   ├── health.py        # Health checks & status endpoints
│   ├── sessions.py      # Session management
│   ├── detection.py     # Core detection endpoints
│   └── external_apis.py # External service integrations
├── core/                # ⚙️ Core application components
│   ├── config.py        # Configuration management
│   ├── exceptions.py    # Custom exception handling
│   └── logging_config.py# Centralized logging
├── services/            # 🔧 Business logic services
│   ├── model_service.py # AI model management
│   ├── session_service.py# Session & tracking logic
│   └── performance_monitor.py # Performance analytics
├── models/              # 📝 Data models & validation
│   └── api_models.py    # Pydantic models for OpenAPI
└── tests/               # 🧪 Comprehensive test suite
    ├── conftest.py      # Test configuration & fixtures
    ├── test_*.py        # Unit & integration tests
    └── ...
```

## ✨ Features

### 🚀 **Performance & Scalability**
- **OpenVINO 2024** optimized inference with intelligent fallback to PyTorch
- **Asynchronous processing** with FastAPI's async capabilities
- **Intelligent model loading** with automatic device selection
- **Performance monitoring** with real-time metrics and alerts

### 🎯 **Computer Vision Excellence**
- **10 Road Hazard Classes**: Alligator Crack, Block Crack, Crosswalk Blur, Lane Blur, Longitudinal Crack, Manhole, Patch Repair, Pothole, Transverse Crack, Wheel Mark Crack
- **Advanced preprocessing** with letterbox padding and normalization
- **Non-Maximum Suppression (NMS)** for optimal detection quality
- **Confidence thresholding** and duplicate detection prevention

### 📊 **Enterprise Features**
- **Session management** with stateful tracking and reporting
- **Real-time performance monitoring** with system health checks
- **Comprehensive error handling** with structured responses
- **OpenAPI documentation** with interactive Swagger UI
- **Type-safe** request/response validation with Pydantic

### 🔧 **Developer Experience**
- **Modular architecture** for easy maintenance and extensibility
- **90%+ test coverage** with pytest and comprehensive mocking
- **CI/CD pipeline** with GitHub Actions
- **Docker containerization** for consistent deployments
- **Environment-based configuration** with `.env` support

## 🚦 API Endpoints

### **Health & Status**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Simple health check |
| `/status` | GET | Detailed service status |
| `/metrics` | GET | Performance metrics & analytics |

### **Session Management**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/session/start` | POST | Create new detection session |
| `/session/{id}/end` | POST | End active session |
| `/session/{id}/summary` | GET | Get session statistics |
| `/session/{id}/report/{report_id}/confirm` | POST | Confirm detection report |
| `/session/{id}/report/{report_id}/dismiss` | POST | Dismiss detection report |

### **Detection Services**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect/{session_id}` | POST | Session-based detection with tracking |
| `/detect` | POST | Legacy detection (backward compatibility) |
| `/detect-batch` | POST | Batch processing for multiple images |

### **External Integrations**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | External services health check |
| `/api/geocode` | POST | Address to coordinates conversion |
| `/api/reverse-geocode` | POST | Coordinates to address conversion |

## 🛠️ Installation & Setup

### **Prerequisites**
- Python 3.9+ (recommended: 3.11)
- 4GB+ RAM for OpenVINO models
- Modern CPU with AVX support (recommended)

### **Quick Start**

1. **Clone & Install**
```bash
git clone <repository-url>
cd hazard-detection-api
python -m pip install -r requirements.txt
```

2. **Configure Environment**
```bash
# Create .env file (optional)
echo "MODEL_BACKEND=auto" > .env
echo "OPENVINO_DEVICE=AUTO" >> .env
echo "LOG_LEVEL=INFO" >> .env
```

3. **Start Development Server**
```bash
python main.py
```

4. **Access Documentation**
- Production API: https://hazard-api-production-production.up.railway.app
- Interactive Docs: https://hazard-api-production-production.up.railway.app/docs
- Performance Metrics: https://hazard-api-production-production.up.railway.app/metrics
- Local API: http://localhost:8080
- Local Docs: http://localhost:8080/docs
- Local Metrics: http://localhost:8080/metrics

## ⚙️ Configuration

### **Environment Variables**
| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8080` | Server port |
| `MODEL_BACKEND` | `auto` | Model backend (`auto`/`openvino`/`pytorch`) |
| `MODEL_DIR` | `/app` | Model files directory |
| `OPENVINO_DEVICE` | `AUTO` | OpenVINO device selection |
| `CONFIDENCE_THRESHOLD` | `0.5` | Detection confidence threshold |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DEBUG` | `false` | Enable debug mode |

### **Model Configuration**
The service automatically locates models in the following priority order:
1. `best0408_openvino_model/best0408.xml` (OpenVINO)
2. `best.pt` (PyTorch)
3. Fallback locations in `models/` subdirectories

## 🧪 Testing

### **Run Test Suite**
```bash
# Run all tests with coverage
pytest app/tests/ -v --cov=app

# Run specific test categories
pytest app/tests/test_detection_api.py -v
pytest app/tests/test_model_service.py -v
pytest app/tests/test_performance_monitor.py -v
```

### **Test Coverage**
- **Unit Tests**: Model service, performance monitoring, session management
- **Integration Tests**: API endpoints with mocked dependencies
- **Mock Testing**: Comprehensive mocking for CI/CD environments
- **Fixture Management**: Reusable test data and configurations

## 🐳 Docker Deployment

### **Build & Run**
```bash
# Build image
docker build -t hazard-detection-api .

# Run container
docker run -p 8080:8080 \
  -e MODEL_BACKEND=auto \
  -e OPENVINO_DEVICE=CPU \
  hazard-detection-api
```

### **Docker Compose** (Optional)
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MODEL_BACKEND=auto
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
```

## ☁️ Cloud Deployment

### **Railway** (Recommended)
1. [![Deploy on Railway](https://railway.app/button.svg)](https://railway.app)
2. Connect your GitHub repository
3. Railway auto-detects configuration from `railway.toml`
4. Set environment variables in Railway dashboard
5. Deploy automatically triggers on git push

### **Other Platforms**
- **Heroku**: Use included `Procfile`
- **DigitalOcean**: Deploy with Docker container
- **AWS ECS**: Use Docker image with ECS task definition
- **Google Cloud Run**: Deploy containerized service

## 📊 Performance Monitoring

### **Built-in Metrics**
- Request/response times and success rates
- Model inference performance by backend
- System resource utilization (CPU, memory, disk)
- Error rates and alert conditions
- Session statistics and detection tracking

### **Performance Endpoints**
```bash
# Get performance summary
curl http://localhost:8080/metrics

# Check detailed service status  
curl http://localhost:8080/status

# Health check for monitoring
curl http://localhost:8080/health
```

## 🔍 Example Usage

### **Python Client Example**
```python
import requests

# Start session
response = requests.post("http://localhost:8080/session/start")
session_id = response.json()["session_id"]

# Detect hazards
with open("road_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        f"http://localhost:8080/detect/{session_id}", 
        files=files
    )

detections = response.json()["detections"]
for detection in detections:
    print(f"Found {detection['class_name']} with {detection['confidence']:.2f} confidence")
```

### **cURL Example**
```bash
# Start session
SESSION_ID=$(curl -X POST http://localhost:8080/session/start | jq -r '.session_id')

# Upload image for detection
curl -X POST http://localhost:8080/detect/$SESSION_ID \
  -F "file=@road_damage.jpg" \
  | jq '.detections[] | {class_name, confidence, bbox}'

# Get session summary
curl http://localhost:8080/session/$SESSION_ID/summary | jq
```

See [CLIENT_README.md](CLIENT_README.md) for a comprehensive client integration guide, including base64 uploads and JavaScript examples.

## 🚀 Development

### **Code Quality**
```bash
# Format code
black app/ main.py --line-length 88

# Lint code  
flake8 app/ main.py

# Type checking
mypy app/ main.py
```

### **Adding New Features**
1. Create feature branch: `git checkout -b feature/new-endpoint`
2. Add business logic in `app/services/`
3. Add API routes in `app/api/`
4. Add comprehensive tests in `app/tests/`
5. Update documentation and commit

### **Project Structure Guidelines**
- **Services**: Business logic and data processing
- **API Routes**: HTTP endpoint definitions only
- **Core**: Shared utilities, configuration, exceptions
- **Models**: Pydantic models for validation
- **Tests**: Mirror the app structure for easy maintenance

## 📈 Production Considerations

### **Security**
- Input validation with Pydantic models
- Structured error handling without information leakage
- CORS configuration for frontend integration
- Environment-based secrets management

### **Scalability**
- Async request handling with FastAPI
- Model caching and performance optimization
- Session cleanup and memory management
- Horizontal scaling support with stateless design

### **Monitoring**
- Structured logging with configurable levels
- Performance metrics collection
- Health check endpoints for load balancers
- Error tracking and alerting capabilities

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the interactive API docs at `/docs`
- **Issues**: Open GitHub issues for bugs or feature requests  
- **Performance**: Monitor metrics at `/metrics` endpoint
- **Health**: Use `/health` and `/status` for diagnostics

---

**Built with ❤️ using FastAPI, OpenVINO, and modern Python practices**