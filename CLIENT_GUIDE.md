# Hazard Detection API Client Guide

This guide explains how to integrate with the Hazard Detection API from any client application.
It covers starting sessions, uploading images, reading results, and computing bounding box
metrics derived from the model output.

## Base URL

Configure the API base URL for your environment. You can override it with the
`HAZARD_API_URL` environment variable.

```text
https://hazard-api-production-production.up.railway.app
# or when running locally
http://localhost:8080
```

## 1. Start a Session (optional but recommended)

```bash
curl -X POST "$BASE_URL/session/start" -H "Content-Type: application/json"
# => {"session_id": "<uuid>"}
```

Store the returned `session_id` to track detections across multiple images.

## 2. Send an Image

### a. Multipart upload

```bash
curl -X POST "$BASE_URL/detect/$SESSION_ID" \
  -F "file=@/path/to/road.jpg"
```

### b. Base64 payload

```bash
curl -X POST "$BASE_URL/detect-base64" \
  -H "Content-Type: application/json" \
  -d '{"image": "$(base64 -w0 road.jpg)", "session_id": "'$SESSION_ID'"}'
```

### c. Legacy endpoint

```bash
curl -X POST "$BASE_URL/detect" -F "file=@/path/to/road.jpg"
```

## 3. Read the Response

A successful detection returns JSON like:

```json
{
  "success": true,
  "detections": [
    {
      "bbox": [123.4, 56.7, 200.1, 150.2],
      "confidence": 0.87,
      "class_id": 7,
      "class_name": "Pothole"
    }
  ],
  "processing_time_ms": 245.67,
  "model_info": {
    "backend": "openvino",
    "classes": ["Alligator Crack", "Block Crack", "Pothole", "…"],
    "confidence_threshold": 0.5
  }
}
```

Each detection item contains:

- `bbox` – bounding box coordinates `[x1, y1, x2, y2]`
- `confidence` – model confidence score (0–1)
- `class_id` / `class_name` – predicted hazard type

### Deriving Bounding Box Metrics

The API only returns the `bbox` array. Clients can compute box center and size as needed:

```javascript
const [x1, y1, x2, y2] = detection.bbox;
const width = x2 - x1;
const height = y2 - y1;
const centerX = x1 + width / 2;
const centerY = y1 + height / 2;
```

## 4. End the Session

```bash
curl -X POST "$BASE_URL/session/$SESSION_ID/end"
```

## Example Clients

### Node.js Quick Start

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const API_URL = process.env.HAZARD_API_URL || 'https://hazard-api-production-production.up.railway.app';

async function quickDetection() {
  const formData = new FormData();
  formData.append('file', fs.createReadStream('test-image.jpg'));
  const res = await axios.post(`${API_URL}/detect`, formData, { headers: formData.getHeaders() });
  res.data.detections.forEach((d, i) => {
    const [x1, y1, x2, y2] = d.bbox;
    const w = x2 - x1;
    const h = y2 - y1;
    const cx = x1 + w / 2;
    const cy = y1 + h / 2;
    console.log(`Hazard ${i + 1}: ${d.class_name} (${(d.confidence*100).toFixed(1)}%)`);
    console.log(` bbox: [${x1.toFixed(1)}, ${y1.toFixed(1)}] → [${x2.toFixed(1)}, ${y2.toFixed(1)}]`);
    console.log(` center: (${cx.toFixed(1)}, ${cy.toFixed(1)}) size: ${w.toFixed(1)} x ${h.toFixed(1)}`);
  });
}

quickDetection();
```

### Python

```python
import requests

BASE_URL = "http://localhost:8080"

session = requests.post(f"{BASE_URL}/session/start").json()["session_id"]
with open("road.jpg", "rb") as f:
    files = {"file": f}
    result = requests.post(f"{BASE_URL}/detect/{session}", files=files).json()
for d in result["detections"]:
    x1, y1, x2, y2 = d["bbox"]
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w/2, y1 + h/2
    print(d["class_name"], cx, cy, w, h)
```

## Further Reading

- [API Fetch Guide](API_FETCH_GUIDE.md) – detailed endpoint reference
- [Node.js Integration Guide](NODEJS_INTEGRATION_GUIDE.md) – reusable client library example
- [Endpoints Guide](ENDPOINTS_GUIDE.md) – image and plot retrieval
- [examples/](examples/README.md) – ready-to-run scripts
