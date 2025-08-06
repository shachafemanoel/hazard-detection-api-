# Hazard Detection API Client Guide

This guide explains how to send images to the Hazard Detection API from a client application and interpret the results.

## Base URL

Use the API base URL for your environment:

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

When using session-based detection, additional fields may include `new_reports` and `session_stats` for hazard tracking.

## 4. End the Session

```bash
curl -X POST "$BASE_URL/session/$SESSION_ID/end"
```

## Example Clients

### Python

```python
import requests

BASE_URL = "http://localhost:8080"

session = requests.post(f"{BASE_URL}/session/start").json()["session_id"]
with open("road.jpg", "rb") as f:
    files = {"file": f}
    result = requests.post(f"{BASE_URL}/detect/{session}", files=files).json()
print(result["detections"])
```

### JavaScript (fetch)

```javascript
const session = await fetch(`${BASE_URL}/session/start`, {method: 'POST'}).then(r => r.json());
const form = new FormData();
form.append('file', fileInput.files[0]);
const res = await fetch(`${BASE_URL}/detect/${session.session_id}`, {method: 'POST', body: form});
const data = await res.json();
console.log(data.detections);
```

## Further Reading

- [API Fetch Guide](API_FETCH_GUIDE.md) – detailed endpoint reference
- [Node.js Integration Guide](NODEJS_INTEGRATION_GUIDE.md) – reusable client library example
- [examples/](examples/README.md) – ready‑to‑run scripts

