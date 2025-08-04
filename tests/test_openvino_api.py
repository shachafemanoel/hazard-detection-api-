import os
from pathlib import Path
import sys
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
from io import BytesIO

# Ensure model directory points to repository root for tests
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))
os.environ.setdefault("MODEL_DIR", str(repo_root))

from app import app

client = TestClient(app)


def create_test_image():
    img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8), 'RGB')
    buf = BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy"}


def test_session_start_and_detect():
    res = client.post("/session/start")
    assert res.status_code == 200
    session_id = res.json()["session_id"]

    image_buf = create_test_image()
    files = {"file": ("test.jpg", image_buf, "image/jpeg")}
    detect_res = client.post(f"/detect/{session_id}", files=files)
    assert detect_res.status_code == 200
    assert "detections" in detect_res.json()
