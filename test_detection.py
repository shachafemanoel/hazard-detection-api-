import pytest
import requests
import subprocess
import time
import os
import socket
from PIL import Image, ImageDraw
from io import BytesIO
import signal


def _get_free_port() -> int:
    """Return an available port on the host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Select a free port for the API server to avoid collisions
API_PORT = _get_free_port()
API_BASE_URL = f"http://localhost:{API_PORT}"

@pytest.fixture(scope="module")
def api_server():
    """Fixture to start and stop the FastAPI server."""
    # Command to start the server
    command = ["python", "app.py"]

    # Ensure the server uses the dynamically selected port
    env = os.environ.copy()
    env["PORT"] = str(API_PORT)

    # Start the server as a subprocess
    server_process = subprocess.Popen(command, env=env)

    # Wait for the server to be ready by polling the health endpoint
    max_wait = 60  # seconds
    start_time = time.time()
    while True:
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                print("API server is ready.")
                break
        except requests.ConnectionError:
            if time.time() - start_time > max_wait:
                pytest.fail("API server did not start within the specified time.")
            time.sleep(1)

    # Yield the process object to the tests
    yield server_process

    # Teardown: stop the server
    print("Stopping API server...")
    server_process.send_signal(signal.SIGINT)
    try:
        server_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_process.kill()
    print("API server stopped.")

def create_test_image_with_object(width=640, height=640):
    """Creates a test image with a white square on a black background."""
    img = Image.new('RGB', (width, height), color="black")
    draw = ImageDraw.Draw(img)
    # Draw a white square in the middle of the image
    box_size = 100
    box_x0 = (width - box_size) // 2
    box_y0 = (height - box_size) // 2
    box_x1 = box_x0 + box_size
    box_y1 = box_y0 + box_size
    draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill="white")

    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer

def test_detect_simple_object(api_server):
    """
    Tests that the /detect/{session_id} endpoint can detect a simple object.
    """
    # 1. Start a session
    start_session_url = f"{API_BASE_URL}/session/start"
    response = requests.post(start_session_url)
    assert response.status_code == 200
    session_id = response.json()["session_id"]

    # 2. Create a test image with an object
    image_buffer = create_test_image_with_object()

    # 3. Send the image to the detection endpoint
    detect_url = f"{API_BASE_URL}/detect/{session_id}"
    files = {'file': ('test_image.jpg', image_buffer, 'image/jpeg')}
    response = requests.post(detect_url, files=files)

    # 4. Assert that the response is successful and contains detections
    assert response.status_code == 200
    response_data = response.json()
    assert "detections" in response_data
    assert isinstance(response_data["detections"], list)
    # Assert that the model detected something
    assert len(response_data["detections"]) > 0

    # 5. End the session
    end_session_url = f"{API_BASE_URL}/session/{session_id}/end"
    requests.post(end_session_url)
