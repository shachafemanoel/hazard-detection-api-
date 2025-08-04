# Hazard Detection API

[![Deployment on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/your-username/your-repo-name)

This is a high-performance FastAPI backend for real-time hazard detection on roads. It uses a YOLOv8 model, optimized with OpenVINO, to identify various types of road damage from images or video streams.

The API is designed for scalability and can be deployed easily to cloud platforms like Railway.

## ✨ Features

*   **High-Performance Inference:** Utilizes OpenVINO for optimized model inference on CPUs, with a fallback to PyTorch.
*   **Intelligent Model Loading:** Automatically selects the best available backend (OpenVINO or PyTorch).
*   **Real-time and Batch Processing:** Endpoints for both single-image and batch processing.
*   **Session Management:** Track and manage detection sessions for continuous monitoring.
*   **Object Tracking:** Basic tracking to identify unique hazards and avoid duplicate reports.
*   **CORS Enabled:** Properly configured for integration with web-based frontends.
*   **Deploy-Ready:** Includes configurations for Railway and a Dockerfile for containerization.

##  Detected Hazard Classes

The model is trained to detect the following 10 classes of road hazards:
1.  Alligator Crack
2.  Block Crack
3.  Crosswalk Blur
4.  Lane Blur
5.  Longitudinal Crack
6.  Manhole
7.  Patch Repair
8.  Pothole
9.  Transverse Crack
10. Wheel Mark Crack

## 🚀 Getting Started

### Prerequisites

*   Python 3.10+
*   Pip for package management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Server Locally

Once the dependencies are installed, you can start the API server using Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at `http://127.0.0.1:8000`.

##  API Endpoints

The API provides the following endpoints:

### Service Health and Status

*   `GET /health`
    *   A lightweight endpoint to check if the service is running. Ideal for platform health checks.
    *   **Response:** `{"status": "ok"}`

*   `GET /status`
    *   Provides a detailed status of the service, including model status, device info, and environment configuration.
    *   **Response:** A detailed JSON object with service diagnostics.

### Core Functionality

*   `POST /session/start`
    *   Starts a new detection session.
    *   **Response:** `{"session_id": "...", "message": "Detection session started"}`

*   `POST /detect/{session_id}`
    *   Upload an image to a specific session for hazard detection. Includes basic object tracking to identify new hazards.
    *   **Request Body:** `multipart/form-data` with an image file.
    *   **Response:** JSON with detections, new reports, and session stats.

*   `POST /detect`
    *   A legacy endpoint for single-image detection without session management.
    *   **Request Body:** `multipart/form-data` with an image file.
    *   **Response:** JSON with detections and processing time.

*   `POST /detect-batch`
    *   Upload multiple images for batch processing.
    *   **Request Body:** `multipart/form-data` with multiple image files.
    *   **Response:** A list of results for each image.

## ⚙️ Configuration

The application can be configured using environment variables:

*   `PORT`: The port on which the server will run. Defaults to `8000`.
*   `MODEL_DIR`: The directory where the model files (`best.pt`, `best_openvino_model/`) are located. Defaults to `/app`.
*   `MODEL_BACKEND`: The preferred model backend. Options are `auto`, `openvino`, or `pytorch`. `auto` will prioritize OpenVINO if available.
*   `PYTHONPATH`: Should be set to the application's root directory to ensure modules are found. Defaults to `/app`.
*   `FRONTEND_URL`: The URL of your frontend application, to be included in the CORS allowed origins.

## 🐳 Docker

A `Dockerfile` is provided to build a container image for the application.

**Build the image:**

```bash
docker build -t hazard-detection-api .
```

**Run the container:**

```bash
docker run -p 8000:8000 -e PORT=8000 hazard-detection-api
```

## 🚂 Deployment on Railway

This project is configured for easy deployment on Railway.

1.  **Click the Deploy Button:** Use the "Deploy on Railway" button at the top of this README.
2.  **Connect Your Repo:** Connect your GitHub repository to the Railway project.
3.  **Configure Environment Variables:** Railway will automatically use the `railway.toml` file. You can add any additional environment variables in the project's settings on Railway.
4.  **Deployment:** Railway will build and deploy the application automatically. The `healthcheckPath` is set to `/health`, ensuring a reliable startup.
