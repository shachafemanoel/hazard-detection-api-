# Real-Time Hazard Detection Streaming Guide

This guide explains how to use the Hazard Detection API to perform real-time monitoring of a stream of images.

The API does not use WebSockets for real-time streaming. Instead, it supports a polling-based approach where a client-side script can watch a directory for new images and send them to the API for processing as they arrive. This is a simple and effective way to achieve real-time monitoring without the complexity of WebSockets.

This guide is based on the `examples/realTimeMonitoring.js` script in this repository.

## How It Works

The real-time monitoring process works as follows:

1.  **Watch a Directory:** A client-side script continuously watches a specified directory for new image files.
2.  **New Image Detected:** When a new image is added to the directory, the script is triggered.
3.  **Send to API:** The script sends the new image to the Hazard Detection API's `/detect/{session_id}` endpoint.
4.  **Receive Detections:** The script receives the detection results from the API.
5.  **Process Results:** The script can then process the results, for example, by saving them to a file, sending an alert, or displaying them in a user interface.

## Setting Up the Client-Side Script

To set up the real-time monitoring script, you will need:

*   **Node.js:** The example script is written in JavaScript and requires Node.js to run.
*   **Dependencies:** You will need to install the following Node.js packages:
    *   `axios`: For making HTTP requests to the API.
    *   `form-data`: For creating `multipart/form-data` requests to upload images.
    *   `chokidar`: For watching the directory for new files.

You can install these dependencies using npm:

```bash
npm install axios form-data chokidar
```

## Client-Side Code Example

Here is a simplified JavaScript code example that demonstrates how to implement real-time monitoring. You can adapt this script to your specific needs.

```javascript
// realTimeMonitoring.js

const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const chokidar = require('chokidar');

// The URL of your Hazard Detection API
const API_URL = 'https://hazard-api-production-production.up.railway.app';

// The directory to watch for new images
const WATCH_DIRECTORY = './incoming-images';

// The directory to save the detection results
const OUTPUT_DIRECTORY = './processed-results';

// --- Helper Functions ---

// Ensure that the necessary directories exist
function setupDirectories() {
  [WATCH_DIRECTORY, OUTPUT_DIRECTORY].forEach(dir => {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
      console.log(`Created directory: ${dir}`);
    }
  });
}

// Start a new session with the API
async function startSession() {
  try {
    const response = await axios.post(`${API_URL}/session/start`);
    console.log(`Session started: ${response.data.session_id}`);
    return response.data.session_id;
  } catch (error) {
    console.error('Failed to start session:', error.message);
    throw error;
  }
}

// Process a single image
async function processImage(imagePath, sessionId) {
  try {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(imagePath));

    const response = await axios.post(
      `${API_URL}/detect/${sessionId}`,
      formData,
      {
        headers: formData.getHeaders(),
      }
    );

    const result = response.data;
    console.log(`Processed: ${path.basename(imagePath)} - ${result.detections.length} detections`);

    // Save the detection results to a file
    const filename = path.basename(imagePath, path.extname(imagePath));
    const resultFile = path.join(OUTPUT_DIRECTORY, `${filename}_result.json`);
    fs.writeFileSync(resultFile, JSON.stringify(result, null, 2));

  } catch (error) {
    console.error(`Error processing ${imagePath}:`, error.message);
  }
}

// --- Main Monitoring Logic ---

async function main() {
  console.log('Starting real-time hazard monitoring...');

  setupDirectories();
  const sessionId = await startSession();

  if (!sessionId) {
    console.error('Could not start a session. Exiting.');
    return;
  }

  const watcher = chokidar.watch(WATCH_DIRECTORY, {
    ignored: /(^|[\/\\])\../, // ignore dotfiles
    persistent: true,
  });

  watcher.on('add', (imagePath) => {
    const ext = path.extname(imagePath).toLowerCase();
    if (['.jpg', '.jpeg', '.png', '.bmp'].includes(ext)) {
      console.log(`New image detected: ${path.basename(imagePath)}`);
      processImage(imagePath, sessionId);
    }
  });

  console.log(`Watching for new images in: ${WATCH_DIRECTORY}`);
  console.log('Press Ctrl+C to stop monitoring.');

  process.on('SIGINT', async () => {
    console.log('Stopping monitoring...');
    watcher.close();
    // You can also end the session here if you want
    // await axios.post(`${API_URL}/session/${sessionId}/end`);
    process.exit(0);
  });
}

main();
```

### How to Use the Script

1.  **Save the code:** Save the code above as a file named `realTimeMonitoring.js`.
2.  **Install dependencies:** Run `npm install axios form-data chokidar` in the same directory.
3.  **Create directories:** Create the `incoming-images` and `processed-results` directories.
4.  **Run the script:** Start the monitoring script by running `node realTimeMonitoring.js`.
5.  **Add images:** Copy or move image files into the `incoming-images` directory. The script will automatically detect them, send them to the API, and save the results in the `processed-results` directory.

## API Response Format

The API will respond with a JSON object containing the detection results. Here is an example of the response format:

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

You can parse this JSON response in your client-side script to access the detection results and use them as needed.
