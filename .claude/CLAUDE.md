
⸻

project: hazard-detection-api
role: Repository operating manual + Claude Code multi‑agent instructions
purpose: A FastAPI service that hosts the OpenVINO object‑detection model for live inference only. It is deployed separately from the main web app (hazard-detection) and exposes simple detection APIs. No long‑term persistence here; the web app handles Redis/Cloudinary and reporting.

Overview
	•	Primary model runtime: OpenVINO (OV)
	•	Fallback runtime: PyTorch/torchvision (or onnxruntime, if configured)
	•	Public base URL (prod): https://hazard-api-production-production.up.railway.app/
	•	Main web app (prod): https://hazard-detection-production-8735.up.railway.app/
	•	Responsibility split
	•	This repo: Load model(s), serve /detect (image + live stream), return results fast. Provide /health, /ready, /status.
	•	Web app repo: Browser ONNX path, Redis, Cloudinary, reports, sessions, and UI.

Architecture

app/
  main.py                # FastAPI app factory + lifespan (model warmup)
  routers/
    detect.py            # /detect/image, /ws/detect (WebSocket), optional /detect/frame
    health.py            # /health (liveness), /ready (model+device), /status (metadata)
  services/
    infer_openvino.py    # OV Core init, model compile, inference, postprocess
    infer_torch.py       # Torch fallback (loaded lazily)
    loader.py            # Model path resolution, class labels, warmup routine
  core/
    config.py            # Env vars, feature flags, defaults
    logging.py           # Structured logging with request id, timing
  models/
    schemas.py           # Pydantic models for requests/responses
  utils/
    nms.py               # NMS/box decode helpers (if not provided by runtime)
  tests/
    test_health.py       # /health, /ready
    test_detect_image.py # single image inference contract
scripts/
  verify_onnx.py         # print IO shapes + test dummy inference (OV and/or torch)
models/
  best0608.onnx          # (kept out of git in prod; mounted via volume or download)

Data Flow
	1.	Startup (lifespan): read MODEL_PATH, init OpenVINO Core, compile model, allocate buffers, warm up (1 dummy run). Record model meta (name, shapes, precision, device).
	2.	Request:
	•	/detect/image (HTTP POST): accepts multipart or JSON/base64; parses options (threshold, max_detections, nms_iou, return_masks), runs inference, returns normalized results.
	•	/ws/detect (WebSocket): accepts binary frames (JPEG/PNG) or base64; streams back JSON messages per frame. Designed for UI live preview.
	3.	Fallback: If OV init fails or device unsupported, switch to BACKEND=torch lazily; log a warning and expose this in /status.

API Contracts

GET /health
	•	Purpose: process liveness. Always 200 {status:"ok"} when the server can answer HTTP.

GET /ready
	•	Purpose: readiness. Returns 200 only if the active backend is initialized and the model is compiled (OV) or loaded (Torch).
	•	Body:

{
  "ready": true,
  "backend": "openvino",
  "device": "CPU",
  "model": {"path": "/app/best0608.onnx", "input": "1x3x480x480", "output": "1x300x6"}
}

GET /status
	•	Purpose: diagnostics for UI/integration.
	•	Body:

{
  "backend": "openvino",
  "fallback": "torch",
  "device": "CPU",
  "model_name": "best0608",
  "classes": ["person","helmet","vest", "hazard"],
  "version": "2025.08.0",
  "uptime_s": 1234,
  "concurrency": {"max": 4, "in_flight": 0}
}

POST /detect/image
	•	Consumes: multipart/form-data (field image) or application/json { "image": "data:image/jpeg;base64,...", "threshold":0.35 }
	•	Returns:

{
  "model": "best0608",
  "backend": "openvino",
  "time_ms": 12.7,
  "detections": [
    { "bbox": {"x": 0.12, "y": 0.31, "w": 0.25, "h": 0.32}, "score": 0.91, "class_id": 1, "class_name": "helmet"}
  ]
}

	•	Notes: bounding boxes normalized to [0,1] in xywh (top-left normalized). Set BOX_FORMAT=xyxy to change format.

WS /ws/detect
	•	Frames in: binary image bytes or base64 JSON ({"image":"...","threshold":0.4})
	•	Frames out: same schema as /detect/image plus frame index.
	•	Backpressure: server may sample frames (FRAME_SAMPLE_RATE) under load.

Out of scope for this repo: creating reports, Redis persistence, Cloudinary uploads—these happen in the web app. The API stays stateless aside from in‑memory caches.

Environment Variables (example .env)

# Model & runtime
MODEL_PATH=/app/best0608.onnx
BACKEND=openvino           # openvino | torch | ort
FALLBACK_BACKEND=torch     # fallback when primary fails
INFERENCE_DEVICE=CPU       # CPU | GPU | AUTO
OV_PRECISION=FP16
WARMUP=1                   # number of dummy runs on startup
MAX_CONCURRENCY=4          # worker pool size

# Contracts / CORS
ALLOWED_ORIGINS=https://hazard-detection-production-8735.up.railway.app
BOX_FORMAT=xywh            # xywh | xyxy
DEFAULT_THRESHOLD=0.35
NMS_IOU=0.5

# Server
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

Guardrails
	•	Stateless: do not write to DB/Redis; do not upload images.
	•	No secrets in logs; redact data URLs over a size threshold.
	•	Small, reversible PRs; add/keep tests; maintain backward compatibility of responses.
	•	If /ready is false, return 503 with {reason} to trigger client fallback (browser ONNX).

Claude Code Agents

Each agent has a crisp mandate. Use explicit invocation lines at the end to dispatch a task to a specific subagent.

Agent A — Work Manager (Server Lead)

Goal: Plan → Assign → Verify → Merge.
	•	Create/maintain TASKS.md with owners & checkboxes.
	•	Sequence Agents B–G; ensure contracts and tests are green before merge.
	•	Deliverables: TASKS.md, PRs, REPORT.md with latency snapshot.

Agent B — OpenVINO Model Integrator

Files: services/infer_openvino.py, core/config.py, scripts/verify_onnx.py
	•	Initialize OV Core, compile network once, pin precision (FP16), pre‑allocate buffers.
	•	Implement preprocess (resize to model input, BGR/RGB swap), postprocess (decode boxes, NMS, class map).
	•	Expose model IO and classes in /status.
	•	DOD: /ready true with backend=openvino; verify_onnx.py prints shapes and runs dummy inference.

Agent C — Torch Fallback Engineer

Files: services/infer_torch.py, core/config.py
	•	Lazy‑load torch model (or ONNX via torchvision/ORT) only if OV unavailable.
	•	Keep identical output schema; unit test parity within ±1 class/frame on sample.
	•	DOD: flip BACKEND=torch and pass test_detect_image.py.

Agent D — API Contract & CORS

Files: routers/detect.py, models/schemas.py, core/config.py, routers/health.py
	•	Define strict Pydantic schemas; support multipart and base64 JSON.
	•	Normalize bbox format and include backend, model, time_ms in response.
	•	Configure CORS to allow the production web app origin.
	•	DOD: test_health.py, test_detect_image.py pass; manual fetch from the web app succeeds (no CORS errors).

Agent E — Performance & Concurrency

Files: services/infer_openvino.py, core/config.py, main.py
	•	Add worker pool or asyncio semaphore with MAX_CONCURRENCY.
	•	Enable OV caching if available; micro‑benchmark throughput vs latency.
	•	DOD: load stays < 70% CPU at 10 FPS with 640×480 frames on CPU target (example numbers; record in report).

Agent F — CI, Packaging & Release

Files: .github/workflows/api.yml (new), requirements.txt / pyproject.toml
	•	CI: lint, tests, build image; publish Railway preview on PR.
	•	Health gates: block merge if /ready fails in preview.
	•	DOD: green pipeline, version tag in /status.version.

Agent G — Repo Sweeper (Safe Cleanup)

Files: .claude/agents/repo-sweeper-api.md (subagent prompt), .gitignore
	•	Dry‑run, quarantine to .trash/<ts>/, validate, report reclaimed bytes.
	•	Never touch models/**, app/**, scripts/**, configs, or CI.
	•	DOD: build/tests pass post‑cleanup, report saved.

Test Plan
	•	Unit: test_health.py, test_detect_image.py with a 1×1 pixel and a real sample.
	•	Integration: Start server → call /detect/image from the web app build; ensure CORS ok and schema unchanged.
	•	Load: simple frame loop @ 5–10 FPS over /ws/detect until 2000 frames; track avg time_ms.

Local Development

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
MODEL_PATH=./models/best0608.onnx uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Deployment (Railway example)
	•	Set env vars from the table above.
	•	Health checks: GET /health for liveness, GET /ready for readiness.
	•	Rollout rule: Only promote if /ready.ready == true and latency p95 < target.

Explicit Invocation Cheatsheet (copy/paste in Claude Code)

> Use the work-manager subagent to create TASKS.md and assign owners for A–G
> Have the openvino-integrator subagent wire up infer_openvino.py and make /ready green
> Ask the torch-fallback subagent to implement and validate the PyTorch fallback
> Use the api-contract subagent to finalize /detect schemas and CORS
> Have the perf-tuner subagent add concurrency control and measure latency
> Ask the ci-release subagent to add GitHub Actions and Railway preview
> Use the repo-sweeper subagent to perform a dry-run cleanup and write CLEANUP_REPORT.md

Definition of Done (Repo‑level)
	•	/health, /ready, /status behave as specified.
	•	/detect/image and /ws/detect return consistent schema across backends.
	•	OpenVINO is default and warmed; Torch fallback works when forced.
	•	Web app can consume results in production origin without CORS issues.
	•	CI is green; a short REPORT.md documents latency, readiness, and model meta.