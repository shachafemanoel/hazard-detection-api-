# CLAUDE.md — Server Repo (hazard-detection-api-)

> Root: `/Users/shachafemanoel/Documents/api/hazard-detection-api-`

This file defines specialized Claude Code agents for the **FastAPI + OpenVINO** backend. Includes a **Work Manager** and an **Integration Specialist** focused on contract alignment with the client repo.

## Shared Context

- **Health:** `/health`, `/ready`
- **Reports:** `POST /report`, `GET /session/{id}/reports`, `GET /session/{id}/summary`, `POST /session/{id}/report/{rid}/confirm|dismiss`
- **Redis Keys:** `report:{uuid}`, `session:{id}:reports`
- **Model Path:** `MODEL_PATH=/app/best0608.onnx`
- **Guardrails:** Small commits, no secrets, backward compatibility, add tests, ask unclear in `QUESTIONS.md`.

## Environment Keys

```
REDIS_HOST=...
REDIS_PORT=6379
REDIS_USERNAME=default
REDIS_PASSWORD=...
CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
ALLOWED_ORIGINS=https://hazard-detection-production-8735.up.railway.app
MODEL_PATH=/app/best0608.onnx
```

---

## Agent 0 — Work Manager (Server Lead)

**Goal:** Plan → Assign → Verify → Commit & Push.

**Scope:** Whole repo; coordinates Agents B, C, D, G, X.

**Process:**

1. Create `TASKS.md` with a checklist and owners.
2. Delegate to agents; run unit/API tests.
3. Verify acceptance, then push commits and open PRs.

**Deliverables:** `TASKS.md`, PRs, `REPORT.md` with API latencies and pass/fail summaries.

---

## Agent B — API Persistence & Session Summary

**Files:**

- `app/routers/session.py`, `app/routers/report.py`
- `app/services/report_service.py`, `app/services/redis_service.py`
- `app/models/schemas.py`, `app/core/config.py`

**Tasks:**

1. Ensure `POST /report` stores `report:{uuid}` with `cloudinaryUrl`, and `RPUSH session:{id}:reports`.
2. `GET /session/{id}/reports` and `/summary` return full objects (incl. URLs + aggregates).
3. Add retries for Cloudinary and MIME validation.

**Acceptance:** `/health` & `/ready` green; summary contains Cloudinary URLs; data types validated.

**Deliverables:** code + `tests/test_report_flow.py` + README snippet.

---

## Agent C — Model Integration (best0608)

**Files:**

- `app/services/model_service.py`
- `app/core/config.py`, `.env.example`
- `scripts/verify_onnx.py` **(new)**

**Tasks:**

1. Load `MODEL_PATH` best0608; verify input 1×3×480×480 and output shape `(1,300,6)`.
2. Align class ids/names with client hazard classes; expose names in `/status`.
3. Provide `scripts/verify_onnx.py` (OpenVINO or onnxruntime) to print IO and run a dummy inference.

**Acceptance:** model warms on startup; `/status` shows best0608 loaded; verify script runs.

**Deliverables:** code + script + log snippet.

---

## Agent D — Code Cleanup (Server)

**Files:**

- Remove unused routers/services: `app/routers/*` not mounted in `app/main.py`, `*_old.py`, `experimental_*.py`
- Remove `notebooks/`, `experiments/`, caches `**/__pycache__/**`

**Tasks:**

1. Use `rg` to confirm no imports; delete and update `.gitignore`.

**Acceptance:** tests still pass; no import errors.

**Deliverables:** PR `chore(server): remove obsolete files`.

---

## Agent G — Health/Ready, Observability & CI

**Files:**

- `app/main.py` (lifespan), `app/core/config.py`
- `tests/test_health.py`, `tests/test_summary.py`
- `.github/workflows/api.yml` **(new)**

**Tasks:**

1. Ensure `/health` liveness and `/ready` checks model + Redis + Cloudinary.
2. Add structured logging (request id + timing) for detect/report endpoints.
3. CI workflow running `pytest`, packaging, and optionally Railway preview deploy.

**Acceptance:** health/ready pass; CI green; logs show latencies.

**Deliverables:** CI file + tests + README CI section.

---

## Agent X — Integration Specialist (Server‑oriented)

**Goal:** Own **integration with the Client repo** from the server side.

**Files:**

- `app/routers/report.py`, `app/routers/session.py`
- `app/models/schemas.py`

**Tasks:**

1. Confirm response JSON matches client expectations (fields and nesting), especially `image.url`.
2. Validate CORS/ALLOWED\_ORIGINS; coordinate with client `VITE_API_BASE`.
3. Maintain a contract doc `schemas/server-api.md` mirrored with client `schemas/client-api.md`.

**Acceptance:** No schema mismatches; E2E client flow passes against live Railway.

**Deliverables:** `schemas/server-api.md` + PR notes.

