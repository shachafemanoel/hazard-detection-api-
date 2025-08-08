# TASKS.md - Hazard Detection API Agent Execution

**Project:** hazard-detection-api- (FastAPI + OpenVINO Backend)  
**Created:** 2025-08-08  
**Status:** 🚀 EXECUTION IN PROGRESS  
**Work Manager:** Agent 0

**HARD RULES:**
- ✅ Inference on server = OpenVINO ONLY (no onnxruntime)
- ✅ Keep contracts stable: /report, /session/{id}/reports, /summary, /health, /ready
- ✅ MODEL_PATH=/app/best0608.onnx (target model)

## Critical Path Execution Order

### 🎯 Phase 1: Agent C - Model Integration (ACTIVE)
**Branch:** `feat/server-model-best0608`  
**Owner:** Model Integration Specialist  
**Status:** IN PROGRESS

**Tasks:**
- [ ] **C1:** Update MODEL_PATH to best0608.onnx in config
- [ ] **C2:** Verify input shape 1×3×480×480 and output shape (1,300,6)
- [ ] **C3:** Update class mapping for 6-class model (vs current 4-class)
- [ ] **C4:** Expose model name and classes in /status endpoint
- [ ] **C5:** Create scripts/verify_onnx.py for OpenVINO smoke test
- [ ] **C6:** Ensure /ready endpoint validates model loading

**Acceptance Criteria:**
- Model loads best0608.onnx on startup ✅
- /ready endpoint returns green ✅
- /status shows best0608 model info + class names ✅
- verify_onnx.py script runs successfully ✅

**On Success:** Commit & push → "feat(api): load best0608 and expose class map"

---

### 📊 Phase 2: Agent B - API Persistence & Session Summary
**Branch:** `feat/api-persistence-summary`  
**Owner:** API Persistence Specialist  
**Status:** PENDING

**Tasks:**
- [ ] **B1:** Verify POST /report stores report:{uuid} with cloudinaryUrl
- [ ] **B2:** Ensure RPUSH to session:{id}:reports index
- [ ] **B3:** GET /session/{id}/reports returns full objects with URLs
- [ ] **B4:** GET /session/{id}/summary includes image.url + aggregates
- [ ] **B5:** Add Cloudinary upload retries and MIME validation
- [ ] **B6:** Create/update tests/test_report_flow.py

**Acceptance Criteria:**
- POST /report persists with cloudinaryUrl ✅
- Session reports indexed properly ✅
- Summary includes full objects with image URLs ✅
- All tests pass ✅

---

### 🏥 Phase 3: Agent G - Health/Ready & CI
**Branch:** `ci/health-ready-and-workflow`  
**Owner:** Health & CI Specialist  
**Status:** PENDING

**Tasks:**
- [ ] **G1:** Enhance /ready endpoint (model + Redis + Cloudinary checks)
- [ ] **G2:** Implement structured logging with request IDs and timing
- [ ] **G3:** Update/create .github/workflows/api.yml for CI
- [ ] **G4:** Add comprehensive health tests
- [ ] **G5:** Ensure pytest runs in CI with proper coverage

**Acceptance Criteria:**
- /health and /ready pass all checks ✅
- CI pipeline runs pytest successfully ✅
- Structured logging shows API latencies ✅

---

### 🧹 Phase 4: Agent D - Code Cleanup
**Branch:** `chore/server-cleanup`  
**Owner:** Code Cleanup Specialist  
**Status:** PENDING

**Tasks:**
- [ ] **D1:** Remove unused routers not mounted in main.py
- [ ] **D2:** Clean up __pycache__ directories
- [ ] **D3:** Remove any obsolete *_old.py files
- [ ] **D4:** Update .gitignore for cache prevention
- [ ] **D5:** Verify no import errors after cleanup

**Acceptance Criteria:**
- No unused routers/services ✅
- Tests still pass after cleanup ✅
- Clean repository structure ✅

---

## Branch Strategy
```
feat/server-model-best0608     → Agent C (Model Integration)
feat/api-persistence-summary   → Agent B (API Persistence)  
ci/health-ready-and-workflow   → Agent G (Health & CI)
chore/server-cleanup          → Agent D (Cleanup)
```

## Environment Configuration
```bash
# Required for best0608 model
MODEL_PATH=/app/best0608.onnx
REDIS_HOST=...
REDIS_PASSWORD=...
CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
ALLOWED_ORIGINS=https://hazard-detection-production-8735.up.railway.app
```

## Success Metrics
- **Model Loading:** best0608.onnx loads successfully on startup
- **API Latency:** P95 ≤150ms for detection requests  
- **Contract Stability:** All client endpoints remain backward compatible
- **Health Checks:** /health and /ready validate full system state
- **Test Coverage:** All tests pass including new test_report_flow.py

---

**Next Action:** Starting Agent C model integration on `feat/server-model-best0608` branch