"""
Minimal FastAPI app for debugging Railway deployment issues
This version skips model loading and complex services to isolate startup problems
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Simple app for testing basic deployment
app = FastAPI(
    title="Hazard Detection API - Debug Mode",
    version="1.0.0-debug",
    description="Minimal version for debugging deployment issues"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Minimal FastAPI service is running",
        "debug": True
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_status": "disabled",
        "version": "1.0.0-debug"
    }

@app.get("/env")
async def environment():
    """Show environment variables for debugging"""
    return {
        "environment": dict(os.environ),
        "cwd": os.getcwd(),
        "port": os.getenv("PORT", "8080")
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "app.main_minimal:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )