"""
Main entry point for the Hazard Detection API
Imports the refactored modular application
"""

if __name__ == "__main__":
    from app.main import app
    import uvicorn
    import os

    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"🚀 Starting Hazard Detection API on {host}:{port}")
    print("📚 API Documentation available at: /docs")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        reload=False,
    )
