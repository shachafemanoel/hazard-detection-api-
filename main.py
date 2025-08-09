"""
Main entry point for the Hazard Detection API
Imports the refactored modular application
"""

if __name__ == "__main__":
    from app.main import app
    from app.core.logging_config import get_logger
    import uvicorn
    import os

    logger = get_logger("main")
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"🚀 Starting Hazard Detection API on {host}:{port}")
    logger.info("📚 API Documentation available at: /docs")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        reload=False,
    )
