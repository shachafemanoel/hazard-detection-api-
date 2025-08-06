"""
Custom exceptions and error handlers for the Hazard Detection API
"""

from typing import Any, Dict, Optional
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .logging_config import get_logger

logger = get_logger("exceptions")


class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: str
    path: Optional[str] = None
    request_id: Optional[str] = None


class HazardDetectionException(Exception):
    """Base exception for hazard detection operations"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ModelNotLoadedException(HazardDetectionException):
    """Raised when model operations are attempted but model isn't loaded"""
    pass


class ModelLoadingException(HazardDetectionException):
    """Raised when model loading fails"""
    pass


class InferenceException(HazardDetectionException):
    """Raised when model inference fails"""
    pass


class SessionNotFoundException(HazardDetectionException):
    """Raised when a session is not found"""
    pass


class InvalidImageException(HazardDetectionException):
    """Raised when image processing fails"""
    pass


class ExternalAPIException(HazardDetectionException):
    """Raised when external API calls fail"""
    pass


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with structured error responses"""
    
    logger.warning(f"HTTP {exc.status_code} at {request.url.path}: {exc.detail}")
    
    error_response = {
        "error": f"HTTP {exc.status_code}",
        "detail": exc.detail,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "path": str(request.url.path)
    }
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )


async def hazard_detection_exception_handler(
    request: Request, 
    exc: HazardDetectionException
) -> JSONResponse:
    """Handle custom hazard detection exceptions"""
    
    logger.error(f"HazardDetectionException at {request.url.path}: {exc.message}")
    if exc.details:
        logger.error(f"Exception details: {exc.details}")
    
    # Map exception types to HTTP status codes
    status_code_map = {
        ModelNotLoadedException: 503,
        ModelLoadingException: 503,
        InferenceException: 500,
        SessionNotFoundException: 404,
        InvalidImageException: 400,
        ExternalAPIException: 502
    }
    
    status_code = status_code_map.get(type(exc), 500)
    
    error_response = {
        "error": exc.__class__.__name__,
        "detail": exc.message,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "path": str(request.url.path)
    }
    
    if exc.details:
        error_response["additional_info"] = exc.details
    
    return JSONResponse(
        status_code=status_code,
        content=error_response
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions"""
    
    logger.error(f"Unhandled exception at {request.url.path}: {str(exc)}", exc_info=True)
    
    error_response = {
        "error": "InternalServerError",
        "detail": "An unexpected error occurred",
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "path": str(request.url.path)
    }
    
    return JSONResponse(
        status_code=500,
        content=error_response
    )