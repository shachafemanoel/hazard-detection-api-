"""
Detection API endpoints for hazard detection
"""

import asyncio
import base64
import time
from typing import Dict, List, Any, Optional
from io import BytesIO
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from pydantic import BaseModel

from ..core.config import settings
from ..core.logging_config import get_logger
from ..core.exceptions import (
    ModelNotLoadedException,
    SessionNotFoundException,
    InvalidImageException,
    InferenceException,
)
from ..services import model_service as model_service_module
from ..services.session_service import session_service

logger = get_logger("detection_api")

router = APIRouter(tags=["detection"])


class Base64DetectionRequest(BaseModel):
    image: str  # Base64 encoded image
    confidence_threshold: Optional[float] = 0.5
    session_id: Optional[str] = None
    timeout_ms: Optional[int] = 30000  # 30 second default timeout


class DetectionResponse(BaseModel):
    success: bool
    detections: List[Dict[str, Any]]
    processing_time_ms: float
    image_size: Dict[str, int]
    model_info: Dict[str, Any]
    
    
class BatchDetectionResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    total_processing_time_ms: float
    processed_count: int
    successful_count: int
    model_info: Dict[str, Any]


@router.post("/detect/{session_id}")
async def detect_hazards_with_session(
    session_id: str, file: Optional[UploadFile] = File(None)
) -> Dict[str, Any]:
    """
    Enhanced detection endpoint with object tracking and report generation
    Returns detections and creates reports for new unique hazards
    """
    if file is None or not file.filename or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=422, detail="File field required with valid image"
        )

    try:
        # Validate session exists before heavy processing
        session_service.get_session(session_id)

        ms = model_service_module.model_service
        # Ensure model is loaded
        if not ms.is_loaded:
            await ms.load_model()

        start_time = time.time()

        # Read and process image
        contents = await file.read()
        image_stream = BytesIO(contents)
        image_stream.seek(0)

        try:
            # Enhanced image validation and loading
            if len(contents) == 0:
                raise InvalidImageException("Uploaded file is empty")
            
            # Validate file is likely an image by checking file signature
            if len(contents) < 10:
                raise InvalidImageException("File too small to be a valid image")
            
            # Check common image headers
            if not (contents.startswith(b'\xff\xd8\xff') or  # JPEG
                   contents.startswith(b'\x89PNG') or        # PNG
                   contents.startswith(b'GIF8') or           # GIF
                   contents.startswith(b'BM') or             # BMP
                   contents.startswith(b'RIFF')):            # WebP (in RIFF container)
                logger.warning("File doesn't match common image signatures, attempting to process anyway")
            
            image = Image.open(image_stream)
            
            # Force load the image to catch any corruption issues early
            try:
                image.load()
            except Exception as load_error:
                raise InvalidImageException(f"Image file appears to be corrupted: {load_error}")
            
            # Validate basic image properties
            if not hasattr(image, 'size') or len(image.size) != 2:
                raise InvalidImageException("Invalid image structure")
            
            # Ensure RGB format for consistent processing
            if image.mode != 'RGB':
                logger.info(f"Converting image from {image.mode} to RGB")
                try:
                    image = image.convert("RGB")
                except Exception as conv_error:
                    raise InvalidImageException(f"Failed to convert image to RGB: {conv_error}")
                
            # Validate image dimensions
            if image.width <= 0 or image.height <= 0:
                raise InvalidImageException(f"Image has invalid dimensions: {image.width}x{image.height}")
                
            # Check for extremely small images that might cause processing issues
            if image.width < 10 or image.height < 10:
                raise InvalidImageException(f"Image too small for processing: {image.width}x{image.height} (minimum 10x10)")
                
            # Additional validation for common issues
            if image.width > 8192 or image.height > 8192:
                logger.warning(f"Large image detected: {image.size}. Resizing for processing.")
                # Resize very large images to prevent memory issues
                max_size = 4096
                if image.width > max_size or image.height > max_size:
                    ratio = min(max_size / image.width, max_size / image.height)
                    new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
                    try:
                        image = image.resize(new_size, Image.Resampling.LANCZOS)
                    except Exception as resize_error:
                        raise InvalidImageException(f"Failed to resize large image: {resize_error}")
                
            logger.info(f"Image processed: {image.size} {image.mode}")
                
        except InvalidImageException:
            raise  # Re-raise our custom exceptions
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            # Try alternative loading method
            try:
                image_stream.seek(0)
                image = Image.open(image_stream)
                image.load()  # Force load the image data
                if image.mode != 'RGB':
                    image = image.convert("RGB")
                
                # Re-validate after alternative loading
                if image.width <= 0 or image.height <= 0:
                    raise InvalidImageException(f"Alternative loading produced invalid dimensions: {image.width}x{image.height}")
                    
                logger.info(f"Alternative loading successful: {image.size} {image.mode}")
            except Exception as e2:
                # Provide more specific error message based on the original error
                if "cannot identify image file" in str(e).lower():
                    raise InvalidImageException("File is not a valid image format or is corrupted")
                elif "truncated" in str(e).lower():
                    raise InvalidImageException("Image file appears to be truncated or incomplete")
                else:
                    raise InvalidImageException(f"Failed to process image with both methods: Primary error: {str(e)}, Fallback error: {str(e2)}")

        # Store original image data for reports (base64 encoded)
        image_base64 = base64.b64encode(contents).decode("utf-8")

        # Run model inference with timeout
        try:
            detections = await asyncio.wait_for(
                ms.predict(image), 
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail={
                    "error": "inference_timeout",
                    "message": "Model inference timed out after 30 seconds",
                    "timeout_ms": 30000
                }
            )

        # Process detections with session tracking
        processing_result = await session_service.process_detections(
            session_id, detections, image_base64
        )

        processing_time = round((time.time() - start_time) * 1000, 2)

        logger.info(
            f"Processed image for session {session_id}: {len(detections)} detections in {processing_time}ms"
        )

        return {
            "success": True,
            "detections": processing_result["detections"],
            "new_reports": processing_result["new_reports"],
            "session_stats": processing_result["session_stats"],
            "processing_time_ms": processing_time,
            "image_size": {"width": image.width, "height": image.height},
            "model_info": {
                **ms.get_model_info(),
                "confidence_threshold": settings.min_confidence_for_report,
                "tracking_enabled": True,
            },
        }

    except SessionNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelNotLoadedException as e:
        raise HTTPException(status_code=503, detail=str(e))
    except InvalidImageException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except InferenceException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post("/detect-base64")
async def detect_hazards_json(detection_request: Base64DetectionRequest) -> Dict[str, Any]:
    """
    Detection endpoint that accepts base64 image data
    Supports the frontend camera detection app format
    """
    try:
        ms = model_service_module.model_service
        # Ensure model is loaded
        if not ms.is_loaded:
            await ms.load_model()

        start_time = time.time()

        # Decode base64 image
        try:
            # Validate base64 string
            if not detection_request.image or len(detection_request.image.strip()) == 0:
                raise InvalidImageException("Base64 image string is empty")
            
            # Remove data URL prefix if present
            image_b64 = detection_request.image.strip()
            if image_b64.startswith('data:'):
                if ',' in image_b64:
                    image_b64 = image_b64.split(',', 1)[1]
                else:
                    raise InvalidImageException("Invalid data URL format in base64 string")
            
            # Validate base64 string length
            if len(image_b64) < 100:  # Very small for any meaningful image
                raise InvalidImageException("Base64 string too short to contain valid image data")
            
            try:
                image_data = base64.b64decode(image_b64, validate=True)
            except Exception as decode_error:
                raise InvalidImageException(f"Invalid base64 encoding: {decode_error}")
            
            if len(image_data) == 0:
                raise InvalidImageException("Decoded base64 image data is empty")
            
            image_stream = BytesIO(image_data)
            image_stream.seek(0)
            
            # Validate image file signature
            if len(image_data) < 10:
                raise InvalidImageException("Decoded image data too small to be valid")
            
            image = Image.open(image_stream)
            
            # Force load to catch corruption early
            try:
                image.load()
            except Exception as load_error:
                raise InvalidImageException(f"Base64 image appears to be corrupted: {load_error}")
            
            # Validate basic image properties
            if not hasattr(image, 'size') or len(image.size) != 2:
                raise InvalidImageException("Invalid base64 image structure")
            
            if image.mode != 'RGB':
                logger.info(f"Converting base64 image from {image.mode} to RGB")
                try:
                    image = image.convert("RGB")
                except Exception as conv_error:
                    raise InvalidImageException(f"Failed to convert base64 image to RGB: {conv_error}")
                
            # Validate dimensions
            if image.width <= 0 or image.height <= 0:
                raise InvalidImageException(f"Base64 image has invalid dimensions: {image.width}x{image.height}")
            
            # Check for extremely small images
            if image.width < 10 or image.height < 10:
                raise InvalidImageException(f"Base64 image too small for processing: {image.width}x{image.height} (minimum 10x10)")
                
            logger.info(f"Base64 image processed: {image.size} {image.mode}")
            
        except InvalidImageException:
            raise  # Re-raise our custom exceptions
        except Exception as e:
            logger.error(f"Base64 image processing error: {e}")
            try:
                # Try alternative decoding
                image_stream.seek(0)
                image = Image.open(image_stream)
                image.load()
                image = image.convert("RGB")
                
                # Re-validate after alternative loading
                if image.width <= 0 or image.height <= 0:
                    raise InvalidImageException(f"Alternative base64 loading produced invalid dimensions: {image.width}x{image.height}")
                    
                logger.info(f"Alternative base64 decoding successful: {image.size} {image.mode}")
            except Exception as e2:
                if "cannot identify image file" in str(e).lower():
                    raise InvalidImageException("Base64 data is not a valid image format")
                elif "invalid base64" in str(e).lower():
                    raise InvalidImageException("Invalid base64 encoding provided")
                else:
                    raise InvalidImageException(f"Failed to decode base64 image with both methods: Primary: {str(e)}, Fallback: {str(e2)}")

        # Run model inference with timeout
        try:
            detections = await asyncio.wait_for(
                ms.predict(image), 
                timeout=detection_request.timeout_ms / 1000.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail={
                    "error": "inference_timeout", 
                    "message": f"Model inference timed out after {detection_request.timeout_ms}ms",
                    "timeout_ms": detection_request.timeout_ms
                }
            )

        # Convert detections to frontend-compatible format
        frontend_detections = []
        for detection in detections:
            # Filter by confidence threshold
            if detection.confidence >= detection_request.confidence_threshold:
                frontend_detections.append({
                    "x1": float(detection.bbox[0]),
                    "y1": float(detection.bbox[1]),
                    "x2": float(detection.bbox[2]),
                    "y2": float(detection.bbox[3]),
                    "score": float(detection.confidence),
                    "classId": int(detection.class_id)
                })

        processing_time = round((time.time() - start_time) * 1000, 2)

        logger.info(
            f"Base64 detection: {len(frontend_detections)} detections in {processing_time}ms"
        )

        return {
            "success": True,
            "detections": frontend_detections,
            "predictions": frontend_detections,  # Alternative key for compatibility
            "processing_time_ms": processing_time,
            "image_size": {"width": image.width, "height": image.height},
            "model_info": {
                **ms.get_model_info(),
                "classes": len(ms.get_model_info().get("classes", [])),
                "confidence_threshold": detection_request.confidence_threshold,
            },
        }

    except ModelNotLoadedException as e:
        raise HTTPException(status_code=503, detail=str(e))
    except InvalidImageException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except InferenceException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Base64 detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post("/detect")
async def detect_hazards_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Legacy detection endpoint that accepts file uploads
    Returns detections without session tracking
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        ms = model_service_module.model_service
        # Ensure model is loaded
        if not ms.is_loaded:
            await ms.load_model()

        start_time = time.time()

        # Read and process image
        contents = await file.read()
        image_stream = BytesIO(contents)
        image_stream.seek(0)

        try:
            image = Image.open(image_stream).convert("RGB")
        except Exception as e:
            raise InvalidImageException(f"Failed to process image: {str(e)}")

        # Run model inference with timeout
        try:
            detections = await asyncio.wait_for(
                ms.predict(image), 
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail={
                    "error": "inference_timeout",
                    "message": "Model inference timed out after 30 seconds", 
                    "timeout_ms": 30000
                }
            )

        # Convert detections to dictionary format
        detection_dicts = [detection.to_dict() for detection in detections]

        processing_time = round((time.time() - start_time) * 1000, 2)

        logger.info(
            f"File detection: {len(detections)} detections in {processing_time}ms"
        )

        return {
            "success": True,
            "detections": detection_dicts,
            "processing_time_ms": processing_time,
            "image_size": {"width": image.width, "height": image.height},
            "model_info": {
                **ms.get_model_info(),
                "classes": len(ms.get_model_info().get("classes", [])),
            },
        }

    except ModelNotLoadedException as e:
        raise HTTPException(status_code=503, detail=str(e))
    except InvalidImageException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except InferenceException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"File detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post("/detect-batch")
async def detect_batch(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Batch detection endpoint for processing multiple images
    """
    try:
        ms = model_service_module.model_service
        # Ensure model is loaded
        if not ms.is_loaded:
            await ms.load_model()

        results = []
        start_time = time.time()

        for i, file in enumerate(files):
            if not file.content_type.startswith("image/"):
                results.append(
                    {
                        "file_index": i,
                        "filename": file.filename,
                        "error": "File must be an image",
                    }
                )
                continue

            try:
                # Process each image
                contents = await file.read()
                image_stream = BytesIO(contents)
                image_stream.seek(0)
                image = Image.open(image_stream).convert("RGB")

                # Run inference with timeout
                try:
                    detections = await asyncio.wait_for(
                        ms.predict(image), 
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    results.append(
                        {
                            "file_index": i,
                            "filename": file.filename,
                            "error": "Model inference timed out after 30 seconds"
                        }
                    )
                    continue

                # Format results
                detection_dicts = [detection.to_dict() for detection in detections]

                results.append(
                    {
                        "file_index": i,
                        "filename": file.filename,
                        "success": True,
                        "detections": detection_dicts,
                        "image_size": {"width": image.width, "height": image.height},
                    }
                )

            except Exception as e:
                logger.error(f"Batch processing error for file {i}: {e}")
                results.append(
                    {"file_index": i, "filename": file.filename, "error": str(e)}
                )

        total_time = round((time.time() - start_time) * 1000, 2)
        successful_count = len([r for r in results if r.get("success", False)])

        logger.info(
            f"Batch processing: {successful_count}/{len(files)} files processed in {total_time}ms"
        )

        return {
            "success": True,
            "results": results,
            "total_processing_time_ms": total_time,
            "processed_count": len(files),
            "successful_count": successful_count,
            "model_info": ms.get_model_info(),
        }

    except ModelNotLoadedException as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")
