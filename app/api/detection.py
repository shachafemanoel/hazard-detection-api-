"""
Detection API endpoints for hazard detection
"""

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
            image = Image.open(image_stream).convert("RGB")
        except Exception as e:
            raise InvalidImageException(f"Failed to process image: {str(e)}")

        # Store original image data for reports (base64 encoded)
        image_base64 = base64.b64encode(contents).decode("utf-8")

        # Run model inference
        detections = await ms.predict(image)

        # Process detections with session tracking
        processing_result = session_service.process_detections(
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
            image_data = base64.b64decode(detection_request.image)
            image_stream = BytesIO(image_data)
            image = Image.open(image_stream).convert("RGB")
        except Exception as e:
            raise InvalidImageException(f"Failed to decode base64 image: {str(e)}")

        # Run model inference
        detections = await ms.predict(image)

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

        # Run model inference
        detections = await ms.predict(image)

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

                # Run inference
                detections = await ms.predict(image)

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
