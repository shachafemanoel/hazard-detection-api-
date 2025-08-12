import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.inference import inference_handler

router = APIRouter()


@router.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        detections = inference_handler.predict(image)
        return {"filename": file.filename, "detections": detections}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
