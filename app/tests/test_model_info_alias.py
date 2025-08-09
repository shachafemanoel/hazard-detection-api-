from app.models.api_models import DetectionResponse, ModelInfo


def test_model_info_alias_serialization():
    info = ModelInfo(
        status="loaded",
        backend="openvino",
        classes=[],
        class_count=0,
    )
    resp = DetectionResponse(
        success=True,
        detections=[],
        new_reports=None,
        session_stats=None,
        processing_time_ms=0.0,
        image_size={"width": 1, "height": 1},
        model_meta=info,
    )
    data = resp.model_dump(by_alias=True)
    assert "model_info" in data
    assert "model_meta" not in data
