"""Simple OpenVINO inference utilities."""

import os

import cv2
import numpy as np
from openvino.runtime import Core

# Global path to the model file. This allows the model location to be
# configured via the ``MODEL_PATH`` environment variable while providing a
# sensible default. All functions and classes in this module refer to this
# path instead of hardcoding file locations.
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/best-11-8-2025.onnx")


class ModelInference:
    def __init__(self, model_path: str = MODEL_PATH, device_name: str = "CPU"):
        """Initialize the inference model.

        Parameters
        ----------
        model_path: str
            Path to the model file. Defaults to the module-level ``MODEL_PATH``
            so that all instances share a consistent global configuration.
        device_name: str
            Target device for OpenVINO inference.
        """

        self.class_names = ["crack", "pothole"]
        self.model_path = model_path

        try:
            core = Core()
            model = core.read_model(model=self.model_path)
            self.compiled_model = core.compile_model(model=model, device_name=device_name)
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
            _, _, self.model_height, self.model_width = self.input_layer.shape
            print(f"✅ Model '{self.model_path}' loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.compiled_model = None

    def _preprocess(self, image: np.ndarray):
        h, w, _ = image.shape
        scale = min(self.model_height / h, self.model_width / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h))
        padded_image = np.full((self.model_height, self.model_width, 3), 0, dtype=np.uint8)
        dx, dy = (self.model_width - new_w) // 2, (self.model_height - new_h) // 2
        padded_image[dy : dy + new_h, dx : dx + new_w] = resized_image
        img_data = np.expand_dims(np.transpose(padded_image, (2, 0, 1)), 0).astype(np.float32)
        return img_data, scale, dx, dy

    def _postprocess(
        self, results: np.ndarray, scale: float, pad_dx: int, pad_dy: int, conf_threshold=0.5
    ):
        detections = []
        for box in results[0]:
            confidence = box[4]
            if confidence >= conf_threshold:
                class_id = int(box[5])
                x1, y1, x2, y2 = (
                    int((box[0] - pad_dx) / scale),
                    int((box[1] - pad_dy) / scale),
                    int((box[2] - pad_dx) / scale),
                    int((box[3] - pad_dy) / scale),
                )
                detections.append(
                    {
                        "class_name": self.class_names[class_id],
                        "confidence": float(confidence),
                        "box": [x1, y1, x2, y2],
                    }
                )
        return detections

    def predict(self, image: np.ndarray):
        if not self.compiled_model:
            raise RuntimeError("Model is not available.")
        input_data, scale, pad_dx, pad_dy = self._preprocess(image)
        raw_result = self.compiled_model([input_data])[self.output_layer]
        return self._postprocess(raw_result, scale, pad_dx, pad_dy)


inference_handler = ModelInference()
