import cv2
import numpy as np
from openvino.runtime import Core


class ModelInference:
    def __init__(self, model_path: str, device_name: str = "CPU"):
        self.class_names = ["crack", "pothole"]
        try:
            core = Core()
            model = core.read_model(model=model_path)
            self.compiled_model = core.compile_model(model=model, device_name=device_name)
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
            _, _, self.model_height, self.model_width = self.input_layer.shape
            print(f"✅ Model '{model_path}' loaded successfully.")
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


inference_handler = ModelInference(model_path="./models/best-11-8-2025.onnx")
