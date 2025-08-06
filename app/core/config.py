"""
Configuration management for the Hazard Detection API
Uses Pydantic Settings for environment-based configuration
"""

import os
from pathlib import Path
from typing import List, Optional, Literal
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Application settings
    app_name: str = "Hazard Detection API"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")

    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8080, env="PORT")

    # Model configuration
    model_dir: str = Field(default="/app", env="MODEL_DIR")
    model_backend: Literal["auto", "openvino", "pytorch"] = Field(
        default="auto", env="MODEL_BACKEND"
    )
    model_input_size: int = Field(default=512, env="MODEL_INPUT_SIZE")

    # OpenVINO settings
    openvino_device: str = Field(default="AUTO", env="OPENVINO_DEVICE")
    openvino_cache_enabled: bool = Field(default=False, env="OPENVINO_CACHE_ENABLED")
    openvino_performance_mode: Literal[
        "LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"
    ] = Field(default="LATENCY", env="OPENVINO_PERFORMANCE_MODE")
    openvino_async_inference: bool = Field(default=True, env="OPENVINO_ASYNC_INFERENCE")

    # Detection settings
    confidence_threshold: float = Field(default=0.5, env="CONFIDENCE_THRESHOLD")
    iou_threshold: float = Field(default=0.45, env="IOU_THRESHOLD")
    min_confidence_for_report: float = Field(
        default=0.6, env="MIN_CONFIDENCE_FOR_REPORT"
    )
    tracking_distance_threshold: int = Field(
        default=50, env="TRACKING_DISTANCE_THRESHOLD"
    )
    tracking_time_threshold: float = Field(default=2.0, env="TRACKING_TIME_THRESHOLD")

    # CORS settings
    cors_origins: List[str] = Field(
        default=["https://*.railway.app"], env="CORS_ORIGINS"
    )
    cors_methods: List[str] = Field(default=["GET", "POST"], env="CORS_METHODS")
    cors_headers: List[str] = Field(default=["*"], env="CORS_HEADERS")

    # External API credentials
    google_maps_api_key: Optional[str] = Field(default=None, env="GOOGLE_MAPS_API_KEY")
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    cloudinary_cloud_name: Optional[str] = Field(
        default=None, env="CLOUDINARY_CLOUD_NAME"
    )
    cloudinary_api_key: Optional[str] = Field(default=None, env="CLOUDINARY_API_KEY")
    cloudinary_api_secret: Optional[str] = Field(
        default=None, env="CLOUDINARY_API_SECRET"
    )
    render_api_key: Optional[str] = Field(default=None, env="RENDER_API_KEY")
    railway_token: Optional[str] = Field(default=None, env="RAILWAY_TOKEN")

    # Deployment settings
    environment: str = Field(default="development", env="ENVIRONMENT")
    railway_environment_name: Optional[str] = Field(
        default=None, env="RAILWAY_ENVIRONMENT_NAME"
    )

    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Health check settings
    healthcheck_timeout: int = Field(default=120, env="HEALTHCHECK_TIMEOUT")

    class Config:
        env_file = ".env"
        case_sensitive = False


class ModelConfig:
    """Model-specific configuration and paths"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_dir = Path(settings.model_dir)

    @property
    def class_names(self) -> List[str]:
        """YOLO class names for hazard detection"""
        return [
            "Alligator Crack",
            "Block Crack",
            "Crosswalk Blur",
            "Lane Blur",
            "Longitudinal Crack",
            "Manhole",
            "Patch Repair",
            "Pothole",
            "Transverse Crack",
            "Wheel Mark Crack",
        ]

    @property
    def openvino_model_paths(self) -> List[Path]:
        """Potential OpenVINO model file paths"""
        return [
            self.base_dir / "best0408_openvino_model" / "best0408.xml",
            self.base_dir / "best_openvino_model" / "last_model_train12052025.xml",
            self.base_dir / "best_openvino_model" / "best.xml",
            self.base_dir / "last_model_train12052025.xml",
            self.base_dir / "best.xml",
            self.base_dir / "openvino" / "best.xml",
            self.base_dir / "model.xml",
        ]

    @property
    def pytorch_model_paths(self) -> List[Path]:
        """Potential PyTorch model file paths"""
        return [
            self.base_dir / "best.pt",
            self.base_dir / "pytorch" / "best.pt",
            self.base_dir / "models" / "best.pt",
            self.base_dir / "road_damage_detection_last_version.pt",
            self.base_dir / "best_yolo12m.pt",
        ]

    @property
    def fallback_directories(self) -> List[Path]:
        """Fallback directories to search for model files"""
        return [
            self.base_dir,
            self.base_dir / "models",
            self.base_dir / "best_openvino_model",
            self.base_dir / "best0408_openvino_model",
            self.base_dir / "models" / "pytorch",
            self.base_dir / "models" / "openvino",
            self.base_dir / "public" / "object_detection_model",
        ]


# Global settings instance
settings = Settings()
model_config = ModelConfig(settings)
