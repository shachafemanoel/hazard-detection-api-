"""
Configuration management for the Hazard Detection API
Uses Pydantic Settings for environment-based configuration
"""

import os
from pathlib import Path
from typing import List, Optional, Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    ml_model_dir: str = Field(default="/app/server", env="MODEL_DIR")
    ml_model_path: Optional[str] = Field(default=None, env="MODEL_PATH")  # Explicit model path (e.g., /app/server/openvino_fp16/best.xml)
    ml_model_backend: Literal["auto", "openvino"] = Field(
        default="openvino", env="MODEL_BACKEND"  # OpenVINO ONLY for server inference
    )
    ml_model_input_size: int = Field(default=640, env="MODEL_INPUT_SIZE")  # Updated for YOLOv12n 640x640 input

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
    
    # Report management settings
    reports_per_page: int = Field(default=20, env="REPORTS_PER_PAGE")
    max_reports_per_request: int = Field(default=100, env="MAX_REPORTS_PER_REQUEST")
    report_image_max_size_mb: int = Field(default=10, env="REPORT_IMAGE_MAX_SIZE_MB")
    report_retention_days: int = Field(default=90, env="REPORT_RETENTION_DAYS")
    auto_create_reports: bool = Field(default=True, env="AUTO_CREATE_REPORTS")
    geocoding_enabled: bool = Field(default=True, env="GEOCODING_ENABLED")

    # CORS settings
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_methods: List[str] = Field(default=["*"], env="CORS_METHODS")
    cors_headers: List[str] = Field(default=["*"], env="CORS_HEADERS")

    # External API credentials
    google_maps_api_key: Optional[str] = Field(default=None, env="GOOGLE_MAPS_API_KEY")
    
    # Redis settings for report storage
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    redis_host: Optional[str] = Field(default=None, env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_username: Optional[str] = Field(default="default", env="REDIS_USERNAME")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Cloudinary settings for image storage
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
    
    # Streaming settings
    streaming_session_timeout: int = Field(default=300, env="STREAMING_SESSION_TIMEOUT")
    streaming_max_fps: int = Field(default=30, env="STREAMING_MAX_FPS")
    streaming_default_fps: int = Field(default=10, env="STREAMING_DEFAULT_FPS")
    streaming_queue_size: int = Field(default=50, env="STREAMING_QUEUE_SIZE")
    streaming_batch_size: int = Field(default=5, env="STREAMING_BATCH_SIZE")
    streaming_enable_tracking: bool = Field(default=True, env="STREAMING_ENABLE_TRACKING")

    model_config = SettingsConfigDict(
        env_file=".env", 
        case_sensitive=False,
        protected_namespaces=()  # Disable protected namespace warnings
    )


class ModelConfig:
    """Model-specific configuration and paths"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_dir = Path(settings.ml_model_dir)
        self._loaded_model_path: Optional[Path] = None

    def set_loaded_model_path(self, model_path: Path) -> None:
        """Set the path of the currently loaded model"""
        self._loaded_model_path = model_path

    @property
    def loaded_model_name(self) -> str:
        """Get the name of the currently loaded model"""
        if self._loaded_model_path:
            if "best0608" in str(self._loaded_model_path):
                return "best0608"
            elif "best0408" in str(self._loaded_model_path):
                return "best0408"
            else:
                return self._loaded_model_path.stem
        return "unknown"

    @property
    def class_names(self) -> List[str]:
        """YOLO class names for hazard detection (YOLOv12n model - 2 classes)"""
        return [
            "crack",    # 0: Crack damage (all types)
            "pothole",  # 1: Pothole damage
        ]

    @property
    def openvino_model_paths(self) -> List[Path]:
        """Potential OpenVINO model file paths (prioritized by preference)"""
        paths = []
        
        # If MODEL_PATH is explicitly set, prioritize it
        if self.settings.ml_model_path:
            paths.append(Path(self.settings.ml_model_path))
        
        # Default search paths for YOLOv12n models
        paths.extend([
            # Primary model (FP16 - faster inference) - RECOMMENDED
            self.base_dir / "openvino_fp16" / "best.xml",
            
            # Secondary model (FP32 - higher precision)  
            self.base_dir / "openvino_fp32" / "best.xml",
            
            # Legacy fallback paths (for backward compatibility)
            self.base_dir / "best0608_openvino_model" / "best0608.xml",
            self.base_dir / "best0408_openvino_model" / "best0408.xml", 
            self.base_dir / "best_openvino_model" / "best.xml",
            self.base_dir / "best.xml",
        ])
        
        return paths

    @property
    def pytorch_model_paths(self) -> List[Path]:
        """Potential PyTorch model file paths"""
        return [
            # Target best0608 PyTorch model
            self.base_dir / "best0608.pt",
            
            # Current and legacy models
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
