"""
Centralized logging configuration for the Hazard Detection API
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

from .config import settings


def setup_logging() -> logging.Logger:
    """Setup centralized logging configuration"""
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    configure_third_party_loggers()
    
    # Create app logger
    app_logger = logging.getLogger("hazard_detection")
    app_logger.info(f"🚀 Logging initialized - Level: {settings.log_level}")
    app_logger.info(f"🌍 Environment: {settings.environment}")
    
    return app_logger


def configure_third_party_loggers():
    """Configure third-party library loggers"""
    
    # Reduce noise from third-party libraries
    noisy_loggers = [
        "urllib3.connectionpool",
        "requests.packages.urllib3.connectionpool",
        "PIL.PngImagePlugin",
        "PIL.TiffImagePlugin"
    ]
    
    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(f"hazard_detection.{name}")


# Initialize logging on import
logger = setup_logging()