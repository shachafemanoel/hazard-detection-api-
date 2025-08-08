"""
Service layer tests for Redis, Cloudinary, and Model services
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from app.services.redis_service import redis_service
from app.services.cloudinary_service import cloudinary_service
from app.services.model_service import model_service


class TestRedisService:
    """Test Redis service functionality"""
    
    def test_health_status(self):
        """Test Redis health status"""
        health = redis_service.get_health_status()
        assert "status" in health
        assert "connected" in health
        
        # Status should be one of the expected values
        assert health["status"] in ["healthy", "not_configured", "unhealthy"]
    
    def test_report_storage_format(self):
        """Test that report storage follows B3 contract format"""
        # Mock Redis client for testing
        if redis_service.redis_client:
            report_id = "test-report-123"
            report_data = {
                "sessionId": "test-session-456",
                "className": "pothole",  # Note: className not 'class'
                "confidence": 0.85,
                "ts": int(time.time() * 1000),
                "geo": {"lat": 40.7128, "lon": -74.0060},
                "cloudinaryUrl": "https://res.cloudinary.com/test/image/upload/test.jpg",
                "status": "pending"
            }
            
            # Test storage
            success = redis_service.store_report(report_id, report_data)
            
            if success:  # Only test if Redis is available
                # Test retrieval
                retrieved = redis_service.get_report(report_id)
                assert retrieved is not None
                assert retrieved["sessionId"] == report_data["sessionId"]
                assert retrieved["className"] == report_data["className"]
                assert retrieved["cloudinaryUrl"] == report_data["cloudinaryUrl"]
                
                # Test session indexing
                success = redis_service.add_report_to_session("test-session-456", report_id)
                assert success
                
                session_reports = redis_service.get_session_reports("test-session-456")
                assert report_id in session_reports
                
                # Cleanup
                redis_service.delete_report(report_id)


class TestCloudinaryService:
    """Test Cloudinary service functionality"""
    
    def test_health_status(self):
        """Test Cloudinary health status"""
        health = cloudinary_service.get_health_status()
        assert "status" in health
        assert "configured" in health
        
        # Status should be one of the expected values
        assert health["status"] in ["healthy", "not_configured", "unhealthy"]
    
    def test_image_validation(self):
        """Test image validation logic (B4 requirement)"""
        # Test with invalid data
        is_valid, error = cloudinary_service._validate_image_data(b"not an image", "test.jpg")
        assert not is_valid
        assert "Invalid image data" in error
        
        # Test file size validation
        large_data = b"x" * (15 * 1024 * 1024)  # 15MB
        is_valid, error = cloudinary_service._validate_image_data(large_data, "large.jpg")
        assert not is_valid
        assert "File too large" in error


class TestModelService:
    """Test Model service functionality"""
    
    def test_health_status(self):
        """Test model health status (B2 requirement)"""
        health = model_service.get_health_status()
        assert "status" in health
        assert "model_loaded" in health
        
        if health["model_loaded"]:
            # If model is loaded, verify B2 requirements
            assert "classes" in health
            assert "expected_classes" in health
            assert "input_size" in health
            assert "classes_valid" in health
            assert "input_size_valid" in health
            
            # B2: Verify class mapping {0: crack, 1: knocked, 2: pothole, 3: surface damage}
            expected_classes = ["crack", "knocked", "pothole", "surface damage"]
            assert health["expected_classes"] == expected_classes
            
            # B2: Verify input size 480x480
            assert health["input_size"] == 480
    
    def test_model_info(self):
        """Test model info structure"""
        info = model_service.get_model_info()
        assert "status" in info
        
        if info["status"] == "loaded":
            assert "backend" in info
            assert "classes" in info
            assert "class_count" in info
            assert info["class_count"] == 4  # B2: 4 classes for best0408 model


@pytest.mark.asyncio
async def test_service_integration():
    """Test integration between services"""
    # Test that all services can report their health
    redis_health = redis_service.get_health_status()
    cloudinary_health = cloudinary_service.get_health_status()
    model_health = model_service.get_health_status()
    
    # All should return valid health dictionaries
    assert isinstance(redis_health, dict)
    assert isinstance(cloudinary_health, dict)
    assert isinstance(model_health, dict)
    
    # Each should have a status field
    assert "status" in redis_health
    assert "status" in cloudinary_health
    assert "status" in model_health