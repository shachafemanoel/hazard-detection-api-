"""
Tests for the report service
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from ..models.report_models import (
    ReportCreateRequest,
    ReportUpdateRequest,
    ReportFilterRequest,
    ReportStatus,
    HazardSeverity,
    DetectionInfo,
    ReportMetadata
)
from ..services.report_service import ReportService
from ..services.model_service import DetectionResult


@pytest.fixture
def report_service():
    """Create a report service instance for testing"""
    service = ReportService()
    # Mock Redis for testing
    service.redis_client = MagicMock()
    service.redis_client.ping.return_value = True
    return service


@pytest.fixture
def sample_detection():
    """Create a sample detection result"""
    return DetectionResult(
        bbox=[100.0, 50.0, 200.0, 150.0],
        confidence=0.85,
        class_id=0,
        class_name="Pothole"
    )


@pytest.fixture
def sample_detection_info():
    """Create sample detection info"""
    return DetectionInfo(
        class_id=0,
        class_name="Pothole",
        confidence=0.85,
        bbox=[100.0, 50.0, 200.0, 150.0],
        area=10000.0,
        center_x=150.0,
        center_y=100.0
    )


@pytest.fixture
def sample_report_request(sample_detection_info):
    """Create a sample report creation request"""
    return ReportCreateRequest(
        detection=sample_detection_info,
        image_data="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//test",
        metadata=ReportMetadata(
            session_id="test-session-123",
            source="test"
        ),
        description="Test pothole report",
        severity=HazardSeverity.HIGH,
        tags=["road", "damage"]
    )


class TestReportService:
    """Test cases for ReportService"""

    @pytest.mark.asyncio
    async def test_create_report_from_detection(self, report_service, sample_detection):
        """Test creating a report from a detection result"""
        with patch('app.services.report_service.cloudinary_service') as mock_cloudinary:
            # Mock cloudinary response
            mock_image_info = MagicMock()
            mock_image_info.url = "https://res.cloudinary.com/test/image.jpg"
            mock_image_info.public_id = "test_image_123"
            mock_cloudinary.upload_base64_image.return_value = mock_image_info
            
            # Mock Redis storage
            report_service.redis_client.set.return_value = True
            
            result = await report_service.create_report_from_detection(
                detection=sample_detection,
                session_id="test-session",
                image_data="base64imagedata"
            )
            
            assert result is not None
            assert result.detection.class_name == "Pothole"
            assert result.detection.confidence == 0.85
            assert result.status == ReportStatus.PENDING
            assert result.metadata.session_id == "test-session"
            
            # Verify Redis storage was called
            report_service.redis_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_report(self, report_service, sample_report_request):
        """Test creating a report from a request"""
        with patch('app.services.report_service.cloudinary_service') as mock_cloudinary:
            # Mock cloudinary response
            mock_image_info = MagicMock()
            mock_image_info.url = "https://res.cloudinary.com/test/image.jpg"
            mock_cloudinary.upload_base64_image.return_value = mock_image_info
            
            # Mock Redis storage
            report_service.redis_client.set.return_value = True
            
            result = await report_service.create_report(sample_report_request)
            
            assert result is not None
            assert result.detection.class_name == "Pothole"
            assert result.description == "Test pothole report"
            assert result.severity == HazardSeverity.HIGH
            assert len(result.tags) == 2
            
            # Verify Redis storage was called
            report_service.redis_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_report(self, report_service):
        """Test retrieving a report by ID"""
        # Mock Redis response
        mock_report_data = {
            "id": "test-report-123",
            "status": "pending",
            "detection": {
                "class_id": 0,
                "class_name": "Pothole",
                "confidence": 0.85,
                "bbox": [100.0, 50.0, 200.0, 150.0],
                "area": 10000.0,
                "center_x": 150.0,
                "center_y": 100.0
            },
            "severity": "high",
            "tags": [],
            "created_at": "2024-01-01T10:00:00",
            "updated_at": "2024-01-01T10:00:00"
        }
        
        report_service.redis_client.get.return_value = json.dumps(mock_report_data)
        
        result = await report_service.get_report("test-report-123")
        
        assert result is not None
        assert result.id == "test-report-123"
        assert result.detection.class_name == "Pothole"
        
        # Verify Redis was queried
        report_service.redis_client.get.assert_called_once_with("report:test-report-123")

    @pytest.mark.asyncio
    async def test_get_report_not_found(self, report_service):
        """Test retrieving a non-existent report"""
        report_service.redis_client.get.return_value = None
        
        result = await report_service.get_report("non-existent")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_update_report(self, report_service):
        """Test updating an existing report"""
        # Mock existing report
        existing_report_data = {
            "id": "test-report-123",
            "status": "pending",
            "detection": {
                "class_id": 0,
                "class_name": "Pothole",
                "confidence": 0.85,
                "bbox": [100.0, 50.0, 200.0, 150.0],
                "area": 10000.0,
                "center_x": 150.0,
                "center_y": 100.0
            },
            "severity": "medium",
            "description": "Original description",
            "tags": [],
            "created_at": "2024-01-01T10:00:00",
            "updated_at": "2024-01-01T10:00:00"
        }
        
        report_service.redis_client.get.return_value = json.dumps(existing_report_data)
        report_service.redis_client.set.return_value = True
        
        update_request = ReportUpdateRequest(
            status=ReportStatus.CONFIRMED,
            description="Updated description",
            severity=HazardSeverity.HIGH
        )
        
        result = await report_service.update_report("test-report-123", update_request)
        
        assert result is not None
        assert result.status == ReportStatus.CONFIRMED
        assert result.description == "Updated description"
        assert result.severity == HazardSeverity.HIGH
        
        # Verify Redis storage was called
        report_service.redis_client.set.assert_called()

    @pytest.mark.asyncio
    async def test_delete_report(self, report_service):
        """Test deleting a report"""
        # Mock existing report for image cleanup
        existing_report_data = {
            "id": "test-report-123",
            "status": "pending",
            "detection": {
                "class_id": 0,
                "class_name": "Pothole",
                "confidence": 0.85,
                "bbox": [100.0, 50.0, 200.0, 150.0],
                "area": 10000.0,
                "center_x": 150.0,
                "center_y": 100.0
            },
            "image": {
                "url": "https://res.cloudinary.com/test/image.jpg",
                "public_id": "test_image_123",
                "width": 640,
                "height": 480
            },
            "severity": "medium",
            "tags": [],
            "created_at": "2024-01-01T10:00:00",
            "updated_at": "2024-01-01T10:00:00"
        }
        
        report_service.redis_client.get.return_value = json.dumps(existing_report_data)
        report_service.redis_client.delete.return_value = 1
        
        with patch('app.services.report_service.cloudinary_service') as mock_cloudinary:
            mock_cloudinary.delete_image.return_value = True
            
            result = await report_service.delete_report("test-report-123")
            
            assert result is True
            
            # Verify image was deleted from Cloudinary
            mock_cloudinary.delete_image.assert_called_once_with("test_image_123")
            
            # Verify report was deleted from Redis
            report_service.redis_client.delete.assert_called_once_with("report:test-report-123")

    @pytest.mark.asyncio
    async def test_list_reports_with_filters(self, report_service):
        """Test listing reports with filters"""
        # Mock Redis keys and data
        report_service.redis_client.keys.return_value = [
            "report:test-1",
            "report:test-2",
            "report:test-3"
        ]
        
        # Mock report data
        reports_data = [
            {
                "id": "test-1",
                "status": "pending",
                "detection": {
                    "class_id": 0,
                    "class_name": "Pothole",
                    "confidence": 0.85,
                    "bbox": [100.0, 50.0, 200.0, 150.0],
                    "area": 10000.0,
                    "center_x": 150.0,
                    "center_y": 100.0
                },
                "severity": "high",
                "tags": [],
                "created_at": "2024-01-01T10:00:00",
                "updated_at": "2024-01-01T10:00:00"
            },
            {
                "id": "test-2",
                "status": "confirmed",
                "detection": {
                    "class_id": 1,
                    "class_name": "Crack",
                    "confidence": 0.75,
                    "bbox": [200.0, 100.0, 300.0, 200.0],
                    "area": 10000.0,
                    "center_x": 250.0,
                    "center_y": 150.0
                },
                "severity": "medium",
                "tags": [],
                "created_at": "2024-01-01T11:00:00",
                "updated_at": "2024-01-01T11:00:00"
            }
        ]
        
        report_service.redis_client.get.side_effect = [
            json.dumps(reports_data[0]),
            json.dumps(reports_data[1])
        ]
        
        filters = ReportFilterRequest(
            status=[ReportStatus.PENDING],
            page=1,
            limit=10
        )
        
        result = await report_service.list_reports(filters)
        
        assert result is not None
        assert len(result.reports) == 1  # Only pending reports
        assert result.reports[0].status == ReportStatus.PENDING
        assert result.total_count == 1
        assert result.pagination["page"] == 1

    @pytest.mark.asyncio
    async def test_get_report_stats(self, report_service):
        """Test getting report statistics"""
        # Mock report data for stats calculation
        report_service.redis_client.keys.return_value = ["report:test-1", "report:test-2"]
        
        reports_data = [
            {
                "id": "test-1",
                "status": "pending",
                "detection": {
                    "class_id": 0,
                    "class_name": "Pothole",
                    "confidence": 0.85,
                    "bbox": [100.0, 50.0, 200.0, 150.0],
                    "area": 10000.0,
                    "center_x": 150.0,
                    "center_y": 100.0
                },
                "severity": "high",
                "tags": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            {
                "id": "test-2",
                "status": "confirmed",
                "detection": {
                    "class_id": 1,
                    "class_name": "Crack",
                    "confidence": 0.75,
                    "bbox": [200.0, 100.0, 300.0, 200.0],
                    "area": 10000.0,
                    "center_x": 250.0,
                    "center_y": 150.0
                },
                "severity": "medium",
                "tags": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
        ]
        
        report_service.redis_client.get.side_effect = [
            json.dumps(reports_data[0]),
            json.dumps(reports_data[1])
        ]
        
        result = await report_service.get_report_stats()
        
        assert result is not None
        assert result.total_reports == 2
        assert result.pending_reports == 1
        assert result.confirmed_reports == 1
        assert result.dismissed_reports == 0
        assert result.avg_confidence == 0.8  # (0.85 + 0.75) / 2

    def test_determine_severity(self, report_service, sample_detection):
        """Test severity determination logic"""
        # Test critical hazard with high confidence
        pothole_detection = DetectionResult(
            bbox=[100.0, 50.0, 200.0, 150.0],
            confidence=0.85,
            class_id=0,
            class_name="Pothole"
        )
        severity = report_service._determine_severity(pothole_detection)
        assert severity == HazardSeverity.CRITICAL
        
        # Test high hazard with medium confidence
        crack_detection = DetectionResult(
            bbox=[100.0, 50.0, 200.0, 150.0],
            confidence=0.75,
            class_id=1,
            class_name="Transverse Crack"
        )
        severity = report_service._determine_severity(crack_detection)
        assert severity == HazardSeverity.HIGH
        
        # Test low-risk hazard
        blur_detection = DetectionResult(
            bbox=[100.0, 50.0, 200.0, 150.0],
            confidence=0.70,
            class_id=2,
            class_name="Lane Blur"
        )
        severity = report_service._determine_severity(blur_detection)
        assert severity == HazardSeverity.LOW

    @pytest.mark.asyncio
    async def test_confirm_report(self, report_service):
        """Test confirming a pending report"""
        # Mock existing pending report
        existing_report_data = {
            "id": "test-report-123",
            "status": "pending",
            "detection": {
                "class_id": 0,
                "class_name": "Pothole",
                "confidence": 0.85,
                "bbox": [100.0, 50.0, 200.0, 150.0],
                "area": 10000.0,
                "center_x": 150.0,
                "center_y": 100.0
            },
            "severity": "high",
            "tags": [],
            "created_at": "2024-01-01T10:00:00",
            "updated_at": "2024-01-01T10:00:00"
        }
        
        report_service.redis_client.get.return_value = json.dumps(existing_report_data)
        report_service.redis_client.set.return_value = True
        
        result = await report_service.confirm_report("test-report-123")
        
        assert result is not None
        assert result.status == ReportStatus.CONFIRMED
        assert result.confirmed_at is not None

    @pytest.mark.asyncio
    async def test_dismiss_report(self, report_service):
        """Test dismissing a pending report"""
        # Mock existing pending report
        existing_report_data = {
            "id": "test-report-123",
            "status": "pending",
            "detection": {
                "class_id": 0,
                "class_name": "Pothole",
                "confidence": 0.85,
                "bbox": [100.0, 50.0, 200.0, 150.0],
                "area": 10000.0,
                "center_x": 150.0,
                "center_y": 100.0
            },
            "severity": "high",
            "tags": [],
            "created_at": "2024-01-01T10:00:00",
            "updated_at": "2024-01-01T10:00:00"
        }
        
        report_service.redis_client.get.return_value = json.dumps(existing_report_data)
        report_service.redis_client.set.return_value = True
        
        result = await report_service.dismiss_report("test-report-123")
        
        assert result is not None
        assert result.status == ReportStatus.DISMISSED