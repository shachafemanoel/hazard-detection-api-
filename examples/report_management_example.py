#!/usr/bin/env python3
"""
Example script demonstrating the Report Management API
Shows how to create, retrieve, update, and manage hazard detection reports
"""

import asyncio
import json
import base64
from datetime import datetime
import httpx

# API Configuration
API_URL = "http://localhost:8080"  # Update this to your API URL
API_URL = "https://hazard-api-production-production.up.railway.app"  # Or use deployed URL


async def example_report_management():
    """Demonstrate complete report management workflow"""
    
    print("🚀 Hazard Detection Report Management Example")
    print("=" * 50)
    
    async with httpx.AsyncClient() as client:
        
        # 1. Create a session for detection
        print("\n📋 Step 1: Creating detection session...")
        session_response = await client.post(
            f"{API_URL}/session/start",
            json={
                "confidence_threshold": 0.5,
                "source": "report_management_example"
            }
        )
        
        if session_response.status_code != 200:
            print(f"❌ Failed to create session: {session_response.status_code}")
            return
            
        session = session_response.json()
        session_id = session["session_id"]
        print(f"✅ Session created: {session_id}")
        
        # 2. Create a manual report
        print("\n📋 Step 2: Creating manual report...")
        
        # Sample detection data
        detection_data = {
            "class_id": 0,
            "class_name": "Pothole",
            "confidence": 0.85,
            "bbox": [100.0, 50.0, 200.0, 150.0],
            "area": 10000.0,
            "center_x": 150.0,
            "center_y": 100.0
        }
        
        # Sample base64 image (1px red image for demo)
        sample_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        report_request = {
            "detection": detection_data,
            "image_data": sample_image,
            "metadata": {
                "session_id": session_id,
                "source": "manual_report"
            },
            "description": "Large pothole detected on Main Street",
            "severity": "high",
            "tags": ["road-damage", "urgent", "main-street"]
        }
        
        report_response = await client.post(
            f"{API_URL}/reports",
            json=report_request
        )
        
        if report_response.status_code != 200:
            print(f"❌ Failed to create report: {report_response.status_code}")
            print(report_response.text)
            return
            
        report = report_response.json()
        report_id = report["id"]
        print(f"✅ Report created: {report_id}")
        print(f"   Status: {report['status']}")
        print(f"   Severity: {report['severity']}")
        print(f"   Confidence: {report['detection']['confidence']}")
        
        # 3. Retrieve the report
        print("\n📋 Step 3: Retrieving report...")
        
        get_response = await client.get(f"{API_URL}/reports/{report_id}")
        
        if get_response.status_code == 200:
            retrieved_report = get_response.json()
            print(f"✅ Retrieved report: {retrieved_report['id']}")
            print(f"   Created at: {retrieved_report['created_at']}")
            print(f"   Description: {retrieved_report['description']}")
            print(f"   Tags: {', '.join(retrieved_report['tags'])}")
        else:
            print(f"❌ Failed to retrieve report: {get_response.status_code}")
        
        # 4. Update the report
        print("\n📋 Step 4: Updating report...")
        
        update_request = {
            "description": "Updated: Large pothole confirmed - repair scheduled",
            "severity": "critical",
            "tags": ["road-damage", "urgent", "main-street", "repair-scheduled"]
        }
        
        update_response = await client.patch(
            f"{API_URL}/reports/{report_id}",
            json=update_request
        )
        
        if update_response.status_code == 200:
            updated_report = update_response.json()
            print(f"✅ Report updated")
            print(f"   New severity: {updated_report['severity']}")
            print(f"   Updated description: {updated_report['description']}")
        else:
            print(f"❌ Failed to update report: {update_response.status_code}")
        
        # 5. List reports with filters
        print("\n📋 Step 5: Listing reports...")
        
        list_response = await client.get(
            f"{API_URL}/reports",
            params={
                "session_id": session_id,
                "status": "pending",
                "limit": 10
            }
        )
        
        if list_response.status_code == 200:
            report_list = list_response.json()
            print(f"✅ Found {len(report_list['reports'])} reports")
            print(f"   Total count: {report_list['total_count']}")
            
            for i, r in enumerate(report_list['reports'], 1):
                print(f"   {i}. {r['detection']['class_name']} "
                      f"(confidence: {r['detection']['confidence']:.2f}, "
                      f"severity: {r['severity']})")
        else:
            print(f"❌ Failed to list reports: {list_response.status_code}")
        
        # 6. Get report statistics
        print("\n📋 Step 6: Getting report statistics...")
        
        stats_response = await client.get(f"{API_URL}/reports/stats")
        
        if stats_response.status_code == 200:
            stats = stats_response.json()
            print(f"✅ Report Statistics:")
            print(f"   Total reports: {stats['total_reports']}")
            print(f"   Pending: {stats['pending_reports']}")
            print(f"   Confirmed: {stats['confirmed_reports']}")
            print(f"   Average confidence: {stats['avg_confidence']:.2f}")
            print(f"   Recent reports (24h): {stats['recent_reports_count']}")
            
            if stats['reports_by_class']:
                print("   By hazard type:")
                for class_name, count in stats['reports_by_class'].items():
                    print(f"     {class_name}: {count}")
        else:
            print(f"❌ Failed to get statistics: {stats_response.status_code}")
        
        # 7. Confirm the report
        print("\n📋 Step 7: Confirming report...")
        
        confirm_response = await client.post(
            f"{API_URL}/reports/{report_id}/confirm",
            json={"notes": "Confirmed after field inspection"}
        )
        
        if confirm_response.status_code == 200:
            confirmed_report = confirm_response.json()
            print(f"✅ Report confirmed")
            print(f"   Status: {confirmed_report['status']}")
            print(f"   Confirmed at: {confirmed_report.get('confirmed_at', 'N/A')}")
        else:
            print(f"❌ Failed to confirm report: {confirm_response.status_code}")
        
        # 8. Example: Create report from session detection (simulated)
        print("\n📋 Step 8: Simulating detection-based report creation...")
        
        # This would normally happen automatically during detection
        # but we can also create reports manually from detection data
        
        auto_detection_data = {
            "class_id": 1,
            "class_name": "Block Crack",
            "confidence": 0.78,
            "bbox": [300.0, 200.0, 450.0, 280.0],
            "area": 12000.0,
            "center_x": 375.0,
            "center_y": 240.0
        }
        
        auto_report_request = {
            "detection": auto_detection_data,
            "metadata": {
                "session_id": session_id,
                "source": "detection_pipeline",
                "processing_time_ms": 234.5,
                "model_version": "YOLOv12s"
            },
            "description": "Automatically detected during camera scan",
            "severity": "medium"
        }
        
        auto_report_response = await client.post(
            f"{API_URL}/reports",
            json=auto_report_request
        )
        
        if auto_report_response.status_code == 200:
            auto_report = auto_report_response.json()
            print(f"✅ Auto-detection report created: {auto_report['id']}")
            print(f"   Class: {auto_report['detection']['class_name']}")
            print(f"   Confidence: {auto_report['detection']['confidence']}")
        else:
            print(f"❌ Failed to create auto-detection report: {auto_report_response.status_code}")
        
        # 9. Cleanup example (optional)
        print("\n📋 Step 9: Cleanup (optional)...")
        
        # In a real scenario, you might want to clean up test data
        # delete_response = await client.delete(f"{API_URL}/reports/{report_id}")
        # if delete_response.status_code == 200:
        #     print(f"✅ Report deleted: {report_id}")
        
        print("\n🎉 Report Management Example Complete!")
        print("=" * 50)
        print("The new report management system provides:")
        print("✅ Automatic report creation from detections")
        print("✅ Manual report creation and management")
        print("✅ Image upload and storage via Cloudinary")
        print("✅ Flexible filtering and search capabilities")
        print("✅ Report status management (pending → confirmed)")
        print("✅ Comprehensive statistics and analytics")
        print("✅ Integration with detection sessions")


async def example_file_upload_report():
    """Example of creating a report with file upload"""
    
    print("\n📁 File Upload Report Example")
    print("=" * 30)
    
    # Create a simple test image file
    import io
    from PIL import Image
    
    # Create a test image
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    detection_data = {
        "class_id": 0,
        "class_name": "Pothole",
        "confidence": 0.92,
        "bbox": [50.0, 25.0, 150.0, 125.0],
        "area": 10000.0,
        "center_x": 100.0,
        "center_y": 75.0
    }
    
    metadata = {
        "session_id": "file-upload-session",
        "source": "file_upload_example"
    }
    
    async with httpx.AsyncClient() as client:
        files = {
            'file': ('test_pothole.jpg', img_bytes, 'image/jpeg')
        }
        data = {
            'detection_data': json.dumps(detection_data),
            'metadata': json.dumps(metadata)
        }
        
        response = await client.post(
            f"{API_URL}/reports/upload",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            report = response.json()
            print(f"✅ File upload report created: {report['id']}")
            print(f"   Image URL: {report['image']['url'] if report['image'] else 'N/A'}")
        else:
            print(f"❌ File upload failed: {response.status_code}")
            print(response.text)


if __name__ == "__main__":
    print("🧪 Starting Hazard Detection Report Management Examples")
    print("Make sure the API server is running at:", API_URL)
    print()
    
    # Run the main example
    asyncio.run(example_report_management())
    
    # Uncomment to run file upload example
    # asyncio.run(example_file_upload_report())