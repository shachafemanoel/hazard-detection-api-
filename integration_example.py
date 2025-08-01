#!/usr/bin/env python3
"""
Example integration for accessing Hazard Detection API from Railway domain
This demonstrates how to integrate the API into another service or application.
"""

import asyncio
import aiohttp
import os
import json
from typing import List, Dict, Optional
from pathlib import Path

# Railway API Configuration
RAILWAY_API_URL = os.getenv('HAZARD_API_URL', 'https://your-project-name.up.railway.app')

class HazardDetectionAPI:
    """Client for Railway-deployed Hazard Detection API"""
    
    def __init__(self, api_url: str = RAILWAY_API_URL):
        self.api_url = api_url.rstrip('/')
        self.session_id = None
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session_id:
            await self.end_session()
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict:
        """Check API health and model status"""
        async with self.session.get(f"{self.api_url}/health") as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Health check failed: {response.status}")
    
    async def start_session(self) -> str:
        """Start a new detection session"""
        async with self.session.post(f"{self.api_url}/session/start") as response:
            if response.status == 200:
                data = await response.json()
                self.session_id = data['session_id']
                return self.session_id
            else:
                raise Exception(f"Failed to start session: {response.status}")
    
    async def detect_image(self, image_path: str) -> Dict:
        """Detect hazards in a single image"""
        if not self.session_id:
            await self.start_session()
        
        # Read image file
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Prepare form data
        form_data = aiohttp.FormData()
        form_data.add_field(
            'file', 
            image_data, 
            filename=Path(image_path).name,
            content_type='image/jpeg'
        )
        
        # Send detection request
        async with self.session.post(
            f"{self.api_url}/detect/{self.session_id}",
            data=form_data
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Detection failed: {response.status} - {error_text}")
    
    async def detect_multiple_images(self, image_paths: List[str]) -> List[Dict]:
        """Detect hazards in multiple images"""
        results = []
        
        for image_path in image_paths:
            try:
                result = await self.detect_image(image_path)
                results.append({
                    'image_path': image_path,
                    'success': True,
                    'data': result
                })
                print(f"✅ Processed {image_path}: {len(result.get('detections', []))} hazards found")
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'success': False,
                    'error': str(e)
                })
                print(f"❌ Failed to process {image_path}: {e}")
        
        return results
    
    async def get_session_summary(self) -> Dict:
        """Get session summary with all reports"""
        if not self.session_id:
            raise Exception("No active session")
        
        async with self.session.get(f"{self.api_url}/session/{self.session_id}/summary") as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Failed to get session summary: {response.status}")
    
    async def confirm_report(self, report_id: str) -> Dict:
        """Confirm a hazard report"""
        if not self.session_id:
            raise Exception("No active session")
        
        async with self.session.post(
            f"{self.api_url}/session/{self.session_id}/report/{report_id}/confirm"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Failed to confirm report: {response.status}")
    
    async def dismiss_report(self, report_id: str) -> Dict:
        """Dismiss a hazard report"""
        if not self.session_id:
            raise Exception("No active session")
        
        async with self.session.post(
            f"{self.api_url}/session/{self.session_id}/report/{report_id}/dismiss"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Failed to dismiss report: {response.status}")
    
    async def end_session(self) -> Dict:
        """End the current session"""
        if not self.session_id:
            return {"message": "No active session"}
        
        async with self.session.post(f"{self.api_url}/session/{self.session_id}/end") as response:
            data = await response.json() if response.status == 200 else {}
            self.session_id = None
            return data

# Example usage functions

async def basic_detection_example():
    """Basic example of detecting hazards in images"""
    print("🚀 Basic Detection Example")
    print("=" * 50)
    
    async with HazardDetectionAPI() as api:
        # Check API health
        try:
            health = await api.health_check()
            print(f"API Status: {health.get('status', 'unknown')}")
            print(f"Model Backend: {health.get('backend_type', 'unknown')}")
            print(f"Model Loaded: {health.get('backend_inference', False)}")
        except Exception as e:
            print(f"❌ API Health Check Failed: {e}")
            return
        
        # Example image paths (replace with your actual images)
        image_paths = [
            "road_image1.jpg",
            "road_image2.jpg",
            "pothole_image.jpg"
        ]
        
        # Filter to only existing files
        existing_images = [path for path in image_paths if Path(path).exists()]
        
        if not existing_images:
            print("⚠️  No image files found. Add some images to test.")
            print("Expected files:", image_paths)
            return
        
        # Process images
        results = await api.detect_multiple_images(existing_images)
        
        # Print results summary
        print("\n📊 Detection Results:")
        print("-" * 30)
        
        total_hazards = 0
        for result in results:
            if result['success']:
                hazard_count = len(result['data'].get('detections', []))
                total_hazards += hazard_count
                print(f"{Path(result['image_path']).name}: {hazard_count} hazards")
                
                # Show hazard details
                for detection in result['data'].get('detections', []):
                    confidence = detection['confidence'] * 100
                    print(f"  - {detection['class_name']}: {confidence:.1f}%")
            else:
                print(f"{Path(result['image_path']).name}: ERROR - {result['error']}")
        
        print(f"\nTotal hazards detected: {total_hazards}")
        
        # Get session summary
        try:
            summary = await api.get_session_summary()
            print(f"Session Summary:")
            print(f"  - Total detections: {summary.get('detection_count', 0)}")
            print(f"  - Unique hazards: {summary.get('unique_hazards', 0)}")
            print(f"  - Reports generated: {len(summary.get('reports', []))}")
        except Exception as e:
            print(f"Could not get session summary: {e}")

async def batch_processing_example():
    """Example of batch processing with error handling"""
    print("\n🔄 Batch Processing Example")
    print("=" * 50)
    
    # Simulate a batch of images from different sources
    image_batch = [
        "street_view_001.jpg",
        "drone_survey_002.jpg", 
        "mobile_capture_003.jpg",
        "dashcam_004.jpg",
        "inspection_005.jpg"
    ]
    
    async with HazardDetectionAPI() as api:
        try:
            # Check API availability first
            health = await api.health_check()
            
            if not health.get('backend_inference'):
                print("⚠️  Model not loaded, cannot perform detection")
                return
            
            print(f"Processing {len(image_batch)} images...")
            
            # Process in smaller batches for better error handling
            batch_size = 2
            all_results = []
            
            for i in range(0, len(image_batch), batch_size):
                batch = image_batch[i:i + batch_size]
                print(f"\nProcessing batch {i//batch_size + 1}...")
                
                batch_results = await api.detect_multiple_images(batch)
                all_results.extend(batch_results)
                
                # Small delay between batches
                await asyncio.sleep(0.5)
            
            # Analyze results
            successful = [r for r in all_results if r['success']]
            failed = [r for r in all_results if not r['success']]
            
            print(f"\n📈 Batch Processing Results:")
            print(f"✅ Successful: {len(successful)}")
            print(f"❌ Failed: {len(failed)}")
            
            if failed:
                print("\nFailed images:")
                for result in failed:
                    print(f"  - {result['image_path']}: {result['error']}")
            
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")

async def report_management_example():
    """Example of managing hazard reports"""
    print("\n📋 Report Management Example")
    print("=" * 50)
    
    async with HazardDetectionAPI() as api:
        # This would typically be called after detecting hazards
        # For demo purposes, we'll simulate the workflow
        
        try:
            # Start session
            session_id = await api.start_session()
            print(f"Started session: {session_id}")
            
            # Simulate detection that creates reports
            # (In real usage, you'd detect actual images here)
            
            # Get session summary to see any reports
            summary = await api.get_session_summary()
            reports = summary.get('reports', [])
            
            print(f"Found {len(reports)} reports in session")
            
            # Example of report management
            for report in reports[:3]:  # Process first 3 reports
                report_id = report['report_id']
                hazard_type = report['detection']['class_name']
                confidence = report['detection']['confidence']
                
                print(f"\nReport {report_id[:8]}...")
                print(f"  Hazard: {hazard_type}")
                print(f"  Confidence: {confidence:.2f}")
                
                # Example decision logic
                if confidence > 0.8:
                    await api.confirm_report(report_id)
                    print("  ✅ Confirmed (high confidence)")
                elif confidence < 0.6:
                    await api.dismiss_report(report_id)
                    print("  ❌ Dismissed (low confidence)")
                else:
                    print("  ⏳ Pending review")
            
        except Exception as e:
            print(f"❌ Report management failed: {e}")

async def integration_health_monitor():
    """Monitor API health for integration purposes"""
    print("\n🔍 Health Monitoring Example")
    print("=" * 50)
    
    async with HazardDetectionAPI() as api:
        try:
            health = await api.health_check()
            
            print("API Health Status:")
            print(f"  Status: {health.get('status', 'unknown')}")
            print(f"  Model Status: {health.get('model_status', 'unknown')}")
            print(f"  Backend: {health.get('backend_type', 'unknown')}")
            print(f"  Active Sessions: {health.get('active_sessions', 0)}")
            
            # Check device info
            device_info = health.get('device_info', {})
            if device_info:
                print(f"  Device: {device_info.get('device', 'unknown')}")
                print(f"  Input Shape: {device_info.get('input_shape', 'unknown')}")
            
            # Check environment
            env_info = health.get('environment', {})
            if env_info:
                print(f"  Platform: {env_info.get('platform', 'unknown')}")
                print(f"  Deployment: {env_info.get('deployment_env', 'unknown')}")
            
            # Check external services
            api_connectors = health.get('api_connectors', {})
            if api_connectors and 'services' in api_connectors:
                print("\nExternal Services:")
                for service, status in api_connectors['services'].items():
                    status_icon = "✅" if status.get('status') == 'healthy' else "⚠️"
                    print(f"  {status_icon} {service}: {status.get('status', 'unknown')}")
            
        except Exception as e:
            print(f"❌ Health monitoring failed: {e}")

async def main():
    """Run all examples"""
    print("🚀 Hazard Detection API Integration Examples")
    print("=" * 60)
    print(f"API URL: {RAILWAY_API_URL}")
    print("=" * 60)
    
    # Set your Railway API URL here
    if "your-project-name" in RAILWAY_API_URL:
        print("⚠️  Please update RAILWAY_API_URL with your actual Railway domain")
        print("   Set environment variable: export HAZARD_API_URL=https://your-actual-domain.up.railway.app")
        return
    
    # Run examples
    await integration_health_monitor()
    await basic_detection_example()
    await batch_processing_example()
    await report_management_example()
    
    print("\n✅ All examples completed!")

if __name__ == "__main__":
    # Configuration
    print("Configuration:")
    print(f"API URL: {RAILWAY_API_URL}")
    print("=" * 50)
    
    # Run examples
    asyncio.run(main())