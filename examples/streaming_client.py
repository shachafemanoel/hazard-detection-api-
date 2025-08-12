#!/usr/bin/env python3
"""
Example client for testing the streaming detection API
Demonstrates both WebSocket and SSE connection types
"""

import asyncio
import websockets
import aiohttp
import json
import base64
import time
from pathlib import Path
from typing import Optional

# Configuration
API_BASE_URL = "http://localhost:8080"
WS_BASE_URL = "ws://localhost:8080"
TEST_IMAGE_PATH = "test_image.jpg"  # You can add a test image here


class StreamingClient:
    """Client for testing streaming detection API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def load_test_image(self, image_path: str) -> str:
        """Load and encode test image"""
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
            return image_data
        except FileNotFoundError:
            # Create a simple test image if none exists
            from PIL import Image
            import io
            
            # Create a simple test image
            img = Image.new('RGB', (640, 640), color='red')
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            image_data = base64.b64encode(buffer.getvalue()).decode()
            return image_data
    
    async def test_websocket_streaming(self, fps_limit: int = 5):
        """Test WebSocket streaming connection"""
        print(f"🔗 Testing WebSocket streaming at {fps_limit} FPS...")
        
        # Load test image
        image_data = self.load_test_image(TEST_IMAGE_PATH)
        
        ws_url = f"{WS_BASE_URL.replace('http://', 'ws://')}/stream/detection?fps_limit={fps_limit}"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                print("✅ WebSocket connected")
                
                # Wait for welcome message
                welcome_msg = await websocket.recv()
                print(f"📨 Welcome: {json.loads(welcome_msg)}")
                
                # Send test frames
                for i in range(10):
                    frame_data = {
                        "type": "frame",
                        "image_data": image_data,
                        "frame_id": f"test_frame_{i}",
                        "timestamp": time.time(),
                        "metadata": {"test": True}
                    }
                    
                    print(f"📤 Sending frame {i+1}")
                    await websocket.send(json.dumps(frame_data))
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        result = json.loads(response)
                        
                        if result.get("type") == "detection_result":
                            data = result.get("data", {})
                            detections = data.get("detections", [])
                            processing_time = data.get("processing_time_ms", 0)
                            
                            print(f"✅ Frame {i+1}: {len(detections)} detections, {processing_time:.1f}ms")
                            
                            # Print first detection if any
                            if detections:
                                det = detections[0]
                                print(f"   🎯 {det['class_name']}: {det['confidence']:.3f}")
                        
                    except asyncio.TimeoutError:
                        print(f"⏱️  Frame {i+1}: Response timeout")
                    
                    # Rate limiting
                    await asyncio.sleep(1.0 / fps_limit)
                
                print("🏁 WebSocket streaming test completed")
                
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
    
    async def test_sse_streaming(self, fps_limit: int = 5):
        """Test SSE streaming connection"""
        print(f"📡 Testing SSE streaming at {fps_limit} FPS...")
        
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        # Load test image
        image_data = self.load_test_image(TEST_IMAGE_PATH)
        
        try:
            # Create streaming session
            print("📝 Creating streaming session...")
            async with self.session.post(
                f"{self.base_url}/stream/sessions",
                json={
                    "fps_limit": fps_limit,
                    "confidence_threshold": 0.5,
                    "enable_tracking": True,
                    "quality_mode": "balanced"
                }
            ) as resp:
                if resp.status != 200:
                    print(f"❌ Failed to create session: {resp.status}")
                    return
                
                session_data = await resp.json()
                print(f"✅ Session created: {session_data['session_id']}")
                print(f"📡 SSE endpoint: {session_data['sse_endpoint']}")
                print(f"📤 Upload endpoint: {session_data['upload_endpoint']}")
            
            client_id = session_data["client_id"]
            sse_url = f"{self.base_url}{session_data['sse_endpoint']}"
            upload_url = f"{self.base_url}{session_data['upload_endpoint']}"
            
            # Start SSE connection
            print("📡 Connecting to SSE endpoint...")
            
            async def listen_to_sse():
                """Listen to SSE events"""
                try:
                    async with self.session.get(sse_url) as resp:
                        if resp.status != 200:
                            print(f"❌ SSE connection failed: {resp.status}")
                            return
                        
                        print("✅ SSE connected")
                        
                        async for line in resp.content:
                            line = line.decode('utf-8').strip()
                            
                            if line.startswith('data: '):
                                data_str = line[6:]  # Remove 'data: ' prefix
                                try:
                                    event_data = json.loads(data_str)
                                    event_type = event_data.get("type", "unknown")
                                    
                                    if event_type == "detection_result":
                                        data = event_data.get("data", {})
                                        detections = data.get("detections", [])
                                        processing_time = data.get("processing_time_ms", 0)
                                        frame_id = data.get("frame_id", "unknown")
                                        
                                        print(f"✅ {frame_id}: {len(detections)} detections, {processing_time:.1f}ms")
                                        
                                        # Print first detection if any
                                        if detections:
                                            det = detections[0]
                                            print(f"   🎯 {det['class_name']}: {det['confidence']:.3f}")
                                    
                                    elif event_type == "stream_started":
                                        print(f"🚀 Stream started for session: {event_data.get('session_id')}")
                                    
                                    elif event_type == "ping":
                                        print("💓 Keepalive ping received")
                                    
                                    elif event_type == "error":
                                        print(f"❌ Error: {event_data.get('error')}")
                                        
                                except json.JSONDecodeError:
                                    print(f"⚠️  Invalid JSON in SSE data: {data_str}")
                
                except Exception as e:
                    print(f"❌ SSE listening error: {e}")
            
            # Start SSE listener
            sse_task = asyncio.create_task(listen_to_sse())
            
            # Give SSE time to connect
            await asyncio.sleep(1.0)
            
            # Send test frames
            print("📤 Sending test frames...")
            for i in range(10):
                frame_data = {
                    "image_data": image_data,
                    "frame_id": f"sse_frame_{i}",
                    "timestamp": time.time(),
                    "metadata": {"test": True, "method": "sse"}
                }
                
                print(f"📤 Sending frame {i+1}")
                
                async with self.session.post(upload_url, json=frame_data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(f"   ✅ Queued: {result.get('status')}")
                    else:
                        print(f"   ❌ Upload failed: {resp.status}")
                
                # Rate limiting
                await asyncio.sleep(1.0 / fps_limit)
            
            # Wait for processing to complete
            print("⏱️  Waiting for processing to complete...")
            await asyncio.sleep(3.0)
            
            # Cancel SSE listener
            sse_task.cancel()
            try:
                await sse_task
            except asyncio.CancelledError:
                pass
            
            print("🏁 SSE streaming test completed")
            
        except Exception as e:
            print(f"❌ SSE streaming error: {e}")
    
    async def test_session_stats(self):
        """Test session statistics endpoint"""
        print("📊 Testing session statistics...")
        
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        try:
            # Get global stats
            async with self.session.get(f"{self.base_url}/stream/stats") as resp:
                if resp.status == 200:
                    stats = await resp.json()
                    print("📊 Global streaming stats:")
                    print(f"   Active sessions: {stats['streaming_stats']['active_sessions']}")
                    print(f"   Total sessions: {stats['streaming_stats']['total_sessions']}")
                    print(f"   Frames processed: {stats['streaming_stats']['frames_processed']}")
                    print(f"   Average FPS: {stats['streaming_stats']['avg_fps']}")
                else:
                    print(f"❌ Failed to get stats: {resp.status}")
            
        except Exception as e:
            print(f"❌ Stats error: {e}")
    
    async def test_health_check(self):
        """Test streaming service health"""
        print("🏥 Testing health check...")
        
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        try:
            async with self.session.get(f"{self.base_url}/stream/health") as resp:
                if resp.status == 200:
                    health = await resp.json()
                    print("🏥 Streaming service health:")
                    print(f"   Status: {health['status']}")
                    print(f"   Model ready: {health['model_service']['model_loaded']}")
                    print(f"   Active sessions: {health['streaming_service']['active_sessions']}")
                else:
                    print(f"❌ Health check failed: {resp.status}")
        
        except Exception as e:
            print(f"❌ Health check error: {e}")


async def main():
    """Run streaming API tests"""
    print("🧪 Starting streaming API tests...")
    
    # Test connection types
    test_websocket = True
    test_sse = True
    fps_limit = 5
    
    async with StreamingClient() as client:
        # Health check first
        await client.test_health_check()
        print()
        
        # Test WebSocket streaming
        if test_websocket:
            await client.test_websocket_streaming(fps_limit=fps_limit)
            print()
        
        # Test SSE streaming
        if test_sse:
            await client.test_sse_streaming(fps_limit=fps_limit)
            print()
        
        # Check final stats
        await client.test_session_stats()
    
    print("🎉 All tests completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")