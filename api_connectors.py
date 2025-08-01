"""
API Connectors for Hazard Detection Backend
Provides integrations with external services like geocoding, caching, and image storage.
"""

import os
import asyncio
import aiohttp
import json
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ApiResponse:
    """Standard API response wrapper"""
    def __init__(self, success: bool = False, data: Any = None, error: str = None):
        self.success = success
        self.data = data
        self.error = error

class GoogleMapsConnector:
    """Google Maps API connector for geocoding services"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GOOGLE_MAPS_API_KEY')
        self.base_url = "https://maps.googleapis.com/maps/api"
        
    async def geocode(self, address: str) -> ApiResponse:
        """Geocode an address to coordinates"""
        if not self.api_key:
            return ApiResponse(
                success=False, 
                error="Google Maps API key not configured"
            )
        
        try:
            url = f"{self.base_url}/geocode/json"
            params = {
                'address': address,
                'key': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['status'] == 'OK' and data.get('results'):
                            result = data['results'][0]
                            location = result['geometry']['location']
                            return ApiResponse(
                                success=True,
                                data={
                                    'lat': location['lat'],
                                    'lng': location['lng'],
                                    'formatted_address': result['formatted_address'],
                                    'place_id': result.get('place_id')
                                }
                            )
                        else:
                            return ApiResponse(
                                success=False,
                                error=f"Geocoding failed: {data.get('status', 'Unknown error')}"
                            )
                    else:
                        return ApiResponse(
                            success=False,
                            error=f"HTTP {response.status}: {await response.text()}"
                        )
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            return ApiResponse(success=False, error=str(e))
    
    async def reverse_geocode(self, lat: float, lng: float) -> ApiResponse:
        """Reverse geocode coordinates to address"""
        if not self.api_key:
            return ApiResponse(
                success=False, 
                error="Google Maps API key not configured"
            )
        
        try:
            url = f"{self.base_url}/geocode/json"
            params = {
                'latlng': f"{lat},{lng}",
                'key': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['status'] == 'OK' and data.get('results'):
                            result = data['results'][0]
                            return ApiResponse(
                                success=True,
                                data={
                                    'formatted_address': result['formatted_address'],
                                    'place_id': result.get('place_id'),
                                    'address_components': result.get('address_components', [])
                                }
                            )
                        else:
                            return ApiResponse(
                                success=False,
                                error=f"Reverse geocoding failed: {data.get('status', 'Unknown error')}"
                            )
                    else:
                        return ApiResponse(
                            success=False,
                            error=f"HTTP {response.status}: {await response.text()}"
                        )
        except Exception as e:
            logger.error(f"Reverse geocoding error: {e}")
            return ApiResponse(success=False, error=str(e))

class RedisConnector:
    """Redis connector for caching detection results"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis_client = None
        
    async def _get_client(self):
        """Get or create Redis client"""
        if self.redis_client is None:
            try:
                import aioredis
                self.redis_client = aioredis.from_url(self.redis_url)
            except ImportError:
                logger.warning("aioredis not available, caching disabled")
                return None
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                return None
        return self.redis_client
    
    async def cache_detection(self, detection_id: str, detection_data: dict, ttl: int = 3600) -> ApiResponse:
        """Cache detection result with TTL"""
        client = await self._get_client()
        if not client:
            return ApiResponse(
                success=False,
                error="Redis client not available"
            )
        
        try:
            data_json = json.dumps(detection_data, default=str)
            await client.setex(f"detection:{detection_id}", ttl, data_json)
            return ApiResponse(
                success=True,
                data={"cached": True, "ttl": ttl}
            )
        except Exception as e:
            logger.error(f"Caching error: {e}")
            return ApiResponse(success=False, error=str(e))
    
    async def get_cached_detection(self, detection_id: str) -> ApiResponse:
        """Retrieve cached detection result"""
        client = await self._get_client()
        if not client:
            return ApiResponse(
                success=False,
                error="Redis client not available"
            )
        
        try:
            cached_data = await client.get(f"detection:{detection_id}")
            if cached_data:
                data = json.loads(cached_data)
                return ApiResponse(success=True, data=data)
            else:
                return ApiResponse(
                    success=False,
                    error="Detection not found in cache"
                )
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return ApiResponse(success=False, error=str(e))

class CloudinaryConnector:
    """Cloudinary connector for image storage"""
    
    def __init__(self, cloud_name: str = None, api_key: str = None, api_secret: str = None):
        self.cloud_name = cloud_name or os.getenv('CLOUDINARY_CLOUD_NAME')
        self.api_key = api_key or os.getenv('CLOUDINARY_API_KEY')
        self.api_secret = api_secret or os.getenv('CLOUDINARY_API_SECRET')
        self.base_url = f"https://api.cloudinary.com/v1_1/{self.cloud_name}"
    
    async def upload_image(self, image_data: str, public_id: str = None) -> ApiResponse:
        """Upload base64 image to Cloudinary"""
        if not all([self.cloud_name, self.api_key, self.api_secret]):
            return ApiResponse(
                success=False,
                error="Cloudinary credentials not configured"
            )
        
        try:
            import hashlib
            import hmac
            from time import time
            
            timestamp = int(time())
            params = {
                'timestamp': timestamp,
                'upload_preset': 'hazard_detection'  # You need to create this preset in Cloudinary
            }
            
            if public_id:
                params['public_id'] = public_id
            
            # Generate signature
            params_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
            signature = hmac.new(
                self.api_secret.encode(),
                params_string.encode(),
                hashlib.sha1
            ).hexdigest()
            
            # Prepare form data
            data = {
                'file': f"data:image/jpeg;base64,{image_data}",
                'api_key': self.api_key,
                'signature': signature,
                **params
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/image/upload", data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return ApiResponse(
                            success=True,
                            data={
                                'url': result['secure_url'],
                                'public_id': result['public_id'],
                                'asset_id': result.get('asset_id'),
                                'version': result.get('version')
                            }
                        )
                    else:
                        error_text = await response.text()
                        return ApiResponse(
                            success=False,
                            error=f"Upload failed: {error_text}"
                        )
        except Exception as e:
            logger.error(f"Image upload error: {e}")
            return ApiResponse(success=False, error=str(e))

class RenderConnector:
    """Render.com API connector for deployment management"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('RENDER_API_KEY')
        self.base_url = "https://api.render.com/v1"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}' if self.api_key else '',
            'Content-Type': 'application/json'
        }
    
    async def get_services(self) -> ApiResponse:
        """Get all services in the account"""
        if not self.api_key:
            return ApiResponse(
                success=False,
                error="Render API key not configured"
            )
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/services", headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return ApiResponse(success=True, data=data)
                    else:
                        error_text = await response.text()
                        return ApiResponse(
                            success=False,
                            error=f"Failed to get services: {error_text}"
                        )
        except Exception as e:
            logger.error(f"Render API error: {e}")
            return ApiResponse(success=False, error=str(e))

class ApiManager:
    """Central API manager for all external services"""
    
    def __init__(self):
        self.google_maps = GoogleMapsConnector()
        self.redis = RedisConnector()
        self.cloudinary = CloudinaryConnector()
        self.render = RenderConnector()
    
    async def health_check(self) -> dict:
        """Check health of all configured services"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'services': {}
        }
        
        # Check Google Maps
        if self.google_maps.api_key:
            # Test with a simple geocoding request
            test_result = await self.google_maps.geocode("New York, NY")
            health_status['services']['google_maps'] = {
                'status': 'healthy' if test_result.success else 'error',
                'message': test_result.error if not test_result.success else 'OK'
            }
        else:
            health_status['services']['google_maps'] = {
                'status': 'disabled',
                'message': 'API key not configured'
            }
        
        # Check Redis
        redis_client = await self.redis._get_client()
        if redis_client:
            try:
                await redis_client.ping()
                health_status['services']['redis'] = {
                    'status': 'healthy',
                    'message': 'Connection OK'
                }
            except Exception as e:
                health_status['services']['redis'] = {
                    'status': 'error',
                    'message': str(e)
                }
        else:
            health_status['services']['redis'] = {
                'status': 'disabled',
                'message': 'Redis not available'
            }
        
        # Check Cloudinary
        if all([self.cloudinary.cloud_name, self.cloudinary.api_key, self.cloudinary.api_secret]):
            health_status['services']['cloudinary'] = {
                'status': 'configured',
                'message': 'Credentials available'
            }
        else:
            health_status['services']['cloudinary'] = {
                'status': 'disabled',
                'message': 'Credentials not configured'
            }
        
        # Check Render
        if self.render.api_key:
            render_result = await self.render.get_services()
            health_status['services']['render'] = {
                'status': 'healthy' if render_result.success else 'error',
                'message': render_result.error if not render_result.success else 'OK'
            }
        else:
            health_status['services']['render'] = {
                'status': 'disabled',
                'message': 'API key not configured'
            }
        
        return health_status

# Global instances
api_manager = ApiManager()

# Convenience functions for backward compatibility
async def geocode_location(address: str) -> ApiResponse:
    """Geocode an address to coordinates"""
    return await api_manager.google_maps.geocode(address)

async def reverse_geocode_location(lat: float, lng: float) -> ApiResponse:
    """Reverse geocode coordinates to address"""
    return await api_manager.google_maps.reverse_geocode(lat, lng)

async def cache_detection_result(detection_id: str, detection_data: dict) -> ApiResponse:
    """Cache detection result"""
    return await api_manager.redis.cache_detection(detection_id, detection_data)

async def upload_detection_image(image_data: str, public_id: str = None) -> ApiResponse:
    """Upload detection image to cloud storage"""
    return await api_manager.cloudinary.upload_image(image_data, public_id)