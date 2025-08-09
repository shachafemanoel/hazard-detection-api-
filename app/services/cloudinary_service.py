"""
Cloudinary service for image upload and management
Includes retries, validation, and MIME type sanitization (B4 requirements)
"""

import io
import base64
import time
import asyncio
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image
import cloudinary
import cloudinary.uploader
import cloudinary.api

from ..core.config import settings
from ..core.logging_config import get_logger

logger = get_logger("cloudinary_service")

# Allowed MIME types for uploaded images
ALLOWED_MIME_TYPES = {
    'image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif', 'image/bmp'
}

# Maximum file size in bytes (10MB as per config)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB


class CloudinaryService:
    """Service for managing image uploads to Cloudinary"""

    def __init__(self):
        self.configured = False
        self._configure_cloudinary()

    def _configure_cloudinary(self):
        """Configure Cloudinary with environment settings"""
        try:
            if all([settings.cloudinary_cloud_name, settings.cloudinary_api_key, settings.cloudinary_api_secret]):
                cloudinary.config(
                    cloud_name=settings.cloudinary_cloud_name,
                    api_key=settings.cloudinary_api_key,
                    api_secret=settings.cloudinary_api_secret,
                    secure=True
                )
                self.configured = True
                logger.info("✅ Cloudinary configured successfully")
            else:
                logger.warning("⚠️ Cloudinary credentials not found in environment")
        except Exception as e:
            logger.error(f"❌ Failed to configure Cloudinary: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get Cloudinary health status for /ready endpoint"""
        if not self.configured:
            return {
                "status": "not_configured",
                "configured": False,
                "error": "Cloudinary credentials not provided"
            }
        
        try:
            # Test connection by getting usage info
            usage = cloudinary.api.usage()
            return {
                "status": "healthy",
                "configured": True,
                "cloud_name": settings.cloudinary_cloud_name,
                "credits_used": usage.get('credits', {}).get('used_percent', 0),
                "storage_used_mb": usage.get('storage', {}).get('used', 0) / (1024 * 1024)
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "configured": True,
                "error": str(e)
            }
    
    def _validate_image_data(self, image_data: bytes, filename: str) -> Tuple[bool, str]:
        """
        Validate image data and MIME type (B4 requirement)
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file size
            if len(image_data) > settings.report_image_max_size_mb * 1024 * 1024:
                return False, f"File too large: {len(image_data)} bytes (max {settings.report_image_max_size_mb}MB)"
            
            # Validate image format using PIL
            try:
                image = Image.open(io.BytesIO(image_data))
                image_format = image.format
                
                if not image_format:
                    return False, "Cannot determine image format"
                
                # Check MIME type
                mime_type = f"image/{image_format.lower()}"
                if mime_type not in ALLOWED_MIME_TYPES:
                    return False, f"Unsupported image format: {image_format} (allowed: {', '.join(ALLOWED_MIME_TYPES)})"
                
                # Additional security checks
                if image.size[0] > 8192 or image.size[1] > 8192:
                    return False, f"Image dimensions too large: {image.size} (max 8192x8192)"
                
                if image.size[0] < 32 or image.size[1] < 32:
                    return False, f"Image dimensions too small: {image.size} (min 32x32)"
                
                return True, ""
                
            except Exception as e:
                return False, f"Invalid image data: {str(e)}"
                
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    async def _upload_with_retries(self, image_data: bytes, upload_options: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """
        Upload to Cloudinary with retry logic (B4 requirement)
        
        Args:
            image_data: Image bytes to upload
            upload_options: Cloudinary upload options
            max_retries: Maximum number of retry attempts
            
        Returns:
            Cloudinary upload result
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff
                    wait_time = (2 ** (attempt - 1)) * 1
                    logger.info(f"Retrying upload in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(wait_time)
                
                result = cloudinary.uploader.upload(image_data, **upload_options)
                
                if attempt > 0:
                    logger.info(f"✅ Upload succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Upload attempt {attempt + 1} failed: {str(e)}")
                
                # Don't retry on certain errors
                if "Invalid image file" in str(e) or "File size too large" in str(e):
                    break
        
        raise Exception(f"Upload failed after {max_retries + 1} attempts. Last error: {last_error}")

    async def upload_image(
        self, 
        image_data: bytes, 
        filename: str,
        folder: str = "hazard-reports",
        create_thumbnail: bool = True
    ) -> Dict[str, Any]:
        """
        Upload image to Cloudinary with validation and retries
        
        Args:
            image_data: Image data as bytes
            filename: Name for the uploaded file
            folder: Cloudinary folder to store the image
            create_thumbnail: Whether to create a thumbnail version
            
        Returns:
            Dict with url, public_id, width, height, etc.
        """
        if not self.configured:
            raise ValueError("Cloudinary not configured")

        try:
            # Validate image data and MIME type (B4 requirement)
            is_valid, error_msg = self._validate_image_data(image_data, filename)
            if not is_valid:
                raise ValueError(f"Image validation failed: {error_msg}")
            
            # Get image metadata
            image = Image.open(io.BytesIO(image_data))
            image_format = image.format.lower() if image.format else 'jpeg'
            
            # Prepare upload options
            upload_options = {
                'folder': folder,
                'public_id': filename,
                'resource_type': 'image',
                'format': image_format,
                'quality': 'auto:good',
                'fetch_format': 'auto',
                'flags': 'progressive',
                'overwrite': True
            }

            # Upload with retries (B4 requirement)
            result = await self._upload_with_retries(image_data, upload_options)

            # Create thumbnail if requested
            thumbnail_url = None
            if create_thumbnail:
                thumbnail_url = self._generate_thumbnail_url(result['public_id'], image_format)

            # Return secure URL and metadata
            upload_result = {
                'url': result['secure_url'],
                'cloudinaryUrl': result['secure_url'],  # B3 contract field
                'public_id': result['public_id'],
                'width': result['width'],
                'height': result['height'],
                'format': image_format,
                'size_bytes': result.get('bytes', len(image_data)),
                'thumbnail_url': thumbnail_url
            }

            logger.info(f"✅ Image uploaded successfully: {filename} ({result['width']}x{result['height']})")
            return upload_result

        except Exception as e:
            logger.error(f"❌ Failed to upload image {filename}: {e}")
            raise

    async def upload_base64_image(
        self,
        base64_data: str,
        filename: str,
        folder: str = "hazard-reports",
        create_thumbnail: bool = True
    ) -> Dict[str, Any]:
        """
        Upload base64 encoded image to Cloudinary
        
        Args:
            base64_data: Base64 encoded image data
            filename: Name for the uploaded file
            folder: Cloudinary folder to store the image
            create_thumbnail: Whether to create a thumbnail version
            
        Returns:
            Dict with upload information including cloudinaryUrl
        """
        try:
            # Decode base64 data
            if base64_data.startswith('data:image/'):
                # Remove data URL prefix if present
                base64_data = base64_data.split(',')[1]
            
            image_bytes = base64.b64decode(base64_data)
            return await self.upload_image(image_bytes, filename, folder, create_thumbnail)
            
        except Exception as e:
            logger.error(f"❌ Failed to decode and upload base64 image {filename}: {e}")
            raise
    
    async def upload_detection_blob(
        self,
        image_blob: bytes,
        detection_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Upload detection image blob and return result for B3 contract
        Used by POST /report endpoint
        
        Args:
            image_blob: Image data as bytes
            detection_meta: Metadata with sessionId, className, confidence, ts, geo
            
        Returns:
            Dict with id and cloudinaryUrl for createReport response
        """
        try:
            # Generate filename from metadata
            session_id = detection_meta.get('sessionId', 'unknown')
            class_name = detection_meta.get('className', 'detection')
            timestamp = detection_meta.get('ts', int(time.time() * 1000))
            
            filename = f"{session_id}_{class_name}_{timestamp}"
            
            # Upload image with retries and validation
            upload_result = await self.upload_image(
                image_blob, 
                filename, 
                folder="hazard-detections"
            )
            
            # Return format expected by B3 contract
            return {
                'id': upload_result['public_id'],
                'cloudinaryUrl': upload_result['cloudinaryUrl']
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to upload detection blob: {e}")
            raise

    def _generate_thumbnail_url(self, public_id: str, image_format: str) -> str:
        """Generate thumbnail URL with Cloudinary transformations"""
        try:
            thumbnail_url = cloudinary.CloudinaryImage(public_id).build_url(
                width=200,
                height=200,
                crop="fill",
                quality="auto:low",
                format=image_format,
                fetch_format="auto"
            )
            return thumbnail_url
        except Exception as e:
            logger.error(f"❌ Failed to generate thumbnail URL for {public_id}: {e}")
            return None

    async def delete_image(self, public_id: str) -> bool:
        """
        Delete image from Cloudinary
        
        Args:
            public_id: Cloudinary public ID of the image to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.configured:
            logger.warning("Cloudinary not configured, cannot delete image")
            return False

        try:
            result = cloudinary.uploader.destroy(public_id)
            success = result.get('result') == 'ok'
            
            if success:
                logger.info(f"✅ Image deleted successfully: {public_id}")
            else:
                logger.warning(f"⚠️ Image deletion may have failed: {public_id} - {result}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to delete image {public_id}: {e}")
            return False

    async def get_image_info(self, public_id: str) -> Optional[Dict[str, Any]]:
        """
        Get image information from Cloudinary
        
        Args:
            public_id: Cloudinary public ID of the image
            
        Returns:
            Dict containing image information or None if not found
        """
        if not self.configured:
            logger.warning("Cloudinary not configured")
            return None

        try:
            result = cloudinary.api.resource(public_id, resource_type="image")
            return result
            
        except cloudinary.exceptions.NotFound:
            logger.warning(f"Image not found: {public_id}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get image info for {public_id}: {e}")
            return None

    async def optimize_image_for_detection(self, image_data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """
        Optimize image for detection processing
        
        Args:
            image_data: Original image data
            
        Returns:
            Tuple of optimized image data and optimization info
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            original_size = image.size
            original_format = image.format
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large (max 2048px on longest side)
            max_size = 2048
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save optimized image
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=85, optimize=True)
            optimized_data = output_buffer.getvalue()
            
            optimization_info = {
                'original_size': original_size,
                'optimized_size': image.size,
                'original_format': original_format,
                'optimized_format': 'JPEG',
                'original_bytes': len(image_data),
                'optimized_bytes': len(optimized_data),
                'compression_ratio': len(optimized_data) / len(image_data)
            }
            
            logger.info(f"📷 Image optimized: {original_size} → {image.size}, "
                       f"{len(image_data)} → {len(optimized_data)} bytes "
                       f"({optimization_info['compression_ratio']:.2f}x)")
            
            return optimized_data, optimization_info
            
        except Exception as e:
            logger.error(f"❌ Failed to optimize image: {e}")
            # Return original data if optimization fails
            return image_data, {'error': str(e)}


# Global cloudinary service instance
cloudinary_service = CloudinaryService()