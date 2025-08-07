"""
Cloudinary service for image upload and management
"""

import io
import base64
from typing import Optional, Dict, Any, Tuple
from PIL import Image
import cloudinary
import cloudinary.uploader
import cloudinary.api

from ..core.config import settings
from ..core.logging_config import get_logger
from ..models.report_models import ImageInfo

logger = get_logger("cloudinary_service")


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

    async def upload_image(
        self, 
        image_data: bytes, 
        filename: str,
        folder: str = "hazard-reports",
        create_thumbnail: bool = True
    ) -> ImageInfo:
        """
        Upload image to Cloudinary and return image information
        
        Args:
            image_data: Image data as bytes
            filename: Name for the uploaded file
            folder: Cloudinary folder to store the image
            create_thumbnail: Whether to create a thumbnail version
            
        Returns:
            ImageInfo: Information about the uploaded image
        """
        if not self.configured:
            raise ValueError("Cloudinary not configured")

        try:
            # Validate image data
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

            # Upload main image
            result = cloudinary.uploader.upload(
                image_data,
                **upload_options
            )

            # Create thumbnail if requested
            thumbnail_url = None
            if create_thumbnail:
                thumbnail_url = self._generate_thumbnail_url(result['public_id'], image_format)

            # Create ImageInfo object
            image_info = ImageInfo(
                url=result['secure_url'],
                public_id=result['public_id'],
                width=result['width'],
                height=result['height'],
                format=image_format,
                size_bytes=result.get('bytes', len(image_data)),
                thumbnail_url=thumbnail_url
            )

            logger.info(f"✅ Image uploaded successfully: {filename} ({image_info.width}x{image_info.height})")
            return image_info

        except Exception as e:
            logger.error(f"❌ Failed to upload image {filename}: {e}")
            raise

    async def upload_base64_image(
        self,
        base64_data: str,
        filename: str,
        folder: str = "hazard-reports",
        create_thumbnail: bool = True
    ) -> ImageInfo:
        """
        Upload base64 encoded image to Cloudinary
        
        Args:
            base64_data: Base64 encoded image data
            filename: Name for the uploaded file
            folder: Cloudinary folder to store the image
            create_thumbnail: Whether to create a thumbnail version
            
        Returns:
            ImageInfo: Information about the uploaded image
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