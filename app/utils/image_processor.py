import io
import base64
import logging
from typing import Optional, Tuple, List
import numpy as np
import rawpy
from PIL import Image
import cv2
from app.config import Config

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Utility class for image processing operations"""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """Load image from path, supporting various formats including RAW"""
        try:
            if image_path.lower().endswith('cr2'):
                with rawpy.imread(image_path) as raw:
                    return raw.postprocess()
            else:
                image = Image.open(image_path)
                return np.asarray(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    @staticmethod
    def resize_image(image: np.ndarray, max_size: int) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        height, width = image.shape[:2]
        
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
        return cv2.resize(image, (new_width, new_height))
    
    @staticmethod
    def crop_face(image_path: str, face_box: List[int]) -> Tuple[str, bool]:
        """Extract face from image and convert to base64"""
        try:            
            # Load image
            if image_path.lower().endswith('cr2'):
                with rawpy.imread(image_path) as raw:
                    image_array = raw.postprocess()
                    image = Image.fromarray(image_array)
            else:
                image = Image.open(image_path)
            
            if image is None:
                return "", False
            
            # Resize if needed
            if image.size[0] > Config.MAX_IMAGE_SIZE or image.size[1] > Config.MAX_IMAGE_SIZE:
                scaling_factor = Config.MAX_IMAGE_SIZE / max(image.size)
                new_size = (int(image.size[0] * scaling_factor), int(image.size[1] * scaling_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Crop face - ensure coordinates are integers
            x, y, w, h = [int(coord) for coord in face_box]
            cropped_image = image.crop((x, y, x+w, y+h))
            
            # Convert to base64
            buffered = io.BytesIO()
            cropped_image.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return img_str, True
            
        except Exception as e:
            logger.error(f"Error cropping face from {image_path}: {e}")
            return "", False

class FileScanner:
    """Utility class for scanning and filtering image files"""
    
    @staticmethod
    def scan_directory(directory: str) -> List[str]:
        """Recursively scan directory for supported image files"""
        import os
        
        image_paths = []
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if FileScanner._is_supported_format(file):
                        image_paths.append(os.path.join(root, file))
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
        
        return image_paths
    
    @staticmethod
    def _is_supported_format(filename: str) -> bool:
        """Check if file has supported image format"""
        extension = filename.lower().split('.')[-1]
        return extension in Config.SUPPORTED_FORMATS
