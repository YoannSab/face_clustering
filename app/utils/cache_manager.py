import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
from app.config import Config

logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k) if isinstance(k, np.integer) else k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

class EmbeddingCache:
    """Efficient caching system for face embeddings"""
    
    def __init__(self, cache_file: str = None):
        self.cache_file = cache_file or Config.EMBEDDINGS_FILE
        self.cache: Dict[str, Dict] = {}
        self.load_cache()
    
    def load_cache(self) -> None:
        """Load cache from disk"""
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                self.cache = data.get('embeddings', {})
                logger.info(f"Loaded {len(self.cache)} embeddings from cache")
        except FileNotFoundError:
            logger.info("No cache file found, starting with empty cache")
            self.cache = {}
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            self.cache = {}
            
    def save_cache(self) -> None:
        """Save cache to disk"""
        try:
            # Clean and convert all data before saving
            clean_data = {
                'embeddings': convert_numpy_types(self.cache),
                'last_updated': datetime.now().isoformat(),
                'version': '2.0'
            }
            with open(self.cache_file, 'w') as f:
                json.dump(clean_data, f, indent=2)
            logger.info(f"Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
            # Try to save without problematic entries
            try:
                clean_cache = {}
                for path, entry in self.cache.items():
                    try:
                        # Test if entry can be serialized
                        cleaned_entry = convert_numpy_types(entry)
                        json.dumps(cleaned_entry)
                        clean_cache[str(path)] = cleaned_entry
                    except Exception as entry_error:
                        logger.warning(f"Skipping problematic cache entry {path}: {entry_error}")
                
                fallback_data = {
                    'embeddings': clean_cache,
                    'last_updated': datetime.now().isoformat(),
                    'version': '2.0'
                }
                with open(self.cache_file, 'w') as f:
                    json.dump(fallback_data, f, indent=2)
                logger.info(f"Saved {len(clean_cache)} clean embeddings to cache (fallback)")
            except Exception as e2:
                logger.error(f"Failed to save even clean cache: {e2}")

    def get(self, image_path: str) -> Optional[List[Dict]]:
        """Get cached embeddings for image"""
        entry = self.cache.get(image_path)
        if entry and self._is_valid_entry(entry):
            return entry['detections']
        return None
    
    def set(self, image_path: str, detections: List[Dict]) -> None:
        """Cache embeddings for image"""
        import os
        try:
            # Convert detections to ensure JSON compatibility
            clean_detections = convert_numpy_types(detections)
            
            self.cache[str(image_path)] = {
                'detections': clean_detections,
                'timestamp': datetime.now().isoformat(),
                'file_size': os.path.getsize(image_path) if os.path.exists(image_path) else 0
            }
        except Exception as e:
            logger.error(f"Error caching embeddings for {image_path}: {e}")
    
    def _is_valid_entry(self, entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        try:
            timestamp = datetime.fromisoformat(entry['timestamp'])
            return datetime.now() - timestamp < Config.CACHE_TIMEOUT
        except:
            return False
    
    def cleanup_invalid_entries(self) -> None:
        """Remove invalid cache entries"""
        import os
        
        invalid_keys = []
        for path, entry in self.cache.items():
            if not os.path.exists(path) or not self._is_valid_entry(entry):
                invalid_keys.append(path)
        
        for key in invalid_keys:
            del self.cache[key]
        
        if invalid_keys:
            logger.info(f"Cleaned up {len(invalid_keys)} invalid cache entries")

class MetadataManager:
    """Manage EXIF metadata operations"""
    
    @staticmethod
    def add_face_tags(image_paths: List[str], face_name: str) -> bool:
        """Add face name to EXIF Subject field"""
        import subprocess
        
        try:
            chunk_size = Config.CHUNK_SIZE
            for i in range(0, len(image_paths), chunk_size):
                chunk = image_paths[i:i+chunk_size]
                subprocess.run([
                    "exiftool", 
                    "-overwrite_original", 
                    f"-Subject+={face_name}"
                ] + chunk, check=True, capture_output=True)
                
                logger.info(f"Tagged {min(i+chunk_size, len(image_paths))}/{len(image_paths)} images")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"ExifTool error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error adding metadata: {e}")
            return False
    
    @staticmethod
    def search_by_face(directory: str, face_names: List[str]) -> List[str]:
        """Search images by face names in EXIF data"""
        import subprocess
        from app.utils.image_processor import FileScanner
        
        try:
            image_paths = FileScanner.scan_directory(directory)
            matching_files = []
            
            chunk_size = Config.CHUNK_SIZE
            for i in range(0, len(image_paths), chunk_size):
                chunk = image_paths[i:i+chunk_size]
                
                result = subprocess.run([
                    "exiftool", "-Subject"
                ] + chunk, capture_output=True, text=True)
                
                current_file = None
                for line in result.stdout.split('\n'):
                    if line.startswith('========'):
                        current_file = line.split(' ')[1]
                    elif 'Subject' in line and current_file:
                        if any(name.lower() in line.lower() for name in face_names):
                            matching_files.append(current_file)
            
            return matching_files
            
        except Exception as e:
            logger.error(f"Error searching by face: {e}")
            return []
