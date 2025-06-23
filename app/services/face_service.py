import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional
import numpy as np
from keras_facenet import FaceNet
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from app.config import Config

logger = logging.getLogger(__name__)

class FaceDetectionService:
    """Service for face detection and embedding generation"""
    
    def __init__(self):
        self.embedder = FaceNet()
        self.executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)
    
    async def detect_faces_async(self, image_paths: List[str]) -> List[Dict]:
        """Asynchronously detect faces in multiple images"""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, self._detect_faces_single, path)
            for path in image_paths
        ]
        
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                detections = await task
                results.extend(detections)
                if i % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(image_paths)} images")
            except Exception as e:                
                logger.error(f"Error processing image: {e}")
                
        return results
    
    def _detect_faces_single(self, image_path: str) -> List[Dict]:
        """Detect faces in a single image"""
        from app.utils.image_processor import ImageProcessor
        from app.utils.cache_manager import convert_numpy_types
        
        try:
            processor = ImageProcessor()
            image = processor.load_image(image_path)
            
            if image is None:
                return []
            
            # Resize if needed
            if image.shape[0] > Config.MAX_IMAGE_SIZE or image.shape[1] > Config.MAX_IMAGE_SIZE:
                image = processor.resize_image(image, Config.MAX_IMAGE_SIZE)
            
            detections = self.embedder.extract(image, threshold=Config.FACE_DETECTION_THRESHOLD)
            
            for detection in detections:
                detection["image_path"] = image_path
                # Convert all numpy types to JSON-serializable types
                detection = convert_numpy_types(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error detecting faces in {image_path}: {e}")
            return []

class ClusteringService:
    """Service for face clustering operations"""
    
    @staticmethod
    def cluster_faces(
        detections: List[Dict], 
        algorithm: str = "dbscan",
        **kwargs
    ) -> Tuple[np.ndarray, List[str]]:
        """Cluster face embeddings using specified algorithm"""
        
        if not detections:
            return np.array([]), []
        
        try:
            embeddings = np.array([detection["embedding"] for detection in detections])
            paths = [detection["image_path"] for detection in detections]
            
            if algorithm == "dbscan":
                eps = kwargs.get('eps', Config.DEFAULT_DBSCAN_EPS)
                min_samples = kwargs.get('min_samples', Config.DEFAULT_DBSCAN_MIN_SAMPLES)
                clustering = DBSCAN(eps=eps, min_samples=min_samples)
                
            elif algorithm == "kmeans":
                n_clusters = kwargs.get('n_clusters', Config.DEFAULT_K_CLUSTERS)
                clustering = KMeans(n_clusters=n_clusters, random_state=42)
                
            elif algorithm == "hierarchical":
                n_clusters = kwargs.get('n_clusters', Config.DEFAULT_K_CLUSTERS)
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                
            else:
                raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
            
            labels = clustering.fit_predict(embeddings)
            return labels, paths
            
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return np.array([]), []
