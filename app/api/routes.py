import asyncio
import logging
from flask import request, jsonify, current_app
from app.api import bp
from app.services.face_service import FaceDetectionService, ClusteringService
from app.utils.image_processor import FileScanner, ImageProcessor
from app.utils.cache_manager import EmbeddingCache, MetadataManager

logger = logging.getLogger(__name__)

# Global services
face_service = FaceDetectionService()
embedding_cache = EmbeddingCache()

@bp.route('/images/count', methods=['POST'])
def get_image_count():
    """Get number of images in directory"""
    try:
        data = request.get_json()
        if not data or 'directory' not in data:
            return jsonify({'error': 'Directory path required'}), 400
        
        directory = data['directory']
        image_paths = FileScanner.scan_directory(directory)
        
        return jsonify({
            'count': len(image_paths),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error counting images: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/faces/cluster', methods=['POST'])
def cluster_faces():
    """Main clustering endpoint with async processing"""
    try:
        data = request.get_json()
        
        # Validate request
        if not data or 'directory' not in data:
            return jsonify({'error': 'Directory path required'}), 400
        
        directory = data['directory']
        algorithm = data.get('algorithm', 'dbscan')
        
        # Algorithm-specific parameters
        params = {}
        if algorithm == 'dbscan':
            params['eps'] = float(data.get('eps', 0.65))
            params['min_samples'] = int(data.get('min_samples', 3))
        elif algorithm in ['kmeans', 'hierarchical']:
            params['n_clusters'] = int(data.get('n_clusters', 20))
        
        # Get image paths
        image_paths = FileScanner.scan_directory(directory)
        if not image_paths:
            return jsonify({'error': 'No images found in directory'}), 404
        
        # Process faces (check cache first)
        all_detections = []
        uncached_paths = []
        
        for path in image_paths:
            cached_detections = embedding_cache.get(path)
            if cached_detections is not None:
                all_detections.extend(cached_detections)
            else:
                uncached_paths.append(path)
        
        # Process uncached images
        if uncached_paths:
            logger.info(f"Processing {len(uncached_paths)} uncached images")
            
            # Use async processing for better performance
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                new_detections = loop.run_until_complete(
                    face_service.detect_faces_async(uncached_paths)
                )
                
                # Cache new detections
                for detection in new_detections:
                    path = detection['image_path']
                    path_detections = [d for d in new_detections if d['image_path'] == path]
                    embedding_cache.set(path, path_detections)
                
                all_detections.extend(new_detections)
                
            finally:
                loop.close()
        
        if not all_detections:
            return jsonify({'error': 'No faces detected in images'}), 404
        
        # Perform clustering
        labels, paths = ClusteringService.cluster_faces(
            all_detections, algorithm, **params
        )
        
        # Create cluster results
        clusters = _create_cluster_results(labels, paths, all_detections)
        
        # Save cache
        embedding_cache.save_cache()
          # Calculate statistics
        total_faces = int(len(labels))
        clustered_faces = int(len([l for l in labels if l != -1]))
        num_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
        
        return jsonify({
            'clusters': clusters,
            'statistics': {
                'total_faces': total_faces,
                'clustered_faces': clustered_faces,
                'num_clusters': num_clusters,
                'clustering_rate': float(clustered_faces / total_faces if total_faces > 0 else 0)
            },
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in face clustering: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/metadata/add', methods=['POST'])
def add_metadata():
    """Add face names to image metadata"""
    try:
        data = request.get_json()
        logger.info(f"Received metadata request: {data}")

        if not data:
            return jsonify({'error': 'Request data required'}), 400
        
        success_count = 0
        total_count = 0
        
        for cluster_id, cluster_data in data.items():
            face_name = cluster_data.get('name')
            image_paths = cluster_data.get('paths', [])
            
            if face_name and image_paths:
                total_count += 1
                if MetadataManager.add_face_tags(image_paths, face_name):
                    success_count += 1
        
        return jsonify({
            'success_count': success_count,
            'total_count': total_count,
            'status': 'success' if success_count == total_count else 'partial'
        })
        
    except Exception as e:
        logger.error(f"Error adding metadata: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/search/faces', methods=['POST'])
def search_faces():
    """Search images by face names"""
    try:
        data = request.get_json()
        
        if not data or 'directory' not in data or 'face_names' not in data:
            return jsonify({'error': 'Directory and face_names required'}), 400
        
        directory = data['directory']
        face_names = data['face_names']
        
        if isinstance(face_names, str):
            face_names = [name.strip() for name in face_names.split(',')]
        
        matching_files = MetadataManager.search_by_face(directory, face_names)
        
        return jsonify({
            'matches': matching_files,
            'count': len(matching_files),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error searching faces: {e}")
        return jsonify({'error': str(e)}), 500

def _create_cluster_results(labels, paths, detections):
    """Create formatted cluster results for frontend"""
    from app.config import Config
    from app.utils.cache_manager import convert_numpy_types
    
    clusters = {}
    
    for i, label in enumerate(labels):
        # Convert numpy int64 to regular int
        label = int(label) if hasattr(label, 'item') else label
        
        if label == -1:  # Noise/unclustered
            continue
        
        if label not in clusters:
            clusters[label] = {
                'faces': [],
                'paths': [],
                'count': 0
            }
        
        clusters[label]['paths'].append(paths[i])
        clusters[label]['count'] += 1
        
        # Add face images (limited number for performance)
        if len(clusters[label]['faces']) < Config.MAX_FACES_PER_CLUSTER:
            face_image, success = ImageProcessor.crop_face(
                paths[i], detections[i]['box']
            )
            if success:
                clusters[label]['faces'].append(face_image)
    
    # Convert all numpy types in the clusters dictionary
    return convert_numpy_types(clusters)
