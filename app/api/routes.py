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

# Global progress tracking
progress_data = {}

# Store extracted faces data for clustering step
extracted_faces_cache = {}

def update_progress(stage, percentage, message, **kwargs):
    """Update progress for a specific stage"""
    progress_data[stage] = {
        'percentage': percentage,
        'message': message,
        **kwargs  # Additional data like processed count, faces found, etc.
    }

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

@bp.route('/faces/extract', methods=['POST'])
def extract_faces():
    """Extract faces and compute embeddings (Step 1)"""
    try:
        data = request.get_json()
        
        # Validate request
        if not data or 'directory' not in data:
            return jsonify({'error': 'Directory path required'}), 400
        
        directory = data['directory']
        
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
        
        new_embeddings = 0
        # Process uncached images
        if uncached_paths:
            logger.info(f"Processing {len(uncached_paths)} uncached images")
            update_progress('extraction', 10, f'Traitement de {len(uncached_paths)} images...')
            
            # Use async processing for better performance
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                new_detections = []
                total_images = len(uncached_paths)
                
                # Process images in batches to update progress
                batch_size = max(1, total_images // 10)  # Process in 10% increments
                for i in range(0, total_images, batch_size):
                    batch = uncached_paths[i:i + batch_size]
                    
                    # Update progress
                    progress_pct = int(10 + (i / total_images) * 80)  # 10% to 90%
                    update_progress('extraction', progress_pct, 
                                  f'Traitement: {i + len(batch)}/{total_images} images',
                                  processed=i + len(batch),
                                  faces_found=len(new_detections))
                    
                    # Process this batch
                    batch_detections = loop.run_until_complete(
                        face_service.detect_faces_async(batch)
                    )
                    new_detections.extend(batch_detections)
                
                update_progress('extraction', 95, 'Mise en cache des résultats...')
                
                # Cache new detections
                for detection in new_detections:
                    path = detection['image_path']
                    path_detections = [d for d in new_detections if d['image_path'] == path]
                    embedding_cache.set(path, path_detections)
                
                all_detections.extend(new_detections)
                new_embeddings = len(new_detections)
                
            finally:
                loop.close()
        else:
            update_progress('extraction', 90, 'Utilisation du cache existant...')
        
        if not all_detections:
            return jsonify({'error': 'No faces detected in images'}), 404
        
        update_progress('extraction', 100, 'Extraction terminée !')
        
        # Store detections in server cache for clustering step
        cache_key = f"extracted_{hash(directory)}"
        extracted_faces_cache[cache_key] = all_detections
        
        # Return extraction results (without the actual detection data)
        return jsonify({
            'status': 'success',
            'total_faces': len(all_detections),
            'total_images': len(image_paths),
            'new_embeddings': new_embeddings,
            'cached_embeddings': len(all_detections) - new_embeddings,
            'vector_dimensions': 512,  # FaceNet embedding dimension
            'cache_key': cache_key  # Key to retrieve data for clustering
        })
        
    except Exception as e:
        logger.error(f"Error extracting faces: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/faces/cluster', methods=['POST'])
def cluster_faces_from_data():
    """Cluster faces from pre-extracted data (Step 2)"""
    try:
        data = request.get_json()
        
        # Validate request
        if not data or 'cache_key' not in data:
            return jsonify({'error': 'Cache key required'}), 400
        
        cache_key = data['cache_key']
        algorithm = data.get('algorithm', 'dbscan')
        
        # Get faces data from server cache
        if cache_key not in extracted_faces_cache:
            return jsonify({'error': 'Extracted faces data not found. Please run extraction first.'}), 404
        
        faces_data = extracted_faces_cache[cache_key]
        
        # Algorithm-specific parameters
        params = {}
        if algorithm == 'dbscan':
            params['eps'] = float(data.get('eps', 0.65))
            params['min_samples'] = int(data.get('min_samples', 3))
        elif algorithm in ['kmeans', 'hierarchical']:
            params['n_clusters'] = int(data.get('n_clusters', 20))
        
        if not faces_data:
            return jsonify({'error': 'No faces data available'}), 404
        
        update_progress('clustering', 10, 'Initialisation du clustering...',
                       total_faces=len(faces_data))
        
        # Perform clustering on the provided faces data
        labels, paths = ClusteringService.cluster_faces(
            faces_data, algorithm, **params
        )
        
        update_progress('clustering', 70, 'Organisation des clusters...')
        
        # Organize results by cluster
        clusters = ClusteringService.organize_clusters(faces_data, labels, paths)
        
        # Calculate statistics
        total_faces = len(faces_data)
        clustered_faces = sum(1 for label in labels if label != -1)
        unique_clusters = len(set(label for label in labels if label != -1))

        update_progress('clustering', 90, 'Calcul des statistiques...',
                       clustered=clustered_faces,
                       total_faces=total_faces)
        
        clustering_rate = clustered_faces / total_faces if total_faces > 0 else 0
        
        statistics = {
            'total_faces': total_faces,
            'clustered_faces': clustered_faces,
            'num_clusters': unique_clusters,
            'clustering_rate': clustering_rate,
            'noise_points': total_faces - clustered_faces
        }
        
        update_progress('clustering', 100, 'Clustering terminé !')
        
        return jsonify({
            'status': 'success',
            'clusters': clusters,
            'statistics': statistics,
            'algorithm_used': algorithm,
            'parameters': params
        })
        
    except Exception as e:
        logger.error(f"Error clustering faces: {e}")
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

@bp.route('/progress/<stage>', methods=['GET'])
def get_progress(stage):
    """Get progress for extraction or clustering"""
    try:
        if stage not in ['extraction', 'clustering']:
            return jsonify({'error': 'Invalid stage'}), 400
        
        # Return actual progress data
        progress = progress_data.get(stage, {
            'percentage': 0,
            'message': f'{stage.capitalize()} en attente...'
        })
        
        return jsonify(progress)
        
    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/cancel', methods=['POST'])
def cancel_process():
    """Cancel current processing"""
    try:
        # Clear progress data
        progress_data.clear()
        
        # In a real implementation, this would signal the processing threads to stop
        logger.info("Process cancellation requested")
        
        return jsonify({
            'status': 'success',
            'message': 'Processus annulé'
        })
        
    except Exception as e:
        logger.error(f"Error cancelling process: {e}")
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
