import os
from datetime import timedelta

class Config:
    """Configuration base class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'face-clustering-secret-key'
    
    # Face Detection Settings
    FACE_DETECTION_THRESHOLD = 0.8
    MAX_IMAGE_SIZE = 800
    MAX_FACES_PER_CLUSTER = 6
    
    # Clustering Settings
    DEFAULT_DBSCAN_EPS = 0.65
    DEFAULT_DBSCAN_MIN_SAMPLES = 3
    DEFAULT_K_CLUSTERS = 20
    
    # Performance Settings
    CHUNK_SIZE = 20
    CACHE_TIMEOUT = timedelta(hours=24)  # Increased cache timeout to 24 hours
    MAX_WORKERS = 4
    
    # File Settings
    EMBEDDINGS_FILE = "./cache/embeddings_cache.json"
    SUPPORTED_FORMATS = {
        'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'webp', 'cr2'
    }
    
    # API Settings
    PAGINATION_PER_PAGE = 50
    MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16MB

    # Cache Settings (personnalisables)
    AUTO_CLEANUP = True                        # Nettoyage automatique des entrées invalides
    CACHE_MAX_SIZE = 10000                     # Limite du nombre d'entrées
    
    # Migration Settings
    LEGACY_CACHE_FILE = "save_emb_keypath.json"  # Ancien fichier à migrer
    BACKUP_ON_MIGRATE = True                     # Sauvegarde avant migration

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
