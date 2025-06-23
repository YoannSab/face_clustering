import logging
import os
from flask import Flask
from flask_cors import CORS
from app.config import Config

def create_app(config_class=Config):
    """Application Factory Pattern"""
    
    # Determine the correct paths
    basedir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.dirname(basedir)
    
    app = Flask(__name__, 
                template_folder=os.path.join(project_root, 'templates'),
                static_folder=os.path.join(project_root, 'static'))
    
    app.config.from_object(config_class)
    
    # Extensions
    CORS(app)
    
    # Configure logging
    if not app.debug:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(name)s %(message)s'
        )
    
    # Log the template and static folders for debugging
    app.logger.info(f"Template folder: {app.template_folder}")
    app.logger.info(f"Static folder: {app.static_folder}")
    app.logger.info(f"Templates exist: {os.path.exists(app.template_folder)}")
    
    # Register Blueprints
    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)
    
    return app
