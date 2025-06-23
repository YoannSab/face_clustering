from flask import render_template, send_from_directory, current_app
from app.main import bp
import os

@bp.route('/')
def index():
    """Serve the main application page"""
    try:
        return render_template('index.html')
    except Exception as e:
        current_app.logger.error(f"Error rendering template: {e}")
        # Fallback en cas d'erreur de template
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Face Clustering</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .btn {{ display: inline-block; padding: 12px 24px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 10px 5px; }}
                .error {{ background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Face Clustering Application</h1>
                <div class="error">
                    <strong>Template Error:</strong> {str(e)}<br>
                    <small>Template directory: {current_app.template_folder}</small>
                </div>
                <p>L'interface moderne rencontre un problème temporaire.</p>
                <h3>Actions disponibles :</h3>
                <a href="/legacy" class="btn">Interface Simple</a>
                <a href="/health" class="btn">Status Système</a>
                <a href="javascript:location.reload()" class="btn">Recharger</a>
                
                <h3>Debug Info :</h3>
                <ul>
                    <li>Templates folder: {current_app.template_folder}</li>
                    <li>Static folder: {current_app.static_folder}</li>
                    <li>Root path: {current_app.root_path}</li>
                </ul>
            </div>
        </body>
        </html>
        """

@bp.route('/health')
def health_check():
    """Point de contrôle de santé de l'application"""
    return {
        'status': 'healthy',
        'version': '2.0',
        'templates_folder': current_app.template_folder,
        'static_folder': current_app.static_folder,
        'template_exists': os.path.exists(os.path.join(current_app.template_folder, 'index.html'))
    }

@bp.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)
