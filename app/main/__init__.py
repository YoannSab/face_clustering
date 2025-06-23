from flask import Blueprint

bp = Blueprint('main', __name__)

# Import routes to register them
from app.main import routes
