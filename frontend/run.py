#!/usr/bin/env python3
"""
Production entry point for the Accent Classifier Frontend
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / 'src'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Set up environment variables and configuration."""
    # Set default port if not specified
    if 'PORT' not in os.environ:
        os.environ['PORT'] = '5000'
    
    # Set production environment if not specified
    if 'FLASK_ENV' not in os.environ:
        os.environ['FLASK_ENV'] = 'production'
    
    # Disable debug mode in production
    os.environ['FLASK_DEBUG'] = 'False'

def check_dependencies():
    """Check if all required dependencies are available."""
    try:
        import flask
        import librosa
        import sklearn
        import numpy
        import scipy
        logger.info("✓ All core dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        logger.error("Please run: pip install -r requirements.txt")
        return False

def check_model():
    """Check if the accent classifier model is trained and ready."""
    model_path = parent_dir / 'models' / 'accent_classifier.joblib'
    if model_path.exists():
        logger.info("✓ Accent classifier model found")
        return True
    else:
        logger.warning("✗ Accent classifier model not found")
        logger.info("The model will be trained automatically on first run")
        return False

def check_ffmpeg():
    """Check if FFmpeg is available for video processing."""
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✓ FFmpeg is available for video processing")
            return True
    except FileNotFoundError:
        pass
    
    logger.warning("✗ FFmpeg not found - video processing will be limited")
    logger.warning("Install FFmpeg: https://ffmpeg.org/download.html")
    return False

def main():
    """Main entry point for the application."""
    logger.info("Starting Accent Classifier Frontend...")
    
    # Set up environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check optional components
    check_model()
    check_ffmpeg()
    
    # Import and run the Flask app
    try:
        from app import app, classifier_app
        
        # Ensure correct model path
        classifier_app.model_path = str(parent_dir / 'models' / 'accent_classifier.joblib')
        
        # Initialize the classifier
        logger.info("Initializing accent classifier...")
        if not classifier_app.setup(train_if_needed=True):
            logger.error("Failed to initialize accent classifier!")
            sys.exit(1)
        
        logger.info("✓ Accent classifier initialized successfully")
        
        # Get configuration
        port = int(os.environ.get('PORT', 5000))
        host = os.environ.get('HOST', '0.0.0.0')
        debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        
        # Log startup information
        logger.info(f"Starting server on {host}:{port}")
        logger.info(f"Debug mode: {debug}")
        logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'production')}")
        
        # Start the Flask application
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 