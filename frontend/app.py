#!/usr/bin/env python3
"""
Accent Classifier Web Application
A Flask-based frontend for the accent classification system.
"""

import os
import sys
import tempfile
import requests
from urllib.parse import urlparse
from pathlib import Path
import json
import subprocess
from typing import Optional, Dict, Any
import uuid
import shutil

from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import youtube_dl

# Add parent directory to path to import our accent classifier
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from accent_classifier import AccentClassifierApp

app = Flask(__name__)
app.secret_key = 'accent-classifier-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Initialize accent classifier with correct model path
classifier_app = AccentClassifierApp()
# Fix model path to point to parent directory
classifier_app.model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'accent_classifier.joblib')

# Create upload directory
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4', 'mov', 'avi', 'mkv', 'webm', 'm4a'}


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_audio_from_video(video_path: str, output_path: str) -> bool:
    """Extract audio from video file using ffmpeg."""
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'libmp3lame',
            '-ab', '192k', '-ar', '16000',
            '-y', output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False


def download_video_from_url(url: str, output_path: str) -> bool:
    """Download video from URL (supports YouTube, Loom, direct links)."""
    try:
        # For direct file URLs
        if url.lower().endswith(('.mp4', '.mp3', '.wav', '.mov', '.avi', '.mkv', '.webm', '.m4a')):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        
        # For YouTube and other supported platforms
        ydl_opts = {
            'outtmpl': output_path.replace('.%(ext)s', '.%(ext)s'),
            'format': 'best[height<=720]/best',
        }
        
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            return True
            
    except Exception as e:
        print(f"Error downloading from URL: {e}")
        return False


def get_confidence_level(confidence: float) -> str:
    """Convert confidence score to human-readable level."""
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.8:
        return "High"
    elif confidence >= 0.7:
        return "Good"
    elif confidence >= 0.6:
        return "Fair"
    else:
        return "Low"


def generate_summary(results: Dict[str, Any]) -> str:
    """Generate a human-readable summary of the accent analysis."""
    accent = results.get('accent_name', 'Unknown')
    confidence = results.get('confidence', 0)
    confidence_level = get_confidence_level(confidence)
    
    summary = f"The speaker has a {accent} accent with {confidence_level.lower()} confidence "
    summary += f"({confidence:.1%}). "
    
    if confidence >= 0.8:
        summary += "This is a strong classification with high confidence."
    elif confidence >= 0.6:
        summary += "This is a reasonable classification with good confidence."
    else:
        summary += "The accent classification has low confidence - consider using a clearer audio sample."
    
    # Add accent-specific insights
    english_accents = ['american', 'british', 'australian', 'canadian', 'irish', 'scottish']
    if accent.lower() in english_accents:
        summary += f" {accent} English is a native English accent."
    else:
        summary += f" {accent} accent indicates English as a second language."
    
    return summary


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_accent():
    """Main endpoint for accent analysis."""
    try:
        # Check if we have a file upload, URL, or test sample
        file = request.files.get('file')
        url = request.form.get('url', '').strip()
        use_default = request.form.get('use-default') == 'true'
        test_sample = request.form.get('test-sample', '').strip()
        
        audio_path = None
        temp_files = []  # Track temporary files for cleanup
        
        # Determine input source
        if use_default:
            # Use default.mp4 from project root
            default_path = os.path.join(os.path.dirname(__file__), '..', 'default.mp4')
            if os.path.exists(default_path):
                # Extract audio from default video first
                audio_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_default.mp3")
                if not extract_audio_from_video(default_path, audio_path):
                    return jsonify({'error': 'Failed to extract audio from default.mp4'})
                temp_files.append(audio_path)
            else:
                return jsonify({'error': 'default.mp4 not found in project root'})
                
        elif test_sample:
            # Use test MP4 files from test_samples directory
            test_files = {
                'american': 'american_test.mp4',
                'british': 'british_test.mp4',
                'french': 'french_test.mp4'
            }
            
            if test_sample not in test_files:
                return jsonify({'error': f'Invalid test sample: {test_sample}'})
            
            test_path = os.path.join(os.path.dirname(__file__), '..', 'test_samples', test_files[test_sample])
            if os.path.exists(test_path):
                # Extract audio from test video
                audio_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{test_sample}.mp3")
                if not extract_audio_from_video(test_path, audio_path):
                    return jsonify({'error': f'Failed to extract audio from {test_files[test_sample]}'})
                temp_files.append(audio_path)
            else:
                return jsonify({'error': f'Test sample {test_files[test_sample]} not found'})
                
        elif file and file.filename:
            # Handle file upload
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Supported: ' + ', '.join(ALLOWED_EXTENSIONS)})
            
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(file_path)
            temp_files.append(file_path)
            
            # If it's a video file, extract audio
            if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
                audio_path = file_path.replace(Path(file_path).suffix, '.mp3')
                if not extract_audio_from_video(file_path, audio_path):
                    return jsonify({'error': 'Failed to extract audio from video'})
                temp_files.append(audio_path)
            else:
                audio_path = file_path
                
        elif url:
            # Handle URL input
            try:
                # Create temporary file for download
                temp_video = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_download.mp4")
                temp_files.append(temp_video)
                
                if not download_video_from_url(url, temp_video):
                    return jsonify({'error': 'Failed to download from URL. Please check the URL and try again.'})
                
                # Extract audio from downloaded video
                audio_path = temp_video.replace('.mp4', '.mp3')
                if not extract_audio_from_video(temp_video, audio_path):
                    return jsonify({'error': 'Failed to extract audio from downloaded video'})
                temp_files.append(audio_path)
                
            except Exception as e:
                return jsonify({'error': f'Error processing URL: {str(e)}'})
        else:
            return jsonify({'error': 'Please provide a file upload, URL, or use the default sample'})
        
        # Classify the accent
        results = classifier_app.classify_file(audio_path, show_details=False)
        
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass  # Ignore cleanup errors
        
        if results.get('error'):
            return jsonify({'error': results['error']})
        
        # Format results for frontend
        accent = results.get('accent_name', 'Unknown')
        confidence = results.get('confidence', 0)
        confidence_percentage = int(confidence * 100)
        
        # Generate summary
        summary = generate_summary(results)
        
        response_data = {
            'accent': accent,
            'confidence': confidence_percentage,
            'confidence_level': get_confidence_level(confidence),
            'summary': summary,
            'all_predictions': results.get('all_predictions', {}),
            'processing_time': results.get('processing_time', 0)
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})


@app.route('/test-media/<sample_type>')
def serve_test_media(sample_type):
    """Serve test media files for preview."""
    try:
        test_samples_dir = os.path.join(os.path.dirname(__file__), '..', 'test_samples')
        
        # Define available media files
        media_files = {
            'american_mp4': 'american_test.mp4',
            'british_mp4': 'british_test.mp4', 
            'french_mp4': 'french_test.mp4',
            'american_wav': 'American Accent_sample.wav',
            'british_wav': 'British Accent_sample.wav',
            'french_wav': 'French Accent_sample.wav',
            'german_wav': 'German Accent_sample.wav',
            'spanish_wav': 'Spanish Accent_sample.wav',
            'russian_wav': 'Russian Accent_sample.wav',
            'italian_wav': 'Italian Accent_sample.wav'
        }
        
        if sample_type not in media_files:
            return jsonify({'error': 'Invalid sample type'}), 404
            
        file_path = os.path.join(test_samples_dir, media_files[sample_type])
            
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(file_path)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint."""
    # Check if model is loaded by testing the pipeline attribute and model path
    model_loaded = (hasattr(classifier_app.classifier, 'pipeline') and 
                   classifier_app.classifier.pipeline is not None and
                   os.path.exists(classifier_app.model_path))
    return jsonify({'status': 'healthy', 'model_loaded': model_loaded})


if __name__ == '__main__':
    # Initialize the classifier
    print("Initializing accent classifier...")
    if not classifier_app.setup(train_if_needed=True):
        print("Failed to initialize accent classifier!")
        sys.exit(1)
    
    print("Accent classifier ready!")
    print("Starting web server...")
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 