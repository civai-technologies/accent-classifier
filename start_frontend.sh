#!/bin/bash

# Accent Classifier Frontend Startup Script

echo "🎯 Starting Accent Classifier Frontend..."
echo "================================================"

# Check if we're in the right directory
if [ ! -f "accent_classifier.py" ]; then
    echo "❌ Error: Please run this script from the accent-classifier project root directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Check if virtual environment is recommended
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Running outside virtual environment. Consider using:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo ""
fi

# Install dependencies if requirements.txt is newer than last install
if [ requirements.txt -nt .requirements_installed ] || [ ! -f .requirements_installed ]; then
    echo "📦 Installing/updating dependencies..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        touch .requirements_installed
        echo "✅ Dependencies installed successfully"
    else
        echo "❌ Failed to install dependencies"
        exit 1
    fi
else
    echo "✅ Dependencies are up to date"
fi

# Check if model exists
if [ ! -f "models/accent_classifier.joblib" ]; then
    echo "🤖 Training accent classifier model (first time setup)..."
    python3 accent_classifier.py --train --use-tts
    if [ $? -ne 0 ]; then
        echo "❌ Failed to train model"
        exit 1
    fi
    echo "✅ Model trained successfully"
else
    echo "✅ Accent classifier model found"
fi

# Check for FFmpeg
if command -v ffmpeg >/dev/null 2>&1; then
    echo "✅ FFmpeg found - video processing enabled"
else
    echo "⚠️  FFmpeg not found - video processing will be limited"
    echo "   Install with: brew install ffmpeg (macOS) or sudo apt install ffmpeg (Ubuntu)"
fi

# Create uploads directory
mkdir -p frontend/uploads

# Set environment variables
export FLASK_ENV=development
export FLASK_DEBUG=True
export PORT=5001

echo ""
echo "🚀 Starting frontend server..."
echo "📱 Access the application at: http://localhost:5001"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Start the frontend
cd frontend
python3 run.py

echo ""
echo "👋 Frontend stopped. Thank you for using Accent Classifier!" 