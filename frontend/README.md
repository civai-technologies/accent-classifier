# Accent Classifier Frontend

A professional web interface for English accent detection and analysis, providing an intuitive way to analyze spoken English.

## 🎯 Purpose

This web application provides a user-friendly interface for analyzing English accents from audio/video inputs. It's designed for researchers, linguists, language teachers, and anyone interested in accent classification and analysis.

## ✨ Features

### Core Functionality
- **File Upload**: Support for audio (MP3, WAV, M4A) and video (MP4, MOV, AVI, MKV, WebM) files
- **URL Processing**: Direct download from video URLs (Loom, YouTube, direct links)
- **Demo Mode**: Built-in sample for testing the system
- **Real-time Analysis**: Live accent classification with confidence scoring

### Analysis Results
- **Accent Classification**: Identifies specific English accents (American, British, Australian, etc.)
- **Confidence Scoring**: 0-100% confidence levels with visual indicators
- **Speaker Classification**: Native vs. non-native speaker identification
- **Detailed Breakdown**: All accent predictions with probability scores
- **Professional Summary**: Human-readable analysis and insights

### User Experience
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Drag & Drop**: Easy file upload with drag-and-drop support
- **Progress Tracking**: Real-time processing status with animated progress bars
- **Results Export**: Download analysis results as JSON
- **Keyboard Shortcuts**: Ctrl+Enter to analyze, Escape to reset

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+
# FFmpeg (for video processing)
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Ubuntu
```

### Installation
```bash
# Clone the repository (if not already done)
cd accent-classifier

# Install dependencies
pip install -r requirements.txt

# Initialize the accent classifier (trains model if needed)
python accent_classifier.py --train --use-tts

# Start the web server
cd frontend
python app.py
```

### Access the Application
Open your browser and navigate to: `http://localhost:5000`

## 📋 Usage Guide

### 1. Upload File
- Click "Upload File" tab
- Select an audio/video file (max 100MB)
- Supported formats: MP3, WAV, MP4, MOV, AVI, MKV, WebM, M4A
- Drag and drop files directly onto the upload area

### 2. Use URL
- Click "URL" tab
- Paste a direct link to a video file
- Supports: Loom, YouTube, direct MP4/MP3 links
- System will automatically download and process

### 3. Try Demo
- Click "Demo" tab
- Uses the default.mp4 sample in the project root
- Perfect for testing system functionality

### 4. Analyze Results
- Click "Analyze Accent" to start processing
- Wait for analysis completion (typically 2-10 seconds)
- Review accent classification and confidence score
- Read the professional summary for linguistic insights

### 5. Export Results
- Click "Download Results" to save analysis as JSON
- Includes all predictions, confidence levels, and timestamps
- Perfect for record-keeping and research documentation

## 🎯 Classification Criteria

### Confidence Levels
- **Very High (90-100%)**: Extremely reliable classification
- **High (80-89%)**: Strong classification, highly reliable
- **Good (70-79%)**: Good classification accuracy
- **Fair (60-69%)**: Acceptable classification with some uncertainty
- **Low (<60%)**: May require clearer audio or manual review

### Accent Categories

#### Native English Accents
- American English
- British English
- Australian English
- Canadian English
- Irish English
- Scottish English

#### Non-Native English Accents
- **High Confidence (>80%)**: Clear accent pattern detected
- **Moderate Confidence (60-80%)**: Recognizable accent features
- **Low Confidence (<60%)**: Unclear accent characteristics

## ⚙️ Configuration

### Environment Variables
```bash
# Optional: Set custom port
export PORT=8080

# Optional: Set Flask environment
export FLASK_ENV=development
```

### File Size Limits
- Maximum file size: 100MB
- Recommended: Keep files under 50MB for faster processing
- Audio duration: Minimum 2 seconds for reliable detection

## 🛠️ Technical Architecture

### Backend Components
- **Flask Web Server**: Handles HTTP requests and file uploads
- **Audio Processor**: Extracts audio from video files using FFmpeg
- **Feature Extractor**: Processes audio for ML analysis
- **ML Classifier**: Machine learning model for accent detection
- **URL Handler**: Downloads content from various video platforms

### Frontend Components
- **Bootstrap 5**: Responsive UI framework
- **Custom CSS**: Professional styling with animations
- **JavaScript**: Interactive functionality and AJAX requests
- **Font Awesome**: Professional icons and visual elements

### Supported Platforms
- **Operating Systems**: Windows, macOS, Linux
- **Browsers**: Chrome, Firefox, Safari, Edge (modern versions)
- **Devices**: Desktop, tablet, mobile (responsive design)

## 📊 API Endpoints

### POST /analyze
Analyzes audio/video for accent classification

**Request:**
```javascript
// Form data with one of:
// - file: multipart file upload
// - url: string URL to video
// - use-default: boolean for demo mode
```

**Response:**
```javascript
{
  "accent": "American",
  "confidence": 87,
  "confidence_level": "High",
  "summary": "The speaker has a American accent with high confidence (87%). This is a strong classification with high confidence. American English is a native English accent.",
  "all_predictions": {
    "American": 0.87,
    "Canadian": 0.08,
    "British": 0.03,
    "Australian": 0.02
  },
  "processing_time": 2.3
}
```

### GET /health
Health check endpoint
```javascript
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🔧 Troubleshooting

### Common Issues

#### Model Not Loading
```bash
# Train the model manually
python accent_classifier.py --train --use-tts
```

#### FFmpeg Not Found
```bash
# Install FFmpeg
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

#### Large File Upload Failures
- Check file size (max 100MB)
- Ensure stable internet connection
- Try converting to MP3 for smaller file size

#### URL Download Issues
- Verify URL is accessible
- Check if video platform is supported
- Try downloading manually and uploading as file

### Performance Optimization
- Use MP3 format for audio-only files
- Keep video files under 50MB when possible
- Ensure adequate server resources for multiple concurrent analyses

## 🔒 Security Considerations

### File Upload Security
- File type validation (whitelist approach)
- File size limits enforced
- Temporary files automatically cleaned up
- No permanent storage of uploaded content

### URL Processing
- Safe URL validation
- Limited to supported video platforms
- No execution of arbitrary code
- Timeout protection for downloads

## 📝 Development

### Local Development Setup
```bash
# Install in development mode
pip install -e .

# Run with debug mode
export FLASK_ENV=development
python frontend/app.py
```

### Adding New Features
1. Update `frontend/app.py` for backend logic
2. Modify `frontend/templates/index.html` for UI changes
3. Enhance `frontend/static/js/main.js` for frontend functionality
4. Update `frontend/static/css/style.css` for styling

### Testing
```bash
# Run backend tests
cd tests
python -m pytest test_integration.py

# Test the web interface
# Start the server and navigate to http://localhost:5000
```

## 📄 License & Attribution

**Developed by:** [Kayode Femi Amoo (Nifemi Alpine)](https://x.com/usecodenaija)  
**Company:** [CIVAI Technologies](https://github.com/civai-technologies/accent-classifier)  
**Twitter:** [@usecodenaija](https://x.com/usecodenaija)  
**Website:** [https://civai.co](https://civai.co)

---

*This frontend is part of the Accent Classifier project - a comprehensive solution for English accent detection and linguistic analysis.* 