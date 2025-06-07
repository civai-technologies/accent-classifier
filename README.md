---
title: Accent Classifier
emoji: 🗣️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Accent Classifier 🗣️

An intelligent audio analysis tool that identifies speaker accents from audio input using machine learning. This application can classify accents including American, British, French, German, Spanish, Russian, and more.

## Features

- **Real-time Audio Analysis**: Upload audio files or use microphone input
- **Multiple Format Support**: WAV, MP3, M4A, and other common audio formats
- **Video Processing**: Extract audio from MP4 videos for analysis
- **Confidence Scoring**: Get confidence levels for each prediction
- **Sample Testing**: Built-in test samples for different accents
- **Modern Web Interface**: Clean, responsive UI with Bootstrap 5

## How to Use

1. **Upload Audio**: Click "Choose File" and select an audio/video file
2. **Record Audio**: Use "Start Recording" for live microphone input
3. **Test Samples**: Try the built-in accent samples (American, British, French)
4. **Analyze**: Click "Analyze Audio" to get accent classification results

## Supported Accents

- 🇺🇸 American English
- 🇬🇧 British English  
- 🇫🇷 French
- 🇩🇪 German
- 🇪🇸 Spanish
- 🇷🇺 Russian
- And more...

## Technical Details

- **Framework**: Flask web application
- **ML Model**: Scikit-learn with audio feature extraction
- **Audio Processing**: Librosa, PyAudio, FFmpeg
- **Features**: MFCC, spectral features, pitch analysis
- **Model Size**: ~355KB (lightweight and fast)

## Use Cases

- **Language Learning**: Identify accent patterns and pronunciation
- **Linguistics Research**: Analyze speech characteristics
- **Voice Analysis**: Understand speaker accent profiles
- **Educational Tools**: Teaching accent recognition

## Model Performance

The classifier uses advanced audio feature extraction including:
- Mel-frequency cepstral coefficients (MFCC)
- Spectral features (centroid, rolloff, bandwidth)
- Chroma features
- Tempo and rhythm analysis

## Privacy

- Audio processing happens locally in the container
- No audio data is stored permanently
- Temporary files are cleaned up after analysis

## Development

This application is built with:
- **Backend**: Python Flask with ML pipeline
- **Frontend**: Bootstrap 5, vanilla JavaScript
- **Audio**: Librosa, SoundFile, PyAudio
- **ML**: Scikit-learn, NumPy, SciPy

## License

MIT License - See LICENSE file for details 