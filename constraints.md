# Accent Classifier - Project Constraints

## Technical Limitations

### Audio Processing Constraints
- **Sample Rate**: Default to 16kHz for consistency, but support multiple rates
- **Audio Duration**: Minimum 2 seconds required for reliable accent detection
- **File Formats**: Primary support for WAV, with MP3 as secondary (requires additional codec)
- **Background Noise**: Performance degrades significantly with SNR < 10dB

### Machine Learning Model Constraints
- **Model Size**: Keep pre-trained models under 500MB for reasonable download/load times
- **Inference Speed**: Target < 3 seconds per audio clip on standard hardware
- **Memory Usage**: Limit peak memory usage to < 2GB during processing
- **Accent Categories**: Start with major accent groups, expand gradually

### Real-time Processing Constraints
- **Latency**: Real-time processing has 5-10 second delay for feature extraction
- **Microphone Input**: Requires continuous 10-second windows for stable predictions
- **Hardware Requirements**: Requires audio input device access

## Workarounds and Solutions

### Audio Quality Issues
- **Low Quality Audio**: Apply noise reduction and normalization preprocessing
- **Variable Sample Rates**: Resample all audio to consistent 16kHz
- **Short Audio Clips**: Pad with silence or reject with clear error message

### Model Performance Issues
- **Confidence Threshold**: Reject predictions below 60% confidence
- **Unknown Accents**: Provide "Unknown/Other" category for unrecognized patterns
- **Multiple Speakers**: Process only the loudest speaker in multi-speaker audio

### System Resource Constraints
- **Memory Management**: Process audio in chunks, clear intermediate results
- **Storage**: Cache models locally, provide download mechanism
- **CPU Intensive**: Provide option for reduced feature extraction for faster processing

### Fallback Strategies
- **Model Loading Failure**: Graceful degradation with basic feature-based classification
- **Audio Input Failure**: Clear error messages with troubleshooting steps
- **Network Issues**: All processing done locally, no external API dependencies

## Known Issues and Limitations
1. Accent classification accuracy varies significantly with speaker gender and age
2. Regional variations within accent categories may not be distinguished
3. Code-switching and multilingual speakers may produce inconsistent results
4. Performance on accented English vs. native language speech differs considerably 