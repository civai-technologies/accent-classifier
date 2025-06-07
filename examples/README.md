# Examples Directory

This directory contains interactive examples and demonstrations of the Accent Classifier system.

## 📓 Jupyter Notebook Demo

### `accent_classifier_demo.ipynb`

A comprehensive interactive demonstration of the accent classification system featuring:

#### 🎯 **What's Included**

1. **Complete Workflow Setup**
   - System initialization and component configuration
   - Google TTS sample generation and management
   - Model training with performance metrics

2. **Interactive Audio Analysis**
   - Real-time audio classification with detailed results
   - Feature extraction and audio processing demonstrations
   - Batch testing and performance evaluation

3. **Advanced Visualizations**
   - Audio waveform and spectrogram analysis
   - MFCC, chroma, and spectral feature visualizations
   - Classification probability distributions
   - Performance metrics and accuracy analysis

4. **Production Examples**
   - Batch processing workflows
   - Real-time classification patterns
   - Integration code examples

#### 🚀 **Getting Started**

1. **Prerequisites**
   ```bash
   # Install required dependencies
   pip install -r ../requirements.txt
   
   # Install Jupyter if not already available
   pip install jupyter notebook
   
   # Configure Google Text-to-Speech API credentials
   # Option 1: Set environment variable
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
   
   # Option 2: Use .env file (copy sample.env to .env and edit)
   cp ../sample.env ../.env
   ```

2. **Launch the Notebook**
   ```bash
   # From the examples directory
   jupyter notebook accent_classifier_demo.ipynb
   
   # Or from the project root
   jupyter notebook examples/accent_classifier_demo.ipynb
   ```

3. **Run the Demo**
   - Execute cells in order for the complete workflow
   - Each cell is self-contained with clear explanations
   - Interactive functions allow testing with custom audio files

#### 📊 **Key Features Demonstrated**

- **Google TTS Integration**: Automatic generation of high-quality training samples
- **Machine Learning Pipeline**: Complete training and evaluation workflow
- **Audio Processing**: Feature extraction and preprocessing techniques
- **Real-time Classification**: Interactive accent detection with confidence scoring
- **Performance Analysis**: Comprehensive metrics and visualization tools
- **Production Patterns**: Ready-to-use code for integration

#### 🎤 **Interactive Functions**

The notebook provides several interactive functions you can use:

```python
# Classify any audio file with detailed analysis
classify_and_analyze('/path/to/your/audio.wav')

# Run batch performance testing
batch_results = batch_test_performance()

# Create comprehensive audio visualizations
visualize_audio_features('/path/to/audio.wav')
```

#### 🔧 **Customization**

- **Add New Languages**: The notebook shows how to extend the system
- **Adjust Parameters**: Experiment with confidence thresholds and model settings
- **Custom Visualizations**: Build on the provided plotting functions
- **Integration Examples**: Adapt the code for your specific use cases

#### 📈 **Expected Results**

When running the complete notebook, you should see:
- **Training Accuracy**: 90%+ on TTS-generated samples
- **Cross-validation**: 85%+ accuracy with proper validation
- **Real-time Performance**: <100ms classification time
- **Visualization Quality**: Professional-grade audio analysis plots

#### 🐛 **Troubleshooting**

**Common Issues:**
- **Import Errors**: Ensure you're running from the correct directory with `sys.path.append('..')`
- **Audio Generation**: Check internet connection for Google TTS API access
- **Visualization**: Install librosa if audio plots don't render: `pip install librosa`
- **Performance**: Reduce sample sizes if processing is slow

**Solutions:**
```python
# Check system status
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Verify audio samples exist
audio_samples_dir = Path('../audio_samples')
print(f"Audio samples found: {audio_samples_dir.exists()}")
```

#### 💡 **Tips for Best Results**

1. **First Run**: Allow extra time for TTS sample generation
2. **Audio Quality**: Use clear, noise-free audio for best classification results
3. **Sample Size**: More training samples generally improve accuracy
4. **Experimentation**: Try different confidence thresholds for your use case
5. **Documentation**: Each function includes detailed docstrings and examples

#### 🌟 **Advanced Usage**

The notebook serves as a foundation for:
- **Research Projects**: Linguistic analysis and accent pattern studies
- **Production Systems**: Call center routing and voice analytics
- **Educational Tools**: Language learning and pronunciation assessment
- **Custom Applications**: Integration into larger audio processing pipelines

---

## 🎯 **Next Steps**

After completing the notebook demo:

1. **Explore the CLI**: Try `python ../accent_classifier.py --help`
2. **Read Documentation**: Check the `../docs/` directory for detailed guides
3. **Run Tests**: Execute `pytest ../tests/` for comprehensive testing
4. **Contribute**: Add new languages or improve the system

## 🤝 **Support**

For questions or issues with the examples:
- Check the main project documentation
- Review the notebook cell outputs for debugging information
- Ensure all dependencies are properly installed
- Verify audio samples are generated correctly

---

## 👨‍💻 **Developer & Company**

**Developed by:** [Kayode Femi Amoo (Nifemi Alpine)](https://twitter.com/usecodenaija)  
**Twitter:** [@usecodenaija](https://twitter.com/usecodenaija)  
**Company:** [CivAI Technologies](https://civai.co)  
**Website:** [https://civai.co](https://civai.co)

This interactive demo and the complete Accent Classifier system represent cutting-edge work in voice AI and linguistic technology, developed as part of CivAI Technologies' mission to advance artificial intelligence solutions.

---

*This interactive demo showcases the complete accent classification pipeline from Google TTS generation to production deployment.* 