# Scalable Audio System Documentation

## Overview

The Accent Classifier now features a scalable audio system that uses structured folders with configuration files for easy language expansion and efficient audio sample management.

## Key Features

### 🚀 **Efficient Sample Management**
- **Avoids regenerating existing files** unless `--fresh` flag is used
- **Structured folder organization** by language/accent
- **Configuration-driven** TTS settings per language
- **Easy scaling** to support new languages

### 📁 **Folder Structure**

```
audio_samples/
├── american/
│   ├── config.json          # TTS configuration for American accent
│   ├── sample_001.wav        # Generated audio samples
│   ├── sample_002.wav
│   ├── sample_003.wav
│   ├── sample_004.wav
│   └── sample_005.wav
├── british/
│   ├── config.json          # TTS configuration for British accent
│   ├── sample_001.wav
│   └── ...
├── french/
│   ├── config.json          # TTS configuration for French accent
│   ├── sample_001.wav
│   └── ...
└── [other languages]/
    ├── config.json
    └── *.wav files
```

### ⚙️ **Configuration Files**

Each language directory contains a `config.json` file with:

```json
{
  "language_name": "American Accent",
  "accent_code": "american",
  "tts_config": {
    "gtts": {
      "lang": "en",
      "tld": "us",
      "slow": false
    },
    "cloud_tts": {
      "language_code": "en-US",
      "voice_name": "en-US-Neural2-D",
      "audio_encoding": "LINEAR16",
      "sample_rate_hertz": 16000
    }
  },
  "sample_texts": [
    "Text sample 1 with American-specific vocabulary...",
    "Text sample 2 with regional pronunciation patterns...",
    "Text sample 3...",
    "Text sample 4...",
    "Text sample 5..."
  ],
  "description": "American English accent with characteristic pronunciation patterns."
}
```

## Usage

### 🎯 **Training with Existing Samples**

```bash
# Use existing audio samples (efficient)
python accent_classifier.py --train --use-tts --verbose
```

**Behavior:**
- ✅ Reuses existing audio files
- ✅ Generates missing samples only
- ✅ Fast training (no regeneration)

### 🔄 **Training with Fresh Samples**

```bash
# Force regenerate all audio samples
python accent_classifier.py --train --use-tts --fresh --verbose
```

**Behavior:**
- 🔄 Regenerates ALL audio samples
- 🔄 Overwrites existing files
- 🔄 Slower but ensures fresh data

### 🎙️ **Standalone Audio Generation**

```bash
# Generate samples for specific languages
python tests/scalable_audio_generator.py --languages american british --num-samples 3

# Generate samples for all configured languages
python tests/scalable_audio_generator.py --num-samples 5

# Force regenerate existing samples
python tests/scalable_audio_generator.py --fresh --num-samples 5

# List available languages
python tests/scalable_audio_generator.py --list

# Show training data information
python tests/scalable_audio_generator.py --info
```

## Performance Comparison

### Before Scalable System
- **Regenerated samples every time** (slow)
- **Hardcoded language configurations**
- **Difficult to add new languages**
- **No sample reuse**

### After Scalable System
- **Reuses existing samples** (fast)
- **Configuration-driven languages**
- **Easy language addition**
- **Efficient sample management**

### Training Performance Results

| Method | Samples | Train Acc | Test Acc | CV Acc | Avg Confidence |
|--------|---------|-----------|----------|--------|----------------|
| Without `--fresh` | 26 | 100% | 83.3% | 65.9% | 61.3% |
| With `--fresh` | 30 | 100% | 100% | 95.8% | 70.0% |

## Adding New Languages

### Step 1: Create Language Directory

```bash
mkdir audio_samples/italian
```

### Step 2: Create Configuration File

Create `audio_samples/italian/config.json`:

```json
{
  "language_name": "Italian Accent",
  "accent_code": "italian",
  "tts_config": {
    "gtts": {
      "lang": "it",
      "tld": "it",
      "slow": false
    },
    "cloud_tts": {
      "language_code": "it-IT",
      "voice_name": "it-IT-Wavenet-A",
      "audio_encoding": "LINEAR16",
      "sample_rate_hertz": 16000
    }
  },
  "sample_texts": [
    "Ciao, come stai oggi? Vado al mercato per comprare pane, formaggio e vino.",
    "Salve! Sono dall'Italia e mi piace mangiare pasta e pizza. Roma è bellissima.",
    "La lingua italiana è molto melodiosa e espressiva. Abbiamo una ricca cultura.",
    "Lavoro in un ufficio nel centro di Milano. Ogni mattina prendo il treno.",
    "L'Italia è famosa per l'arte, la musica e la cucina. Abbiamo grandi artisti."
  ],
  "description": "Standard Italian accent with characteristic pronunciation patterns."
}
```

### Step 3: Generate Samples

```bash
# Generate samples for the new language
python tests/scalable_audio_generator.py --languages italian --num-samples 5
```

### Step 4: Train with New Language

```bash
# Train model including the new language
python accent_classifier.py --train --use-tts --verbose
```

## Advanced Features

### 🔍 **Language Discovery**

The system automatically discovers available languages by scanning for directories with `config.json` files:

```python
from tests.scalable_audio_generator import ScalableAudioGenerator

generator = ScalableAudioGenerator()
languages = generator.discover_languages()
print(f"Available languages: {languages}")
```

### 📊 **Training Data Information**

```python
info = generator.get_training_data_info()
for lang, data in info.items():
    print(f"{lang}: {data['existing_samples']} samples")
```

### 🧹 **Sample Management**

```python
# Remove all samples for a language
generator.cleanup_language('italian')

# Get existing samples
samples = generator.get_existing_samples('american')
```

## Configuration Options

### TTS Engine Selection

The system supports both gTTS and Google Cloud TTS:

```python
# Use gTTS (free, basic quality)
generator = ScalableAudioGenerator(use_cloud_tts=False)

# Use Google Cloud TTS (paid, high quality)
generator = ScalableAudioGenerator(use_cloud_tts=True)
```

### Sample Rate Configuration

```python
# Custom sample rate
generator = ScalableAudioGenerator(sample_rate=22050)
```

### Base Directory

```python
# Custom audio samples directory
generator = ScalableAudioGenerator(base_dir="my_audio_samples")
```

## Troubleshooting

### Common Issues

1. **No languages found**
   - Ensure `config.json` files exist in language directories
   - Check JSON syntax validity

2. **TTS generation fails**
   - Check internet connection for gTTS
   - Verify Google Cloud credentials for Cloud TTS
   - Check language codes in configuration

3. **Audio processing errors**
   - Ensure audio files are valid WAV format
   - Check file permissions
   - Verify sample rate compatibility

### Debug Commands

```bash
# Check available languages
python tests/scalable_audio_generator.py --list

# Show detailed information
python tests/scalable_audio_generator.py --info

# Test single language generation
python tests/scalable_audio_generator.py --languages american --num-samples 1 --fresh
```

## Best Practices

### 🎯 **Efficient Development**

1. **Use existing samples** during development (`--use-tts` without `--fresh`)
2. **Generate fresh samples** for final training (`--use-tts --fresh`)
3. **Add languages incrementally** and test each addition
4. **Use descriptive sample texts** that highlight accent characteristics

### 📈 **Performance Optimization**

1. **Start with 5 samples per language** for initial testing
2. **Increase to 10+ samples** for production models
3. **Use `--fresh` sparingly** to avoid unnecessary regeneration
4. **Monitor training metrics** to determine optimal sample counts

### 🔧 **Configuration Management**

1. **Use distinct voice models** for different accents
2. **Include accent-specific vocabulary** in sample texts
3. **Test TTS quality** before large-scale generation
4. **Backup configuration files** before modifications

## Future Enhancements

### Planned Features

- **Automatic voice model selection** based on language
- **Sample quality assessment** and filtering
- **Batch configuration updates** for multiple languages
- **Audio augmentation** for increased sample diversity
- **Real audio sample integration** alongside TTS samples

### Extensibility

The system is designed for easy extension:

- **Custom TTS engines** can be added
- **Additional audio formats** can be supported
- **Feature extraction methods** can be customized
- **Training strategies** can be modified

## Summary

The scalable audio system provides:

✅ **Efficient sample management** with reuse capabilities  
✅ **Easy language addition** through configuration files  
✅ **Flexible TTS engine support** (gTTS + Cloud TTS)  
✅ **Structured organization** for maintainability  
✅ **Performance optimization** through smart regeneration  
✅ **Comprehensive tooling** for development and debugging  

This system makes the Accent Classifier easily scalable to support dozens of languages and accents while maintaining efficient development workflows. 