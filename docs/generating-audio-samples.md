# Generating Audio Samples

This guide explains how to generate audio samples for training and testing the Accent Classifier.

## Overview

The Accent Classifier includes a powerful audio generation system that can:

- **Generate samples for specific languages** or all configured languages
- **Avoid regenerating existing files** for efficiency (unless forced)
- **Use multiple TTS engines** (gTTS and Google Cloud TTS)
- **Scale to dozens of languages** with configuration-driven approach
- **Provide detailed reporting** on generation progress and results

## Audio Generator Tool

The main tool for generating audio samples is located at `src/audio_generator.py`. It provides a comprehensive command-line interface for all audio generation tasks.

### Basic Usage

```bash
# Generate samples for all configured languages (5 samples each)
python src/audio_generator.py

# Generate samples for specific languages
python src/audio_generator.py --languages american british --num-samples 3

# Force regenerate existing samples
python src/audio_generator.py --fresh --num-samples 5

# List available languages
python src/audio_generator.py --list

# Show detailed information about existing samples
python src/audio_generator.py --info
```

## Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--languages` | Specify languages to generate (space-separated) | `--languages american british french` |
| `--num-samples` | Number of samples per language (default: 5) | `--num-samples 10` |
| `--fresh` | Force regenerate existing files | `--fresh` |
| `--cloud-tts` | Use Google Cloud TTS instead of gTTS | `--cloud-tts` |
| `--base-dir` | Custom base directory for samples | `--base-dir my_samples` |
| `--list` | List available languages | `--list` |
| `--info` | Show training data information | `--info` |

## Generation Strategies

### 1. Efficient Development Workflow

For daily development, use existing samples to save time:

```bash
# Generate missing samples only (efficient)
python src/audio_generator.py --num-samples 5

# Check what already exists
python src/audio_generator.py --info
```

**Benefits:**
- ⚡ Fast execution (skips existing files)
- 💰 Saves API costs (for Cloud TTS)
- 🔄 Consistent training data across runs

### 2. Fresh Sample Generation

For final training or when you need new variations:

```bash
# Regenerate ALL samples (slower but fresh)
python src/audio_generator.py --fresh --num-samples 5

# Regenerate specific languages only
python src/audio_generator.py --languages american british --fresh --num-samples 3
```

**Benefits:**
- 🆕 Fresh audio variations
- 🎯 Better model generalization
- 📈 Potentially improved accuracy

### 3. Targeted Language Development

When working on specific languages:

```bash
# Work on just one language
python src/audio_generator.py --languages italian --num-samples 10

# Compare regional variants
python src/audio_generator.py --languages american british --num-samples 5 --fresh
```

### 4. Production-Ready Generation

For production models with high sample counts:

```bash
# Generate many samples for robust training
python src/audio_generator.py --num-samples 20 --cloud-tts

# Or target specific high-priority languages
python src/audio_generator.py --languages american british french german --num-samples 15 --cloud-tts
```

## TTS Engine Selection

### gTTS (Default)

**Advantages:**
- ✅ Free to use
- ✅ No setup required
- ✅ Supports many languages
- ✅ Good for development and testing

**Limitations:**
- ⚠️ Basic voice quality
- ⚠️ Limited voice options
- ⚠️ Requires internet connection
- ⚠️ Rate limiting possible

```bash
# Use gTTS (default)
python src/audio_generator.py --languages american --num-samples 3
```

### Google Cloud TTS

**Advantages:**
- ✅ High-quality neural voices
- ✅ Many voice options per language
- ✅ Professional audio quality
- ✅ Consistent pronunciation

**Requirements:**
- 💳 Requires Google Cloud account with billing
- 🔑 Requires authentication setup
- 🔧 More complex setup

```bash
# Use Google Cloud TTS
python src/audio_generator.py --languages american --num-samples 3 --cloud-tts

# For production-quality samples
python src/audio_generator.py --cloud-tts --num-samples 10
```

#### Setting up Google Cloud TTS

1. **Create Google Cloud Project**
2. **Enable Text-to-Speech API**
3. **Create Service Account and download credentials**
4. **Set environment variable:**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
   ```

## Sample Generation Process

### How It Works

1. **Language Discovery:** Scans `audio_samples/` for directories with `config.json`
2. **Configuration Loading:** Reads TTS settings and sample texts for each language
3. **Existing File Check:** Identifies which samples already exist (unless `--fresh`)
4. **TTS Generation:** Generates audio using configured TTS engine
5. **Audio Processing:** Converts to standard format (16kHz WAV)
6. **Progress Reporting:** Shows detailed progress and summary

### File Organization

Generated files follow a consistent structure:

```
audio_samples/
├── american/
│   ├── config.json
│   ├── sample_001.wav
│   ├── sample_002.wav
│   ├── sample_003.wav
│   ├── sample_004.wav
│   └── sample_005.wav
├── british/
│   ├── config.json
│   ├── sample_001.wav
│   └── ...
└── [other languages]/
```

### Sample Naming Convention

- **Format:** `sample_XXX.wav` where XXX is zero-padded number
- **Sequence:** `sample_001.wav`, `sample_002.wav`, etc.
- **Consistency:** Same naming across all languages
- **Sorting:** Files sort correctly alphabetically

## Advanced Usage

### Custom Base Directory

Generate samples in a custom location:

```bash
# Use custom directory
python src/audio_generator.py --base-dir my_custom_samples --num-samples 3

# Useful for experiments or different model versions
python src/audio_generator.py --base-dir experiments/high_quality --cloud-tts --num-samples 10
```

### Batch Operations

Generate samples for multiple specific languages efficiently:

```bash
# European languages
python src/audio_generator.py --languages french german spanish italian --num-samples 5

# Asian languages (if configured)
python src/audio_generator.py --languages japanese korean mandarin --num-samples 3

# English variants
python src/audio_generator.py --languages american british australian --num-samples 8
```

### Quality Control

Monitor generation quality and adjust as needed:

```bash
# Generate one sample to test quality
python src/audio_generator.py --languages american --num-samples 1 --fresh

# Check detailed information about samples
python src/audio_generator.py --info

# Listen to generated samples manually
# (Use audio player of your choice)
```

## Integration with Training

### Training with Generated Samples

The accent classifier automatically uses generated samples for training:

```bash
# Train with existing samples (efficient)
python accent_classifier.py --train --use-tts --verbose

# Train with fresh samples (higher quality but slower)
python accent_classifier.py --train --use-tts --fresh --verbose
```

### Sample Count Recommendations

| Use Case | Samples per Language | TTS Engine | Command |
|----------|---------------------|------------|---------|
| **Quick Testing** | 3-5 | gTTS | `--num-samples 3` |
| **Development** | 5-8 | gTTS | `--num-samples 5` |
| **Production** | 10-15 | Cloud TTS | `--num-samples 10 --cloud-tts` |
| **High Accuracy** | 15+ | Cloud TTS | `--num-samples 20 --cloud-tts` |

## Monitoring and Reporting

### Generation Summary

The tool provides detailed summaries after generation:

```
                 Generation Summary                 
┏━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Language ┃ Samples ┃ Total Duration ┃ Total Size ┃
┡━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ American │ 5       │ 37.7s          │ 1178.4 KB  │
│ British  │ 5       │ 40.0s          │ 1249.6 KB  │
│ French   │ 5       │ 42.3s          │ 1325.1 KB  │
└──────────┴─────────┴────────────────┴────────────┘

📊 Total: 15 files, 3752.5 KB
```

### Training Data Information

Check current state of training data:

```bash
python src/audio_generator.py --info
```

```
                     Training Data Information                     
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Language        ┃ Accent Code ┃ Existing Samples ┃ Sample Texts ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ American Accent │ american    │ 5                │ 5            │
│ British Accent  │ british     │ 5                │ 5            │
│ French Accent   │ french      │ 5                │ 5            │
│ German Accent   │ german      │ 5                │ 5            │
│ Italian Accent  │ italian     │ 3                │ 5            │
│ Russian Accent  │ russian     │ 5                │ 5            │
│ Spanish Accent  │ spanish     │ 5                │ 5            │
└─────────────────┴─────────────┴──────────────────┴──────────────┘
```

## Troubleshooting

### Common Issues

#### 1. Generation Fails

**Symptoms:**
- Error messages during generation
- No audio files created
- TTS timeouts

**Solutions:**
```bash
# Test internet connection
ping google.com

# Test single language first
python src/audio_generator.py --languages american --num-samples 1

# Check configuration syntax
python -c "import json; print('Valid JSON' if json.load(open('audio_samples/american/config.json')) else 'Invalid JSON')"
```

#### 2. Poor Audio Quality

**Symptoms:**
- Distorted audio
- Low volume
- Unnatural speech

**Solutions:**
```bash
# Try different TTS engine
python src/audio_generator.py --languages american --num-samples 1 --cloud-tts --fresh

# Check sample rate settings in config.json
# Adjust voice selection in config.json
```

#### 3. Slow Generation

**Symptoms:**
- Very slow generation process
- Timeouts
- High CPU usage

**Solutions:**
```bash
# Generate fewer samples first
python src/audio_generator.py --languages american --num-samples 2

# Use existing samples when possible (remove --fresh)
python src/audio_generator.py --num-samples 5

# Generate one language at a time
python src/audio_generator.py --languages american --num-samples 5
```

#### 4. File Permission Errors

**Symptoms:**
- Permission denied errors
- Cannot create directories
- Cannot write files

**Solutions:**
```bash
# Check directory permissions
ls -la audio_samples/

# Create directories manually if needed
mkdir -p audio_samples/american

# Check disk space
df -h
```

### Debug Commands

```bash
# Test configuration loading
python -c "
from src.audio_generator import ScalableAudioGenerator
gen = ScalableAudioGenerator()
print('Languages:', gen.discover_languages())
"

# Test single sample generation
python src/audio_generator.py --languages american --num-samples 1 --fresh

# Validate all configurations
for dir in audio_samples/*/; do
    echo "Testing $(basename "$dir"):"
    python -c "import json; json.load(open('$dir/config.json')); print('✓ Valid')" 2>/dev/null || echo "✗ Invalid JSON"
done
```

## Performance Optimization

### Efficient Workflows

1. **During Development:**
   ```bash
   # Use existing samples
   python src/audio_generator.py --num-samples 5
   ```

2. **Before Training:**
   ```bash
   # Generate missing samples only
   python src/audio_generator.py --num-samples 10
   ```

3. **Final Production:**
   ```bash
   # Fresh high-quality samples
   python src/audio_generator.py --fresh --num-samples 15 --cloud-tts
   ```

### Batch Generation Strategy

```bash
# Week 1: Setup core languages
python src/audio_generator.py --languages american british french german spanish --num-samples 5

# Week 2: Add more languages
python src/audio_generator.py --languages italian russian --num-samples 5

# Week 3: Increase sample counts
python src/audio_generator.py --num-samples 10

# Week 4: Production quality
python src/audio_generator.py --fresh --num-samples 15 --cloud-tts
```

## Best Practices

### 1. Start Small
- Begin with 3-5 samples per language
- Test with a few languages first
- Scale up gradually based on results

### 2. Monitor Quality
- Listen to generated samples periodically
- Check training performance after generation
- Adjust configurations based on results

### 3. Use Version Control
- Track configuration changes
- Document generation settings used
- Keep backup copies of working setups

### 4. Optimize for Your Use Case
- Development: Use gTTS with existing samples
- Testing: Generate fresh samples periodically
- Production: Use Cloud TTS with many samples

The audio generation system is designed to be flexible, efficient, and scalable for any accent classification project needs. 