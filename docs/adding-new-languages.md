# Adding New Languages to Accent Classifier

This guide explains how to add new languages and accents to the Accent Classifier system.

## Overview

The Accent Classifier uses a scalable, configuration-driven approach that makes adding new languages simple and straightforward. Each language requires:

1. **A dedicated directory** in `audio_samples/`
2. **A configuration file** (`config.json`) with TTS settings and sample texts
3. **Audio sample generation** using the built-in tools

## Step-by-Step Guide

### Step 1: Create Language Directory

Create a new directory for your language in the `audio_samples/` folder:

```bash
mkdir audio_samples/portuguese
```

**Naming Convention:**
- Use lowercase language names
- Use simple, descriptive names (e.g., `italian`, `portuguese`, `mandarin`)
- For regional variants, use descriptive names (e.g., `brazilian`, `australian`, `canadian`)

### Step 2: Create Configuration File

Create a `config.json` file in your language directory with the following structure:

```json
{
  "language_name": "Portuguese Accent",
  "accent_code": "portuguese",
  "tts_config": {
    "gtts": {
      "lang": "pt",
      "tld": "pt",
      "slow": false
    },
    "cloud_tts": {
      "language_code": "pt-PT",
      "voice_name": "pt-PT-Wavenet-A",
      "audio_encoding": "LINEAR16",
      "sample_rate_hertz": 16000
    }
  },
  "sample_texts": [
    "Olá, como está hoje? Vou ao mercado comprar pão, queijo e vinho.",
    "Oi! Sou de Portugal e gosto de comer pastéis de nata e beber café.",
    "A língua portuguesa é falada por mais de 250 milhões de pessoas no mundo.",
    "Trabalho num escritório no centro de Lisboa. Todas as manhãs apanho o metro.",
    "Portugal é famoso pela sua história, literatura e cultura rica em tradições."
  ],
  "description": "European Portuguese accent with characteristic pronunciation patterns."
}
```

#### Configuration Fields Explained

| Field | Description | Required |
|-------|-------------|----------|
| `language_name` | Human-readable language name (displayed in UI) | ✅ Yes |
| `accent_code` | Unique identifier for the accent (used internally) | ✅ Yes |
| `tts_config.gtts.lang` | gTTS language code (ISO 639-1) | ✅ Yes |
| `tts_config.gtts.tld` | Top-level domain for regional variants | ✅ Yes |
| `tts_config.gtts.slow` | Whether to use slow speech (usually `false`) | ✅ Yes |
| `tts_config.cloud_tts.language_code` | Google Cloud TTS language code | ✅ Yes |
| `tts_config.cloud_tts.voice_name` | Specific voice model to use | ✅ Yes |
| `tts_config.cloud_tts.audio_encoding` | Audio format (usually `LINEAR16`) | ✅ Yes |
| `tts_config.cloud_tts.sample_rate_hertz` | Sample rate (usually `16000`) | ✅ Yes |
| `sample_texts` | Array of text samples for audio generation | ✅ Yes |
| `description` | Brief description of the accent characteristics | ❌ No |

### Step 3: Choose Appropriate TTS Settings

#### gTTS Settings

Find the correct gTTS settings:

```python
# Common gTTS language codes
languages = {
    'pt': 'Portuguese',
    'es': 'Spanish', 
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'hi': 'Hindi'
}

# Common TLD variants
tlds = {
    'us': 'United States',
    'co.uk': 'United Kingdom', 
    'ca': 'Canada',
    'com.au': 'Australia',
    'co.in': 'India',
    'pt': 'Portugal',
    'es': 'Spain',
    'fr': 'France',
    'de': 'Germany'
}
```

#### Google Cloud TTS Settings

Find available voices at: [Google Cloud TTS Voices](https://cloud.google.com/text-to-speech/docs/voices)

Example voice configurations:

```json
{
  "Portuguese (Portugal)": {
    "language_code": "pt-PT",
    "voice_name": "pt-PT-Wavenet-A"
  },
  "Portuguese (Brazil)": {
    "language_code": "pt-BR", 
    "voice_name": "pt-BR-Wavenet-A"
  },
  "Japanese": {
    "language_code": "ja-JP",
    "voice_name": "ja-JP-Wavenet-A"
  },
  "Korean": {
    "language_code": "ko-KR",
    "voice_name": "ko-KR-Wavenet-A"
  }
}
```

### Step 4: Write Quality Sample Texts

#### Guidelines for Sample Texts

1. **Use native language** for the target accent
2. **Include accent-specific words** and pronunciation patterns
3. **Vary sentence structure** and vocabulary
4. **Keep reasonable length** (10-20 words per sentence)
5. **Include common sounds** that distinguish the accent

#### Example Strategies

**For Regional English Variants:**
```json
{
  "Australian English": [
    "G'day mate! I'm going to the bottle shop to grab some tinnies for the barbie.",
    "Fair dinkum, that's a beaut of a day! Let's take the ute to the footy match.",
    "No worries mate, I'll chuck a U-ey and meet you at Maccas for brekkie."
  ]
}
```

**For Non-English Languages:**
```json
{
  "Mandarin Chinese": [
    "你好，今天天气很好。我要去商店买一些东西。",
    "我来自中国北京，很高兴认识你。中文是世界上使用最广泛的语言之一。",
    "我在一家公司工作，每天乘地铁上班。中国有着悠久的历史和丰富的文化。"
  ]
}
```

### Step 5: Generate Audio Samples

Once your configuration is ready, generate audio samples:

```bash
# Generate samples for your new language
python src/audio_generator.py --languages portuguese --num-samples 5

# Verify the language was discovered
python src/audio_generator.py --list

# Check the generated samples
python src/audio_generator.py --info
```

### Step 6: Train Model with New Language

Include your new language in training:

```bash
# Train with the new language included
python accent_classifier.py --train --use-tts --verbose

# For fresh training with all new samples
python accent_classifier.py --train --use-tts --fresh --verbose
```

### Step 7: Test the New Language

Test your new language:

```bash
# Test a specific audio file
python accent_classifier.py --file audio_samples/portuguese/sample_001.wav --verbose

# Test with microphone (if you can speak the language)
python accent_classifier.py --microphone --duration 10 --verbose
```

## Advanced Configuration

### Custom Sample Rate

For higher quality audio:

```json
{
  "tts_config": {
    "cloud_tts": {
      "sample_rate_hertz": 22050
    }
  }
}
```

### Multiple Regional Variants

Create separate directories for regional variants:

```
audio_samples/
├── portuguese-pt/     # European Portuguese
├── portuguese-br/     # Brazilian Portuguese
├── english-us/        # American English
├── english-uk/        # British English
└── english-au/        # Australian English
```

### Voice Gender Selection

Choose different voice genders in Cloud TTS:

```json
{
  "voice_name": "pt-PT-Wavenet-A",  // Female
  "voice_name": "pt-PT-Wavenet-B",  // Male
  "voice_name": "pt-PT-Wavenet-C",  // Female
  "voice_name": "pt-PT-Wavenet-D"   // Male
}
```

## Troubleshooting

### Common Issues

#### 1. Language Not Discovered

**Problem:** New language doesn't appear in `--list`
**Solution:** 
- Ensure `config.json` exists in the language directory
- Validate JSON syntax using online validators
- Check file permissions

#### 2. TTS Generation Fails

**Problem:** Audio generation fails with errors
**Solution:**
- Verify internet connection for gTTS
- Check language codes are correct
- Test with simpler sample texts first

#### 3. Poor Accent Classification

**Problem:** Model doesn't distinguish the new accent well
**Solution:**
- Add more diverse sample texts
- Include accent-specific vocabulary
- Increase number of training samples
- Use `--fresh` flag to regenerate training data

#### 4. Audio Quality Issues

**Problem:** Generated audio sounds poor or distorted
**Solution:**
- Try different voice models in Cloud TTS
- Adjust sample rate settings
- Use shorter, clearer sample texts

### Debug Commands

```bash
# Test single language generation
python src/audio_generator.py --languages portuguese --num-samples 1 --fresh

# Validate configuration
python -c "import json; print(json.load(open('audio_samples/portuguese/config.json')))"

# Check generated files
ls -la audio_samples/portuguese/

# Test TTS settings manually
python -c "
from gtts import gTTS
tts = gTTS('Hello world', lang='pt', tld='pt')
tts.save('test.wav')
print('TTS test successful')
"
```

## Best Practices

### 1. Configuration Management
- Use version control for configuration files
- Document any custom voice settings
- Test both gTTS and Cloud TTS configurations
- Keep backup copies of working configurations

### 2. Sample Text Quality
- Review texts with native speakers when possible
- Include diverse vocabulary and sentence structures
- Test pronunciation of difficult sounds
- Avoid overly technical or uncommon words

### 3. Incremental Development
- Start with one language at a time
- Test each language thoroughly before adding more
- Monitor training performance as you add languages
- Use version control to track changes

### 4. Performance Optimization
- Start with 5 samples per language for testing
- Scale up to 10+ samples for production
- Use `--fresh` sparingly to avoid unnecessary generation
- Monitor model accuracy and adjust as needed

## Examples

### Complete Example: Adding Japanese

**1. Create directory:**
```bash
mkdir audio_samples/japanese
```

**2. Create `audio_samples/japanese/config.json`:**
```json
{
  "language_name": "Japanese Accent",
  "accent_code": "japanese", 
  "tts_config": {
    "gtts": {
      "lang": "ja",
      "tld": "jp",
      "slow": false
    },
    "cloud_tts": {
      "language_code": "ja-JP",
      "voice_name": "ja-JP-Wavenet-A",
      "audio_encoding": "LINEAR16",
      "sample_rate_hertz": 16000
    }
  },
  "sample_texts": [
    "こんにちは、今日はいい天気ですね。市場にパンとチーズとワインを買いに行きます。",
    "はじめまして！日本から来ました。寿司とラーメンが大好きです。東京はとても美しい都市です。",
    "日本語は世界で約1億2千万人に話されています。豊かな文学と文化の伝統があります。",
    "私は東京の中心部のオフィスで働いています。毎朝電車で通勤し、夕方に家に帰ります。",
    "日本は芸術、音楽、料理で有名です。葛飾北斎や歌川広重などの偉大な芸術家がいます。"
  ],
  "description": "Standard Japanese accent with characteristic phonetic patterns and pitch accent."
}
```

**3. Generate samples:**
```bash
python src/audio_generator.py --languages japanese --num-samples 5
```

**4. Train and test:**
```bash
python accent_classifier.py --train --use-tts --verbose
```

## Support

If you encounter issues adding new languages:

1. Check existing language configurations for reference
2. Validate your JSON configuration files
3. Test TTS generation manually before training
4. Start with simple sample texts and gradually add complexity
5. Monitor training metrics to ensure the new language is learning correctly

The scalable architecture makes it easy to support dozens of languages and accents while maintaining high performance and accuracy. 