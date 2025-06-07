#!/usr/bin/env python3
"""
Scalable Audio Generator for Accent Classifier
Uses structured folders with configuration files for easy scaling.
"""

import os
import json
import tempfile
from typing import Dict, Optional, List, Tuple
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import logging
from pathlib import Path

# Import TTS libraries
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("gTTS not available. Install with: pip install gtts")

try:
    from google.cloud import texttospeech
    GOOGLE_CLOUD_TTS_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_TTS_AVAILABLE = False
    print("Google Cloud TTS not available. Install with: pip install google-cloud-texttospeech")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

class ScalableAudioGenerator:
    """
    Scalable audio generator that uses structured folders with configuration files.
    
    Folder structure:
    audio_samples/
    ├── american/
    │   ├── config.json
    │   ├── sample_001.wav
    │   ├── sample_002.wav
    │   └── ...
    ├── british/
    │   ├── config.json
    │   ├── sample_001.wav
    │   └── ...
    └── ...
    """
    
    def __init__(self, 
                 base_dir: str = "audio_samples",
                 use_cloud_tts: bool = False,
                 sample_rate: int = 16000):
        """
        Initialize the scalable audio generator.
        
        Args:
            base_dir: Base directory for audio samples
            use_cloud_tts: Whether to use Google Cloud TTS
            sample_rate: Target sample rate for audio
        """
        self.base_dir = Path(base_dir)
        self.use_cloud_tts = use_cloud_tts
        self.sample_rate = sample_rate
        
        # Initialize Cloud TTS client if needed
        if use_cloud_tts and GOOGLE_CLOUD_TTS_AVAILABLE:
            self.cloud_client = texttospeech.TextToSpeechClient()
        elif use_cloud_tts:
            raise ImportError("Google Cloud TTS not available")
        
        # Ensure base directory exists
        self.base_dir.mkdir(exist_ok=True)
    
    def discover_languages(self) -> List[str]:
        """
        Discover available language directories with config files.
        
        Returns:
            List of language directory names
        """
        languages = []
        
        for item in self.base_dir.iterdir():
            if item.is_dir():
                config_file = item / "config.json"
                if config_file.exists():
                    languages.append(item.name)
        
        return sorted(languages)
    
    def load_language_config(self, language: str) -> Dict:
        """
        Load configuration for a specific language.
        
        Args:
            language: Language directory name
            
        Returns:
            Configuration dictionary
        """
        config_file = self.base_dir / language / "config.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return config
    
    def get_existing_samples(self, language: str) -> List[Path]:
        """
        Get list of existing audio samples for a language.
        
        Args:
            language: Language directory name
            
        Returns:
            List of existing audio file paths
        """
        lang_dir = self.base_dir / language
        
        if not lang_dir.exists():
            return []
        
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(lang_dir.glob(ext))
        
        return sorted(audio_files)
    
    def generate_sample(self, 
                       language: str, 
                       text: str, 
                       output_path: Path, 
                       force: bool = False) -> bool:
        """
        Generate a single audio sample.
        
        Args:
            language: Language directory name
            text: Text to convert to speech
            output_path: Output file path
            force: Force regeneration even if file exists
            
        Returns:
            True if successful, False otherwise
        """
        # Skip if file exists and not forcing
        if output_path.exists() and not force:
            console.print(f"[blue]  ⏭ Skipping existing: {output_path.name}[/blue]")
            return True
        
        try:
            config = self.load_language_config(language)
            
            # Try Cloud TTS first if configured
            if self.use_cloud_tts and GOOGLE_CLOUD_TTS_AVAILABLE:
                success = self._generate_cloud_tts(config, text, output_path)
                if success:
                    return True
            
            # Fallback to gTTS
            if GTTS_AVAILABLE:
                return self._generate_gtts(config, text, output_path)
            else:
                console.print(f"[red]No TTS engines available for {language}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]Error generating {language} sample: {e}[/red]")
            return False
    
    def _generate_gtts(self, config: Dict, text: str, output_path: Path) -> bool:
        """Generate audio using gTTS."""
        try:
            tts_config = config['tts_config']['gtts']
            
            tts = gTTS(
                text=text,
                lang=tts_config['lang'],
                tld=tts_config['tld'],
                slow=tts_config.get('slow', False)
            )
            
            # Save to temporary MP3 file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                temp_mp3 = tmp_file.name
            
            tts.save(temp_mp3)
            
            # Convert to WAV with target sample rate
            audio = AudioSegment.from_mp3(temp_mp3)
            audio = audio.set_frame_rate(self.sample_rate).set_channels(1)
            audio.export(str(output_path), format="wav")
            
            # Clean up
            os.unlink(temp_mp3)
            
            console.print(f"[green]  ✓ Generated: {output_path.name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]  ✗ gTTS failed: {e}[/red]")
            return False
    
    def _generate_cloud_tts(self, config: Dict, text: str, output_path: Path) -> bool:
        """Generate audio using Google Cloud TTS."""
        try:
            tts_config = config['tts_config']['cloud_tts']
            
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=tts_config['language_code'],
                name=tts_config['voice_name']
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=tts_config.get('sample_rate_hertz', self.sample_rate)
            )
            
            response = self.cloud_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            with open(output_path, 'wb') as f:
                f.write(response.audio_content)
            
            console.print(f"[green]  ✓ Generated (Cloud): {output_path.name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]  ✗ Cloud TTS failed: {e}[/red]")
            return False
    
    def generate_language_samples(self, 
                                 language: str, 
                                 num_samples: int = 5,
                                 force: bool = False) -> List[Path]:
        """
        Generate audio samples for a specific language.
        
        Args:
            language: Language directory name
            num_samples: Number of samples to generate
            force: Force regeneration of existing files
            
        Returns:
            List of generated audio file paths
        """
        try:
            config = self.load_language_config(language)
            lang_dir = self.base_dir / language
            lang_dir.mkdir(exist_ok=True)
            
            console.print(f"\n[cyan]📢 Generating {language} samples...[/cyan]")
            
            sample_texts = config.get('sample_texts', [])
            if not sample_texts:
                console.print(f"[red]No sample texts found for {language}[/red]")
                return []
            
            generated_files = []
            
            # Generate samples up to the requested number
            for i in range(num_samples):
                # Use texts cyclically if we need more samples than texts
                text = sample_texts[i % len(sample_texts)]
                output_file = lang_dir / f"sample_{i+1:03d}.wav"
                
                if self.generate_sample(language, text, output_file, force):
                    generated_files.append(output_file)
            
            console.print(f"[green]✅ Generated {len(generated_files)} samples for {language}[/green]")
            return generated_files
            
        except Exception as e:
            console.print(f"[red]Error generating {language} samples: {e}[/red]")
            return []
    
    def generate_all_samples(self, 
                            num_samples_per_language: int = 5,
                            force: bool = False,
                            languages: Optional[List[str]] = None) -> Dict[str, List[Path]]:
        """
        Generate audio samples for all discovered languages.
        
        Args:
            num_samples_per_language: Number of samples per language
            force: Force regeneration of existing files
            languages: Specific languages to generate (None for all)
            
        Returns:
            Dictionary mapping language names to generated file paths
        """
        console.print(Panel.fit("🎙️ Scalable Audio Sample Generation", style="bold blue"))
        
        # Discover available languages
        available_languages = self.discover_languages()
        
        if not available_languages:
            console.print("[red]No language configurations found![/red]")
            return {}
        
        # Use specified languages or all available
        target_languages = languages if languages else available_languages
        
        # Filter to only available languages
        target_languages = [lang for lang in target_languages if lang in available_languages]
        
        if not target_languages:
            console.print("[red]No valid target languages found![/red]")
            return {}
        
        console.print(f"[blue]Target languages: {', '.join(target_languages)}[/blue]")
        
        all_generated_files = {}
        
        for language in target_languages:
            generated_files = self.generate_language_samples(
                language, 
                num_samples_per_language, 
                force
            )
            
            if generated_files:
                all_generated_files[language] = generated_files
        
        # Display summary
        self._display_generation_summary(all_generated_files)
        
        return all_generated_files
    
    def _display_generation_summary(self, generated_files: Dict[str, List[Path]]) -> None:
        """Display a summary of generated files."""
        table = Table(title="Generation Summary")
        table.add_column("Language", style="cyan")
        table.add_column("Samples", style="green")
        table.add_column("Total Duration", style="yellow")
        table.add_column("Total Size", style="blue")
        
        total_files = 0
        total_size = 0
        
        for language, files in generated_files.items():
            total_files += len(files)
            
            # Calculate total size and duration
            lang_size = 0
            lang_duration = 0
            
            for file_path in files:
                if file_path.exists():
                    lang_size += file_path.stat().st_size
                    
                    try:
                        audio_data, sr = sf.read(str(file_path))
                        lang_duration += len(audio_data) / sr
                    except:
                        pass
            
            total_size += lang_size
            
            table.add_row(
                language.title(),
                str(len(files)),
                f"{lang_duration:.1f}s",
                f"{lang_size / 1024:.1f} KB"
            )
        
        console.print(table)
        
        console.print(f"\n[green]📊 Total: {total_files} files, {total_size / 1024:.1f} KB[/green]")
    
    def get_training_data_info(self) -> Dict[str, Dict]:
        """
        Get information about available training data.
        
        Returns:
            Dictionary with training data information
        """
        info = {}
        
        for language in self.discover_languages():
            try:
                config = self.load_language_config(language)
                existing_samples = self.get_existing_samples(language)
                
                info[language] = {
                    'config': config,
                    'existing_samples': len(existing_samples),
                    'sample_paths': existing_samples,
                    'accent_code': config.get('accent_code', language)
                }
                
            except Exception as e:
                console.print(f"[red]Error loading {language}: {e}[/red]")
        
        return info
    
    def cleanup_language(self, language: str) -> None:
        """
        Remove all audio samples for a specific language.
        
        Args:
            language: Language directory name
        """
        lang_dir = self.base_dir / language
        
        if not lang_dir.exists():
            return
        
        # Remove all audio files
        for audio_file in self.get_existing_samples(language):
            audio_file.unlink()
            console.print(f"[yellow]Removed: {audio_file}[/yellow]")


def main():
    """Command-line interface for the scalable audio generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scalable Audio Sample Generator")
    parser.add_argument('--base-dir', default='audio_samples', help='Base directory for samples')
    parser.add_argument('--languages', nargs='+', help='Specific languages to generate')
    parser.add_argument('--num-samples', type=int, default=5, help='Samples per language')
    parser.add_argument('--fresh', action='store_true', help='Force regenerate existing files')
    parser.add_argument('--cloud-tts', action='store_true', help='Use Google Cloud TTS')
    parser.add_argument('--list', action='store_true', help='List available languages')
    parser.add_argument('--info', action='store_true', help='Show training data info')
    
    args = parser.parse_args()
    
    generator = ScalableAudioGenerator(
        base_dir=args.base_dir,
        use_cloud_tts=args.cloud_tts
    )
    
    if args.list:
        languages = generator.discover_languages()
        console.print(f"[blue]Available languages: {', '.join(languages)}[/blue]")
        return
    
    if args.info:
        info = generator.get_training_data_info()
        
        table = Table(title="Training Data Information")
        table.add_column("Language", style="cyan")
        table.add_column("Accent Code", style="green")
        table.add_column("Existing Samples", style="yellow")
        table.add_column("Sample Texts", style="blue")
        
        for lang, data in info.items():
            table.add_row(
                data['config']['language_name'],
                data['accent_code'],
                str(data['existing_samples']),
                str(len(data['config'].get('sample_texts', [])))
            )
        
        console.print(table)
        return
    
    # Generate samples
    generated_files = generator.generate_all_samples(
        num_samples_per_language=args.num_samples,
        force=args.fresh,
        languages=args.languages
    )
    
    if generated_files:
        console.print("[green]🎉 Audio generation completed successfully![/green]")
    else:
        console.print("[red]❌ No audio files were generated![/red]")


if __name__ == "__main__":
    main() 