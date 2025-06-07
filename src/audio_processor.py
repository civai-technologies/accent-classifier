"""
Audio processing module for accent classifier.
Handles audio input, preprocessing, and format conversion.
"""

import os
import warnings
from typing import Tuple, Optional, Union
import numpy as np
import librosa
import soundfile as sf
import pyaudio
from pydub import AudioSegment
from rich.console import Console

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)

console = Console()


class AudioProcessor:
    """Handles audio file loading, preprocessing, and real-time capture."""
    
    def __init__(self, target_sr: int = 16000, duration_threshold: float = 2.0):
        """
        Initialize AudioProcessor.
        
        Args:
            target_sr: Target sample rate for audio processing
            duration_threshold: Minimum duration in seconds for valid audio
        """
        self.target_sr = target_sr
        self.duration_threshold = duration_threshold
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        
    def load_audio_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and preprocess it.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio is too short or invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        try:
            # Try loading with librosa first
            audio_data, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
        except Exception as e:
            # Fallback to pydub for other formats
            try:
                audio_segment = AudioSegment.from_file(file_path)
                audio_segment = audio_segment.set_frame_rate(self.target_sr).set_channels(1)
                audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize
                sr = self.target_sr
            except Exception as fallback_error:
                raise ValueError(f"Could not load audio file: {e}, {fallback_error}")
        
        # Validate audio duration
        duration = len(audio_data) / sr
        if duration < self.duration_threshold:
            raise ValueError(f"Audio too short: {duration:.2f}s < {self.duration_threshold}s")
            
        # Preprocess audio
        audio_data = self._preprocess_audio(audio_data)
        
        return audio_data, sr
    
    def record_audio(self, duration: float = 10.0) -> Tuple[np.ndarray, int]:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            RuntimeError: If microphone access fails
        """
        try:
            p = pyaudio.PyAudio()
            
            # Open stream
            stream = p.open(
                format=self.audio_format,
                channels=1,
                rate=self.target_sr,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            console.print(f"[green]Recording for {duration} seconds...[/green]")
            
            frames = []
            for _ in range(int(self.target_sr / self.chunk_size * duration)):
                data = stream.read(self.chunk_size)
                frames.append(data)
            
            console.print("[green]Recording complete![/green]")
            
            # Close stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
            
            # Preprocess audio
            audio_data = self._preprocess_audio(audio_data)
            
            return audio_data, self.target_sr
            
        except Exception as e:
            raise RuntimeError(f"Failed to record audio: {e}")
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data with normalization and noise reduction.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Preprocessed audio data
        """
        # Remove DC offset
        audio_data = audio_data - np.mean(audio_data)
        
        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Simple noise reduction using spectral gating
        audio_data = self._spectral_noise_reduction(audio_data)
        
        return audio_data
    
    def _spectral_noise_reduction(self, audio_data: np.ndarray, 
                                  noise_factor: float = 0.02) -> np.ndarray:
        """
        Apply basic spectral noise reduction.
        
        Args:
            audio_data: Input audio data
            noise_factor: Noise reduction factor
            
        Returns:
            Noise-reduced audio data
        """
        # Compute STFT
        stft = librosa.stft(audio_data)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise floor from first 0.5 seconds
        noise_samples = int(0.5 * self.target_sr / 512)  # 512 is hop length
        noise_profile = np.mean(magnitude[:, :noise_samples], axis=1, keepdims=True)
        
        # Apply spectral gating
        mask = magnitude > (noise_profile * (1 + noise_factor))
        magnitude_clean = magnitude * mask
        
        # Reconstruct audio
        stft_clean = magnitude_clean * np.exp(1j * phase)
        audio_clean = librosa.istft(stft_clean)
        
        return audio_clean
    
    def save_audio(self, audio_data: np.ndarray, output_path: str, 
                   sample_rate: int = None) -> None:
        """
        Save audio data to file.
        
        Args:
            audio_data: Audio data to save
            output_path: Output file path
            sample_rate: Sample rate (uses target_sr if None)
        """
        if sample_rate is None:
            sample_rate = self.target_sr
            
        sf.write(output_path, audio_data, sample_rate)
        console.print(f"[green]Audio saved to: {output_path}[/green]")
    
    def get_audio_info(self, audio_data: np.ndarray, sample_rate: int) -> dict:
        """
        Get basic information about audio data.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            Dictionary with audio information
        """
        duration = len(audio_data) / sample_rate
        rms_energy = np.sqrt(np.mean(audio_data**2))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        return {
            'duration': duration,
            'sample_rate': sample_rate,
            'samples': len(audio_data),
            'rms_energy': rms_energy,
            'zero_crossing_rate': zero_crossing_rate,
            'max_amplitude': np.max(np.abs(audio_data))
        } 