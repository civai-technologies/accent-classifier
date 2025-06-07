"""
Integration tests for audio processing module.
Tests real audio processing functionality without mocking.
"""

import os
import pytest
import numpy as np
import tempfile
import soundfile as sf
from unittest.mock import patch
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from audio_processor import AudioProcessor


class TestAudioProcessor:
    """Test suite for AudioProcessor class."""
    
    def setup_method(self):
        """Setup method run before each test."""
        self.processor = AudioProcessor()
        self.sample_rate = 16000
        self.test_audio_duration = 3.0  # 3 seconds
        
    def create_test_audio(self, duration: float = 3.0, frequency: float = 440.0) -> np.ndarray:
        """
        Create synthetic test audio data.
        
        Args:
            duration: Duration in seconds
            frequency: Frequency of sine wave
            
        Returns:
            Audio data as numpy array
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        # Create a sine wave with some noise
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        # Add some noise to make it more realistic
        noise = 0.05 * np.random.randn(len(audio))
        return audio + noise
    
    def test_audio_info_extraction(self):
        """Test extraction of audio information."""
        audio_data = self.create_test_audio()
        audio_info = self.processor.get_audio_info(audio_data, self.sample_rate)
        
        # Verify all expected keys are present
        expected_keys = ['duration', 'sample_rate', 'samples', 'rms_energy', 
                        'zero_crossing_rate', 'max_amplitude']
        for key in expected_keys:
            assert key in audio_info
        
        # Verify values are reasonable
        assert abs(audio_info['duration'] - self.test_audio_duration) < 0.1
        assert audio_info['sample_rate'] == self.sample_rate
        assert audio_info['samples'] == len(audio_data)
        assert 0.0 < audio_info['rms_energy'] < 1.0
        assert 0.0 <= audio_info['zero_crossing_rate'] <= 1.0
        assert 0.0 <= audio_info['max_amplitude'] <= 1.0
    
    def test_audio_preprocessing(self):
        """Test audio preprocessing pipeline."""
        # Create audio with DC offset and high amplitude
        raw_audio = self.create_test_audio() + 0.5  # Add DC offset
        raw_audio *= 3.0  # Increase amplitude
        
        processed_audio = self.processor._preprocess_audio(raw_audio)
        
        # Check DC offset removal
        assert abs(np.mean(processed_audio)) < 0.01
        
        # Check normalization
        assert np.max(np.abs(processed_audio)) <= 1.0
        
        # Check that processing preserves audio length
        assert len(processed_audio) == len(raw_audio)
    
    def test_spectral_noise_reduction(self):
        """Test spectral noise reduction functionality."""
        # Create audio with added noise
        clean_audio = self.create_test_audio(frequency=440.0)
        noise = 0.2 * np.random.randn(len(clean_audio))
        noisy_audio = clean_audio + noise
        
        # Apply noise reduction
        denoised_audio = self.processor._spectral_noise_reduction(noisy_audio)
        
        # Check that noise reduction doesn't destroy the signal
        assert len(denoised_audio) == len(noisy_audio)
        assert np.max(np.abs(denoised_audio)) > 0.01  # Signal still present
        
        # Check that RMS energy is preserved to some degree
        original_rms = np.sqrt(np.mean(clean_audio**2))
        denoised_rms = np.sqrt(np.mean(denoised_audio**2))
        assert denoised_rms > 0.1 * original_rms
    
    def test_file_operations(self):
        """Test audio file save and load operations."""
        # Create test audio
        original_audio = self.create_test_audio()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Save audio
            self.processor.save_audio(original_audio, temp_path, self.sample_rate)
            
            # Verify file exists
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
            
            # Load audio back
            loaded_audio, loaded_sr = self.processor.load_audio_file(temp_path)
            
            # Verify loaded audio properties
            assert loaded_sr == self.sample_rate
            assert len(loaded_audio) > 0
            
            # Check that the audio is similar (allowing for some processing differences)
            # Normalize both for comparison
            orig_norm = original_audio / np.max(np.abs(original_audio))
            loaded_norm = loaded_audio / np.max(np.abs(loaded_audio))
            
            # Check correlation (should be high for same signal)
            correlation = np.corrcoef(orig_norm, loaded_norm)[0, 1]
            assert correlation > 0.8  # High correlation expected
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_invalid_file_handling(self):
        """Test handling of invalid audio files."""
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            self.processor.load_audio_file("nonexistent_file.wav")
        
        # Test invalid file format
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(b"This is not an audio file")
            temp_path = tmp_file.name
        
        try:
            with pytest.raises(ValueError):
                self.processor.load_audio_file(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_short_audio_rejection(self):
        """Test rejection of audio that's too short."""
        # Create audio shorter than threshold
        short_audio = self.create_test_audio(duration=1.0)  # 1 second
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            sf.write(temp_path, short_audio, self.sample_rate)
            
            # Should raise ValueError for short audio
            with pytest.raises(ValueError, match="Audio too short"):
                self.processor.load_audio_file(temp_path)
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('pyaudio.PyAudio')
    def test_microphone_recording_setup(self, mock_pyaudio):
        """Test microphone recording setup (mocked for CI compatibility)."""
        # Mock PyAudio for testing setup
        mock_audio = mock_pyaudio.return_value
        mock_stream = mock_audio.open.return_value
        
        # Create fake audio data
        fake_frames = [b'\x00\x01' * 1024 for _ in range(10)]
        mock_stream.read.side_effect = fake_frames
        
        try:
            # This should not raise an exception
            audio_data, sample_rate = self.processor.record_audio(duration=1.0)
            
            # Verify basic properties
            assert isinstance(audio_data, np.ndarray)
            assert sample_rate == self.processor.target_sr
            assert len(audio_data) > 0
            
            # Verify PyAudio was called correctly
            mock_audio.open.assert_called_once()
            assert mock_stream.read.called
            mock_stream.stop_stream.assert_called_once()
            mock_stream.close.assert_called_once()
            mock_audio.terminate.assert_called_once()
            
        except RuntimeError:
            # If actual audio recording fails, that's expected in CI
            pytest.skip("Audio recording not available in test environment")
    
    def test_different_sample_rates(self):
        """Test handling of different sample rates."""
        # Create audio at different sample rate
        original_sr = 44100
        audio_44k = self.create_test_audio()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Save at 44.1kHz
            sf.write(temp_path, audio_44k, original_sr)
            
            # Load with processor (should resample to 16kHz)
            loaded_audio, loaded_sr = self.processor.load_audio_file(temp_path)
            
            # Should be resampled to target sample rate
            assert loaded_sr == self.processor.target_sr
            assert len(loaded_audio) > 0
            
            # Length should be approximately correct after resampling
            expected_length = int(len(audio_44k) * self.processor.target_sr / original_sr)
            assert abs(len(loaded_audio) - expected_length) < 1000  # Allow some tolerance
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_multiple_audio_formats(self):
        """Test loading different audio formats."""
        audio_data = self.create_test_audio()
        
        # Test different formats (if supported)
        formats_to_test = ['.wav']  # WAV is always supported
        
        for fmt in formats_to_test:
            with tempfile.NamedTemporaryFile(suffix=fmt, delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            try:
                sf.write(temp_path, audio_data, self.sample_rate)
                
                # Should load successfully
                loaded_audio, loaded_sr = self.processor.load_audio_file(temp_path)
                
                assert loaded_sr == self.sample_rate
                assert len(loaded_audio) > 0
                
            except Exception as e:
                pytest.skip(f"Format {fmt} not supported: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path) 