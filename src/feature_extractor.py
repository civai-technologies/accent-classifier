"""
Feature extraction module for accent classifier.
Extracts relevant audio features for accent classification.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import librosa
from python_speech_features import mfcc, delta
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class FeatureExtractor:
    """Extracts audio features for accent classification."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize FeatureExtractor.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.n_mfcc = 13
        self.n_mels = 128
        self.hop_length = 512
        self.n_fft = 2048
        
    def extract_all_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive feature set from audio.
        
        Args:
            audio_data: Audio time series
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        # MFCC features
        features.update(self.extract_mfcc_features(audio_data))
        
        # Spectral features
        features.update(self.extract_spectral_features(audio_data))
        
        # Prosodic features
        features.update(self.extract_prosodic_features(audio_data))
        
        # Rhythm and timing features
        features.update(self.extract_rhythm_features(audio_data))
        
        # Formant features
        features.update(self.extract_formant_features(audio_data))
        
        return features
    
    def extract_mfcc_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract MFCC and related features.
        
        Args:
            audio_data: Audio time series
            
        Returns:
            Dictionary with MFCC-based features
        """
        # Extract MFCCs using python_speech_features (more traditional approach)
        mfcc_features = mfcc(
            audio_data, 
            self.sample_rate,
            numcep=self.n_mfcc,
            nfilt=26,
            nfft=self.n_fft,
            winstep=0.01
        )
        
        # Extract MFCCs using librosa (for comparison)
        mfcc_librosa = librosa.feature.mfcc(
            y=audio_data,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        
        # Delta and delta-delta features
        delta_features = delta(mfcc_features, 2)
        delta_delta_features = delta(delta_features, 2)
        
        # Statistical summaries
        mfcc_mean = np.mean(mfcc_features, axis=0)
        mfcc_std = np.std(mfcc_features, axis=0)
        mfcc_skew = skew(mfcc_features, axis=0)
        mfcc_kurt = kurtosis(mfcc_features, axis=0)
        
        return {
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'mfcc_skew': mfcc_skew,
            'mfcc_kurtosis': mfcc_kurt,
            'mfcc_delta_mean': np.mean(delta_features, axis=0),
            'mfcc_delta_std': np.std(delta_features, axis=0),
            'mfcc_delta_delta_mean': np.mean(delta_delta_features, axis=0),
            'mfcc_delta_delta_std': np.std(delta_delta_features, axis=0),
            'mfcc_raw': mfcc_features,
            'mfcc_librosa': mfcc_librosa.T
        }
    
    def extract_spectral_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract spectral features.
        
        Args:
            audio_data: Audio time series
            
        Returns:
            Dictionary with spectral features
        """
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            audio_data, hop_length=self.hop_length
        )[0]
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data, sr=self.sample_rate, n_mels=self.n_mels,
            hop_length=self.hop_length, n_fft=self.n_fft
        )
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        return {
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_rolloff_std': np.std(spectral_rolloff),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_bandwidth_std': np.std(spectral_bandwidth),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'chroma_mean': np.mean(chroma, axis=1),
            'chroma_std': np.std(chroma, axis=1),
            'mel_spectrogram_mean': np.mean(mel_spectrogram, axis=1),
            'mel_spectrogram_std': np.std(mel_spectrogram, axis=1),
            'spectral_contrast_mean': np.mean(spectral_contrast, axis=1),
            'spectral_contrast_std': np.std(spectral_contrast, axis=1)
        }
    
    def extract_prosodic_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract prosodic features (pitch, energy, etc.).
        
        Args:
            audio_data: Audio time series
            
        Returns:
            Dictionary with prosodic features
        """
        # Fundamental frequency (pitch)
        pitches, magnitudes = librosa.piptrack(
            y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Extract pitch values (remove zeros)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        pitch_values = np.array(pitch_values)
        
        # RMS energy
        rms = librosa.feature.rms(
            y=audio_data, hop_length=self.hop_length
        )[0]
        
        # Tempo estimation
        tempo, _ = librosa.beat.beat_track(
            y=audio_data, sr=self.sample_rate
        )
        
        features = {
            'rms_mean': np.mean(rms),
            'rms_std': np.std(rms),
            'tempo': tempo,
        }
        
        # Pitch statistics (if pitch values exist)
        if len(pitch_values) > 0:
            features.update({
                'pitch_mean': np.mean(pitch_values),
                'pitch_std': np.std(pitch_values),
                'pitch_min': np.min(pitch_values),
                'pitch_max': np.max(pitch_values),
                'pitch_range': np.max(pitch_values) - np.min(pitch_values),
                'pitch_skew': skew(pitch_values),
                'pitch_kurtosis': kurtosis(pitch_values)
            })
        else:
            # Default values if no pitch detected
            features.update({
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'pitch_min': 0.0,
                'pitch_max': 0.0,
                'pitch_range': 0.0,
                'pitch_skew': 0.0,
                'pitch_kurtosis': 0.0
            })
        
        return features
    
    def extract_rhythm_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract rhythm and timing features.
        
        Args:
            audio_data: Audio time series
            
        Returns:
            Dictionary with rhythm features
        """
        # Onset detection
        onset_frames = librosa.onset.onset_detect(
            y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Convert to time
        onset_times = librosa.frames_to_time(
            onset_frames, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Inter-onset intervals
        if len(onset_times) > 1:
            ioi = np.diff(onset_times)
            ioi_mean = np.mean(ioi)
            ioi_std = np.std(ioi)
            ioi_cv = ioi_std / ioi_mean if ioi_mean > 0 else 0
        else:
            ioi_mean = ioi_std = ioi_cv = 0
        
        # Rhythm regularity (based on autocorrelation of onset pattern)
        onset_strength = librosa.onset.onset_strength(
            y=audio_data, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Autocorrelation of onset strength
        autocorr = np.correlate(onset_strength, onset_strength, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation (indicates rhythmic regularity)
        peaks, _ = signal.find_peaks(autocorr, height=np.max(autocorr) * 0.1)
        rhythm_regularity = len(peaks) / len(autocorr) if len(autocorr) > 0 else 0
        
        return {
            'onset_rate': len(onset_times) / (len(audio_data) / self.sample_rate),
            'ioi_mean': ioi_mean,
            'ioi_std': ioi_std,
            'ioi_cv': ioi_cv,
            'rhythm_regularity': rhythm_regularity,
            'onset_strength_mean': np.mean(onset_strength),
            'onset_strength_std': np.std(onset_strength)
        }
    
    def extract_formant_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract formant frequencies (simplified approach).
        
        Args:
            audio_data: Audio time series
            
        Returns:
            Dictionary with formant features
        """
        # Frame the audio
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.01 * self.sample_rate)     # 10ms hop
        
        frames = librosa.util.frame(
            audio_data, frame_length=frame_length, 
            hop_length=hop_length, axis=0
        )
        
        formant_features = {
            'f1_mean': 0.0, 'f1_std': 0.0,
            'f2_mean': 0.0, 'f2_std': 0.0,
            'f3_mean': 0.0, 'f3_std': 0.0,
            'formant_bandwidth_mean': 0.0
        }
        
        # Simplified formant estimation using spectral peaks
        formants_all = []
        
        for frame in frames.T:
            if np.sum(frame**2) < 1e-6:  # Skip silent frames
                continue
                
            # Apply window
            windowed = frame * np.hanning(len(frame))
            
            # FFT
            fft = np.fft.rfft(windowed, n=2048)
            magnitude = np.abs(fft)
            
            # Find spectral peaks (simplified formant estimation)
            peaks, _ = signal.find_peaks(magnitude, height=np.max(magnitude) * 0.1)
            
            # Convert peak indices to frequencies
            freqs = peaks * self.sample_rate / (2 * len(magnitude))
            
            # Keep only reasonable formant frequencies (80-4000 Hz)
            formant_freqs = freqs[(freqs >= 80) & (freqs <= 4000)]
            
            if len(formant_freqs) >= 3:
                formants_all.append(formant_freqs[:3])  # First 3 formants
        
        # Calculate statistics if formants were found
        if formants_all:
            formants_array = np.array(formants_all)
            formant_features.update({
                'f1_mean': np.mean(formants_array[:, 0]),
                'f1_std': np.std(formants_array[:, 0]),
                'f2_mean': np.mean(formants_array[:, 1]),
                'f2_std': np.std(formants_array[:, 1]),
                'f3_mean': np.mean(formants_array[:, 2]),
                'f3_std': np.std(formants_array[:, 2]),
                'formant_bandwidth_mean': np.mean(np.std(formants_array, axis=1))
            })
        
        return formant_features
    
    def get_feature_vector(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract all features and return as a single feature vector.
        
        Args:
            audio_data: Audio time series
            
        Returns:
            Feature vector as numpy array
        """
        all_features = self.extract_all_features(audio_data)
        
        # Flatten all features into a single vector
        feature_vector = []
        
        for key, value in all_features.items():
            if key in ['mfcc_raw', 'mfcc_librosa']:
                # Skip raw MFCC matrices for now
                continue
            
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    feature_vector.append(value.item())
                else:
                    feature_vector.extend(value.flatten())
            else:
                feature_vector.append(value)
        
        return np.array(feature_vector)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all features in the feature vector.
        
        Returns:
            List of feature names
        """
        # This would need to be implemented based on the exact features
        # For now, return a placeholder
        return [f"feature_{i}" for i in range(100)]  # Approximate number 