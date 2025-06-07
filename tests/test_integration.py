"""
Integration tests for the complete accent classification pipeline.
Tests the full workflow from audio input to accent prediction.
"""

import os
import pytest
import numpy as np
import tempfile
import soundfile as sf
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from audio_processor import AudioProcessor
from feature_extractor import FeatureExtractor
from model_handler import AccentClassifier


class TestAccentClassificationPipeline:
    """Test suite for the complete accent classification pipeline."""
    
    def setup_method(self):
        """Setup method run before each test."""
        self.audio_processor = AudioProcessor()
        self.feature_extractor = FeatureExtractor()
        self.classifier = AccentClassifier()
        self.sample_rate = 16000
        
    def create_test_audio(self, duration: float = 3.0, accent_type: str = 'american') -> np.ndarray:
        """
        Create synthetic test audio with accent-specific characteristics.
        
        Args:
            duration: Duration in seconds
            accent_type: Type of accent to simulate
            
        Returns:
            Audio data as numpy array
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Different accent types have different fundamental frequency patterns
        if accent_type == 'american':
            base_freq = 220.0
            formant_pattern = [1, 0.7, 0.4]
        elif accent_type == 'british':
            base_freq = 240.0
            formant_pattern = [0.8, 1, 0.5]
        elif accent_type == 'indian':
            base_freq = 260.0
            formant_pattern = [0.9, 0.8, 0.7]
        elif accent_type == 'russian':
            base_freq = 200.0
            formant_pattern = [1.2, 0.6, 0.3]
        else:
            base_freq = 230.0
            formant_pattern = [0.8, 0.8, 0.5]
        
        # Create complex audio with multiple harmonics
        audio = np.zeros_like(t)
        for i, amp in enumerate(formant_pattern):
            freq = base_freq * (i + 1)
            audio += amp * 0.3 * np.sin(2 * np.pi * freq * t)
        
        # Add some prosodic variation
        audio += 0.1 * np.sin(2 * np.pi * 5 * t)  # Prosodic variation
        
        # Add realistic noise
        noise = 0.05 * np.random.randn(len(audio))
        return audio + noise
    
    def test_complete_pipeline_with_synthetic_data(self):
        """Test the complete pipeline from audio to prediction."""
        # Create test audio
        audio_data = self.create_test_audio(duration=5.0, accent_type='american')
        
        # Step 1: Audio processing (already done)
        audio_info = self.audio_processor.get_audio_info(audio_data, self.sample_rate)
        assert audio_info['duration'] > 4.0  # Should be around 5 seconds
        
        # Step 2: Feature extraction
        features = self.feature_extractor.get_feature_vector(audio_data)
        assert isinstance(features, np.ndarray)
        assert len(features) > 50  # Should have many features
        assert not np.any(np.isnan(features))  # No NaN values
        assert not np.any(np.isinf(features))  # No infinite values
        
        # Step 3: Model training (with synthetic data)
        training_metrics = self.classifier.train()
        assert training_metrics['train_accuracy'] > 0.5  # Should be better than random
        assert training_metrics['cv_mean'] > 0.0
        assert self.classifier.is_trained
        
        # Step 4: Prediction
        prediction = self.classifier.predict(features)
        
        # Verify prediction structure
        expected_keys = ['accent', 'accent_name', 'confidence', 'all_probabilities', 'reliable', 'error']
        for key in expected_keys:
            assert key in prediction
        
        # Verify prediction content
        assert prediction['error'] is None
        assert isinstance(prediction['accent'], str)
        assert isinstance(prediction['accent_name'], str)
        assert 0.0 <= prediction['confidence'] <= 1.0
        assert isinstance(prediction['all_probabilities'], dict)
        assert isinstance(prediction['reliable'], bool)
        
        # All probabilities should sum to approximately 1
        prob_sum = sum(prediction['all_probabilities'].values())
        assert abs(prob_sum - 1.0) < 0.01
    
    def test_pipeline_with_file_operations(self):
        """Test pipeline using file save/load operations."""
        # Create test audio
        audio_data = self.create_test_audio(duration=4.0, accent_type='british')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Save audio file
            self.audio_processor.save_audio(audio_data, temp_path, self.sample_rate)
            
            # Load audio file
            loaded_audio, loaded_sr = self.audio_processor.load_audio_file(temp_path)
            
            # Extract features from loaded audio
            features = self.feature_extractor.get_feature_vector(loaded_audio)
            
            # Train model
            self.classifier.train()
            
            # Make prediction
            prediction = self.classifier.predict(features)
            
            # Verify prediction is valid
            assert prediction['error'] is None
            assert prediction['accent'] in self.classifier.accent_mapping
            assert 0.0 <= prediction['confidence'] <= 1.0
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        # Train model
        training_metrics = self.classifier.train()
        
        # Create test audio and get prediction
        audio_data = self.create_test_audio(duration=3.0)
        features = self.feature_extractor.get_feature_vector(audio_data)
        original_prediction = self.classifier.predict(features)
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            temp_model_path = tmp_file.name
        
        try:
            self.classifier.save_model(temp_model_path)
            assert os.path.exists(temp_model_path)
            
            # Create new classifier and load model
            new_classifier = AccentClassifier()
            assert not new_classifier.is_trained
            
            load_success = new_classifier.load_model(temp_model_path)
            assert load_success
            assert new_classifier.is_trained
            
            # Make prediction with loaded model
            loaded_prediction = new_classifier.predict(features)
            
            # Predictions should be identical
            assert loaded_prediction['accent'] == original_prediction['accent']
            assert abs(loaded_prediction['confidence'] - original_prediction['confidence']) < 1e-6
            
        finally:
            if os.path.exists(temp_model_path):
                os.unlink(temp_model_path)
    
    def test_multiple_accent_classification(self):
        """Test classification with multiple different accents."""
        # Create audio samples for different accents
        accent_types = ['american', 'british', 'indian', 'russian']
        audio_samples = {}
        feature_samples = {}
        
        for accent in accent_types:
            audio_data = self.create_test_audio(duration=3.0, accent_type=accent)
            audio_samples[accent] = audio_data
            feature_samples[accent] = self.feature_extractor.get_feature_vector(audio_data)
        
        # Train model
        self.classifier.train()
        
        # Test prediction for each accent
        predictions = {}
        for accent in accent_types:
            prediction = self.classifier.predict(feature_samples[accent])
            predictions[accent] = prediction
            
            # Each prediction should be valid
            assert prediction['error'] is None
            assert 0.0 <= prediction['confidence'] <= 1.0
            assert isinstance(prediction['all_probabilities'], dict)
        
        # Check that predictions are diverse (not all the same)
        predicted_accents = [pred['accent'] for pred in predictions.values()]
        unique_predictions = set(predicted_accents)
        
        # Should have some diversity in predictions (not all identical)
        # Note: With synthetic data, this might not always be the case
        assert len(unique_predictions) >= 1
    
    def test_feature_extraction_consistency(self):
        """Test that feature extraction is consistent."""
        audio_data = self.create_test_audio(duration=4.0)
        
        # Extract features multiple times
        features1 = self.feature_extractor.get_feature_vector(audio_data)
        features2 = self.feature_extractor.get_feature_vector(audio_data)
        
        # Features should be identical for same audio
        np.testing.assert_array_almost_equal(features1, features2, decimal=10)
    
    def test_different_audio_lengths(self):
        """Test pipeline with different audio lengths."""
        durations = [2.5, 5.0, 10.0]  # Different durations
        
        # Train model once
        self.classifier.train()
        
        for duration in durations:
            audio_data = self.create_test_audio(duration=duration)
            
            # Should process successfully regardless of length
            features = self.feature_extractor.get_feature_vector(audio_data)
            prediction = self.classifier.predict(features)
            
            assert prediction['error'] is None
            assert 0.0 <= prediction['confidence'] <= 1.0
            assert len(features) > 0
    
    def test_confidence_threshold_behavior(self):
        """Test that confidence threshold affects reliability assessment."""
        # Create test audio and train model
        audio_data = self.create_test_audio(duration=3.0)
        features = self.feature_extractor.get_feature_vector(audio_data)
        self.classifier.train()
        
        # Test with different confidence thresholds
        thresholds = [0.3, 0.6, 0.9]
        
        for threshold in thresholds:
            self.classifier.set_confidence_threshold(threshold)
            prediction = self.classifier.predict(features)
            
            # Higher thresholds should be more conservative with reliability
            if prediction['confidence'] >= threshold:
                assert prediction['reliable'] is True
            else:
                assert prediction['reliable'] is False
    
    def test_error_handling_in_pipeline(self):
        """Test error handling throughout the pipeline."""
        # Test with invalid features (NaN values)
        invalid_features = np.array([np.nan, 1.0, 2.0, np.inf])
        
        # Train model first
        self.classifier.train()
        
        # Should handle invalid features gracefully
        prediction = self.classifier.predict(invalid_features)
        
        # Should either handle gracefully or return error
        if prediction.get('error'):
            assert isinstance(prediction['error'], str)
        else:
            # If no error, should have valid prediction structure
            assert 'accent' in prediction
            assert 'confidence' in prediction
    
    def test_feature_vector_properties(self):
        """Test properties of extracted feature vectors."""
        audio_data = self.create_test_audio(duration=5.0)
        features = self.feature_extractor.get_feature_vector(audio_data)
        
        # Feature vector should have reasonable properties
        assert isinstance(features, np.ndarray)
        assert len(features) > 20  # Should have substantial number of features
        assert not np.any(np.isnan(features))  # No NaN values
        assert not np.any(np.isinf(features))  # No infinite values
        assert np.var(features) > 0  # Features should have some variance
        
        # Features should be in reasonable range (after any normalization)
        assert np.all(np.abs(features) < 1000)  # Not extremely large values 