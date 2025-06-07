"""
Model handler for accent classification.
Manages model loading, training, and inference.
"""

import os
import joblib
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
from rich.console import Console
import sys

warnings.filterwarnings("ignore", category=UserWarning)

console = Console()


class AccentClassifier:
    """Handles accent classification model training and inference."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize AccentClassifier.
        
        Args:
            model_type: Type of model ('random_forest', 'svm', 'ensemble')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.accent_mapping = self._get_accent_mapping()
        self.confidence_threshold = 0.6
        self.is_trained = False
        
    def _get_accent_mapping(self) -> Dict[str, str]:
        """
        Get mapping of accent categories.
        
        Returns:
            Dictionary mapping accent codes to full names
        """
        return {
            'american': 'American English',
            'british': 'British English',
            'australian': 'Australian English',
            'indian': 'Indian English',
            'african': 'African English',
            'russian': 'Russian Accent',
            'german': 'German Accent',
            'french': 'French Accent',
            'spanish': 'Spanish Accent',
            'chinese': 'Chinese Accent',
            'arabic': 'Arabic Accent',
            'unknown': 'Unknown/Other'
        }
    
    def _create_model(self) -> Any:
        """
        Create the machine learning model.
        
        Returns:
            Initialized ML model
        """
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            )
        else:
            # Default to random forest
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
    
    def create_synthetic_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic training data for demonstration.
        In a real implementation, this would load actual audio data.
        
        Args:
            n_samples: Number of synthetic samples to generate
            
        Returns:
            Tuple of (features, labels)
        """
        console.print("[yellow]Creating synthetic training data...[/yellow]")
        
        # Import feature extractor to get actual feature dimensions
        from feature_extractor import FeatureExtractor
        
        # Create a temporary feature extractor to get feature dimensions
        temp_extractor = FeatureExtractor()
        
        # Create a sample audio to determine feature vector size
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        sample_audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        sample_features = temp_extractor.get_feature_vector(sample_audio)
        n_features = len(sample_features)
        
        console.print(f"[blue]Using {n_features} features for training[/blue]")
        
        # Create synthetic features (in reality, these would come from real audio)
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        
        # Create synthetic accent labels with some patterns
        accents = list(self.accent_mapping.keys())[:-1]  # Exclude 'unknown'
        y = []
        
        for i in range(n_samples):
            accent_idx = i % len(accents)
            accent = accents[accent_idx]
            
            # Add some accent-specific patterns to make classification meaningful
            feature_chunk_size = n_features // len(accents)
            start_idx = accent_idx * feature_chunk_size
            end_idx = min(start_idx + feature_chunk_size, n_features)
            
            if end_idx > start_idx:
                X[i, start_idx:end_idx] += np.random.normal(1.0, 0.5, end_idx - start_idx)
            
            y.append(accent)
        
        return X, np.array(y)
    
    def create_tts_training_data(self, fresh: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Create training data using TTS-generated audio samples from structured folders.
        This provides much more realistic training data than synthetic random features.
        
        Returns:
            Tuple of (features, labels) or (None, None) if TTS generation fails
        """
        try:
            console.print("[blue]Creating TTS-based training data from audio samples...[/blue]")
            
            # Import required modules
            from audio_generator import ScalableAudioGenerator
            from audio_processor import AudioProcessor
            from feature_extractor import FeatureExtractor
            
            # Initialize components
            generator = ScalableAudioGenerator(use_cloud_tts=False)
            audio_processor = AudioProcessor()
            feature_extractor = FeatureExtractor()
            
            # Get available training data info
            training_info = generator.get_training_data_info()
            
            if not training_info:
                console.print("[red]No language configurations found in audio_samples/[/red]")
                return None, None
            
            all_features = []
            all_labels = []
            
            console.print(f"[yellow]Processing audio samples from {len(training_info)} languages...[/yellow]")
            
            for language, info in training_info.items():
                try:
                    accent_code = info['accent_code']
                    existing_samples = info['sample_paths']
                    
                    if not existing_samples or fresh:
                        message = "No existing samples" if not existing_samples else "Regenerating samples"
                        console.print(f"[yellow]{message} for {language}...[/yellow]")
                        # Generate samples if none exist or if fresh generation is requested
                        generated_files = generator.generate_language_samples(language, num_samples=5, force=fresh)
                        sample_paths = generated_files
                    else:
                        sample_paths = existing_samples
                    
                    # Process existing audio files
                    for sample_path in sample_paths:
                        try:
                            # Load and process audio
                            audio_data, _ = audio_processor.load_audio_file(str(sample_path))
                            
                            # Extract features
                            features = feature_extractor.get_feature_vector(audio_data)
                            
                            # Add to training data
                            all_features.append(features)
                            all_labels.append(accent_code)
                            
                            console.print(f"[green]  ✓ Processed: {sample_path.name} ({accent_code})[/green]")
                            
                        except Exception as e:
                            console.print(f"[red]  ✗ Error processing {sample_path}: {e}[/red]")
                            continue
                            
                except Exception as e:
                    console.print(f"[red]Error processing {language}: {e}[/red]")
                    continue
            
            if len(all_features) < 6:  # At least one sample per expected language
                console.print("[red]Insufficient audio samples found, falling back to synthetic[/red]")
                return None, None
            
            X = np.array(all_features)
            y = np.array(all_labels)
            
            console.print(f"[green]✓ Created TTS training data: {X.shape[0]} samples, {X.shape[1]} features[/green]")
            console.print(f"[blue]Languages: {list(set(y))}[/blue]")
            
            return X, y
            
        except Exception as e:
            console.print(f"[red]TTS training data generation failed: {e}[/red]")
            console.print("[yellow]Falling back to synthetic data[/yellow]")
            return None, None
    
    def train(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, use_tts: bool = True, fresh: bool = False) -> Dict[str, float]:
        """
        Train the accent classification model.
        
        Args:
            X: Feature matrix (if None, creates training data)
            y: Labels (if None, creates training data)
            use_tts: Whether to use TTS-generated training data (more realistic)
            
        Returns:
            Dictionary with training metrics
        """
        console.print("[green]Training accent classifier...[/green]")
        
        # Use TTS or synthetic data if no real data provided
        if X is None or y is None:
            if use_tts:
                X_tts, y_tts = self.create_tts_training_data(fresh=fresh)
                if X_tts is not None and y_tts is not None:
                    X, y = X_tts, y_tts
                    console.print("[green]Using TTS-generated training data[/green]")
                else:
                    console.print("[yellow]TTS generation failed, using synthetic data[/yellow]")
                    X, y = self.create_synthetic_data()
            else:
                X, y = self.create_synthetic_data()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Cross-validation (adjust CV folds based on dataset size)
        n_samples_per_class = len(y_train) // len(np.unique(y_train))
        cv_folds = min(3, n_samples_per_class)  # Use fewer folds for small datasets
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)
        
        # Predictions for detailed evaluation
        y_pred = self.model.predict(X_test_scaled)
        
        console.print(f"[green]Training completed![/green]")
        console.print(f"Train accuracy: {train_score:.3f}")
        console.print(f"Test accuracy: {test_score:.3f}")
        console.print(f"CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Predict accent from audio features.
        
        Args:
            features: Feature vector from audio
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained or self.model is None:
            return {
                'accent': 'unknown',
                'accent_name': 'Unknown/Other',
                'confidence': 0.0,
                'all_probabilities': {},
                'reliable': False,
                'error': 'Model not trained'
            }
        
        try:
            # Ensure features are 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction and probabilities
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Decode prediction
            accent_code = self.label_encoder.inverse_transform([prediction])[0]
            accent_name = self.accent_mapping.get(accent_code, 'Unknown/Other')
            
            # Get confidence (max probability)
            confidence = np.max(probabilities)
            
            # Get all probabilities
            all_probs = {}
            for i, prob in enumerate(probabilities):
                label = self.label_encoder.inverse_transform([i])[0]
                all_probs[self.accent_mapping.get(label, label)] = float(prob)
            
            # Check if prediction is reliable
            reliable = confidence >= self.confidence_threshold
            
            # Always show the top prediction regardless of confidence
            # (commented out the reliability override)
            # if not reliable:
            #     accent_code = 'unknown'
            #     accent_name = 'Unknown/Other'
            
            return {
                'accent': accent_code,
                'accent_name': accent_name,
                'confidence': float(confidence),
                'all_probabilities': all_probs,
                'reliable': reliable,
                'error': None
            }
            
        except Exception as e:
            return {
                'accent': 'unknown',
                'accent_name': 'Unknown/Other',
                'confidence': 0.0,
                'all_probabilities': {},
                'reliable': False,
                'error': str(e)
            }
    
    def save_model(self, model_path: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            model_path: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'accent_mapping': self.accent_mapping,
            'confidence_threshold': self.confidence_threshold
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        joblib.dump(model_data, model_path)
        console.print(f"[green]Model saved to: {model_path}[/green]")
    
    def load_model(self, model_path: str) -> bool:
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to load model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                console.print(f"[red]Model file not found: {model_path}[/red]")
                return False
            
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.model_type = model_data.get('model_type', 'random_forest')
            self.accent_mapping = model_data.get('accent_mapping', self._get_accent_mapping())
            self.confidence_threshold = model_data.get('confidence_threshold', 0.6)
            self.is_trained = True
            
            console.print(f"[green]Model loaded from: {model_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'confidence_threshold': self.confidence_threshold,
            'supported_accents': list(self.accent_mapping.keys()),
            'accent_mapping': self.accent_mapping
        }
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Set confidence threshold for predictions.
        
        Args:
            threshold: Confidence threshold (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self.confidence_threshold = threshold
        console.print(f"[blue]Confidence threshold set to: {threshold}[/blue]") 