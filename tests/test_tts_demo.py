"""
Demonstration test showing TTS audio generation and accent classification
for multiple languages (American, British, French, German, Spanish, Russian).
"""

import os
import sys
import pytest
from typing import Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.dirname(__file__))

from audio_processor import AudioProcessor
from feature_extractor import FeatureExtractor
from model_handler import AccentClassifier
from audio_generator import AudioSampleGenerator


def test_tts_accent_classification_demo():
    """
    Demonstration test: Generate TTS audio for 4+ languages and classify their accents.
    This test shows the complete pipeline working with real audio data.
    """
    # Initialize components
    generator = AudioSampleGenerator()
    audio_processor = AudioProcessor()
    feature_extractor = FeatureExtractor()
    classifier = AccentClassifier()
    
    # Define test languages (covering the requested 4+ languages)
    test_languages = ['american', 'british', 'french', 'german', 'spanish', 'russian']
    
    print("\n" + "="*60)
    print("TTS ACCENT CLASSIFICATION DEMONSTRATION")
    print("="*60)
    
    # Generate TTS audio samples
    print(f"\n1. Generating TTS audio samples for {len(test_languages)} languages...")
    generated_files = generator.generate_test_samples(test_languages)
    
    assert len(generated_files) >= 4, f"Should generate at least 4 samples, got {len(generated_files)}"
    print(f"   ✓ Successfully generated {len(generated_files)} audio samples")
    
    # Train the model
    print("\n2. Training accent classification model...")
    metrics = classifier.train()
    print(f"   ✓ Model trained with {metrics['cv_mean']:.1%} cross-validation accuracy")
    
    # Test each language sample
    print(f"\n3. Classifying accent for each TTS-generated sample...")
    results = {}
    
    for lang, file_path in generated_files.items():
        print(f"\n   Testing {lang.upper()} TTS sample:")
        print(f"   File: {file_path}")
        
        # Load and process audio
        audio_data, sample_rate = audio_processor.load_audio_file(file_path)
        audio_info = audio_processor.get_audio_info(audio_data, sample_rate)
        
        print(f"   Duration: {audio_info['duration']:.2f}s, "
              f"Energy: {audio_info['rms_energy']:.3f}")
        
        # Extract features and classify
        features = feature_extractor.get_feature_vector(audio_data)
        prediction = classifier.predict(features)
        
        results[lang] = prediction
        
        print(f"   → Predicted accent: {prediction['accent']}")
        print(f"   → Confidence: {prediction['confidence']:.1%}")
        print(f"   → Top 3 predictions:")
        
        # Show top 3 predictions
        top_3 = sorted(prediction['all_probabilities'].items(), 
                      key=lambda x: x[1], reverse=True)[:3]
        for i, (accent, prob) in enumerate(top_3, 1):
            print(f"     {i}. {accent}: {prob:.1%}")
        
        # Verify basic prediction properties
        assert prediction['error'] is None, f"Prediction should succeed for {lang}"
        assert 0.0 <= prediction['confidence'] <= 1.0, f"Confidence should be valid for {lang}"
        assert abs(sum(prediction['all_probabilities'].values()) - 1.0) < 0.01, \
            f"Probabilities should sum to 1 for {lang}"
    
    # Summary analysis
    print(f"\n4. SUMMARY ANALYSIS:")
    print(f"   Languages tested: {', '.join(generated_files.keys())}")
    
    # Analyze prediction diversity
    predicted_accents = [result['accent'] for result in results.values()]
    unique_predictions = set(predicted_accents)
    print(f"   Unique accent predictions: {len(unique_predictions)}")
    
    # Analyze confidence levels
    confidences = [result['confidence'] for result in results.values()]
    avg_confidence = sum(confidences) / len(confidences)
    print(f"   Average confidence: {avg_confidence:.1%}")
    
    # Check for reasonable predictions
    reliable_count = sum(1 for result in results.values() if result['reliable'])
    print(f"   Reliable predictions: {reliable_count}/{len(results)}")
    
    # Language-specific insights
    print(f"\n5. LANGUAGE-SPECIFIC INSIGHTS:")
    
    # English variants analysis
    english_variants = ['american', 'british']
    english_results = {lang: results[lang] for lang in english_variants if lang in results}
    
    if len(english_results) >= 2:
        print(f"   English variants tested: {', '.join(english_results.keys())}")
        for lang, result in english_results.items():
            # Check if English accents appear in top predictions
            english_accents = ['american', 'british', 'australian', 'indian']
            top_3_accents = [accent.lower() for accent, _ in 
                           sorted(result['all_probabilities'].items(), 
                                 key=lambda x: x[1], reverse=True)[:3]]
            
            english_in_top = any(any(eng in accent for eng in english_accents) 
                               for accent in top_3_accents)
            print(f"     {lang}: English accent in top 3: {'Yes' if english_in_top else 'No'}")
    
    # Non-English languages analysis
    non_english_langs = ['french', 'german', 'spanish', 'russian']
    non_english_results = {lang: results[lang] for lang in non_english_langs if lang in results}
    
    if non_english_results:
        print(f"   Non-English languages: {', '.join(non_english_results.keys())}")
        for lang, result in non_english_results.items():
            expected_accent = f"{lang} accent"
            actual_top_3 = [accent.lower() for accent, _ in 
                          sorted(result['all_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]]
            
            lang_in_top = lang in ' '.join(actual_top_3)
            print(f"     {lang}: Expected accent in top 3: {'Yes' if lang_in_top else 'No'}")
    
    # Cleanup
    generator.cleanup_samples()
    
    print(f"\n6. TEST CONCLUSION:")
    print(f"   ✓ Successfully generated TTS audio for {len(generated_files)} languages")
    print(f"   ✓ All {len(results)} samples were successfully classified")
    print(f"   ✓ Feature extraction produced {len(features)} features per sample")
    print(f"   ✓ Model predictions were consistent and properly formatted")
    print(f"   ✓ Demonstrated end-to-end accent classification pipeline")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("="*60)
    
    # Final assertions
    assert len(generated_files) >= 4, "Should test at least 4 languages"
    assert all(result['error'] is None for result in results.values()), \
        "All predictions should succeed"
    assert len(unique_predictions) >= 1, "Should have at least one predicted accent"
    assert avg_confidence > 0.0, "Should have positive average confidence"


if __name__ == "__main__":
    test_tts_accent_classification_demo() 