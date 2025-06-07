#!/usr/bin/env python3
"""
Script to analyze TTS samples and check for differences between accents.
"""

import sys
import os
import numpy as np
import soundfile as sf

# Add the current directory to Python path
sys.path.append(os.getcwd())

from tests.audio_generator import AudioSampleGenerator
from src.feature_extractor import FeatureExtractor
from rich.console import Console

console = Console()

def analyze_audio_samples():
    """Analyze generated audio samples to check for differences."""
    
    # Generate samples with both TTS engines
    console.print("[bold blue]🔍 TTS SAMPLE ANALYSIS[/bold blue]")
    
    # Use gTTS for this analysis
    generator = AudioSampleGenerator(use_cloud_tts=False)
    
    # Generate fresh samples with specific languages
    test_languages = ['American Accent', 'British Accent', 'French Accent', 'German Accent']
    files = generator.generate_test_samples(languages=test_languages, use_cloud_tts=False)
    
    console.print(f"\nGenerated {len(files)} samples:")
    for lang, path in files.items():
        info = generator.get_sample_info(path)
        console.print(f"  {lang}: {info.get('duration', 0):.2f}s")
    
    # Compare American vs British specifically
    american_file = files.get('American Accent')
    british_file = files.get('British Accent')
    
    if american_file and british_file:
        console.print("\n[yellow]🇺🇸 vs 🇬🇧 AMERICAN vs BRITISH COMPARISON[/yellow]")
        
        # Load audio data
        american_data, sr1 = sf.read(american_file)
        british_data, sr2 = sf.read(british_file)
        
        console.print(f"American sample: {len(american_data)} samples, {sr1}Hz")
        console.print(f"British sample: {len(british_data)} samples, {sr2}Hz")
        
        # Check if identical
        if len(american_data) == len(british_data):
            are_identical = np.array_equal(american_data, british_data)
            console.print(f"Audio samples identical: [red]{are_identical}[/red]" if are_identical else f"Audio samples identical: [green]{are_identical}[/green]")
            
            if not are_identical:
                # Calculate similarity metrics
                max_length = min(len(american_data), len(british_data))
                am_clip = american_data[:max_length]
                br_clip = british_data[:max_length]
                
                # Cosine similarity
                dot_product = np.dot(am_clip, br_clip)
                norm_am = np.linalg.norm(am_clip)
                norm_br = np.linalg.norm(br_clip)
                cosine_sim = dot_product / (norm_am * norm_br) if norm_am > 0 and norm_br > 0 else 0
                
                # Euclidean distance
                euclidean_dist = np.linalg.norm(am_clip - br_clip)
                
                console.print(f"Cosine similarity: {cosine_sim:.4f}")
                console.print(f"Euclidean distance: {euclidean_dist:.4f}")
                
                # Calculate RMS difference
                rms_diff = np.sqrt(np.mean((am_clip - br_clip) ** 2))
                console.print(f"RMS difference: {rms_diff:.6f}")
        else:
            console.print("Different lengths - definitely different audio")
            
        # Extract and compare features
        console.print("\n[blue]📊 FEATURE COMPARISON[/blue]")
        
        feature_extractor = FeatureExtractor()
        
        # Extract features for both
        american_features = feature_extractor.get_feature_vector(american_data)
        british_features = feature_extractor.get_feature_vector(british_data)
        
        if american_features is not None and british_features is not None:
            console.print(f"American features shape: {american_features.shape}")
            console.print(f"British features shape: {british_features.shape}")
            
            # Check if features are identical
            features_identical = np.array_equal(american_features, british_features)
            console.print(f"Features identical: [red]{features_identical}[/red]" if features_identical else f"Features identical: [green]{features_identical}[/green]")
            
            if not features_identical:
                # Feature similarity metrics
                feature_cosine = np.dot(american_features, british_features) / (
                    np.linalg.norm(american_features) * np.linalg.norm(british_features)
                )
                feature_euclidean = np.linalg.norm(american_features - british_features)
                
                console.print(f"Feature cosine similarity: {feature_cosine:.4f}")
                console.print(f"Feature euclidean distance: {feature_euclidean:.4f}")
                
                # Find most different features
                feature_diff = np.abs(american_features - british_features)
                max_diff_idx = np.argmax(feature_diff)
                console.print(f"Max feature difference at index {max_diff_idx}: {feature_diff[max_diff_idx]:.6f}")
                console.print(f"American[{max_diff_idx}]: {american_features[max_diff_idx]:.6f}")
                console.print(f"British[{max_diff_idx}]: {british_features[max_diff_idx]:.6f}")
            else:
                console.print("Features are completely identical!")
        
    else:
        console.print("Could not find American or British audio files")
        
    console.print("\n[bold blue]✅ Analysis Complete[/bold blue]")
    
    # Test with Google Cloud TTS if available
    try:
        console.print("\n[bold blue]☁️ Testing Google Cloud TTS[/bold blue]")
        cloud_generator = AudioSampleGenerator(use_cloud_tts=True)
        
        # Generate only American and British for comparison with Cloud TTS
        cloud_files = cloud_generator.generate_test_samples(
            languages=['American Accent', 'British Accent'], 
            use_cloud_tts=True
        )
        
        for lang, path in cloud_files.items():
            info = cloud_generator.get_sample_info(path)
            console.print(f"  {lang}: {info.get('duration', 0):.2f}s")
        
        # Compare cloud-generated samples
        if len(cloud_files) == 2:
            cloud_american = cloud_files['American Accent']
            cloud_british = cloud_files['British Accent']
            
            am_data, _ = sf.read(cloud_american)
            br_data, _ = sf.read(cloud_british)
            
            cloud_identical = np.array_equal(am_data, br_data) if len(am_data) == len(br_data) else False
            console.print(f"Cloud TTS samples identical: [red]{cloud_identical}[/red]" if cloud_identical else f"Cloud TTS samples identical: [green]{cloud_identical}[/green]")
            
            if not cloud_identical and len(am_data) == len(br_data):
                cosine_sim = np.dot(am_data, br_data) / (np.linalg.norm(am_data) * np.linalg.norm(br_data))
                console.print(f"Cloud TTS cosine similarity: {cosine_sim:.4f}")
    
    except Exception as e:
        console.print(f"Cloud TTS test failed: {e}")

if __name__ == "__main__":
    analyze_audio_samples() 