#!/usr/bin/env python3
"""
Accent Classifier - Main Script
Classifies accents from audio input (file or microphone).
"""

import os
import sys
import argparse
from typing import Optional, List
import click
from rich.console import Console
from rich.panel import Panel

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from audio_processor import AudioProcessor
from feature_extractor import FeatureExtractor
from model_handler import AccentClassifier
from utils import (
    format_audio_info, format_prediction_results, create_results_table,
    save_results_to_json, validate_audio_file, get_supported_formats,
    create_progress_context, print_dependency_status
)

console = Console()


class AccentClassifierApp:
    """Main application class for accent classification."""
    
    def __init__(self):
        """Initialize the accent classifier application."""
        self.audio_processor = AudioProcessor()
        self.feature_extractor = FeatureExtractor()
        self.classifier = AccentClassifier()
        self.model_path = os.path.join('models', 'accent_classifier.joblib')
        
    def setup(self, train_if_needed: bool = True) -> bool:
        """
        Setup the application (load or train model).
        
        Args:
            train_if_needed: Whether to train model if not found
            
        Returns:
            True if setup successful, False otherwise
        """
        console.print("[blue]Setting up Accent Classifier...[/blue]")
        
        # Try to load existing model
        if os.path.exists(self.model_path):
            if self.classifier.load_model(self.model_path):
                console.print("[green]✓ Model loaded successfully[/green]")
                return True
        
        # Train model if needed
        if train_if_needed:
            console.print("[yellow]No pre-trained model found. Training new model...[/yellow]")
            try:
                metrics = self.classifier.train()
                self.classifier.save_model(self.model_path)
                console.print("[green]✓ Model trained and saved successfully[/green]")
                return True
            except Exception as e:
                console.print(f"[red]✗ Failed to train model: {e}[/red]")
                return False
        else:
            console.print("[red]✗ No model available and training disabled[/red]")
            return False
    
    def classify_file(self, file_path: str, show_details: bool = True) -> dict:
        """
        Classify accent from audio file.
        
        Args:
            file_path: Path to audio file
            show_details: Whether to show detailed information
            
        Returns:
            Classification results dictionary
        """
        try:
            # Validate file
            if not validate_audio_file(file_path):
                return {'error': f'Invalid or unsupported audio file: {file_path}'}
            
            with create_progress_context() as progress:
                # Load audio
                task = progress.add_task("Loading audio...", total=None)
                audio_data, sample_rate = self.audio_processor.load_audio_file(file_path)
                
                # Show audio info if requested
                if show_details:
                    audio_info = self.audio_processor.get_audio_info(audio_data, sample_rate)
                    console.print(format_audio_info(audio_info))
                
                # Extract features
                progress.update(task, description="Extracting features...")
                features = self.feature_extractor.get_feature_vector(audio_data)
                
                # Classify
                progress.update(task, description="Classifying accent...")
                results = self.classifier.predict(features)
                
                progress.remove_task(task)
            
            # Add file information to results
            results['file_path'] = file_path
            results['file_size'] = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def classify_microphone(self, duration: float = 10.0, show_details: bool = True) -> dict:
        """
        Classify accent from microphone input.
        
        Args:
            duration: Recording duration in seconds
            show_details: Whether to show detailed information
            
        Returns:
            Classification results dictionary
        """
        try:
            with create_progress_context() as progress:
                # Record audio
                task = progress.add_task("Recording audio...", total=None)
                audio_data, sample_rate = self.audio_processor.record_audio(duration)
                
                # Show audio info if requested
                if show_details:
                    audio_info = self.audio_processor.get_audio_info(audio_data, sample_rate)
                    console.print(format_audio_info(audio_info))
                
                # Extract features
                progress.update(task, description="Extracting features...")
                features = self.feature_extractor.get_feature_vector(audio_data)
                
                # Classify
                progress.update(task, description="Classifying accent...")
                results = self.classifier.predict(features)
                
                progress.remove_task(task)
            
            # Add recording information to results
            results['input_type'] = 'microphone'
            results['duration'] = duration
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def batch_classify(self, file_paths: List[str], output_dir: Optional[str] = None) -> List[dict]:
        """
        Classify multiple audio files.
        
        Args:
            file_paths: List of audio file paths
            output_dir: Directory to save individual results
            
        Returns:
            List of classification results
        """
        results = []
        
        console.print(f"[blue]Processing {len(file_paths)} files...[/blue]")
        
        for i, file_path in enumerate(file_paths, 1):
            console.print(f"\n[cyan]Processing file {i}/{len(file_paths)}: {file_path}[/cyan]")
            
            result = self.classify_file(file_path, show_details=False)
            results.append(result)
            
            # Show brief result
            if result.get('error'):
                console.print(f"[red]✗ Error: {result['error']}[/red]")
            else:
                accent = result['accent_name']
                confidence = result['confidence']
                console.print(f"[green]✓ {accent} ({confidence:.3f})[/green]")
            
            # Save individual result if output directory specified
            if output_dir and not result.get('error'):
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_file = os.path.join(output_dir, f"{base_name}_result.json")
                save_results_to_json(result, output_file)
        
        return results


def test_trained_model_with_tts(app: AccentClassifierApp) -> None:
    """Test the trained model with fresh TTS samples to verify performance."""
    try:
        # Import required modules
        sys.path.append('src')
        from audio_generator import ScalableAudioGenerator
        from pathlib import Path
        
        # Generate fresh test samples using scalable system
        generator = ScalableAudioGenerator(use_cloud_tts=False)
        
        # Discover all available languages automatically
        available_languages = generator.discover_languages()
        training_info = generator.get_training_data_info()
        
        console.print("[blue]Generating fresh TTS samples for testing...[/blue]")
        
        # Generate test samples for each language
        generated_files = {}
        for language in available_languages:
            try:
                config = generator.load_language_config(language)
                language_name = config['language_name']
                
                # Generate a single test sample
                temp_file = f"test_samples/{language_name}_sample.wav"
                os.makedirs("test_samples", exist_ok=True)
                
                # Use first sample text for testing
                sample_texts = config.get('sample_texts', [])
                if sample_texts:
                    test_text = sample_texts[0]
                    if generator.generate_sample(language, test_text, Path(temp_file), force=True):
                        generated_files[language_name] = temp_file
                        console.print(f"Generated {language_name} audio: {temp_file}")
                
            except Exception as e:
                console.print(f"[red]Error generating test sample for {language}: {e}[/red]")
        
        # Test each sample
        test_results = []
        for lang, file_path in generated_files.items():
            try:
                result = app.classify_file(file_path, show_details=False)
                expected_accent = lang.lower().replace(' accent', '')
                
                test_results.append({
                    'language': lang,
                    'expected': expected_accent,
                    'predicted': result['accent'],
                    'confidence': result['confidence'],
                    'correct': result['accent'] == expected_accent
                })
                
            except Exception as e:
                console.print(f"[red]Error testing {lang}: {e}[/red]")
        
        # Display results
        from rich.table import Table
        results_table = Table(title="Model Performance on TTS Samples")
        results_table.add_column("Language", style="cyan")
        results_table.add_column("Expected", style="blue") 
        results_table.add_column("Predicted", style="green")
        results_table.add_column("Confidence", style="yellow")
        results_table.add_column("Result", style="magenta")
        
        correct_count = 0
        for result in test_results:
            status = "✅ Correct" if result['correct'] else "❌ Wrong"
            if result['correct']:
                correct_count += 1
                
            results_table.add_row(
                result['language'],
                result['expected'],
                result['predicted'],
                f"{result['confidence']:.1%}",
                status
            )
        
        console.print(results_table)
        
        # Summary
        accuracy = correct_count / len(test_results) if test_results else 0
        avg_confidence = sum(r['confidence'] for r in test_results) / len(test_results) if test_results else 0
        
        summary_table = Table(title="Test Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Accuracy", f"{accuracy:.1%}")
        summary_table.add_row("Average Confidence", f"{avg_confidence:.1%}")
        summary_table.add_row("Samples Tested", str(len(test_results)))
        
        console.print(summary_table)
        
        if accuracy >= 0.5:
            console.print("[green]🎉 Model is performing well on TTS samples![/green]")
        else:
            console.print("[yellow]⚠️ Model accuracy could be improved. Consider more training data.[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error during TTS testing: {e}[/red]")


@click.command()
@click.option('--file', '-f', type=str, help='Audio file to classify')
@click.option('--microphone', '-m', is_flag=True, help='Record from microphone')
@click.option('--duration', '-d', type=float, default=10.0, help='Recording duration (seconds)')
@click.option('--batch', '-b', type=str, help='Directory containing audio files for batch processing')
@click.option('--output', '-o', type=str, help='Output file/directory for results')
@click.option('--model-path', type=str, help='Path to trained model')
@click.option('--train', is_flag=True, help='Train new model')
@click.option('--use-tts', is_flag=True, help='Use TTS-generated audio for training (more realistic)')
@click.option('--fresh', is_flag=True, help='Force regenerate audio samples even if they exist (only with --use-tts)')
@click.option('--confidence-threshold', type=float, default=0.6, help='Confidence threshold')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
@click.option('--check-deps', is_flag=True, help='Check dependencies')
def main(file, microphone, duration, batch, output, model_path, train, use_tts, fresh, confidence_threshold, verbose, check_deps):
    """
    Accent Classifier - Classify accents from audio input.
    
    Examples:
        python accent_classifier.py --file audio.wav
        python accent_classifier.py --microphone --duration 15
        python accent_classifier.py --batch audio_files/ --output results/
        python accent_classifier.py --train
        python accent_classifier.py --train --use-tts  # Train with realistic TTS data
    """
    console.print(Panel.fit(
        "[bold blue]Accent Classifier[/bold blue]\n"
        "Classifies speaker accents from audio input",
        border_style="blue"
    ))
    
    # Check dependencies if requested
    if check_deps:
        print_dependency_status()
        return
    
    # Initialize application
    app = AccentClassifierApp()
    
    # Use custom model path if provided
    if model_path:
        app.model_path = model_path
    
    # Setup application
    if not app.setup(train_if_needed=train):
        console.print("[red]Failed to setup application. Exiting.[/red]")
        return
    
    # Set confidence threshold
    app.classifier.set_confidence_threshold(confidence_threshold)
    
    # Process based on options
    if file:
        # Single file classification
        console.print(f"\n[cyan]Classifying file: {file}[/cyan]")
        results = app.classify_file(file, show_details=verbose)
        
        if results.get('error'):
            console.print(f"[red]Error: {results['error']}[/red]")
        else:
            console.print(format_prediction_results(results))
            
            # Show detailed table
            if verbose:
                table = create_results_table(results)
                console.print(table)
            
            # Save results if output specified
            if output:
                save_results_to_json(results, output)
    
    elif microphone:
        # Microphone classification
        console.print(f"\n[cyan]Recording from microphone for {duration} seconds...[/cyan]")
        results = app.classify_microphone(duration, show_details=verbose)
        
        if results.get('error'):
            console.print(f"[red]Error: {results['error']}[/red]")
        else:
            console.print(format_prediction_results(results))
            
            # Show detailed table
            if verbose:
                table = create_results_table(results)
                console.print(table)
            
            # Save results if output specified
            if output:
                save_results_to_json(results, output)
    
    elif batch:
        # Batch processing
        if not os.path.isdir(batch):
            console.print(f"[red]Error: {batch} is not a valid directory[/red]")
            return
        
        # Find audio files
        supported_formats = get_supported_formats()
        audio_files = []
        
        for root, dirs, files in os.walk(batch):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_formats):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            console.print(f"[yellow]No audio files found in {batch}[/yellow]")
            return
        
        # Process files
        results = app.batch_classify(audio_files, output)
        
        # Show summary
        from utils import create_batch_summary, format_batch_summary
        summary = create_batch_summary(results)
        console.print(format_batch_summary(summary))
        
        # Save batch results
        if output:
            if os.path.isdir(output):
                batch_results_path = os.path.join(output, 'batch_results.json')
            else:
                batch_results_path = output
            
            batch_data = {
                'summary': summary,
                'results': results
            }
            save_results_to_json(batch_data, batch_results_path)
    
    elif train:
        # Train new model
        training_method = "TTS-based" if use_tts else "synthetic"
        console.print(f"\n[cyan]Training new model with {training_method} data...[/cyan]")
        
        if use_tts:
            console.print("[blue]Using TTS-generated audio for training (more realistic but slower)[/blue]")
        else:
            console.print("[blue]Using synthetic data for training (faster but less realistic)[/blue]")
        
        # Train with specified method
        metrics = app.classifier.train(use_tts=use_tts, fresh=fresh)
        app.classifier.save_model(app.model_path)
        
        # Display detailed training results
        console.print(f"\n[green]✅ Model training completed![/green]")
        
        from rich.table import Table
        metrics_table = Table(title="Training Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        
        metrics_table.add_row("Training Accuracy", f"{metrics['train_accuracy']:.3f}")
        metrics_table.add_row("Test Accuracy", f"{metrics['test_accuracy']:.3f}")
        metrics_table.add_row("CV Mean", f"{metrics['cv_mean']:.3f}")
        metrics_table.add_row("CV Std", f"{metrics['cv_std']:.3f}")
        metrics_table.add_row("Training Method", training_method.title())
        
        console.print(metrics_table)
        
        # Test the trained model with TTS samples if TTS training was used
        if use_tts:
            console.print("\n[cyan]Testing trained model with fresh TTS samples...[/cyan]")
            test_trained_model_with_tts(app)
    
    else:
        # Show help if no action specified
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        console.print("\n[yellow]Please specify an action: --file, --microphone, --batch, or --train[/yellow]")


if __name__ == '__main__':
    main() 