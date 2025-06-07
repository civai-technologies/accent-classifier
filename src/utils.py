"""
Utility functions for accent classifier.
Contains helper functions and common utilities.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def format_audio_info(audio_info: Dict[str, Any]) -> str:
    """
    Format audio information for display.
    
    Args:
        audio_info: Dictionary with audio information
        
    Returns:
        Formatted string
    """
    return f"""
Audio Information:
  Duration: {audio_info['duration']:.2f} seconds
  Sample Rate: {audio_info['sample_rate']} Hz
  Samples: {audio_info['samples']:,}
  RMS Energy: {audio_info['rms_energy']:.4f}
  Zero Crossing Rate: {audio_info['zero_crossing_rate']:.4f}
  Max Amplitude: {audio_info['max_amplitude']:.4f}
"""


def format_prediction_results(results: Dict[str, Any]) -> str:
    """
    Format prediction results for display.
    
    Args:
        results: Prediction results dictionary
        
    Returns:
        Formatted string
    """
    if results.get('error'):
        return f"[red]Error: {results['error']}[/red]"
    
    accent_name = results['accent_name']
    confidence = results['confidence']
    reliable = results['reliable']
    
    status = "[green]Reliable[/green]" if reliable else "[yellow]Low Confidence[/yellow]"
    
    output = f"""
Accent Classification Results:
  Detected Accent: {accent_name}
  Confidence: {confidence:.3f} ({confidence*100:.1f}%)
  Status: {status}
"""
    
    # Add top probabilities
    if results.get('all_probabilities'):
        sorted_probs = sorted(
            results['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        output += "\nTop Predictions:\n"
        for accent, prob in sorted_probs[:5]:
            bar = "█" * int(prob * 20)  # Simple bar chart
            output += f"  {accent:<20} {prob:.3f} {bar}\n"
    
    return output


def create_results_table(results: Dict[str, Any]) -> Table:
    """
    Create a rich table for prediction results.
    
    Args:
        results: Prediction results dictionary
        
    Returns:
        Rich Table object
    """
    table = Table(title="Accent Classification Results")
    table.add_column("Accent", style="cyan")
    table.add_column("Probability", style="magenta")
    table.add_column("Confidence Bar", style="green")
    
    if results.get('all_probabilities'):
        sorted_probs = sorted(
            results['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for accent, prob in sorted_probs:
            bar = "█" * int(prob * 20)
            table.add_row(accent, f"{prob:.3f}", bar)
    
    return table


def save_results_to_json(results: Dict[str, Any], output_path: str) -> None:
    """
    Save prediction results to JSON file.
    
    Args:
        results: Prediction results dictionary
        output_path: Path to save JSON file
    """
    # Add timestamp
    results['timestamp'] = time.time()
    results['datetime'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"[green]Results saved to: {output_path}[/green]")


def validate_audio_file(file_path: str) -> bool:
    """
    Validate if file is a supported audio format.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    # Check file extension
    valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
    file_ext = os.path.splitext(file_path)[1].lower()
    
    return file_ext in valid_extensions


def get_supported_formats() -> List[str]:
    """
    Get list of supported audio formats.
    
    Returns:
        List of supported file extensions
    """
    return ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']


def create_progress_context():
    """
    Create a progress context for long-running operations.
    
    Returns:
        Rich Progress context manager
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    )


def estimate_processing_time(audio_duration: float) -> float:
    """
    Estimate processing time based on audio duration.
    
    Args:
        audio_duration: Duration of audio in seconds
        
    Returns:
        Estimated processing time in seconds
    """
    # Rough estimate: 0.5 seconds per second of audio
    return audio_duration * 0.5


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        file_path: Path to file
        
    Returns:
        Formatted file size string
    """
    if not os.path.exists(file_path):
        return "Unknown"
    
    size_bytes = os.path.getsize(file_path)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} TB"


def create_batch_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create summary statistics for batch processing results.
    
    Args:
        results: List of prediction results
        
    Returns:
        Summary statistics dictionary
    """
    if not results:
        return {'total_files': 0, 'successful': 0, 'failed': 0}
    
    successful = [r for r in results if not r.get('error')]
    failed = [r for r in results if r.get('error')]
    
    # Count accent predictions
    accent_counts = {}
    confidence_scores = []
    
    for result in successful:
        accent = result.get('accent_name', 'Unknown')
        accent_counts[accent] = accent_counts.get(accent, 0) + 1
        confidence_scores.append(result.get('confidence', 0.0))
    
    summary = {
        'total_files': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'success_rate': len(successful) / len(results) if results else 0,
        'accent_distribution': accent_counts,
        'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
        'confidence_std': np.std(confidence_scores) if confidence_scores else 0.0
    }
    
    return summary


def format_batch_summary(summary: Dict[str, Any]) -> str:
    """
    Format batch processing summary for display.
    
    Args:
        summary: Summary statistics dictionary
        
    Returns:
        Formatted summary string
    """
    output = f"""
Batch Processing Summary:
  Total Files: {summary['total_files']}
  Successful: {summary['successful']}
  Failed: {summary['failed']}
  Success Rate: {summary['success_rate']:.1%}
  Average Confidence: {summary['average_confidence']:.3f}
"""
    
    if summary.get('accent_distribution'):
        output += "\nAccent Distribution:\n"
        for accent, count in summary['accent_distribution'].items():
            percentage = count / summary['successful'] * 100
            output += f"  {accent:<20} {count:>3} ({percentage:.1f}%)\n"
    
    return output


def setup_logging(log_level: str = 'INFO') -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    import logging
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('accent_classifier.log')
        ]
    )


def cleanup_temp_files(temp_dir: str = 'temp') -> None:
    """
    Clean up temporary files.
    
    Args:
        temp_dir: Temporary directory to clean
    """
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
        console.print(f"[blue]Cleaned up temporary directory: {temp_dir}[/blue]")


def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are available.
    
    Returns:
        Dictionary with dependency status
    """
    dependencies = {
        'librosa': False,
        'sklearn': False,
        'numpy': False,
        'scipy': False,
        'pyaudio': False,
        'rich': False
    }
    
    for dep in dependencies:
        try:
            if dep == 'sklearn':
                import sklearn
            else:
                __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return dependencies


def print_dependency_status() -> None:
    """Print status of all dependencies."""
    deps = check_dependencies()
    
    table = Table(title="Dependency Status")
    table.add_column("Package", style="cyan")
    table.add_column("Status", style="green")
    
    for dep, available in deps.items():
        status = "✓ Available" if available else "✗ Missing"
        color = "green" if available else "red"
        table.add_row(dep, f"[{color}]{status}[/{color}]")
    
    console.print(table) 