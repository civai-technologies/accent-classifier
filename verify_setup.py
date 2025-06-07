#!/usr/bin/env python3
"""
Verification script for Accent Classifier setup
Checks model compatibility and test sample availability
"""

import os
import sys
import joblib
from pathlib import Path

def check_model_compatibility():
    """Check if the model is compatible with current sklearn version."""
    print("🔍 Checking model compatibility...")
    
    try:
        import sklearn
        print(f"   Current scikit-learn version: {sklearn.__version__}")
        
        # Load the model
        model_path = Path(__file__).parent / 'models' / 'accent_classifier.joblib'
        if not model_path.exists():
            print("   ❌ Model file not found!")
            return False
            
        model = joblib.load(model_path)
        print(f"   Model type: {type(model)}")
        
        if isinstance(model, dict):
            classifier = model.get('model')
            if classifier:
                print(f"   Classifier type: {type(classifier)}")
                # Try to access attributes to check compatibility
                try:
                    _ = classifier.n_estimators  # This should work for RandomForest
                    print("   ✅ Model loaded successfully!")
                    return True
                except AttributeError as e:
                    print(f"   ❌ Model compatibility issue: {e}")
                    return False
        else:
            print("   ❌ Unexpected model format!")
            return False
            
    except Exception as e:
        print(f"   ❌ Error checking model: {e}")
        return False

def check_test_samples():
    """Check if test samples are available."""
    print("\n🔍 Checking test samples...")
    
    test_samples_dir = Path(__file__).parent / 'test_samples'
    if not test_samples_dir.exists():
        print("   ❌ test_samples directory not found!")
        return False
    
    expected_files = [
        'american_test.mp4',
        'british_test.mp4',
        'french_test.mp4',
        'American Accent_sample.wav',
        'British Accent_sample.wav',
        'French Accent_sample.wav',
        'German Accent_sample.wav',
        'Spanish Accent_sample.wav',
        'Russian Accent_sample.wav',
        'Italian Accent_sample.wav'
    ]
    
    missing_files = []
    for file in expected_files:
        file_path = test_samples_dir / file
        if file_path.exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n   Missing {len(missing_files)} files!")
        return False
    else:
        print(f"\n   ✅ All {len(expected_files)} test files found!")
        return True

def check_dependencies():
    """Check if required dependencies are available."""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        'flask',
        'librosa', 
        'sklearn',
        'numpy',
        'scipy',
        'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   Missing {len(missing_packages)} packages!")
        return False
    else:
        print(f"\n   ✅ All {len(required_packages)} packages available!")
        return True

def main():
    """Run all verification checks."""
    print("🚀 Starting Accent Classifier Verification...\n")
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Model Compatibility", check_model_compatibility),
        ("Test Samples", check_test_samples)
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    print("\n" + "="*50)
    print("📋 VERIFICATION SUMMARY")
    print("="*50)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:<20}: {status}")
        if not passed:
            all_passed = False
    
    print("="*50)
    if all_passed:
        print("🎉 All checks passed! Ready for deployment.")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix issues before deployment.")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 