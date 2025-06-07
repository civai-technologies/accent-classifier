# Scalable ML Project Template: Dynamic Sample Generation & Training

This document provides a comprehensive blueprint for creating scalable machine learning projects with dynamic sample generation, modular architecture, and automated training pipelines. Based on the successful patterns from the Accent Classifier project.

## 🎯 Project Blueprint Overview

This template enables you to build ML projects with:
- **Scalable Sample Generation**: Automated data creation with configuration-driven expansion
- **Modular Architecture**: Clean separation of concerns with reusable components
- **Dynamic Training**: Automated model training with performance validation
- **Production-Ready Structure**: Comprehensive testing, documentation, and deployment patterns

---

## 📋 Template Application Instructions

### Step 1: Define Your Use Case

Replace the following placeholders with your specific use case:

```markdown
**Project Name**: [YOUR_PROJECT_NAME]
**Primary Task**: [CLASSIFICATION/REGRESSION/DETECTION/GENERATION]
**Input Data Type**: [AUDIO/IMAGE/TEXT/VIDEO/SENSOR_DATA]
**Output Categories**: [LIST_OF_CLASSES_OR_TARGETS]
**Sample Generation Method**: [API/SYNTHETIC/AUGMENTATION/SIMULATION]
```

**Example Applications**:
- Image classification (dog breeds, medical images, document types)
- Text sentiment analysis (emotions, topics, intent classification)
- Time series prediction (stock prices, sensor readings, weather)
- Object detection (faces, vehicles, products)
- Natural language processing (language detection, spam classification)

---

## 🏗 Scalable Project Structure

```
[project-name]/
├── src/                           # Core modules
│   ├── data_processor.py          # Data input and preprocessing
│   ├── feature_extractor.py       # Feature engineering pipeline
│   ├── model_handler.py           # ML model management
│   ├── sample_generator.py        # Dynamic sample generation
│   └── utils.py                   # Utility functions
├── [data_type]_samples/           # Generated training data
│   ├── [category_1]/
│   │   ├── config.json            # Generation configuration
│   │   ├── sample_001.[ext]       # Generated samples
│   │   ├── sample_002.[ext]
│   │   └── ...
│   ├── [category_2]/
│   │   ├── config.json
│   │   └── sample_*.ext
│   └── [category_n]/
├── tests/                         # Comprehensive test suite
│   ├── test_data_processing.py
│   ├── test_feature_extraction.py
│   ├── test_integration.py
│   └── test_sample_generation.py
├── docs/                          # Documentation
│   ├── README.md
│   ├── adding-new-categories.md
│   ├── generating-samples.md
│   └── api-documentation.md
├── models/                        # Trained model artifacts
├── examples/                      # Interactive demos
│   ├── [project]_demo.ipynb
│   └── README.md
├── [main_script].py               # Main CLI interface
├── requirements.txt               # Dependencies
├── constraints.md                 # Project constraints
└── README.md                     # Project documentation
```

---

## 🔧 Core Component Templates

### 1. Sample Generator (`src/sample_generator.py`)

```python
"""
Scalable sample generation system for [YOUR_USE_CASE].
Supports configuration-driven category expansion and automated data creation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

class ScalableSampleGenerator:
    """
    Generates training samples for [YOUR_USE_CASE] using [GENERATION_METHOD].
    
    Features:
    - Configuration-driven category management
    - Automated sample generation with caching
    - Scalable architecture for adding new categories
    - Quality validation and error handling
    """
    
    def __init__(self, base_dir: str = "[data_type]_samples"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def discover_categories(self) -> Dict[str, Dict]:
        """
        Automatically discover configured categories from directory structure.
        
        Returns:
            Dict mapping category codes to configuration data
        """
        categories = {}
        
        for category_dir in self.base_dir.iterdir():
            if category_dir.is_dir():
                config_file = category_dir / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        categories[category_dir.name] = config
                    except Exception as e:
                        self.logger.warning(f"Failed to load config for {category_dir.name}: {e}")
        
        return categories
    
    def get_training_info(self) -> Dict[str, Dict]:
        """
        Get comprehensive information about available training data.
        
        Returns:
            Dict with sample counts and metadata for each category
        """
        categories = self.discover_categories()
        info = {}
        
        for category_code, config in categories.items():
            category_dir = self.base_dir / category_code
            sample_files = list(category_dir.glob("sample_*.[ext]"))  # Replace [ext] with your file extension
            
            info[category_code] = {
                'category_name': config.get('category_name', category_code),
                'sample_count': len(sample_files),
                'config': config,
                'samples_path': str(category_dir)
            }
        
        return info
    
    def generate_samples(
        self, 
        categories: Optional[List[str]] = None,
        num_samples: int = 5,
        force_regenerate: bool = False
    ) -> Dict[str, Dict]:
        """
        Generate training samples for specified categories.
        
        Args:
            categories: List of category codes to generate (None for all)
            num_samples: Number of samples to generate per category
            force_regenerate: Whether to regenerate existing samples
            
        Returns:
            Dict with generation results for each category
        """
        available_categories = self.discover_categories()
        
        if categories is None:
            categories = list(available_categories.keys())
        
        results = {}
        
        for category in categories:
            if category not in available_categories:
                results[category] = {
                    'success': False,
                    'message': f'Category {category} not configured'
                }
                continue
                
            try:
                result = self._generate_category_samples(
                    category, 
                    available_categories[category],
                    num_samples,
                    force_regenerate
                )
                results[category] = result
                
            except Exception as e:
                results[category] = {
                    'success': False,
                    'message': f'Generation failed: {str(e)}'
                }
                self.logger.error(f"Failed to generate samples for {category}: {e}")
        
        return results
    
    def _generate_category_samples(
        self, 
        category_code: str, 
        config: Dict, 
        num_samples: int,
        force_regenerate: bool
    ) -> Dict:
        """
        Generate samples for a specific category.
        
        CUSTOMIZE THIS METHOD FOR YOUR USE CASE:
        - Replace sample generation logic with your specific method
        - Handle your data format and generation parameters
        - Implement quality validation for your domain
        """
        category_dir = self.base_dir / category_code
        category_dir.mkdir(exist_ok=True)
        
        # Check existing samples
        existing_samples = list(category_dir.glob("sample_*.[ext]"))
        
        if len(existing_samples) >= num_samples and not force_regenerate:
            return {
                'success': True,
                'message': f'Using {len(existing_samples)} existing samples'
            }
        
        # Generate new samples
        generated_count = 0
        
        for i in range(1, num_samples + 1):
            sample_path = category_dir / f"sample_{i:03d}.[ext]"
            
            if sample_path.exists() and not force_regenerate:
                continue
            
            # IMPLEMENT YOUR SAMPLE GENERATION LOGIC HERE
            # Examples:
            # - API calls for data generation
            # - Synthetic data creation
            # - Data augmentation
            # - Simulation or procedural generation
            
            try:
                # Placeholder for your generation method
                sample_data = self._create_sample(config, i)
                self._save_sample(sample_data, sample_path)
                generated_count += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to generate sample {i} for {category_code}: {e}")
        
        return {
            'success': True,
            'message': f'Generated {generated_count} new samples',
            'total_samples': len(list(category_dir.glob("sample_*.[ext]")))
        }
    
    def _create_sample(self, config: Dict, sample_index: int) -> Any:
        """
        Create a single sample based on configuration.
        
        IMPLEMENT YOUR SPECIFIC GENERATION LOGIC HERE:
        - Use config parameters for generation
        - Handle different generation methods
        - Implement quality controls
        
        Args:
            config: Category configuration dictionary
            sample_index: Index of the sample being generated
            
        Returns:
            Generated sample data
        """
        # PLACEHOLDER - Replace with your implementation
        # Examples:
        # - Generate synthetic images
        # - Create text samples via API
        # - Simulate sensor data
        # - Generate audio samples
        pass
    
    def _save_sample(self, sample_data: Any, file_path: Path) -> None:
        """
        Save generated sample to file.
        
        CUSTOMIZE FOR YOUR DATA FORMAT:
        - Handle your specific file format
        - Implement compression if needed
        - Add metadata or annotations
        """
        # PLACEHOLDER - Replace with your implementation
        # Examples:
        # - Save images (PIL, OpenCV)
        # - Save audio (librosa, soundfile)
        # - Save text (JSON, CSV)
        # - Save tensors (NumPy, PyTorch)
        pass

# Category Configuration Template
CATEGORY_CONFIG_TEMPLATE = {
    "category_name": "[Human Readable Name]",
    "category_code": "[short_code]",
    "generation_settings": {
        "api_endpoint": "[API_URL]",
        "parameters": {
            "param1": "value1",
            "param2": "value2"
        }
    },
    "sample_texts": [  # For text-based generation
        "Sample input 1",
        "Sample input 2"
    ],
    "quality_thresholds": {
        "min_quality": 0.8,
        "max_duration": 10.0  # Customize for your domain
    },
    "metadata": {
        "description": "Category description",
        "source": "Generation method",
        "version": "1.0"
    }
}
```

### 2. Model Handler Template (`src/model_handler.py`)

```python
"""
ML Model management for [YOUR_USE_CASE].
Handles training, evaluation, and inference with dynamic sample generation.
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier  # Customize for your use case
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import logging

from .sample_generator import ScalableSampleGenerator
from .feature_extractor import FeatureExtractor
from .data_processor import DataProcessor

class ModelHandler:
    """
    Handles ML model training and inference for [YOUR_USE_CASE].
    
    Features:
    - Dynamic training with generated samples
    - Multiple model support and evaluation
    - Automated performance validation
    - Production-ready inference pipeline
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.data_processor = DataProcessor()
        self.sample_generator = ScalableSampleGenerator()
        
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.model_config = {
            'random_forest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'random_state': 42,
                    'max_depth': 10
                }
            }
            # Add other model types as needed
        }
    
    def train_model(
        self,
        use_generated_samples: bool = True,
        custom_data_path: Optional[str] = None,
        model_type: str = 'random_forest',
        force_retrain: bool = False,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Train the ML model using generated or custom samples.
        
        Args:
            use_generated_samples: Whether to use generated training data
            custom_data_path: Path to custom training data
            model_type: Type of model to train
            force_retrain: Whether to force retraining
            verbose: Whether to show detailed output
            
        Returns:
            Training results and performance metrics
        """
        model_path = self.model_dir / f"{model_type}_model.joblib"
        
        # Check if model exists and skip if not forcing retrain
        if model_path.exists() and not force_retrain:
            self.model = joblib.load(model_path)
            return {
                'message': 'Loaded existing model',
                'model_path': str(model_path)
            }
        
        # Prepare training data
        if use_generated_samples:
            X, y, category_names = self._load_generated_samples()
        else:
            X, y, category_names = self._load_custom_samples(custom_data_path)
        
        if len(X) == 0:
            raise ValueError("No training data available")
        
        # Initialize model
        model_class = self.model_config[model_type]['class']
        model_params = self.model_config[model_type]['params']
        self.model = model_class(**model_params)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        if verbose:
            print(f"Training {model_type} with {len(X_train)} samples...")
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        
        # Predictions for detailed evaluation
        y_pred = self.model.predict(X_test)
        
        # Save model
        joblib.dump(self.model, model_path)
        
        results = {
            'model_type': model_type,
            'n_samples': len(X),
            'n_features': X.shape[1] if len(X.shape) > 1 else len(X[0]),
            'n_classes': len(category_names),
            'class_names': category_names,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model_path': str(model_path)
        }
        
        if verbose:
            print(f"Training completed:")
            print(f"  Training accuracy: {train_score:.1%}")
            print(f"  Test accuracy: {test_score:.1%}")
            print(f"  CV score: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
            print(f"  Model saved: {model_path}")
        
        return results
    
    def _load_generated_samples(self) -> Tuple[List, List, List]:
        """
        Load training data from generated samples.
        
        Returns:
            Features, labels, and category names
        """
        training_info = self.sample_generator.get_training_info()
        
        X, y = [], []
        category_names = []
        
        for category_code, info in training_info.items():
            category_names.append(info['category_name'])
            samples_dir = Path(info['samples_path'])
            
            # Load all samples for this category
            sample_files = list(samples_dir.glob("sample_*.[ext]"))  # Replace [ext]
            
            for sample_file in sample_files:
                try:
                    # Load and process sample
                    data = self.data_processor.load_data(str(sample_file))
                    features = self.feature_extractor.extract_features(data)
                    
                    X.append(features)
                    y.append(category_code)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process {sample_file}: {e}")
        
        return X, y, category_names
    
    def _load_custom_samples(self, data_path: Optional[str]) -> Tuple[List, List, List]:
        """
        Load training data from custom dataset.
        
        IMPLEMENT FOR YOUR SPECIFIC DATA FORMAT
        """
        # PLACEHOLDER - Implement based on your data format
        # Examples:
        # - Load from CSV/JSON files
        # - Process image directories
        # - Handle audio file collections
        # - Parse text datasets
        pass
    
    def predict(self, input_data: Any) -> Dict[str, Any]:
        """
        Make prediction on input data.
        
        Args:
            input_data: Raw input data for classification
            
        Returns:
            Prediction results with confidence and metadata
        """
        if self.model is None:
            # Try to load existing model
            model_files = list(self.model_dir.glob("*_model.joblib"))
            if not model_files:
                raise ValueError("No trained model available")
            
            self.model = joblib.load(model_files[0])
        
        # Process input data
        processed_data = self.data_processor.process_input(input_data)
        features = self.feature_extractor.extract_features(processed_data)
        
        # Make prediction
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        
        # Get category names
        category_names = self.model.classes_
        
        # Create probability mapping
        prob_mapping = {
            name: float(prob) 
            for name, prob in zip(category_names, probabilities)
        }
        
        # Calculate confidence
        confidence = float(max(probabilities))
        
        # Determine reliability (customize threshold)
        reliable = confidence > 0.6
        
        return {
            'prediction': prediction,
            'category_name': prediction,  # Customize if needed
            'confidence': confidence,
            'reliable': reliable,
            'all_probabilities': prob_mapping,
            'timestamp': np.datetime64('now').item()
        }

    def classify_data(self, data_path: str) -> Dict[str, Any]:
        """
        Classify data from file path.
        
        Args:
            data_path: Path to data file
            
        Returns:
            Classification results
        """
        # Load data
        data = self.data_processor.load_data(data_path)
        
        # Get prediction
        result = self.predict(data)
        result['file_path'] = data_path
        
        return result
```

### 3. Configuration Template (for each category)

Create this structure for each category in `[data_type]_samples/[category]/config.json`:

```json
{
  "category_name": "[Human Readable Category Name]",
  "category_code": "[short_category_code]",
  "generation_settings": {
    "api_endpoint": "[API_URL_OR_SERVICE]",
    "method": "[GENERATION_METHOD]",
    "parameters": {
      "quality": "high",
      "format": "[OUTPUT_FORMAT]",
      "custom_param1": "value1",
      "custom_param2": "value2"
    }
  },
  "sample_inputs": [
    "[Input text/prompt/parameter 1]",
    "[Input text/prompt/parameter 2]",
    "[Input text/prompt/parameter 3]",
    "[Input text/prompt/parameter 4]",
    "[Input text/prompt/parameter 5]"
  ],
  "quality_thresholds": {
    "min_confidence": 0.8,
    "max_file_size": "10MB",
    "min_duration": 2.0,
    "max_duration": 10.0
  },
  "augmentation_settings": {
    "enable_augmentation": true,
    "augmentation_ratio": 0.3,
    "techniques": [
      "rotation",
      "noise_addition",
      "scaling"
    ]
  },
  "metadata": {
    "description": "[Category description]",
    "source": "[Data source or generation method]",
    "version": "1.0",
    "created_date": "2024-01-01",
    "tags": ["tag1", "tag2", "tag3"]
  }
}
```

---

## 🚀 Implementation Workflow

### Phase 1: Project Setup
1. **Create Project Structure**
   ```bash
   mkdir [project-name]
   cd [project-name]
   # Create directory structure as shown above
   ```

2. **Initialize Core Components**
   - Copy and customize the template files
   - Replace placeholders with your specific use case
   - Set up data processing pipeline
   - Configure feature extraction methods

3. **Create Category Configurations**
   - Define your categories/classes
   - Create config.json files for each category
   - Set up generation parameters
   - Configure quality thresholds

### Phase 2: Sample Generation System
1. **Implement Sample Generator**
   - Customize `_create_sample()` method
   - Implement `_save_sample()` method
   - Set up your data generation API/method
   - Add quality validation

2. **Test Sample Generation**
   ```python
   from src.sample_generator import ScalableSampleGenerator
   
   generator = ScalableSampleGenerator()
   results = generator.generate_samples(
       categories=['category1', 'category2'],
       num_samples=5
   )
   ```

### Phase 3: Model Training Pipeline
1. **Customize Feature Extraction**
   - Implement domain-specific feature extraction
   - Handle your data format
   - Optimize for your use case

2. **Set up Model Training**
   - Choose appropriate ML algorithms
   - Configure hyperparameters
   - Set up evaluation metrics

3. **Test Training Pipeline**
   ```python
   from src.model_handler import ModelHandler
   
   model_handler = ModelHandler()
   results = model_handler.train_model(
       use_generated_samples=True,
       verbose=True
   )
   ```

### Phase 4: Integration & Testing
1. **Create Main CLI Interface**
   - Implement command-line interface
   - Add batch processing capabilities
   - Set up output formatting

2. **Add Comprehensive Testing**
   - Unit tests for each component
   - Integration tests for full pipeline
   - Performance benchmarks

3. **Create Documentation**
   - Usage examples and tutorials
   - API documentation
   - Deployment guides

---

## 🎯 Use Case Examples

### Example 1: Image Classification (Dog Breeds)
```markdown
**Project Name**: Dog Breed Classifier
**Primary Task**: CLASSIFICATION
**Input Data Type**: IMAGE
**Output Categories**: [golden_retriever, labrador, poodle, bulldog, ...]
**Sample Generation Method**: Stock photo APIs + Data augmentation
```

**Key Customizations**:
- Use image processing libraries (PIL, OpenCV)
- Implement data augmentation (rotation, scaling, color adjustment)
- Extract visual features (CNN features, color histograms)
- Use image classification models (ResNet, EfficientNet)

### Example 2: Sentiment Analysis
```markdown
**Project Name**: Social Media Sentiment Analyzer
**Primary Task**: CLASSIFICATION
**Input Data Type**: TEXT
**Output Categories**: [positive, negative, neutral, mixed]
**Sample Generation Method**: Text generation API + Template-based creation
```

**Key Customizations**:
- Use NLP libraries (spaCy, NLTK, transformers)
- Implement text preprocessing and tokenization
- Extract linguistic features (TF-IDF, embeddings)
- Use text classification models (BERT, RoBERTa)

### Example 3: Time Series Prediction
```markdown
**Project Name**: Stock Price Predictor
**Primary Task**: REGRESSION
**Input Data Type**: TIME_SERIES
**Output Categories**: [price_movement_values]
**Sample Generation Method**: Financial data simulation + Historical augmentation
```

**Key Customizations**:
- Use time series libraries (pandas, sklearn)
- Implement temporal feature extraction
- Generate synthetic market data
- Use regression models (LSTM, GRU, XGBoost)

---

## 📋 Customization Checklist

### Core Modifications Required:
- [ ] Replace `[YOUR_USE_CASE]` with your specific application
- [ ] Update file extensions and data formats
- [ ] Implement `_create_sample()` method in generator
- [ ] Implement `_save_sample()` method for your data format
- [ ] Customize feature extraction for your domain
- [ ] Configure appropriate ML models for your task
- [ ] Set up category configurations with proper parameters
- [ ] Implement data loading and preprocessing
- [ ] Add domain-specific quality validation
- [ ] Configure evaluation metrics for your use case

### Optional Enhancements:
- [ ] Add real-time processing capabilities
- [ ] Implement data augmentation techniques
- [ ] Add model ensemble methods
- [ ] Create web API interface
- [ ] Add monitoring and logging
- [ ] Implement A/B testing framework
- [ ] Add deployment automation
- [ ] Create interactive visualization tools

---

## 🎉 Success Patterns

### Scalability Features:
1. **Configuration-Driven Expansion**: Easy addition of new categories
2. **Automated Sample Management**: Intelligent caching and regeneration
3. **Modular Architecture**: Independent, reusable components
4. **Dynamic Training**: Automated model updates with new data
5. **Production-Ready**: Comprehensive testing and deployment support

### Performance Optimization:
1. **Efficient Caching**: Avoid unnecessary regeneration
2. **Batch Processing**: Handle large datasets efficiently
3. **Feature Optimization**: Domain-specific feature engineering
4. **Model Selection**: Choose appropriate algorithms for your data
5. **Quality Gates**: Automated validation and error handling

### Maintainability:
1. **Clear Documentation**: Comprehensive guides and examples
2. **Extensive Testing**: Unit, integration, and performance tests
3. **Error Handling**: Graceful failure and recovery mechanisms
4. **Logging**: Detailed operation tracking and debugging
5. **Version Control**: Model and data versioning strategies

---

## 🚀 Deployment Considerations

### Local Development:
```bash
# Train model with generated samples
python [main_script].py --train --use-generated --verbose

# Classify single item
python [main_script].py --input data/sample.[ext] --verbose

# Batch processing
python [main_script].py --batch input_folder/ --output results/
```

### Production Deployment:
- **API Service**: REST/GraphQL endpoints for classification
- **Batch Processing**: Scheduled data processing pipelines
- **Real-time Processing**: Streaming data classification
- **Monitoring**: Performance metrics and model drift detection
- **Scaling**: Load balancing and auto-scaling capabilities

---

## 📚 Additional Resources

### Key Libraries by Domain:
- **Images**: OpenCV, PIL, scikit-image, torchvision
- **Audio**: librosa, soundfile, PyAudio, torchaudio  
- **Text**: spaCy, NLTK, transformers, gensim
- **Time Series**: pandas, statsmodels, prophet, tslearn
- **General ML**: scikit-learn, XGBoost, TensorFlow, PyTorch

### Development Tools:
- **Testing**: pytest, unittest, hypothesis
- **Documentation**: Sphinx, MkDocs, Jupyter
- **Monitoring**: MLflow, Weights & Biases, TensorBoard
- **Deployment**: Docker, Kubernetes, FastAPI, Flask

---

*This template provides a comprehensive foundation for building scalable ML projects with dynamic sample generation and automated training pipelines. Customize the components based on your specific domain and requirements.* 