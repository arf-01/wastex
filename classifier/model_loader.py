"""
Keras Model Loader for Image Classification
Loads and runs inference with .keras format models
"""
import os
import numpy as np
from PIL import Image
from pathlib import Path
from django.conf import settings


class KerasModelLoader:
    """Load and use Keras models for image classification."""
    
    def __init__(self, model_path=None):
        self.model = None
        self.class_names = []
        self.input_shape = (224, 224)  # Default, will be updated from model
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            self.load_class_names()
    
    def load_model(self, model_path):
        """Load a Keras model from .keras file."""
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            
            # Get input shape from model
            input_shape = self.model.input_shape
            if input_shape[1] and input_shape[2]:
                self.input_shape = (input_shape[1], input_shape[2])
            
            print(f"Model loaded: {model_path}")
            print(f"Input shape: {self.input_shape}")
            print(f"Output classes: {self.model.output_shape[-1]}")
            
        except ImportError:
            raise ImportError(
                "TensorFlow not installed. Install with: pip install tensorflow"
            )
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
    
    def load_class_names(self):
        """Load class names from classes.txt file."""
        models_dir = Path(settings.BASE_DIR) / 'models'
        class_file = models_dir / 'classes.txt'
        
        if class_file.exists():
            with open(class_file, 'r') as f:
                self.class_names = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(self.class_names)} class names")
        else:
            # Generate default class names
            if self.model:
                num_classes = self.model.output_shape[-1]
                self.class_names = [f"Class_{i}" for i in range(num_classes)]
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model input.
        Applies standard preprocessing for Keras models.
        """
        # Load and resize image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.input_shape)
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_path, top_k=5):
        """
        Predict class probabilities for an image.
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
        
        Returns:
            List of tuples (class_name, probability)
        """
        if self.model is None:
            # Return demo predictions if no model loaded
            return [
                ("cat", 0.85),
                ("dog", 0.10),
                ("bird", 0.03),
                ("fish", 0.01),
                ("horse", 0.01),
            ][:top_k]
        
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top K predictions
        top_indices = np.argsort(predictions)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            class_name = (
                self.class_names[idx] 
                if idx < len(self.class_names) 
                else f"Class_{idx}"
            )
            confidence = float(predictions[idx])
            results.append((class_name, confidence))
        
        return results


# Global model instance
_model = None


def get_model():
    """Get or create the global model instance."""
    global _model
    
    if _model is None:
        # Look for .keras model in models directory
        models_dir = Path(settings.BASE_DIR) / 'models'
        keras_models = list(models_dir.glob('*.keras'))
        
        if keras_models:
            model_path = str(keras_models[0])
            _model = KerasModelLoader(model_path)
        else:
            # No model found, create empty loader (demo mode)
            _model = KerasModelLoader()
    
    return _model
