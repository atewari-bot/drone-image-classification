"""
Drone Detection Flask API
Provides REST endpoints for drone image classification using trained models.
"""

import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Try to import TensorFlow with fallback handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
    print("✅ TensorFlow loaded successfully")
except ImportError as e:
    print(f"⚠️  TensorFlow not available: {e}")
    print("📝 See installation instructions in README")
    TF_AVAILABLE = False
    # Mock functions for development
    def load_model(path):
        print(f"Mock: Loading model from {path}")
        return None

import cv2
from PIL import Image
import io
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for models
classification_model = None
detection_model = None

# Class labels
CLASS_LABELS = ['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER']

# Model paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFICATION_MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'best_classification_model.h5')
DETECTION_MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'best_detection_model.h5')

def load_models():
    """Load the trained models"""
    global classification_model, detection_model
    
    if not TF_AVAILABLE:
        logger.warning("TensorFlow not available - models cannot be loaded")
        return
    
    try:
        if os.path.exists(CLASSIFICATION_MODEL_PATH):
            classification_model = load_model(CLASSIFICATION_MODEL_PATH)
            logger.info("Classification model loaded successfully")
        else:
            logger.warning(f"Classification model not found at {CLASSIFICATION_MODEL_PATH}")
            
        if os.path.exists(DETECTION_MODEL_PATH):
            detection_model = load_model(DETECTION_MODEL_PATH)
            logger.info("Detection model loaded successfully")
        else:
            logger.warning(f"Detection model not found at {DETECTION_MODEL_PATH}")
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

def preprocess_image(img, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    try:
        # Convert PIL Image to numpy array
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        # Resize image
        img_resized = cv2.resize(img, target_size)
        
        # Normalize pixel values
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def decode_base64_image(base64_string):
    """Decode base64 image string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        image_pil = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
            
        return image_pil
        
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'tensorflow_available': TF_AVAILABLE,
        'models_loaded': {
            'classification': classification_model is not None,
            'detection': detection_model is not None
        }
    })

@app.route('/classify', methods=['POST'])
def classify_image():
    """Classify uploaded image"""
    try:
        if classification_model is None:
            return jsonify({'error': 'Classification model not loaded'}), 500
        
        # Get image from request
        if 'image' in request.files:
            # File upload
            file = request.files['image']
            img = Image.open(file.stream)
        elif 'image_base64' in request.json:
            # Base64 encoded image
            img = decode_base64_image(request.json['image_base64'])
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess image
        processed_img = preprocess_image(img)
        
        # Make prediction
        predictions = classification_model.predict(processed_img)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = CLASS_LABELS[predicted_class_idx]
        
        # Get all class probabilities
        class_probabilities = {
            CLASS_LABELS[i]: float(predictions[0][i]) 
            for i in range(len(CLASS_LABELS))
        }
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probabilities
        })
        
    except Exception as e:
        logger.error(f"Error in classification: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Detect objects in uploaded image"""
    try:
        if detection_model is None:
            return jsonify({'error': 'Detection model not loaded'}), 500
        
        # Get image from request
        if 'image' in request.files:
            file = request.files['image']
            img = Image.open(file.stream)
        elif 'image_base64' in request.json:
            img = decode_base64_image(request.json['image_base64'])
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess image
        processed_img = preprocess_image(img)
        
        # Make prediction
        detections = detection_model.predict(processed_img)
        
        # Process detection results (this will depend on your model output format)
        # For now, returning basic classification as detection might use same model
        predicted_class_idx = np.argmax(detections[0])
        confidence = float(detections[0][predicted_class_idx])
        predicted_class = CLASS_LABELS[predicted_class_idx]
        
        return jsonify({
            'detections': [{
                'class': predicted_class,
                'confidence': confidence,
                'bbox': [0, 0, img.width, img.height]  # Full image bbox as placeholder
            }]
        })
        
    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get available classes"""
    return jsonify({'classes': CLASS_LABELS})

if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)