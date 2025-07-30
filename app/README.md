# Drone Detection API & UI

This directory contains the Flask API and Streamlit UI for the drone detection system.

## Files

- `drone_detection.py` - Flask API server
- `streamlit_ui.py` - Streamlit web interface
- `start_services.sh` - Startup script for both services

## Quick Start

### Option 1: Use the startup script (Recommended)
```bash
./start_services.sh
```

### Option 2: Manual startup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Flask API:**
   ```bash
   python app/drone_detection.py
   ```

3. **Start Streamlit UI (in another terminal):**
   ```bash
   streamlit run app/streamlit_ui.py
   ```

## Access Points

- **Flask API:** http://localhost:5000
- **Streamlit UI:** http://localhost:8501

## API Endpoints

### Health Check
```bash
GET /health
```

### Classification
```bash
POST /classify
Content-Type: application/json

{
  "image_base64": "base64_encoded_image_string"
}
```

Or with file upload:
```bash
POST /classify
Content-Type: multipart/form-data

image: file
```

### Detection
```bash
POST /detect
Content-Type: application/json

{
  "image_base64": "base64_encoded_image_string"
}
```

### Get Classes
```bash
GET /classes
```

## Response Format

### Classification Response
```json
{
  "predicted_class": "DRONE",
  "confidence": 0.95,
  "class_probabilities": {
    "AIRPLANE": 0.02,
    "BIRD": 0.01,
    "DRONE": 0.95,
    "HELICOPTER": 0.02
  }
}
```

### Detection Response
```json
{
  "detections": [
    {
      "class": "DRONE",
      "confidence": 0.95,
      "bbox": [x, y, width, height]
    }
  ]
}
```

## Usage Examples

### Python Client Example
```python
import requests
import base64

# Encode image to base64
with open("image.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

# Make classification request
response = requests.post(
    "http://localhost:5000/classify",
    json={"image_base64": img_base64}
)

result = response.json()
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### curl Example
```bash
# Health check
curl http://localhost:5000/health

# Classification with file upload
curl -X POST \
  -F "image=@path/to/image.jpg" \
  http://localhost:5000/classify

# Get available classes
curl http://localhost:5000/classes
```

## Model Information

The API uses trained models from the `data/models/` directory:
- `best_classification_model.h5` - Classification model
- `best_detection_model.h5` - Detection model

## Troubleshooting

1. **Models not loading:** Ensure model files exist in `data/models/`
2. **API not accessible:** Check if Flask server is running on port 5000
3. **CORS issues:** The API includes CORS headers for web requests
4. **Memory issues:** Large images are automatically resized to 224x224

## Features

### Streamlit UI Features:
- 📤 Image upload interface
- 🎯 Real-time classification
- 🔍 Object detection
- 📊 Interactive probability charts
- 🔧 API status monitoring
- 📱 Responsive design

### Flask API Features:
- 🚀 RESTful endpoints
- 🔄 Multiple input formats (file upload, base64)
- 📏 Automatic image preprocessing
- 🛡️ Error handling and logging
- 🌐 CORS support
- 💾 Model caching
