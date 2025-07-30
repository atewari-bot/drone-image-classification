"""
Streamlit UI for Drone Detection
Interactive web interface for uploading and classifying drone images.
"""

import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import base64
import json
import plotly.express as px
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Drone Detection System",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:5000"

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def call_classification_api(image):
    """Call the Flask API for classification"""
    try:
        # Convert image to base64
        img_base64 = encode_image_to_base64(image)
        
        # Prepare request
        payload = {"image_base64": img_base64}
        
        # Make API call
        response = requests.post(
            f"{API_BASE_URL}/classify",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the API. Make sure the Flask server is running on localhost:5000")
        return None
    except Exception as e:
        st.error(f"❌ Error calling API: {str(e)}")
        return None

def call_detection_api(image):
    """Call the Flask API for detection"""
    try:
        # Convert image to base64
        img_base64 = encode_image_to_base64(image)
        
        # Prepare request
        payload = {"image_base64": img_base64}
        
        # Make API call
        response = requests.post(
            f"{API_BASE_URL}/detect",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the API. Make sure the Flask server is running on localhost:5000")
        return None
    except Exception as e:
        st.error(f"❌ Error calling API: {str(e)}")
        return None

def check_api_health():
    """Check if the API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def display_prediction_results(result):
    """Display prediction results with visualizations"""
    if result:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Prediction Result")
            
            # Main prediction
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            # Display with confidence styling
            if confidence > 0.8:
                confidence_color = "🟢"
            elif confidence > 0.6:
                confidence_color = "🟡"
            else:
                confidence_color = "🔴"
            
            st.markdown(f"""
            **Predicted Class:** {predicted_class}  
            **Confidence:** {confidence_color} {confidence:.2%}
            """)
            
            # Confidence meter
            st.progress(confidence)
        
        with col2:
            st.subheader("📊 Class Probabilities")
            
            # Create probability chart
            prob_data = result['class_probabilities']
            df = pd.DataFrame(
                list(prob_data.items()),
                columns=['Class', 'Probability']
            )
            
            # Create bar chart
            fig = px.bar(
                df, 
                x='Class', 
                y='Probability',
                title="Class Prediction Probabilities",
                color='Probability',
                color_continuous_scale='viridis'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display probabilities as table
            st.dataframe(
                df.style.format({'Probability': '{:.2%}'}),
                hide_index=True
            )

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🚁 Drone Detection System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        # API Health Check
        st.subheader("API Status")
        health_status = check_api_health()
        
        if health_status:
            st.success("✅ API is healthy")
            models_status = health_status.get('models_loaded', {})
            st.write("**Models loaded:**")
            st.write(f"• Classification: {'✅' if models_status.get('classification') else '❌'}")
            st.write(f"• Detection: {'✅' if models_status.get('detection') else '❌'}")
        else:
            st.error("❌ API is not accessible")
            st.info("Please ensure the Flask server is running:\n```bash\npython app/drone_detection.py\n```")
        
        st.markdown("---")
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        model_type = st.selectbox(
            "Choose model type:",
            ["Classification", "Detection"],
            help="Classification: Identify the type of aircraft\nDetection: Locate and identify objects in the image"
        )
        
        st.markdown("---")
        
        # About
        st.subheader("ℹ️ About")
        st.info("""
        This system can classify aerial objects into:
        • ✈️ **AIRPLANE**
        • 🦅 **BIRD** 
        • 🚁 **DRONE**
        • 🚁 **HELICOPTER**
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image containing an aircraft, bird, drone, or helicopter"
        )
        
        # Sample images section
        st.subheader("📷 Or try sample images")
        sample_images = {
            "Drone": "https://via.placeholder.com/300x200/4CAF50/FFFFFF?text=DRONE",
            "Airplane": "https://via.placeholder.com/300x200/2196F3/FFFFFF?text=AIRPLANE",
            "Helicopter": "https://via.placeholder.com/300x200/FF9800/FFFFFF?text=HELICOPTER",
            "Bird": "https://via.placeholder.com/300x200/9C27B0/FFFFFF?text=BIRD"
        }
        
        sample_cols = st.columns(2)
        for i, (name, url) in enumerate(sample_images.items()):
            with sample_cols[i % 2]:
                if st.button(f"Use {name} sample", key=f"sample_{i}"):
                    st.info(f"Sample {name} image selected (placeholder)")
    
    with col2:
        st.header("🖼️ Image Preview")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.write(f"**Image size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**File size:** {uploaded_file.size} bytes")
            st.write(f"**Format:** {image.format}")
            
        else:
            st.info("👆 Upload an image to see preview")
    
    # Analysis section
    if uploaded_file is not None:
        st.markdown("---")
        st.header("🔍 Analysis Results")
        
        image = Image.open(uploaded_file)
        
        # Analysis buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 Run Classification", type="primary"):
                with st.spinner("Analyzing image..."):
                    result = call_classification_api(image)
                    if result:
                        display_prediction_results(result)
        
        with col2:
            if st.button("🔍 Run Detection", type="secondary"):
                with st.spinner("Detecting objects..."):
                    result = call_detection_api(image)
                    if result:
                        st.subheader("🎯 Detection Results")
                        detections = result.get('detections', [])
                        
                        if detections:
                            for i, detection in enumerate(detections):
                                st.write(f"**Object {i+1}:**")
                                st.write(f"• Class: {detection['class']}")
                                st.write(f"• Confidence: {detection['confidence']:.2%}")
                                st.write(f"• Bounding Box: {detection['bbox']}")
                        else:
                            st.info("No objects detected")
        
        with col3:
            if st.button("📊 Both Models"):
                with st.spinner("Running both models..."):
                    # Run classification
                    classification_result = call_classification_api(image)
                    
                    # Run detection
                    detection_result = call_detection_api(image)
                    
                    if classification_result:
                        st.subheader("🎯 Classification Results")
                        display_prediction_results(classification_result)
                    
                    if detection_result:
                        st.subheader("🔍 Detection Results")
                        detections = detection_result.get('detections', [])
                        for i, detection in enumerate(detections):
                            st.write(f"**Detection {i+1}:** {detection['class']} ({detection['confidence']:.2%})")

if __name__ == "__main__":
    main()
