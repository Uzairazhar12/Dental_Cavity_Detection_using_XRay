"""
Dental Cavity Detection - Web Application
A simple Streamlit web app for dental cavity detection from X-ray images.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Configuration
IMG_SIZE = 150
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model', 'dental_cavity_model.h5')

# Page configuration
st.set_page_config(
    page_title="Dental Cavity Detection",
    page_icon="🦷",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1E88E5;
        padding: hh20px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .cavity-detected {
        background-color: #FFEBEE;
        color: #C62828;
        border: 2px solid #C62828;
    }
    .normal {
        background-color: #E8F5E9;
        color: #2E7D32;
        border: 2px solid #2E7D32;
    }
    .confidence-text {
        font-size: 18px;
        color: #666;
        text-align: center;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    """Load the trained model (cached)."""
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        return model
    return None

def preprocess_image(image):
    """Preprocess image for prediction."""
    # Convert to numpy array
    img = np.array(image)
    
    # Convert to RGB if needed
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict(model, image):
    """Make prediction on image."""
    # Preprocess
    processed_img = preprocess_image(image)
    
    # Predict
    prediction = model.predict(processed_img, verbose=0)[0][0]
    
    # Get result
    if prediction > 0.5:
        result = "Cavity Detected"
        confidence = prediction * 100
        is_cavity = True
    else:
        result = "Normal (No Cavity)"
        confidence = (1 - prediction) * 100
        is_cavity = False
    
    return result, confidence, is_cavity

def main():
    # Header
    st.markdown("<h1 class='main-header'>🦷 Dental Cavity Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Upload a dental X-ray image to detect cavities</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please train the model first.")
        st.info("""
        **To train the model:**
        1. Prepare your dataset in the `data/` folder
        2. Run: `python train.py`
        3. Refresh this page
        """)
        
        st.markdown("### 📁 Expected folder structure:")
        st.code("""
data/
├── train/
│   ├── cavity/      (X-ray images with cavities)
│   └── normal/      (Normal X-ray images)
└── validation/
    ├── cavity/      (X-ray images with cavities)
    └── normal/      (Normal X-ray images)
        """)
        return
    
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    st.markdown("### 📤 Upload Dental X-Ray Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a dental X-ray image (JPG, PNG, or BMP format)"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🖼️ Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Analysis Result")
            
            with st.spinner("Analyzing X-ray image..."):
                result, confidence, is_cavity = predict(model, image)
            
            # Display result
            if is_cavity:
                st.markdown(f"""
                <div class='result-box cavity-detected'>
                    ⚠️ {result}
                </div>
                """, unsafe_allow_html=True)
                st.warning("Please consult a dentist for proper diagnosis and treatment.")
            else:
                st.markdown(f"""
                <div class='result-box normal'>
                    ✅ {result}
                </div>
                """, unsafe_allow_html=True)
                st.success("No cavity detected. Keep up good oral hygiene!")
            
            # Confidence meter
            st.markdown(f"<p class='confidence-text'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)
            st.progress(float(confidence / 100))
    
    # Information section
    st.markdown("---")
    st.markdown("### ℹ️ About This Application")
    
    st.markdown("""
    <div class='info-box'>
    <b>How it works:</b><br>
    This application uses a Convolutional Neural Network (CNN) trained on dental X-ray images 
    to detect the presence of cavities. Simply upload an X-ray image, and the AI will analyze 
    it and provide a prediction.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <b>⚠️ Disclaimer:</b><br>
    This tool is for educational purposes only and should NOT be used as a substitute for 
    professional dental diagnosis. Always consult a qualified dentist for proper examination 
    and treatment.
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #999;'>Made with ❤️ using Python & Streamlit</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

