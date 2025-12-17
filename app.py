import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(page_title="BDT Banknote Classifier - CNN", page_icon="৳", layout="centered")

# Custom CSS to match the design
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: #FFFFFF;
    }
    .prediction-box {
        font-size: 1.5rem;
        font-weight: 600;
        color: #008000;
        padding: 1rem;
        background-color: #f1f8f4;
        border-radius: 8px;
        text-align: center;
        margin: 0 0 1.5rem 0;
    }
    .confidence-box {
        font-size: 1.3rem;
        color: #000080;
        text-align: center;
        margin: 0.5rem 0 1.5rem 0;
        padding: 1rem;
        background-color: #f1f8f4;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">৳ BDT Banknote Classifier - CNN ৳</div>', unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_bdt_model():
    try:
        model = load_model('bdt_cnn_model.h5')
        return model
    except Exception as e:
        st.error(f"⚠️ Model file 'bdt_cnn_model.h5' not found! Error: {str(e)}")
        return None

model = load_bdt_model()

# Class labels
class_labels = ['1000_BDT', '100_BDT', '200_BDT', '500_BDT']

def preprocess_image(image):
    """Preprocess image for CNN model"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Main layout
st.markdown("---")

# Upload Section
st.markdown("### Upload BDT Banknote")
st.markdown("*Ex: 100, 200, 500 or 1000 BDT*")

uploaded_file = st.file_uploader(
    "Drag and drop file here",
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed"
)

st.markdown("---")

# Results Section
if uploaded_file is not None and model is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📷 Uploaded Banknote")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Results")
        
        # Make prediction
        with st.spinner("Analyzing banknote..."):
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img, verbose=0)[0]
        
        # Get prediction and confidence
        predicted_idx = np.argmax(predictions)
        predicted_class = class_labels[predicted_idx]
        confidence = predictions[predicted_idx] * 100
        
        # Display prediction
        st.markdown(f'<div class="prediction-box">Prediction: <br>{predicted_class.replace("_BDT", "")} Taka</div>', 
                   unsafe_allow_html=True)
        
        st.markdown(f'<div class="confidence-box">Confidence: {confidence:.0f}%</div>', 
                   unsafe_allow_html=True)


     # All Class Probabilities
        st.markdown("#### All Class Probabilities")
        
        prob_data = {
            "Denomination": ["1000 Taka", "500 Taka", "200 Taka", "100 Taka"],
            "Probability": [
                f"{predictions[class_labels.index('1000_BDT')]*100:.1f}%",
                f"{predictions[class_labels.index('500_BDT')]*100:.1f}%",
                f"{predictions[class_labels.index('200_BDT')]*100:.1f}%",
                f"{predictions[class_labels.index('100_BDT')]*100:.1f}%"
            ]
        }
        
        # Create table
        for denom, prob in zip(prob_data["Denomination"], prob_data["Probability"]):
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.write(f"**{denom}**")
            with col_b:
                st.write(f"`{prob}`")
            st.progress(float(prob.strip('%'))/100)
        
        # Confidence warning
        if confidence < 80:
            st.warning("⚠️ **Low Confidence Warning**\n\nThe model is uncertain about this prediction. Consider:\n- Better lighting\n- Clearer image\n- Flat, unfolded note")
        else:
            st.success("✅ **High Confidence - Prediction is reliable**")

elif model is None:
    st.error("❌ Cannot make predictions without the model file. Please ensure 'bdt_cnn_model.h5' is in the same directory.")
else:
    st.info("👆 Please upload a banknote image to classify")       
