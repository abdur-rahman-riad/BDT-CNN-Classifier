import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="BDT Banknote Classifier - CNN",
    page_icon="৳",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
    body {
        background-color: #0e1117;
    }

    .main-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: #ffffff;
    }

    .card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        margin-bottom: 1rem;
    }

    .prediction-box {
        font-size: 1.4rem;
        font-weight: 600;
        color: #000000;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .confidence-box {
        font-size: 1.2rem;
        color: #000000;
        text-align: center;
    }

    .section-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.markdown(
    '<div class="main-title">৳ BDT Banknote Classifier - CNN ৳</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load TFLite model
# --------------------------------------------------
@st.cache_resource
def load_tflite_model():
    """Load TensorFlow Lite model"""
    try:
        interpreter = tf.lite.Interpreter(model_path="bdt_cnn_model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

interpreter = load_tflite_model()

# --------------------------------------------------
# Class labels
# --------------------------------------------------
class_labels = ["1000_BDT", "100_BDT", "200_BDT", "500_BDT"]

# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
def preprocess_image(image):
    """Preprocess image for model input"""
    if image.mode != "RGB":
        image = image.convert("RGB")

    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict_banknote(image, interpreter):
    """Run prediction on image"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess image
    input_data = preprocess_image(image)
    
    # Run inference
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])[0]
    
    return predictions

# --------------------------------------------------
# Upload section
# --------------------------------------------------
st.markdown("### Upload BDT Banknote")
st.markdown("*Supported: 100, 200, 500, 1000 BDT*")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

st.markdown("---")

# --------------------------------------------------
# Results section
# --------------------------------------------------
if uploaded_file and interpreter:
    col1, col2 = st.columns(2)

    # ---------- LEFT: IMAGE ----------
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📷 Uploaded Banknote</div>', unsafe_allow_html=True)

        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- RIGHT: RESULTS ----------
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Prediction</div>', unsafe_allow_html=True)

        with st.spinner("Analyzing banknote..."):
            predictions = predict_banknote(image, interpreter)

        predicted_idx = np.argmax(predictions)
        predicted_class = class_labels[predicted_idx]
        confidence = predictions[predicted_idx] * 100

        st.markdown(
            f'<div class="prediction-box">{predicted_class.replace("_BDT", "")} Taka</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence-box">Confidence: {confidence:.1f}%</div>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- PROBABILITIES ----------
    st.markdown("### All Class Probabilities")

    for label in ["1000_BDT", "500_BDT", "200_BDT", "100_BDT"]:
        prob = predictions[class_labels.index(label)] * 100
        st.write(f"**{label.replace('_BDT','')} Taka** — `{prob:.1f}%`")
        st.progress(prob / 100)

    # ---------- CONFIDENCE MESSAGE ----------
    if confidence < 80:
        st.warning(
            "⚠️ **Low Confidence**\n\n"
            "- Ensure good lighting\n"
            "- Use a flat, unfolded note\n"
            "- Avoid blur"
        )
    else:
        st.success("✅ **High Confidence – Prediction is reliable**")

elif not interpreter:
    st.error("❌ Model not loaded. Ensure `bdt_cnn_model.tflite` exists in repository.")
else:
    st.info("👆 Please upload a banknote image to classify")