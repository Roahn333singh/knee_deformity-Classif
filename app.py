import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# === Config ===
IMG_SIZE = 224
class_names = ['(Genu Varum)', 'Genu Valgum', 'Normal Knee']  # Replace with actual class labels

# === Load model ===
@st.cache_resource
def load_trained_model():
    return load_model("/Users/rohansingh/Desktop/ComputerVision/LegDeformaties/mobilenet_v1.h5")

model = load_trained_model()

# === Preprocessing ===
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Rescale like during training
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# === Streamlit UI ===
st.title("Leg Deformity Classification 🦵📷")
st.write("Upload an X-ray or image to classify the type of leg deformity.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner('Classifying...'):
        img_array = preprocess_image(uploaded_file)
        preds = model.predict(img_array)
        pred_class = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100

    st.success(f"**Prediction:** {pred_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")
