import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# ---------------- CONFIG ----------------
IMG_SIZE = 256
MODEL_PATH = "model/flood_unet_sen1.h5"
# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

st.title("🌊 Flood Detection System")
st.write("Upload an RGB image (JPG / PNG) to detect flooded regions")

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image)

    # Normalize
    img_norm = img_array / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # ---------------- PREDICTION ----------------
    prediction = model.predict(img_input)[0, :, :, 0]

    # Binary mask
    mask = (prediction > 0.5).astype(np.uint8)

    # ---------------- CONFIDENCE SCORE ----------------
    if np.sum(mask) > 0:
        confidence = np.mean(prediction[mask == 1]) * 100
    else:
        confidence = 0.0

    # ---------------- HEATMAP ----------------
    heatmap = (prediction * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)

    # ---------------- DISPLAY ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with col2:
        st.image(mask * 255, caption="Flood Mask", use_container_width=True)

    with col3:
        st.image(overlay, caption="Flood Risk Heatmap", use_container_width=True)

    st.markdown("### 📊 Flood Confidence Score")
    st.progress(int(confidence))
    st.write(f"**{confidence:.2f}% Flood Probability**")

else:
    st.info("👆 Upload a JPG or PNG image to begin")
