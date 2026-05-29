import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ── Model ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_final.keras')

model = load_model()

# ── UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")
st.title("🐾 Cat vs Dog Classifier")
st.markdown("**MobileNetV2 Transfer Learning — 98.70% Test Accuracy**")
st.divider()

uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((224, 224))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    prob = model.predict(x, verbose=0)[0][0]
    label = "🐶 Dog" if prob > 0.5 else "🐱 Cat"
    conf  = max(prob, 1 - prob) * 100

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.metric("Prediction", label)
        st.metric("Confidence", f"{conf:.1f}%")
        st.progress(int(conf))

    if conf > 95:
        st.success(f"High confidence: **{label}**")
    else:
        st.info(f"Moderate confidence: **{label}**")