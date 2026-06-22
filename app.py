import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

st.set_page_config(page_title="Road Sign Detection", layout="centered")
st.title("🚦 Road Sign Detection")
st.write("Upload a photo and the model will detect road signs.")

CLASS_NAMES = {
    0: 'Green Light',
    1: 'Red Light',
    2: 'Speed Limit 10',
    3: 'Speed Limit 100',
    4: 'Speed Limit 110',
    5: 'Speed Limit 120',
    6: 'Speed Limit 20',
    7: 'Speed Limit 30',
    8: 'Speed Limit 40',
    9: 'Speed Limit 50',
    10: 'Speed Limit 60',
    11: 'Speed Limit 70',
    12: 'Speed Limit 80',
    13: 'Speed Limit 90',
    14: 'Stop',
}

# Load model (cached so it only loads once)
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    model.model.names = CLASS_NAMES
    return model

model = load_model()

# Confidence slider
conf = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)

# Upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Detecting..."):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            results = model.predict(tmp.name, conf=conf)
            os.unlink(tmp.name)

        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    st.image(annotated_rgb, caption="Detections", use_container_width=True)

    boxes = results[0].boxes
    if len(boxes) == 0:
        st.info("No signs detected. Try lowering the confidence threshold.")
    else:
        st.success(f"Found {len(boxes)} detection(s):")
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = CLASS_NAMES.get(cls_id, f'class_{cls_id}')
            confidence = float(box.conf[0])
            st.write(f"• **{cls_name}** — {confidence:.0%} confidence")
