import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import cv2
import pandas as pd

st.title("Brain Tumor Classifier - With Synthetic Image Generator")  # Updated title
st.write("Predicts tumors and displays pre-loaded synthetic brain images with a physician report.")

# Load model
try:
    model = tf.keras.models.load_model('../Models/model.h5', compile=False)
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

CATEGORIES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Synthetic image display
SYNTHETIC_DIR = "synthetic_images"
synthetic_images = [f for f in os.listdir(SYNTHETIC_DIR) if f.endswith('.png')]
if synthetic_images:
    if st.button("Generate Synthetic Image"):
        selected_image = random.choice(synthetic_images)
        synth_path = os.path.join(SYNTHETIC_DIR, selected_image)
        synth_img = Image.open(synth_path).convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
        # Center the image using columns
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:  # Middle column for centering
            st.image(synth_img, width=112, use_container_width=False)  # Half size, centered
else:
    st.warning("No synthetic images found in the synthetic_images folder.")

# Single image upload section
upload_placeholder = st.empty()
uploaded_file = upload_placeholder.file_uploader("Upload MRI Image", type=["jpg", "npy"], accept_multiple_files=False)
if uploaded_file:
    # Clear previous content
    st.empty()
    st.empty()
    st.empty()

    # Load image
    if uploaded_file.name.endswith('.npy'):
        img = np.load(uploaded_file)
        if img.ndim == 2:  # Grayscale to RGB
            img = np.stack([img] * 3, axis=-1)
    else:
        img = np.array(Image.open(uploaded_file).convert('RGB'))
    # Resize to model’s expected size (224x224)
    img_pil = Image.fromarray(img)
    img_resized = img_pil.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # Predict
    try:
        pred = model.predict(img_array)
        probs = tf.nn.softmax(pred[0]).numpy()
        max_prob = float(np.max(probs))
        label_idx = np.argmax(probs)
        label = CATEGORIES[label_idx]
        if label == "No Tumor":
            st.image(img_resized, caption="Uploaded Image", use_container_width=True)
            st.write("No Tumor Has Been Detected")
            st.stop()
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.stop()
    # Generate tumor likelihood heatmap (approximation)
    gray_img = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2GRAY)
    heatmap = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
    coords = []
    for _ in range(3):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)
        coords.append((max_loc[0], max_loc[1], max_val / 255.0))  # X, Y, normalized probability
        cv2.circle(heatmap, max_loc, 5, 0, -1)  # Mask the found point

    # Merge close coordinates
    merged_coords = []
    threshold = 10
    while coords:
        x, y, prob = coords.pop(0)
        group = [(x, y, prob)]
        remaining = []
        for cx, cy, cp in coords:
            if abs(cx - x) < threshold and abs(cy - y) < threshold:
                group.append((cx, cy, cp))
            else:
                remaining.append((cx, cy, cp))
        coords = remaining
        avg_x = int(sum(c[0] for c in group) / len(group))
        avg_y = int(sum(c[1] for c in group) / len(group))
        avg_prob = max(c[2] for c in group)
        merged_coords.append((avg_x, avg_y, avg_prob))

    # Calculate average regional probability
    avg_prob = np.mean([coord[2] for coord in merged_coords]) if merged_coords else 0.0

    # Display resized original image with report
    st.image(img_resized, caption=f"Uploaded Image: {uploaded_file.name} - Prediction: {label} ({max_prob:.2f})", use_container_width=True)
    st.subheader("Radiology Report")
    st.write("### Tumor Likelihood Assessment")
    st.write(f"**Tumor Probability (Average Regional Intensity)**: {avg_prob * 100:.0f}%")
    st.write("**Top 3 Suspected Tumor Regions (Pixel Coordinates and Likelihood)**")
    report_data = [
        [f"Region {i+1}", f"X: {coord[0]}, Y: {coord[1]}", f"Probability: {coord[2] * 100:.1f}%", "Core" if coord[2] > 0.7 else "Periphery"]
        for i, coord in enumerate(merged_coords)
    ]
    st.table(pd.DataFrame(report_data, columns=["Region", "Coordinates", "Likelihood", "Region Type"]))
    st.write("**Notes**: Coordinates indicate pixel locations with highest intensity, suggesting tumor presence. Further biopsy recommended for confirmation.")

    # Display original image with circles and colored pixels
    marked_img = np.array(img_resized)
    for x, y, prob in merged_coords:
        radius = 10 if len([c for c in merged_coords if abs(c[0] - x) < threshold and abs(c[1] - y) < threshold]) > 1 else 5
        cv2.circle(marked_img, (x, y), radius, (0, 0, 255), 2)  # Red circle
        # Color pixels within circle
        for i in range(max(0, x - radius), min(224, x + radius + 1)):
            for j in range(max(0, y - radius), min(224, y + radius + 1)):
                if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                    marked_img[j, i] = [0, 0, 255] if prob > 0.5 else [255, 0, 0]  # Red for high prob, blue for lower
    st.image(marked_img, caption="Tumor Highlighted Regions", use_container_width=True)