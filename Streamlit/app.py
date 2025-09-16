import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2

st.title("Brain Tumor Classifier - CNN")
st.write("Predicts Glioma, Meningioma, Pituitary, or No Tumor with confidence percentages.")

# Load model
try:
    model = tf.keras.models.load_model('../Models/model.h5', compile=False)
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

CATEGORIES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

uploaded_files = st.file_uploader("Upload MRI Images", type=["jpg", "npy"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Load image
        if uploaded_file.name.endswith('.npy'):
            img = np.load(uploaded_file)
            if img.ndim == 2:  # Grayscale to RGB
                img = np.stack([img] * 3, axis=-1)
        else:
            img = np.array(Image.open(uploaded_file).convert('RGB').resize((224, 224)))
        img_array = img.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        # Predict
        try:
            pred = model.predict(img_array)
            probs = tf.nn.softmax(pred[0]).numpy()  # Normalize probabilities
            max_prob = float(np.max(probs))
            label_idx = np.argmax(probs)
            label = CATEGORIES[label_idx]
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            continue
        # Display prediction and image
        st.write(f"Image: {uploaded_file.name}, Tumor Probability: {max_prob:.3f}, Prediction: {label}")
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Prediction: {label}")
        st.pyplot(fig)
        # Draw arrow pointing toward center
        superimposed = np.uint8(img * 255)
        height, width = img.shape[:2]
        center_x, center_y = width // 2, height // 2
        arrow_end = (center_x, center_y)
        arrow_start = (center_x + 30, center_y - 30)  # Start outside, point inward
        cv2.arrowedLine(superimposed, arrow_start, arrow_end, (0, 255, 0), 2, tipLength=0.3)
        st.image(superimposed, caption=f"{label} Center Indicated (Confidence: {max_prob:.2%})", use_column_width=True)
