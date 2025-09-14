import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

st.title("Brain Tumor Classifier - CNN")
# Load model
try:
    model = tf.keras.models.load_model('../Models/model_retrained.h5')
except FileNotFoundError:
    st.error("Model weights not found at Models/model_retrained.h5")
    st.stop()

uploaded_files = st.file_uploader("Upload MRI Images", type=["jpg", "npy"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Load image
        if uploaded_file.name.endswith('.npy'):
            img = np.load(uploaded_file)
        else:
            img = np.array(Image.open(uploaded_file).resize((224, 224)))  # ResNet50 input size
        img_array = img.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        # Predict
        prob = model.predict(img_array)[0][0]
        label = "Tumor" if prob > 0.5 else "No Tumor"
        # Display
        st.write(f"Image: {uploaded_file.name}, Tumor Probability: {prob:.3f}, Prediction: {label}")
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Prediction: {label}")
        st.pyplot(fig)
