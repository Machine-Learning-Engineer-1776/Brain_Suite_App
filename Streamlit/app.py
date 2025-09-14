import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2

st.title("Brain Tumor Classifier - CNN")
# Load model
try:
    model = tf.keras.models.load_model('../Models/model_retrained.h5')
except FileNotFoundError:
    st.error("Model weights not found at Models/model_retrained.h5")
    st.stop()

def grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 1]  # Tumor class

    grads = tape.gradient(loss, conv_outputs)
    conv_outputs = conv_outputs[0]
    grads = grads[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))

    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)
    for index, w in enumerate(weights):
        cam += w * conv_outputs[:, :, index]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_array.shape[1], img_array.shape[2]))
    cam = cam / np.max(cam)
    return cam

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
        # Grad-CAM if tumor
        if prob > 0.5:
            layer_name = 'conv5_block3_out'  # Last conv layer in ResNet50, adjust if different
            cam = grad_cam(model, img_array, layer_name)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            superimposed = cv2.addWeighted(np.uint8(img * 255), 0.6, heatmap, 0.4, 0)
            # Find contour for circle
            gray = np.uint8(cam * 255)
            _, thresh = cv2.threshold(gray, 127, 255, 0)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                (x, y), r = cv2.minEnclosingCircle(contours[0])
                cv2.circle(superimposed, (int(x), int(y)), int(r), (0, 255, 0), 2)
            st.image(superimposed, caption="Tumor Circled", use_column_width=True)
