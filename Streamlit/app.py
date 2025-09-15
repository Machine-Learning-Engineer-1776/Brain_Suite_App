import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2

st.title("Brain Tumor Classifier - CNN")
st.write("Note: Model predicts binary Tumor/No Tumor due to dataset mapping (glioma, meningioma, pituitary as Tumor).")

# Load model with compile=False to avoid config errors
try:
    model = tf.keras.models.load_model('../Models/model_retrained.h5', compile=False)
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

def grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[0]  # Binary sigmoid output
    grads = tape.gradient(loss, conv_outputs)
    conv_outputs = conv_outputs[0]
    grads = grads[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)
    for index, w in enumerate(weights):
        cam += w * conv_outputs[:, :, index]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_array.shape[2], img_array.shape[1]))
    cam = cam / np.max(cam)
    return cam

uploaded_files = st.file_uploader("Upload MRI Images", type=["jpg", "npy"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Load image
        if uploaded_file.name.endswith('.npy'):
            img = np.load(uploaded_file)
        else:
            img = np.array(Image.open(uploaded_file).resize((224, 224)))
        img_array = img.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        # Predict
        prob = model.predict(img_array)[0][0]
        label = "Tumor Detected" if prob > 0.5 else "No Tumor"
        confidence = prob if label == "Tumor Detected" else (1 - prob)
        # Display prediction and image
        st.write(f"Image: {uploaded_file.name}, Tumor Probability: {confidence:.3f}, Prediction: {label}")
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Prediction: {label}")
        st.pyplot(fig)
        # Grad-CAM and circle if tumor
        if prob > 0.5:
            layer_name = 'conv5_block3_out'  # ResNet50 last conv layer
            try:
                cam = grad_cam(model, img_array, layer_name)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                superimposed = cv2.addWeighted(np.uint8(img * 255), 0.6, heatmap, 0.4, 0)
                gray = np.uint8(cam * 255)
                _, thresh = cv2.threshold(gray, 127, 255, 0)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    (x, y), r = cv2.minEnclosingCircle(contours[0])
                    cv2.circle(superimposed, (int(x), int(y)), int(r), (0, 255, 0), 2)
                    st.image(superimposed, caption=f"Tumor Circled (Confidence: {confidence:.2%})", use_column_width=True)
                else:
                    st.write("No clear tumor region detected for circling.")
            except Exception as e:
                st.error(f"Grad-CAM failed: {str(e)}")
