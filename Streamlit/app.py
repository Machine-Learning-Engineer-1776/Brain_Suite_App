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

def get_heatmap(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0], axis=-1).numpy().item()
        loss = predictions[0][class_idx]
    grads = tape.gradient(loss, conv_outputs)
    conv_outputs = conv_outputs[0]
    grads = grads[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(conv_outputs, weights), axis=-1).numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    cam = np.uint8(255 * cam)
    return cam

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
        # Display original image
        st.write(f"Image: {uploaded_file.name}, Tumor Probability: {max_prob:.3f}, Prediction: {label}")
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Image 1 - {label}")
        st.pyplot(fig)
        # Display second image with heatmap overlay
        superimposed = img.copy()  # Exact copy
        if label != "No Tumor":
            layer_name = 'conv5_block3_out'  # ResNet50 last conv layer
            try:
                heatmap = get_heatmap(model, img_array, layer_name)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed = cv2.addWeighted(np.uint8(img * 255), 0.7, heatmap, 0.3, 0)
            except Exception as e:
                st.error(f"Heatmap generation failed: {str(e)}")
        superimposed_uint8 = superimposed  # Already in uint8 from addWeighted
        st.image(superimposed_uint8, caption=f"Image 2 - {label} (Confidence: {max_prob:.2%})", use_column_width=True)
