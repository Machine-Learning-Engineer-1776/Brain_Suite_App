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

def grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0], axis=-1).numpy().item()  # Scalar index
        loss = predictions[0][class_idx]
    grads = tape.gradient(loss, conv_outputs)
    conv_outputs = conv_outputs[0]
    grads = grads[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)
    for index, w in enumerate(weights):
        cam += w * conv_outputs[:, :, index]
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-10)  # Avoid division by zero
    cam = cv2.resize(cam, (img_array.shape[2], img_array.shape[1]))
    # Enhance contrast
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)
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
        # Display prediction and image
        st.write(f"Image: {uploaded_file.name}, Tumor Probability: {max_prob:.3f}, Prediction: {label}")
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Prediction: {label}")
        st.pyplot(fig)
        # Grad-CAM and circle if not No Tumor
        if label != "No Tumor":
            layer_name = 'conv5_block3_out'  # ResNet50 last conv layer
            try:
                cam = grad_cam(model, img_array, layer_name)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                superimposed = cv2.addWeighted(np.uint8(img * 255), 0.6, heatmap, 0.4, 0)
                gray = np.uint8(cam * 255)
                _, thresh = cv2.threshold(gray, 0.01, 255, 0)  # Ultra-low threshold
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    (x, y), r = cv2.minEnclosingCircle(contours[0])
                    r = max(r, 20)  # Larger circle
                    cv2.circle(superimposed, (int(x), int(y)), int(r), (0, 255, 0), 2)
                    st.image(superimposed, caption=f"{label} Circled (Confidence: {max_prob:.2%})", use_column_width=True)
                else:
                    st.write("No clear tumor region detected. Try a higher contrast image.")
            except Exception as e:
                st.error(f"Grad-CAM failed: {str(e)}")
