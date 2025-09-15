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
        class_idx = tf.argmax(predictions[0], axis=-1).numpy().item()
        loss = predictions[0][class_idx]
    grads = tape.gradient(loss, conv_outputs)
    conv_outputs = conv_outputs[0]
    grads = grads[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1).numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    cam = cv2.resize(cam, (img_array.shape[2], img_array.shape[1]), interpolation=cv2.INTER_LINEAR)
    cam = np.clip(cam, 0, 1)
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
        # Grad-CAM and arrow if not No Tumor
        if label != "No Tumor":
            layer_name = 'conv5_block3_out'  # ResNet50 last conv layer
            try:
                cam = grad_cam(model, img_array, layer_name)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                superimposed = cv2.addWeighted(np.uint8(img * 255), 0.7, heatmap, 0.3, 0)
                # Find max activation point for arrow
                y, x = np.unravel_index(np.argmax(cam), cam.shape)
                arrow_start = (int(x), int(y))
                arrow_end = (int(x + 30), int(y - 30))  # Longer arrow, upward-right
                cv2.arrowedLine(superimposed, arrow_start, arrow_end, (0, 255, 0), 2, tipLength=0.3)
                st.image(superimposed, caption=f"{label} Indicated (Confidence: {max_prob:.2%})", use_column_width=True)
            except Exception as e:
                st.error(f"Grad-CAM failed: {str(e)}")
