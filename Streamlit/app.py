import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Load U-Net model
model = torch.load('unet_weights.pth', map_location=torch.device('cpu'))
model.eval()

st.title("Brain Tumor Classifier - U-Net")
uploaded_files = st.file_uploader("Upload MRI Images", type=["npy", "jpg"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Load image
        if uploaded_file.name.endswith('.npy'):
            img = np.load(uploaded_file)
        else:
            img = np.array(Image.open(uploaded_file))
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            pred = model(img_tensor)
            prob = torch.sigmoid(pred).item()
            mask = (pred > 0.5).float().numpy()

        # Display
        st.write(f"Image: {uploaded_file.name}, Tumor Probability: {prob:.3f}")
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img, cmap='gray')
        ax1.set_title("MRI Image")
        ax2.imshow(mask, cmap='hot')
        ax2.set_title("Tumor Mask")
        st.pyplot(fig)
