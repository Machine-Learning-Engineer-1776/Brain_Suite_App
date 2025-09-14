import streamlit as st
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

st.title("Brain Tumor Classifier - CNN")
# Load model (adjust path and architecture as per Brain_Tumor_Classifier.ipynb)
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Binary: tumor/no tumor
try:
    model.load_state_dict(torch.load('../Models/model.pth', map_location=torch.device('cpu')))
except FileNotFoundError:
    st.error("Model weights not found. Ensure Brain_Tumor_Classifier.ipynb saves weights as Models/model.pth")
    st.stop()
model.eval()

uploaded_files = st.file_uploader("Upload MRI Images", type=["jpg", "npy"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Load image
        if uploaded_file.name.endswith('.npy'):
            img = np.load(uploaded_file)
        else:
            img = np.array(Image.open(uploaded_file).resize((224, 224)))  # Adjust size for your CNN
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        # Predict
        with torch.no_grad():
            pred = model(img_tensor)
            prob = torch.softmax(pred, dim=1)[0][1].item()  # Tumor probability
            label = "Tumor" if prob > 0.5 else "No Tumor"
        # Display
        st.write(f"Image: {uploaded_file.name}, Tumor Probability: {prob:.3f}, Prediction: {label}")
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Prediction: {label}")
        st.pyplot(fig)
