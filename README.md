# Brain Suite: Brain MRI Tumor Classifier + Synthetic MRI Viewer

A Streamlit web app that runs a deep learning classifier on brain MRI images and shows synthetic MRI scans produced by a separate diffusion-model project. Built as a portfolio demo, **not a medical device**.

<div align="center">
<img width="261" height="173" alt="Brain Suite app" src="https://github.com/user-attachments/assets/37318e63-48f5-450b-8cd5-b73e2651c075" />
</div>

**Live app:** http://44.246.164.107/ (may be offline when the server isn't running)  
**Synthetic MRI model:** [Synthetic_Image_Generation_Diffusion_Model](https://github.com/Machine-Learning-Engineer-1776/Synthetic_Image_Generation_Diffusion_Model)

---

## What it does

- **Tumor classification:** upload a JPG/NPY brain MRI (or pick a test image) and the app runs a TensorFlow/Keras model on a 224×224 RGB version of it.
- **"Radiology report" and highlighted regions:** the app marks the 3 brightest areas of the image (Gaussian-blurred pixel intensity) and lists their coordinates. This is a **simple intensity heuristic**, not model-based tumor localization or segmentation.
- **Synthetic MRI viewer:** displays pre-generated synthetic brain MRI images from the diffusion-model project. Images are not generated live in the app.

<img width="1177" height="321" alt="Test image selection" src="https://github.com/user-attachments/assets/146515cc-84bd-40af-9fb5-75dac838de03" />

<div align="center">
<img width="235" height="241" alt="Highlighted regions" src="https://github.com/user-attachments/assets/c28293ef-c146-4f8d-a04f-ff7355b4a199" />
</div>

<img width="1105" height="1045" alt="App results view" src="https://github.com/user-attachments/assets/d60de2a4-839f-4647-a360-f7c68e4963d7" />

---

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) on Kaggle (`masoudnickparvar/brain-tumor-mri-dataset`), loaded in the notebook with `kagglehub`:
- 5,712 training images and 1,311 test images
- 4 classes: glioma, meningioma, pituitary, no tumor

Sample test images from this dataset are in `Test Images/`.

---

## Model and results

All numbers come from `Models/Brain_Tumor_Classifier.ipynb`:

| Model | Task | Result |
|-------|------|--------|
| Custom CNN (baseline) | 4-class | Best validation accuracy 52.6% (10 epochs); dropped in favour of transfer learning |
| **ResNet50 (ImageNet weights) + dense head** | **Binary: tumor vs. no tumor** | **83.9% test accuracy** on 1,311 test images |

Binary test results (ResNet50):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| No Tumor | 0.72 | 0.80 | 0.75 | 405 |
| Tumor | 0.90 | 0.86 | 0.88 | 906 |

### Known limitations / next steps
- The notebook's final model is **binary** (single sigmoid output), while `app.py` applies softmax and maps the output to 4 labels (Glioma / Meningioma / No Tumor / Pituitary). These need to be aligned: either train and evaluate a 4-class model, or show tumor / no tumor in the app.
- The highlighted regions are brightness-based, not learned. Grad-CAM or a segmentation model would be the next step.
- For research and demo use only; not a substitute for professional medical diagnosis.

---

## How to run locally

```bash
pip install -r requirements.txt
```

1. Run `Models/Brain_Tumor_Classifier.ipynb` (Colab) to train the ResNet50 model; it saves `model.h5`. The trained model file is **not** included in this repo.
2. `Streamlit/app.py` loads the model from `../Models/model.h5` and synthetic images from a `synthetic_images/` folder in the working directory (not included here; sample synthetic images are in the [diffusion repo](https://github.com/Machine-Learning-Engineer-1776/Synthetic_Image_Generation_Diffusion_Model)).
3. From the `Streamlit/` folder: `streamlit run app.py`

`Dockerfile.txt` contains the container setup used for deployment (rename to `Dockerfile` to build).

---

## Tech stack
TensorFlow / Keras (ResNet50 transfer learning), Streamlit, OpenCV, NumPy, Pillow, Matplotlib, kagglehub, Google Colab, Docker

---

## Project structure
```
Architecture/Design_Deployment_Solution_Architecture.ipynb   # deployment design notes
Models/Brain_Tumor_Classifier.ipynb   # data loading, CNN baseline, ResNet50 training + evaluation
Models/app.py                         # copy of the Streamlit app
Streamlit/app.py                      # Streamlit app
Streamlit/requirements.txt
Test Images/Images For Testing/       # sample MRIs: glioma, meningioma, pituitary, no tumor
Dockerfile.txt, Dockerignore.txt
requirements.txt
```
