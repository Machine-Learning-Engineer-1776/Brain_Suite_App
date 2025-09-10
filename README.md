**<H1>Brain Tumor Classifier Web App:</H1>**

<div align="center">
  
![DaBrain](https://github.com/user-attachments/assets/52c5e786-9fcd-4da7-a0bf-121ae4dcab15)

</div>


This project builds a deep learning model to classify brain MRI images as Tumor or No Tumor, deployed as a Flask web app for real-time predictions via a drag-and-drop interface. Using a ResNet50-based model and the Brain Tumor MRI Dataset from Kaggle, it showcases expertise in machine learning, computer vision, and web deployment.

**Key Achievements:**

+ Boosted model accuracy to 84% (from 69%) and cut loss to 0.41 (from 387.35) by fixing a bias where all predictions were Tumor.
+ Achieved balanced predictions with a confusion matrix of 322 true No Tumor, 83 false Tumor, 128 false No Tumor, 778 true Tumor.
+ Built a user-friendly Flask web app for seamless MRI classification with confidence scores.

**Technical Details Dataset:**

+ Source: Brain Tumor MRI Dataset, https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
+ Training Set: 5712 images (1595 No Tumor, 4117 Tumor: 1321 glioma, 1339 meningioma, 1457 pituitary)
+ Test Set: 1311 images (405 No Tumor, 906 Tumor: 300 glioma, 306 meningioma, 300 pituitary)
+ Preprocessing: Images resized to 224x224, RGB, rescaled by 1/255; binary labels (No Tumor: 0, Tumor: 1)

**Model Architecture:**
+ ResNet50 (ImageNet weights, last 10 layers fine-tuned), GlobalAveragePooling2D, Dense(1024, relu), Dense(1, sigmoid)
+ Training: Binary cross-entropy loss, Adam optimizer (1e-4 learning rate), class weights for 28% No Tumor vs. 72% Tumor, augmentation (rotation, flipping)
+ Performance: 84% accuracy, 0.41 loss, 80% recall for No Tumor, 86% for Tumor

**Web App:**

+ Platform: Flask with drag-and-drop interface
+ Functionality: Upload MRI images, get Tumor or No Tumor predictions with confidence scores
+ Access:
<div align="center">
  
  [Brain Tumor Classification Web App](https://40ba98880c7b.ngrok-free.app/)

</div>
  + Don't Forget to Read the Instructions!

    [Read Me](https://github.com/Machine-Learning-Engineer-1776/Brain_Tumor_Classifier_For_Web_App/blob/main/README%20Using%20The%20WebApp.md/)
  

+ Preprocessing: Images resized to 224x224, RGB, rescaled by 1/255; prediction threshold at 0.5
