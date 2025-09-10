**<h1>Brain Tumor Classifier Web App:**
  **Using the Web App</h1>**



This guide explains how to use the Brain Tumor Classifier web app, a deep learning tool that classifies brain MRI images as Tumor or No Tumor. Built with a ResNet50-based model and deployed using Flask, the app allows users to upload MRI images via a drag-and-drop interface and receive real-time predictions with confidence scores. The app leverages the Brain Tumor MRI Dataset from Kaggle for accurate medical classification.



**How to Use the Web App:**

+ **Access the App:**
Visit the link below to open the Brain Tumor Classifier web app.

<div align="center">
  
[Brain Tumor Classification Web App](https://40ba98880c7b.ngrok-free.app/) 

</div>

+ **Upload an Image:** Use the drag-and-drop interface to upload a brain MRI image from the Test Images folder of the Brain Tumor MRI Dataset.

  +  Test Images for you to use can be found here.

<div align="center">
    
  [Test Images](https://github.com/Machine-Learning-Engineer-1776/Brain_Tumor_Classifier_For_Web_App/tree/main/Test%20Images/Images%20For%20Testing)

</div>

+ **Upload Even More Images :-)**
  +  Thousands of more images available at the link below
  +  Found in the Testing folder.

<div align="center">
      
  [Kaggle Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset.)

</div>



+ **View Results:**
The app displays Tumor Detected or No Tumor with a confidence score (e.g., Confidence: 88.61%). Results are based on a ResNet50 model with 84 percent accuracy, 80 percent recall for No Tumor, and 86 percent recall for Tumor.

<div align="center">

![results](https://github.com/user-attachments/assets/2b473de0-87ac-43f2-99fb-b0c1004a17f6)


</div>

+ **Test with Other Images:** You can upload any image (JPG or PNG), but the model is trained on brain MRIs. Non-MRI images may produce unpredictable results, though the model should output varied confidence scores.

+ **What You Need to Know**

+ **Dataset:**

The Testing folder contains 1311 images: 405 No Tumor, 906 Tumor (300 glioma, 306 meningioma, 300 pituitary). Images are preprocessed to 224x224 pixels, RGB, rescaled by 1/255.

+ **Model Performance:**

The model achieves 84 percent accuracy and 0.41 loss, with a confusion matrix of 322 true No Tumor, 83 false Tumor, 128 false No Tumor, 778 true Tumor. Expect 80 percent of No Tumor images and 86 percent of Tumor images to be correctly classified.


+ **Limitations:**

The model is optimized for brain MRI images. Non-MRI images (e.g., photos of objects) may yield less reliable results. Confidence scores reflect the model’s prediction strength, not medical certainty.
**
