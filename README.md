**Brain Tumor Classifier Web App**

This project develops a deep learning model to classify brain MRI images as Tumor or No Tumor using a ResNet50-based convolutional neural network. The model is deployed as a Flask web application, allowing users to upload MRI images via a drag-and-drop interface for real-time predictions. Using the Brain Tumor MRI Dataset from Kaggle, the project demonstrates skills in machine learning, computer vision, and web deployment.

**Key Achievements:**
Improved model accuracy to 84 percent from 69 percent and reduced loss to 0.41 from 387.35 by fixing a critical bias where all predictions were Tumor. Achieved balanced predictions with a confusion matrix of 322 true No Tumor, 83 false Tumor, 128 false No Tumor, 778 true Tumor, ensuring reliable medical classification. Developed a user-friendly Flask web app for seamless MRI classification with confidence scores.

**Technical Details:**
Dataset: The project uses the Brain Tumor MRI Dataset from Kaggle, available at https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset. The training set includes 5712 images: 1595 No Tumor, 4117 Tumor (1321 glioma, 1339 meningioma, 1457 pituitary). The test set includes 1311 images: 405 No Tumor, 906 Tumor (300 glioma, 306 meningioma, 300 pituitary). Images are resized to 224 by 224 pixels, converted to RGB, rescaled by 1/255, and mapped to binary labels: No Tumor as 0, Tumor as 1.


**Model:** 
The model uses ResNet50 pre-trained on ImageNet, with the last 10 layers fine-tuned, followed by GlobalAveragePooling2D, Dense 1024 with relu activation, and Dense 1 with sigmoid activation for binary classification. Training uses binary cross-entropy loss, Adam optimizer with 1e-4 learning rate, class weights to balance 28 percent No Tumor versus 72 percent Tumor, and data augmentation (rotation, flipping). The model achieves 84 percent accuracy, 0.41 loss, with 80 percent recall for No Tumor and 86 percent for Tumor.


**Web App:** 
The Flask web app provides a drag-and-drop interface for uploading MRI images and receiving Tumor or No Tumor predictions with confidence scores. Access the app at [Insert Web App Link Here]. The app preprocesses images by resizing to 224 by 224, converting to RGB, and rescaling by 1/255, then uses the model to predict a result based on a 0.5 threshold.


