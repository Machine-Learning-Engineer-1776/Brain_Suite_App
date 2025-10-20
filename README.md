**<H1>Brain Suite: AI-Powered Brain Tumor Classifier and Synthetic MRI Generator:</H1>**

The Brain Suite is a live web application hosted at http://44.246.164.107/, designed for medical research and diagnostics. It leverages deep learning to classify brain MRI images and generate synthetic MRI scans. Built with Streamlit and TensorFlow, the app provides a user-friendly interface for researchers and clinicians to analyze brain MRIs for tumors (Glioma, Meningioma, Pituitary, or No Tumor) and visualize high-fidelity synthetic brain images. Key features include a pre-trained classification model, preloaded test images, detailed radiology reports, tumor localization visualization, and synthetic image generation for research and training purposes.


**Features:**

**•	Synthetic Image Generator:** Produces high-fidelity synthetic brain MRI images for visualization, research, and training using advanced generative AI techniques. These images are not classified for tumors but serve as realistic simulations.
Suggested Screenshot: Display of a generated synthetic MRI image from the "Synthetic Image Generator" section (e.g., a centered 224x224 PNG from the app’s synthetic image output).

**•	Tumor Classification:** Analyzes user-uploaded or preloaded MRI images using a pre-trained deep learning model, classifying them into Glioma, Meningioma, Pituitary, or No Tumor with confidence scores.
Suggested Screenshot: Dropdown menu from the "Select a Test Image" section, showing options (Glioma, Meningioma, Pituitary, No Tumor).

**•	Preloaded Test Images:** Includes four preloaded MRI images (Glioma.jpg, Meningioma.jpg, Pituitary.jpg, Tumor FREEDOM.jpg) for immediate classification, enabling users without personal scans to test the app.


<img width="1177" height="321" alt="{B2CC4B40-4384-4B3E-9316-CC34D65B3EC6}" src="https://github.com/user-attachments/assets/146515cc-84bd-40af-9fb5-75dac838de03" />




**•	Radiology Report:** Generates a detailed report for tumor classifications, including tumor probability, top three suspected tumor regions with coordinates and likelihoods, and region type (Core or Periphery).
Suggested Screenshot: Radiology report table showing tumor probability and region details.

**•	Tumor Localization Visualization:** Highlights suspected tumor regions on classified images with colored circles (red for high probability, blue for lower), aiding visual interpretation of tumor locations.


<img width="235" height="241" alt="{1D0090DB-AB0A-4CC2-AA0B-E153A8F11AD2}" src="https://github.com/user-attachments/assets/c28293ef-c146-4f8d-a04f-ff7355b4a199" />



**How to Use the Brain Suite Web App:**

**1.	Access the App:**
Visit the live Brain Suite web app at http://44.246.164.107/.



**2.	Generate Synthetic Images:**
Navigate to the "Synthetic Image Generator" section and click "Generate Synthetic Image" to view a pre-generated synthetic brain MRI. The app cycles between two sample images stored on the server.
**Note:** Synthetic images are for visualization only and cannot be classified.


**3.	Classify an Image:**

•	Select a Test Image: Choose a preloaded MRI from the dropdown menu (Glioma, Meningioma, Pituitary, No Tumor) in the "Select a Test Image" section. 

•	Upload Your Own MRI: Use the "Upload Your Own MRI" section to upload a JPG or NPY image.
Click "Classify Image" to process the selected or uploaded image.
Suggested Screenshot: Upload interface with a file selected or dropdown selection.

**4.	View Results:**

•	If no tumor is detected, a "No Tumor Has Been Detected" message is displayed. 

•	If a tumor is detected, the app shows: 

  •	The classified image with prediction and confidence score. 
  
  •	A radiology report detailing tumor probability and suspected regions. 
  
  •	A visualized image with highlighted tumor regions (red/blue circles).
  


<img width="1105" height="1045" alt="{BC5F30C2-8705-4BF9-B45D-46F4ED264C50}" src="https://github.com/user-attachments/assets/d60de2a4-839f-4647-a360-f7c68e4963d7" />



**Dataset**

The app leverages the BraTS2017 dataset (https://www.med.upenn.edu/cbica/brats/) for training the classification model, containing T1-weighted MRI scans with tumor annotations (Glioma, Meningioma, Pituitary). Images are preprocessed to 224x224 pixels, RGB, and normalized to [0, 1]. The app includes four preloaded test images (Glioma.jpg, Meningioma.jpg, Pituitary.jpg, Tumor FREEDOM.jpg) for classification. Synthetic images are pre-generated and stored on the server for visualization purposes.

**Model Performance**

The classification model, trained on BraTS2017, achieves reliable detection and classification of brain tumors. It uses a pre-trained TensorFlow architecture to provide accurate predictions with detailed radiology reports. [Note: Specific performance metrics, e.g., accuracy, to be added if available.]

**Limitations**

  •	The classification model is optimized for brain MRI images in JPG or NPY format. Non-MRI or unsupported formats may yield unreliable results. 
  
  •	Synthetic images are for visualization only and not suitable for classification. 
  
  •	Confidence scores and radiology reports reflect AI predictions and are not a substitute for professional medical diagnosis. Further clinical evaluation (e.g., biopsy) is recommended for tumor confirmation.

**Repository Structure**

  •	/data/: Dataset samples and preprocessed images. 
  
  •	/notebooks/: Jupyter notebooks for data wrangling and model training. 
  
  •	/test-images/: Preloaded test images for classification. 
  
  •	/synthetic_images/: Pre-generated synthetic MRI images. 
  
  •	app.py: Main Streamlit application script. 
  
  •	README.md: Project overview and usage instructions.



