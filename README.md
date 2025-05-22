# Mask R-CNN Defect Detection

A Flask web application for detecting defects (e.g., cracks) in images using Mask R-CNN models. Users can upload images and select between two pre-trained models ("Model1" and "Model2") to generate predictions with bounding boxes and masks.

## Features
- Upload images (PNG, JPG, JPEG) for defect detection.
- Select between two Mask R-CNN models ("Model1" or "Model2").
- Visualize predictions with bounding boxes and masks.
- Robust TensorFlow graph and session management to prevent errors.

## Directory Structure

FinalDeploymentFlask/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
├── templates/
│   └── index.html          # HTML template for web interface
├── Uploads/
│   ├── trainedModels/
│   │   ├── Model1/
│   │   │   └── model1.h5   # Model1 weights (not in repo)
│   │   ├── Model2/
│   │   │   └── model2.h5   # Model2 weights (not in repo)
│   └── (temporary uploads) # Temporary image storage



## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/MaskRCNN-Defect-Detection.git
   cd MaskRCNN-Defect-Detection

2. Create a Virtual Environment:
  conda create -n test_env python=3.6.13
  
  
3.Install Dependencies:

> pip install -r requirements.txt

4. Place Model Weights:

* Obtain model1.h5 and model2.h5 (pre-trained Mask R-CNN weights).
* Place them in:
    - Uploads/trainedModels/Model1/model1.h5
    - Uploads/trainedModels/Model2/model2.h5
Ensure the directory structure matches the above.


5. Run the Application
python app.py

* Open a browser and navigate to http://localhost:8880 or http://<your-ip>:8880.


## USAGE

1. Access the web interface at http://localhost:8880.
2. Upload an image (PNG, JPG, or JPEG).
3. Select a model ("Defect Detection Model 1" or "Defect Detection Model 2").
4. Click "Analyze" to view the prediction results with defect visualizations.
5. The UI displays the selected model and any errors (e.g., missing weights).

# Notes

* Model Weights: model1.h5 and model2.h5 are not included due to size. You must provide these files, trained for crack detection with Mask R-CNN.
* Image Size: Images are resized to 512x512 pixels. Adjust in app.py (process_image) if your models require a different size.
* Performance: Loading Mask R-CNN models is slow. For production, consider caching models.
* Production: Use a WSGI server (e.g., Gunicorn) instead of Flask’s development server.
