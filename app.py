from flask import Flask, request, render_template
import os
import cv2
import numpy as np
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from mrcnn import visualize
import tensorflow.compat.v1 as tf
import logging
from tensorflow.keras import backend as K

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable TensorFlow 2.x behaviors for Mask R-CNN compatibility
tf.disable_v2_behavior()

app = Flask(__name__)
UPLOAD_FOLDER = 'Uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Prediction configuration
class PredictionConfig(Config):
    NAME = "object"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + cracks
    STEPS_PER_EPOCH = 10
    DETECTION_MIN_CONFIDENCE = 0.9

# Predefined weights paths using current working directory
WEIGHTS_PATHS = {
    'Model1': os.path.join(os.getcwd(), 'Uploads', 'trainedModels', 'Model1', 'model1.h5'),
    'Model2': os.path.join(os.getcwd(), 'Uploads', 'trainedModels', 'Model2', 'model2.h5')
}

def load_model(weights_path, model_name, graph, session):
    """Load Mask R-CNN model with given weights in the specified graph and session."""
    logger.info(f"Loading model {model_name} with weights: {weights_path}")
    with graph.as_default():
        with session.as_default():
            config = PredictionConfig()
            model = MaskRCNN(mode='inference', model_dir='./logs', config=config)
            try:
                model.load_weights(weights_path, by_name=True)
                logger.info(f"Model {model_name} weights loaded successfully")
                return model
            except Exception as e:
                logger.error(f"Failed to load weights for {model_name}: {str(e)}")
                raise

def process_image(image_path):
    """Read and preprocess image for Mask R-CNN."""
    logger.info(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    # Resize image to 512x512 for Mask R-CNN
    image = cv2.resize(image, (512, 512))
    logger.info(f"Image loaded: shape={image.shape}, dtype={image.dtype}")
    return image

def visualize_predictions(image, results):
    """Visualize predictions with bounding boxes and masks."""
    logger.info("Visualizing prediction results")
    fig, ax = plt.subplots(1, figsize=(10, 10))
    visualize.display_instances(
        image,
        results['rois'],
        results['masks'],
        results['class_ids'],
        ['BG', 'cracks'],  # Class names
        results['scores'],
        ax=ax,
        title="Predictions"
    )
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    logger.info("Visualization completed")
    return image_base64

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received prediction request")
    # Check for uploaded image and weights selection
    if 'image' not in request.files or 'weights' not in request.form:
        logger.error("Missing image or weights selection")
        return render_template('index.html', error='Please upload an image and select a model')

    image_file = request.files['image']
    weights_selection = request.form['weights']

    if image_file.filename == '':
        logger.error("No image selected")
        return render_template('index.html', error='No image selected')

    if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        logger.error("Invalid image format")
        return render_template('index.html', error='Image must be PNG, JPG, or JPEG')

    if weights_selection not in WEIGHTS_PATHS:
        logger.error(f"Invalid weights selection: {weights_selection}")
        return render_template('index.html', error='Invalid model selection')

    # Save uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)
    logger.info(f"Image saved: {image_path}")

    # Get weights path
    weights_path = WEIGHTS_PATHS[weights_selection]
    if not os.path.exists(weights_path):
        logger.error(f"Weights file not found: {weights_path}")
        try:
            os.remove(image_path)
        except:
            pass
        return render_template('index.html', error=f'Model weights not found: {weights_selection}')

    try:
        # Load and process image
        image = process_image(image_path)

        # Clear Keras session and create a new TensorFlow graph and session
        K.clear_session()
        graph = tf.Graph()
        session = tf.Session(graph=graph)

        # Initialize model based on selection
        model = None
        if weights_selection == 'Model1':
            logger.info("Processing with Model1")
            model = load_model(WEIGHTS_PATHS['Model1'], 'Model1', graph, session)
            with graph.as_default():
                with session.as_default():
                    results = model.detect([image], verbose=0)[0]
            logger.info("Prediction completed for Model1")

        elif weights_selection == 'Model2':
            logger.info("Processing with Model2")
            model = load_model(WEIGHTS_PATHS['Model2'], 'Model2', graph, session)
            with graph.as_default():
                with session.as_default():
                    results = model.detect([image], verbose=0)[0]
            logger.info("Prediction completed for Model2")

        # Close the session to free resources
        session.close()

        # Visualize results
        result_image = visualize_predictions(image, results)

        # Clean up uploaded image
        try:
            os.remove(image_path)
        except:
            logger.warning(f"Failed to remove image: {image_path}")

        return render_template('index.html', result_image=result_image, model_used=weights_selection)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Clean up uploaded image
        try:
            os.remove(image_path)
        except:
            logger.warning(f"Failed to remove image: {image_path}")
        return render_template('index.html', error=f'Error processing prediction: {str(e)}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8880, debug=False, threaded=False)