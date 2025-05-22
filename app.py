from flask import Flask, request, render_template, session, send_from_directory
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
app.secret_key = 'super_secret_key'  # Required for session management
UPLOAD_FOLDER = 'ImagesUploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
LAST_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, 'last_image.jpg')

# Prediction configuration
class PredictionConfig(Config):
    NAME = "object"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + cracks
    STEPS_PER_EPOCH = 10
    DETECTION_MIN_CONFIDENCE = 0.9

# Predefined weights paths
WEIGHTS_PATHS = {
    'Model1': os.path.join(os.getcwd(), 'Uploads', 'trainedModels', 'Model1', 'model1.h5'),
    'Model2': os.path.join(os.getcwd(), 'Uploads', 'trainedModels', 'Model2', 'model2.h5')
}

# Cache for models, graphs, and sessions
MODEL_CACHE = {
    'Model1': {'model': None, 'graph': None, 'tf_session': None},
    'Model2': {'model': None, 'graph': None, 'tf_session': None}
}

def load_model(weights_path, model_name):
    """Load Mask R-CNN model with given weights in a new graph and session."""
    logger.info(f"Loading model {model_name} with weights: {weights_path}")
    graph = tf.Graph()
    tf_session = tf.Session(graph=graph)
    with graph.as_default():
        with tf_session.as_default():
            config = PredictionConfig()
            model = MaskRCNN(mode='inference', model_dir='./logs', config=config)
            try:
                model.load_weights(weights_path, by_name=True)
                logger.info(f"Model {model_name} weights loaded successfully")
                return model, graph, tf_session
            except Exception as e:
                logger.error(f"Failed to load weights for {model_name}: {str(e)}")
                tf_session.close()
                raise

def get_or_load_model(model_name):
    """Get cached model or load a new one."""
    if MODEL_CACHE[model_name]['model'] is None:
        model, graph, tf_session = load_model(WEIGHTS_PATHS[model_name], model_name)
        MODEL_CACHE[model_name]['model'] = model
        MODEL_CACHE[model_name]['graph'] = graph
        MODEL_CACHE[model_name]['tf_session'] = tf_session
    return MODEL_CACHE[model_name]['model'], MODEL_CACHE[model_name]['graph'], MODEL_CACHE[model_name]['tf_session']

def process_image(image_path):
    """Read and preprocess image for Mask R-CNN."""
    logger.info(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (1024, 1024))
    logger.info(f"Image loaded: shape={image.shape}, dtype={image.dtype}")
    return image

def encode_image(image_path):
    """Convert image to base64 string, resized to 512x512."""
    logger.info(f"Encoding image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to encode image: {image_path}")
        raise ValueError(f"Failed to encode image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (1024, 1024))  # Resize to match prediction input
    _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    logger.info("Image encoded successfully")
    return image_base64

def visualize_predictions(image, results):
    """Visualize predictions with bounding boxes and masks."""
    logger.info("Visualizing prediction results")
    fig, ax = plt.subplots(1, figsize=(5, 5))  # Smaller figsize for consistent output
    visualize.display_instances(
        image,
        results['rois'],
        results['masks'],
        results['class_ids'],
        ['BG', 'cracks'],
        results['scores'],
        ax=ax,
        title="Predictions"
    )
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)  # DPI for consistent size
    plt.close(fig)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    logger.info("Visualization completed")
    return image_base64

@app.route('/image/<filename>')
def serve_image(filename):
    """Serve images from ImagesUploaded folder."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/', methods=['GET'])
def index():
    last_image_exists = os.path.exists(LAST_IMAGE_PATH)
    last_image_name = session.get('last_image_name', 'None')
    return render_template('index.html', last_image_exists=last_image_exists, last_image_name=last_image_name)

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received prediction request")
    use_last_image = request.form.get('use_last_image') == 'true'
    weights_selection = request.form.get('weights')

    # Validate model selection
    if not weights_selection or weights_selection not in WEIGHTS_PATHS:
        logger.error(f"Invalid weights selection: {weights_selection}")
        return render_template('index.html', error='Invalid model selection', 
                             last_image_exists=os.path.exists(LAST_IMAGE_PATH), 
                             last_image_name=session.get('last_image_name', 'None'))

    # Handle image input
    image_path = LAST_IMAGE_PATH if use_last_image else None
    if not use_last_image:
        if 'image' not in request.files:
            logger.error("Missing image")
            return render_template('index.html', error='Please upload an image', 
                                 last_image_exists=os.path.exists(LAST_IMAGE_PATH), 
                                 last_image_name=session.get('last_image_name', 'None'))
        image_file = request.files['image']
        if image_file.filename == '':
            logger.error("No image selected")
            return render_template('index.html', error='No image selected', 
                                 last_image_exists=os.path.exists(LAST_IMAGE_PATH), 
                                 last_image_name=session.get('last_image_name', 'None'))
        if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            logger.error("Invalid image format")
            return render_template('index.html', error='Image must be PNG, JPG, or JPEG', 
                                 last_image_exists=os.path.exists(LAST_IMAGE_PATH), 
                                 last_image_name=session.get('last_image_name', 'None'))
        # Save new image as last_image.jpg and store filename
        image_path = LAST_IMAGE_PATH
        image_file.save(image_path)
        session['last_image_name'] = image_file.filename
        logger.info(f"Image saved: {image_path}, original name: {image_file.filename}")

    # Check if image exists
    if not image_path or not os.path.exists(image_path):
        logger.error("No valid image available")
        return render_template('index.html', error='No valid image available', 
                             last_image_exists=False, last_image_name='None')

    # Verify weights file exists
    weights_path = WEIGHTS_PATHS[weights_selection]
    if not os.path.exists(weights_path):
        logger.error(f"Weights file not found: {weights_path}")
        return render_template('index.html', error=f'Model weights not found: {weights_selection}', 
                             last_image_exists=os.path.exists(LAST_IMAGE_PATH), 
                             last_image_name=session.get('last_image_name', 'None'))

    try:
        # Load and process image
        image = process_image(image_path)

        # Get or load model
        model, graph, tf_session = get_or_load_model(weights_selection)

        # Make predictions
        logger.info(f"Running model.detect for {weights_selection}...")
        with graph.as_default():
            with tf_session.as_default():
                results = model.detect([image], verbose=0)[0]
        logger.info(f"Prediction completed for {weights_selection}")

        # Visualize results
        result_image = visualize_predictions(image, results)

        # Encode original image for side-by-side display
        original_image = encode_image(image_path)

        return render_template('index.html', result_image=result_image, original_image=original_image, 
                             model_used=weights_selection, last_image_exists=True, 
                             last_image_name=session.get('last_image_name', 'None'))

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html', error=f'Error processing prediction: {str(e)}', 
                             last_image_exists=os.path.exists(LAST_IMAGE_PATH), 
                             last_image_name=session.get('last_image_name', 'None'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8880, debug=False, threaded=False)