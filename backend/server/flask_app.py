# from flask import Flask, request, jsonify
# import numpy as np
# from flask_cors import CORS
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import os
# import sys
# import traceback
# import logging
# import io
# from PIL import Image

# # Add ai_agents path to sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ai_agents')))

# from agent1_identifier import Agent1Identifier
# from agent2_advisor import Agent2Advisor

# app = Flask(__name__)
# CORS(app)

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)  # This ensures logs are printed at DEBUG level
# logger = logging.getLogger(__name__)

# # --- Load model once at startup ---
# MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_training', 'mobilenet_planville_model.h5'))
# try:
#     model = load_model(MODEL_PATH)
#     logger.info("Model loaded successfully.")
# except Exception as e:
#     logger.error(f"Error loading the model: {str(e)}")
#     sys.exit(1)  # Stop the server if the model fails to load

# # --- Class index mapping ---
# class_indices = {
#     'Pepper__bell___Bacterial_spot': 0,
#     'Pepper__bell___healthy': 1,
#     'Potato___Early_blight': 2,
#     'Potato___Late_blight': 3,
#     'Potato___healthy': 4,
#     'Tomato_Bacterial_spot': 5,
#     'Tomato_Early_blight': 6,
#     'Tomato_Late_blight': 7,
#     'Tomato_Leaf_Mold': 8,
#     'Tomato_Septoria_leaf_spot': 9,
#     'Tomato_Spider_mites_Two_spotted_spider_mite': 10
# }
# inv_class_indices = {v: k for k, v in class_indices.items()}

# # --- Initialize Agents ---
# identifier = Agent1Identifier(class_indices)
# knowledge_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ai_agents', 'knowledge_base.json'))
# advisor = Agent2Advisor(knowledge_base_path)

# # --- Allowed file types ---
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # --- Predict Endpoint ---
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         logger.error('No file part in the request')
#         return jsonify({'error': 'No file part'}), 400

#     file = request.files['file']

#     if file.filename == '':
#         logger.error('No selected file')
#         return jsonify({'error': 'No selected file'}), 400

#     if not allowed_file(file.filename):
#         logger.error('Unsupported file type')
#         return jsonify({'error': 'Unsupported file type'}), 400

#     try:
#         # Convert the file stream into a BytesIO object
#         img_bytes = file.read()
#         img_stream = io.BytesIO(img_bytes)

#         # Load image and resize to the model input size (224x224)
#         img = Image.open(img_stream)
#         img = img.resize((224, 224))

#         # Convert image to numpy array and normalize it
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

#         # Predict using the model
#         predictions = model.predict(img_array)
#         predicted_index = np.argmax(predictions[0])
#         predicted_class = inv_class_indices.get(predicted_index, "Unknown")

#         # Use agents for disease identification and advice
#         disease_name = identifier.identify(predicted_index)
#         advice = advisor.get_advice(disease_name)

#         logger.info(f"Prediction: {disease_name}, Advice: {advice}")

#         return jsonify({
#             'predicted_class': disease_name,
#             'advice': advice
#         })

#     except Exception as e:
#         logger.error(f"Exception occurred during prediction: {str(e)}")
#         logger.error("Full traceback:")
#         logger.error(traceback.format_exc())  # This prints the full traceback of the error
#         return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500

# # --- Run App ---
# if __name__ == '__main__':
#     try:
#         app.run(debug=True, host='0.0.0.0', port=5000)
#     except Exception as e:
#         logger.error(f"Error starting the Flask app: {str(e)}")











from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
import os
import sys
import traceback
import logging
import io
from PIL import Image

# Configure paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ai_agents')))

from agent1_identifier import Agent1Identifier
from agent2_advisor import Agent2Advisor

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model_training', 'mobilenet_planville_model.h5')
KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'ai_agents', 'knowledge_base.json')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
INPUT_SHAPE = (224, 224)

# Load resources at startup
try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    sys.exit(1)

class_indices = {
    'Pepper__bell___Bacterial_spot': 0,
    'Pepper__bell___healthy': 1,
    'Potato___Early_blight': 2,
    'Potato___Late_blight': 3,
    'Potato___healthy': 4,
    'Tomato_Bacterial_spot': 5,
    'Tomato_Early_blight': 6,
    'Tomato_Late_blight': 7,
    'Tomato_Leaf_Mold': 8,
    'Tomato_Septoria_leaf_spot': 9,
    'Tomato_Spider_mites_Two_spotted_spider_mite': 10
}

# Initialize AI agents
identifier = Agent1Identifier(class_indices)
advisor = Agent2Advisor(KNOWLEDGE_BASE_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    
    if not file or file.filename == '':
        return jsonify({'error': 'Empty file submission'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Read and validate image
        img_stream = io.BytesIO(file.read())
        img = Image.open(img_stream).convert('RGB')
        
        # Preprocess image
        img = img.resize(INPUT_SHAPE)
        img_array = image.img_to_array(img)
        img_array = preprocess_input(np.expand_dims(img_array, axis=0))  # MobileNet preprocessing
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        
        # Get results from agents
        disease_name = identifier.identify(predicted_index)
        advice = advisor.get_advice(disease_name)
        
        # Debug logging
        logger.debug(f"Raw predictions: {predictions[0]}")
        logger.info(f"Prediction: {disease_name}")
        
        return jsonify({
            'predicted_class': disease_name,
            'confidence': float(predictions[0][predicted_index]),
            'advice': advice
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
