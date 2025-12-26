"""
Flask web application for hand-written digit recognition.
"""
from flask import Flask, render_template, request, jsonify, send_file
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64
import os
import visualkeras

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'mnist_model.h5'
model = None

def load_model():
    """Load the trained model."""
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model = keras.models.load_model(MODEL_PATH)
        # Build the model by making a dummy prediction
        dummy_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
        model.predict(dummy_input, verbose=0)
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model file '{MODEL_PATH}' not found!")
        print("Please run 'python train_model.py' first to train the model.")

# Load model on startup
load_model()

def preprocess_image(image_data):
    """
    Preprocess the image from canvas to match model input requirements.

    Args:
        image_data: Base64 encoded image data from canvas

    Returns:
        Preprocessed numpy array ready for prediction
    """
    # Remove data URL prefix if present
    if 'base64,' in image_data:
        image_data = image_data.split('base64,')[1]

    # Decode base64 image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to grayscale
    image = image.convert('L')

    # Resize to 28x28 (MNIST size)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize
    image_array = np.array(image).astype('float32') / 255.0

    # Invert colors (canvas has white background, MNIST has black)
    image_array = 1.0 - image_array

    # Reshape to match model input (1, 28, 28, 1)
    image_array = np.expand_dims(image_array, axis=(0, -1))

    return image_array

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the digit from the canvas image.

    Expects JSON with 'image' field containing base64 encoded image data.
    Returns JSON with prediction and confidence scores.
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500

    try:
        # Get image data from request
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # Preprocess the image
        processed_image = preprocess_image(image_data)

        # Make prediction
        predictions = model.predict(processed_image, verbose=0)[0]

        # Get activations by manually calling each layer
        activations = []
        layer_names = []

        # Pass input through each layer to get activations
        x = processed_image
        for layer in model.layers:
            x = layer(x, training=False)
            # Only save activations from dense and flatten layers
            if 'dense' in layer.name or 'flatten' in layer.name:
                activations.append(x.numpy())
                layer_names.append(layer.name)

        # Get predicted digit and confidence
        predicted_digit = int(np.argmax(predictions))
        confidence = float(predictions[predicted_digit])

        # Get all probabilities
        all_probabilities = {
            str(i): float(predictions[i]) for i in range(10)
        }

        # Process layer activations for visualization
        network_activations = []
        if activations and layer_names:
            for act, name in zip(activations, layer_names):
                act_array = act[0]  # Remove batch dimension
                if len(act_array.shape) > 1:  # Flatten if needed
                    act_array = act_array.flatten()

                # Limit to reasonable number of nodes for visualization
                if len(act_array) > 64:
                    # Sample evenly if too many nodes
                    indices = np.linspace(0, len(act_array)-1, 64, dtype=int)
                    act_array = act_array[indices]

                network_activations.append({
                    'layer': name,
                    'activations': [float(x) for x in act_array]
                })

        return jsonify({
            'digit': predicted_digit,
            'confidence': confidence,
            'probabilities': all_probabilities,
            'network_activations': network_activations
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model-architecture')
def model_architecture():
    """Generate and return model architecture visualization."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Try visualkeras first for prettier output
        try:
            img = visualkeras.layered_view(
                model,
                legend=True,
                spacing=30,
                draw_volume=False,
                to_file=None
            )

            # Convert PIL image to bytes
            img_io = io.BytesIO()
            img.save(img_io, 'PNG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/png')

        except Exception as vk_error:
            print(f"Visualkeras failed: {vk_error}, falling back to keras plot_model")

            # Fallback to Keras plot_model
            img_path = '/tmp/model_architecture.png'
            tf.keras.utils.plot_model(
                model,
                to_file=img_path,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=True,
                dpi=96
            )

            return send_file(img_path, mimetype='image/png')

    except Exception as e:
        print(f"Error generating model architecture: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
