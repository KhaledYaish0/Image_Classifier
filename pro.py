import argparse
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def process_image(image_path):
    """
    Process the image into a format suitable for prediction by the model.
    """
    img = load_img(image_path, target_size=(224, 224))  # Resize to the model's input size
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

def predict(image_path, model, top_k=1):
    """
    Predict the top K classes for the given image using the model.
    """
    processed_image = process_image(image_path)
    predictions = model.predict(processed_image)[0]  # Get predictions for the image
    top_indices = np.argsort(predictions)[-top_k:][::-1]  # Top K indices, sorted by probability
    top_probs = predictions[top_indices]
    return top_probs, top_indices

# Set up argument parser
parser = argparse.ArgumentParser(
    description='Predict the flower name from an image along with the probability of that name.'
)
parser.add_argument('image_path', help='Path to the image file')
parser.add_argument('saved_model', help='Path to the saved model')
parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
parser.add_argument('--category_names', default='label_map.json', help='Path to category label mapping file')

# Parse arguments
args = parser.parse_args()

# Load the model
model = load_model(args.saved_model)

# Make predictions
probs, predicted_classes = predict(args.image_path, model, args.top_k)

# Load category names
with open(args.category_names, 'r') as f:
    label_map = json.load(f)

# Map predicted classes to names
class_names = [label_map[str(cls + 1)] for cls in predicted_classes]  # Adjust index if class indices start from 1

# Display results
print("Probabilities:", probs)
print("Class Names:", class_names)
