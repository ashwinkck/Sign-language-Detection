import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    print(f"Attempting to load image at: {image_path}")  # Debugging info
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist at: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image at: {image_path}. Check if the file is a valid image.")
    
    img = cv2.resize(img, (128, 128))  # Resize to the same dimensions as during training
    img = img / 255.0  # Normalize pixel values
    img = img.reshape((1, 128, 128, 3))  # Reshape for the model input
    return img

# Function to make predictions
def make_prediction(model, image_path):
    processed_image = load_and_preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class] * 100
    
    return predicted_class, confidence

# Load the trained model
model_path = r'C:\Users\ashwi\OneDrive\Desktop\Sycamore\SLD mark1.5\sign_language_model.h5'  # Change this to your model's path
model = load_model(model_path)

# Directory containing test images
test_images_dir = r'C:\Users\ashwi\OneDrive\Desktop\Sycamore\SLD mark1.5\Test'  # Your specified test directory
test_images = os.listdir(test_images_dir)  # Get all files in the directory

# Loop through all images in the test directory
for image_file in test_images:
    test_image_path = os.path.join(test_images_dir, image_file)  # Full path to the image
    try:
        predicted_class, confidence = make_prediction(model, test_image_path)
        print(f"Image: {image_file} - Predicted class: {predicted_class}, Confidence: {confidence:.2f}%")
    except Exception as e:
        print(f"Error processing image {image_file}: {e}")
