import os

# Suppress TensorFlow warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Import TensorFlow after setting environment variables
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Define paths
model_path = r"C:\Users\fazil\Desktop\rttd\optimized_railway_crack_detector_cnn.h5"
image_path = r"C:\Users\fazil\Desktop\rttd\Test\Defective\IMG_20201114_102203.jpg"  
#image_path = r"C:\Users\fazil\Desktop\rttd\Test\Non defective\IMG_20201114_102253.jpg"  


# Decision threshold (adjust as needed)
threshold = 0.4  

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Model file not found at: {model_path}")
    exit()

# Load the trained model
try:
    model = load_model(model_path)

    # Recompile the model to suppress the "compiled metrics" warning
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Model loaded and recompiled successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Function to preprocess an image
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image file not found at: {image_path}")
        return None

    # Read and process the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    plt.imshow(img)
    plt.title("Input Image")
    plt.show()

    # Resize to match the model's expected input (128, 128)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize like in training
    img = np.reshape(img, [1, 128, 128, 3])  # Add batch dimension
    return img

# Function to make predictions
def predict_image(model, img, threshold):
    prediction = model.predict(img)
    confidence = prediction[0][0]

    print(f"Raw Prediction: {confidence:.4f}")
    if confidence < threshold:
        print(f"✅ Non-defective track detected (Confidence: {1 - confidence:.2f})")
    else:
        print(f"🚨 Defective track detected (Confidence: {confidence:.2f})")

# Main Script
if __name__ == "__main__":
    # Preprocess the image
    img = preprocess_image(image_path)
    if img is None:
        exit()

    # Make prediction
    predict_image(model, img, threshold)