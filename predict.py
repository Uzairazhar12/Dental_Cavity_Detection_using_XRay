"""
Dental Cavity Detection - Prediction Script
This script loads the trained model and makes predictions on new X-ray images.
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configuration
IMG_SIZE = 150
MODEL_PATH = 'model/dental_cavity_model.h5'

# Class labels
CLASSES = {0: 'Normal', 1: 'Cavity'}

def load_trained_model():
    """Load the trained model from disk."""
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at: {MODEL_PATH}")
        print("[INFO] Please train the model first by running: python train.py")
        return None
    
    print("[INFO] Loading model...")
    model = load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully!")
    return model

def preprocess_image(image_path):
    """Preprocess image for prediction."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_single_image(model, image_path):
    """Make prediction on a single image."""
    
    # Preprocess image
    img = preprocess_image(image_path)
    if img is None:
        return None, None
    
    # Make prediction
    prediction = model.predict(img, verbose=0)[0][0]
    
    # Get class and confidence
    if prediction > 0.5:
        predicted_class = 'Cavity Detected'
        confidence = prediction * 100
    else:
        predicted_class = 'Normal (No Cavity)'
        confidence = (1 - prediction) * 100
    
    return predicted_class, confidence

def predict_from_folder(model, folder_path):
    """Make predictions on all images in a folder."""
    
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return
    
    # Get all image files
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(folder_path) 
                   if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not image_files:
        print(f"[INFO] No image files found in: {folder_path}")
        return
    
    print(f"\n[INFO] Found {len(image_files)} images")
    print("=" * 60)
    
    results = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        predicted_class, confidence = predict_single_image(model, image_path)
        
        if predicted_class:
            results.append({
                'file': image_file,
                'prediction': predicted_class,
                'confidence': confidence
            })
            print(f"Image: {image_file}")
            print(f"  Result: {predicted_class}")
            print(f"  Confidence: {confidence:.2f}%")
            print("-" * 60)
    
    # Summary
    cavity_count = sum(1 for r in results if 'Cavity' in r['prediction'])
    normal_count = len(results) - cavity_count
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Images: {len(results)}")
    print(f"Cavity Detected: {cavity_count}")
    print(f"Normal: {normal_count}")
    
    return results

def main():
    """Main function to run predictions."""
    
    print("=" * 60)
    print("DENTAL CAVITY DETECTION - PREDICTION")
    print("=" * 60)
    
    # Load model
    model = load_trained_model()
    if model is None:
        return
    
    while True:
        print("\n" + "-" * 40)
        print("Options:")
        print("1. Predict single image")
        print("2. Predict folder of images")
        print("3. Exit")
        print("-" * 40)
        
        choice = input("Enter your choice (1/2/3): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                predicted_class, confidence = predict_single_image(model, image_path)
                if predicted_class:
                    print("\n" + "=" * 40)
                    print("PREDICTION RESULT")
                    print("=" * 40)
                    print(f"Image: {image_path}")
                    print(f"Result: {predicted_class}")
                    print(f"Confidence: {confidence:.2f}%")
            else:
                print(f"[ERROR] Image not found: {image_path}")
                
        elif choice == '2':
            folder_path = input("Enter folder path: ").strip()
            predict_from_folder(model, folder_path)
            
        elif choice == '3':
            print("\nGoodbye!")
            break
        else:
            print("[ERROR] Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()

