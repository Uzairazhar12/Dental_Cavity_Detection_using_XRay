"""
Dental Cavity Detection - Training Script
This script trains a CNN model to detect cavities in dental X-ray images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
IMG_SIZE = 150  # Image size (150x150)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# Data paths - Update these paths according to your dataset location
TRAIN_DIR = 'data/train'
VALIDATION_DIR = 'data/validation'
MODEL_SAVE_PATH = 'model/dental_cavity_model.h5'

# ==================== DATA PREPARATION ====================
def create_data_generators():
    """Create training and validation data generators with augmentation."""
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation data - only rescaling
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )
    
    # Load validation data
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, validation_generator

# ==================== MODEL ARCHITECTURE ====================
def create_model():
    """Create CNN model for dental cavity detection."""
    
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ==================== TRAINING ====================
def train_model():
    """Train the dental cavity detection model."""
    
    print("=" * 50)
    print("DENTAL CAVITY DETECTION - MODEL TRAINING")
    print("=" * 50)
    
    # Create directories if they don't exist
    os.makedirs('model', exist_ok=True)
    
    # Check if data directories exist
    if not os.path.exists(TRAIN_DIR):
        print(f"\n[ERROR] Training data directory not found: {TRAIN_DIR}")
        print("\nPlease create the following folder structure:")
        print("data/")
        print("├── train/")
        print("│   ├── cavity/      (X-ray images with cavities)")
        print("│   └── normal/      (Normal X-ray images)")
        print("└── validation/")
        print("    ├── cavity/      (X-ray images with cavities)")
        print("    └── normal/      (Normal X-ray images)")
        return None
    
    # Create data generators
    print("\n[INFO] Loading data...")
    train_generator, validation_generator = create_data_generators()
    
    print(f"\n[INFO] Training samples: {train_generator.samples}")
    print(f"[INFO] Validation samples: {validation_generator.samples}")
    print(f"[INFO] Classes: {train_generator.class_indices}")
    
    # Check if we have data
    if train_generator.samples == 0:
        print("\n" + "=" * 50)
        print("[ERROR] No training images found!")
        print("=" * 50)
        print("\nPlease add dental X-ray images to these folders:")
        print("  data/train/cavity/   - X-rays with cavities")
        print("  data/train/normal/   - Normal X-rays")
        print("  data/validation/cavity/")
        print("  data/validation/normal/")
        print("\nDownload dataset from Kaggle: 'dental cavity dataset'")
        return None
    
    # Create model
    print("\n[INFO] Creating model...")
    model = create_model()
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train the model
    print("\n[INFO] Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"\n[INFO] Model saved to: {MODEL_SAVE_PATH}")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    evaluate_model(model, validation_generator)
    
    return model, history

# ==================== VISUALIZATION ====================
def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss', color='blue')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', color='red')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('model/training_history.png')
    plt.show()
    print("\n[INFO] Training history plot saved to: model/training_history.png")

def evaluate_model(model, validation_generator):
    """Evaluate the model on validation data."""
    
    print("\n[INFO] Evaluating model...")
    
    # Get predictions
    validation_generator.reset()
    predictions = model.predict(validation_generator, steps=len(validation_generator))
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = validation_generator.classes
    
    # Classification report
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(true_classes, predicted_classes, 
                                target_names=['Normal', 'Cavity']))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    print("\nConfusion Matrix:")
    print(cm)

# ==================== MAIN ====================
if __name__ == "__main__":
    train_model()

