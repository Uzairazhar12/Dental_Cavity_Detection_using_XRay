# 🦷 Dental Cavity Detection using X-Rays

A deep learning project to detect dental cavities from X-ray images using Convolutional Neural Networks (CNN).

## 📋 Project Structure

```
Dental Cavity Detection using X-Rays/
├── data/
│   ├── train/
│   │   ├── cavity/      # Training images with cavities
│   │   └── normal/      # Normal training
│   └── validation/
│       ├── cavity/      # Validation images with cavities
│       └── normal/      # Normal validation images
├── model/
│   └── dental_cavity_model.h5  # Trained model (created after training)
├── train.py             # Model training script
├── predict.py           # Command-line prediction script
├── app.py               # Streamlit web application
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🚀 Installation

### Step 1: Clone na or Download the Project

Download or clone this project to your local machine.

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 📁 Dataset Preparation

1. Create the following folder structure inside the project:

```
data/
├── train/
│   ├── cavity/      # Put training X-ray images WITH cavities here
│   └── normal/      # Put training X-ray images WITHOUT cavities here
└── validation/
    ├── cavity/      # Put validation X-ray images WITH cavities here
    └── normal/      # Put validation X-ray images WITHOUT cavities here
```

2. Download dental X-ray dataset from one of these sources:
   - [Kaggle - Dental X-rays Dataset](https://www.kaggle.com/datasets)
   - Your own collected dataset

3. Split your images:
   - ~80% for training
   - ~20% for validation

## 🎯 Usage

### 1. Train the Model

```bash
python train.py
```

This will:
- Load images from `data/train` and `data/validation`
- Train the CNN model
- Save the best model to `model/dental_cavity_model.h5`
- Generate training history plots

### 2. Make Predictions (Command Line)

```bash
python predict.py
```

Follow the prompts to:
- Predict on a single image
- Predict on a folder of images

### 3. Run Web Application

```bash
streamlit run app.py
```

This will open a web browser with a user-friendly interface where you can:
- Upload dental X-ray images
- Get instant cavity detection results
- View confidence scores

## 🧠 Model Architecture

The CNN model consists of:
- 4 Convolutional blocks with BatchNormalization and MaxPooling
- Flatten layer
- 2 Dense layers with Dropout for regularization
- Sigmoid output for binary classification

```
Input (150x150x3)
    ↓
Conv2D (32 filters) → BatchNorm → MaxPool
    ↓
Conv2D (64 filters) → BatchNorm → MaxPool
    ↓
Conv2D (128 filters) → BatchNorm → MaxPool
    ↓
Conv2D (256 filters) → BatchNorm → MaxPool
    ↓
Flatten
    ↓
Dense (512) → Dropout (0.5)
    ↓
Dense (256) → Dropout (0.3)
    ↓
Dense (1, sigmoid) → Output
```

## 📊 Results

After training, you will see:
- Training/Validation accuracy plots
- Training/Validation loss plots
- Classification report
- Confusion matrix

## ⚙️ Configuration

You can modify these parameters in `train.py`:

```python
IMG_SIZE = 150       # Image size
BATCH_SIZE = 32      # Training batch size
EPOCHS = 20          # Number of training epochs
LEARNING_RATE = 0.0001  # Learning rate
```

## 📝 Notes

- For better accuracy, use a larger and diverse dataset
- Ensure images are good quality X-rays
- The model works best with properly cropped dental X-rays

## ⚠️ Disclaimer

This project is for **educational purposes only**. It should NOT be used as a substitute for professional dental diagnosis. Always consult a qualified dentist for proper examination and treatment.

## 🛠️ Technologies Used

- Python 3.8+
- TensorFlow / Keras
- OpenCV
- Streamlit
- NumPy
- Matplotlib
- scikit-learn

## 📧 Contact

For any questions or suggestions, feel free to reach out!

---
Made with ❤️ for Dental Health Awareness

