"""
Download sample dental X-ray images for testing
"""
import os
import urllib.request
import ssl
import shutil

# Create SSL context that doesn't verify certificates (for downloading)
ssl._create_default_https_context = ssl._create_unverified_context

# Sample dental X-ray image URLs (public domain/free images)
# These are placeholder URLs - in real scenario you'd use actual dental X-ray dataset

def create_sample_images():
    """Create sample grayscale images for testing."""
    import numpy as np
    from PIL import Image
    
    print("Creating sample dental X-ray images for testing...")
    
    folders = [
        'data/train/cavity',
        'data/train/normal',
        'data/validation/cavity',
        'data/validation/normal'
    ]
    
    # Create folders
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    # Generate sample images
    np.random.seed(42)
    
    # Training images
    print("\nGenerating training images...")
    for i in range(50):
        # Cavity images - darker spots
        img = np.random.randint(100, 200, (150, 150), dtype=np.uint8)
        # Add dark spots (simulating cavities)
        for _ in range(np.random.randint(2, 5)):
            cx, cy = np.random.randint(30, 120, 2)
            radius = np.random.randint(10, 25)
            y, x = np.ogrid[:150, :150]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            img[mask] = np.random.randint(20, 60)
        
        img_pil = Image.fromarray(img, mode='L').convert('RGB')
        img_pil.save(f'data/train/cavity/cavity_{i+1}.jpg')
    
    for i in range(50):
        # Normal images - uniform texture
        img = np.random.randint(120, 200, (150, 150), dtype=np.uint8)
        # Add some natural variation
        noise = np.random.normal(0, 10, (150, 150))
        img = np.clip(img + noise, 100, 220).astype(np.uint8)
        
        img_pil = Image.fromarray(img, mode='L').convert('RGB')
        img_pil.save(f'data/train/normal/normal_{i+1}.jpg')
    
    # Validation images
    print("Generating validation images...")
    for i in range(15):
        img = np.random.randint(100, 200, (150, 150), dtype=np.uint8)
        for _ in range(np.random.randint(2, 5)):
            cx, cy = np.random.randint(30, 120, 2)
            radius = np.random.randint(10, 25)
            y, x = np.ogrid[:150, :150]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            img[mask] = np.random.randint(20, 60)
        
        img_pil = Image.fromarray(img, mode='L').convert('RGB')
        img_pil.save(f'data/validation/cavity/cavity_{i+1}.jpg')
    
    for i in range(15):
        img = np.random.randint(120, 200, (150, 150), dtype=np.uint8)
        noise = np.random.normal(0, 10, (150, 150))
        img = np.clip(img + noise, 100, 220).astype(np.uint8)
        
        img_pil = Image.fromarray(img, mode='L').convert('RGB')
        img_pil.save(f'data/validation/normal/normal_{i+1}.jpg')
    
    print("\n" + "="*50)
    print("SAMPLE DATA CREATED SUCCESSFULLY!")
    print("="*50)
    print(f"\nTraining: 50 cavity + 50 normal images")
    print(f"Validation: 15 cavity + 15 normal images")
    print(f"\nNow run: python train.py")

if __name__ == "__main__":
    create_sample_images()

