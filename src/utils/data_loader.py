import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_process_image(path, target_size=(128, 128)):
    img = cv2.imread(path)
    img = cv2.resize(img, target_size)
    return img.astype(np.float32) / 255.0

def create_data_generator(data_dir, batch_size=32):
    """Create consistent-sized batches"""
    image_paths = []
    ages = []
    
    # Collect all valid files
    for img_name in os.listdir(data_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                age = int(img_name.split('_')[0])
                image_paths.append(os.path.join(data_dir, img_name))
                ages.append(age)
            except (ValueError, IndexError):
                continue
    
    # Split data
    train_paths, val_paths, train_ages, val_ages = train_test_split(
        image_paths, ages, test_size=0.2, random_state=42
    )
    
    def generator(paths, age_labels):
        while True:  # Infinite generator for Keras
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i:i+batch_size]
                batch_ages = age_labels[i:i+batch_size]
                
                batch_images = []
                for path in batch_paths:
                    batch_images.append(load_and_process_image(path))
                
                yield np.array(batch_images), np.array(batch_ages).reshape(-1, 1)
    
    return generator(train_paths, train_ages), generator(val_paths, val_ages), len(train_paths), len(val_paths)

def get_dataset_size(data_dir):
    return sum(1 for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))