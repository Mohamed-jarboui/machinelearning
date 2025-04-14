import os
import cv2
import numpy as np
from tqdm import tqdm

def resize_images(input_dir, output_dir, target_size=(128, 128)):
    """
    Resize all images in input_dir and save to output_dir.
    Args:
        input_dir: Path to raw images (e.g., 'data/raw/UTKFace')
        output_dir: Where to save resized images (e.g., 'data/processed/resized')
        target_size: (width, height)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, target_size)
        cv2.imwrite(os.path.join(output_dir, img_name), img)
        
if __name__ == "__main__":
    resize_images(
        input_dir="C:\\Users\\mjarb\\OneDrive\\Bureau\\Project 4\\MachineLearningProject\\data\\raw\\UTKFace",
        output_dir="C:\\Users\\mjarb\\OneDrive\\Bureau\\Project 4\\MachineLearningProject\\data\\processed\\resized"
    )