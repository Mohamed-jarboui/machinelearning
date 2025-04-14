from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def create_augmentor():
    """Returns configured image augmentor"""
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

def save_augmented_images(generator, input_dir, output_dir, samples_per_class=1000):
    """
    Generate augmented images and save to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Implement augmentation logic here
    # (For full implementation, use generator.flow_from_directory())
    
if __name__ == "__main__":
    augmentor = create_augmentor()