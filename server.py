import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow_addons as tfa  # For Option 1

# Load model with R2Score (Option 1: TensorFlow Addons)
model = load_model(
    'C:\\Users\\mjarb\\OneDrive\\Bureau\\Project 4\\MachineLearningProject\\outputs\\models\\best_model.h5',
    custom_objects={'R2Score': tfa.metrics.RSquare()}
)

# Preprocess image
def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Predict
img_path = "C:\\Users\\mjarb\\OneDrive\\Bureau\\Project 4\\MachineLearningProject\\mon_image\\souhaib.jpg"
processed_img = preprocess_image(img_path)
predictions = model.predict(processed_img)

print("Predicted Value:", predictions[0][0])  # For regression
# For classification: np.argmax(predictions, axis=1)