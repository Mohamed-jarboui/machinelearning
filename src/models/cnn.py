from tensorflow.keras import layers, models

def build_model(input_shape=(128, 128, 3)):
    """Basic CNN for age regression"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'), 
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1)  # Linear activation for regression
    ])
    
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()