# from tensorflow.keras import layers, models, regularizers
# from tensorflow.keras.optimizers import Adam

# from tensorflow.keras import layers, models, regularizers
# from tensorflow.keras.optimizers import Adam

# # Add this after Feature Aggregation (before GlobalAveragePooling)
# def channel_attention(x):
#     channel = layers.GlobalAveragePooling2D()(x)
#     channel = layers.Dense(x.shape[-1]//8, activation='relu')(channel)
#     channel = layers.Dense(x.shape[-1], activation='sigmoid')(channel)
#     return layers.Multiply()([x, channel])


# def multi_scale_block(x):
#     # Branch 1 (original)
#     b1 = layers.Conv2D(128, (3,3), padding='same')(x)
    
#     # Branch 2 (dilated)
#     b2 = layers.Conv2D(128, (3,3), dilation_rate=2, padding='same')(x)
    
#     # Branch 3 (depthwise separable)
#     b3 = layers.SeparableConv2D(128, (3,3), padding='same')(x)
    
#     return layers.Concatenate()([b1, b2, b3])

# def build_model(input_shape=(128, 128, 3)):
#     inputs = layers.Input(shape=input_shape)
    
#     # Initial Conv Block
#     x = layers.Conv2D(32, (7,7), padding='same')(inputs)  # Larger initial kernel
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.MaxPooling2D((2, 2))(x)
#     x = layers.Dropout(0.2)(x)
    
#     # Improved Residual Blocks
#     def residual_block(x, filters, downsample=True):
#         shortcut = x
#         stride = 2 if downsample else 1
        
#         x = layers.Conv2D(filters, (3,3), padding='same', strides=stride)(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Conv2D(filters, (3,3), padding='same')(x)
#         x = layers.BatchNormalization()(x)
        
#         if downsample:
#             shortcut = layers.Conv2D(filters, (1,1), strides=2)(shortcut)
#             shortcut = layers.BatchNormalization()(shortcut)
        
#         return layers.Add()([x, shortcut])
    
#     x = residual_block(x, 64)
#     x = residual_block(x, 128)
    
#     # Multi-Scale Feature Extraction
#     x = multi_scale_block(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
    
#     # Attention Mechanism
#     x = channel_attention(x)
    
#     # Feature Aggregation
#     x = layers.GlobalAveragePooling2D()(x)
    
#     # Regression Head
#     x = layers.Dense(256, activation='relu',
#                     kernel_regularizer=regularizers.l2(0.01))(x)
#     x = layers.Dropout(0.5)(x)
#     outputs = layers.Dense(1, activation='linear')(x)
    
#     model = models.Model(inputs=inputs, outputs=outputs)
    
#     model.compile(
#         optimizer=Adam(learning_rate=0.0005),
#         loss='huber_loss',
#         metrics=['mae', 'mse']
#     )
    
#     return model

# if __name__ == "__main__":
#     model = build_model()
#     model.summary()

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
