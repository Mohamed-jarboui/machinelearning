import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.data_loader import create_data_generator, get_dataset_size
from src.models.cnn import build_model
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.metrics import r2_score
import numpy as np

def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
            print("GPU configured with memory growth")
        except RuntimeError as e:
            print(e)

class R2Score(tf.keras.metrics.Metric):
    """Custom R² score metric for TensorFlow"""
    def __init__(self, name='r2_score', **kwargs):
        super(R2Score, self).__init__(name=name, **kwargs)
        self.total_sum = self.add_weight(name='total_sum', initializer='zeros')
        self.residual = self.add_weight(name='residual', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        total_error = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
        unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
        r2 = tf.subtract(1.0, tf.divide(unexplained_error, total_error))
        self.total_sum.assign_add(tf.reduce_sum(r2))
        self.count.assign_add(tf.cast(tf.size(r2), tf.float32))

    def result(self):
        return tf.divide(self.total_sum, self.count)

    def reset_states(self):
        self.total_sum.assign(0.)
        self.residual.assign(0.)
        self.count.assign(0.)

def train_model(data_dir="C:\\Users\\mjarb\\OneDrive\\Bureau\\Project 4\\MachineLearningProject\\data\\processed\\resized", 
               epochs=30, batch_size=32):
    configure_gpu()
    
    # Create generators
    train_gen, val_gen, train_size, val_size = create_data_generator(data_dir, batch_size)
    
    # Calculate steps
    steps_per_epoch = train_size // batch_size
    val_steps = val_size // batch_size
    
    # Verify output shapes
    sample_x, sample_y = next(train_gen)
    print(f"Sample batch - images: {sample_x.shape}, ages: {sample_y.shape}")
    print(f"Training with {steps_per_epoch} steps/epoch, {val_steps} validation steps")
    
    # Build and train model
    model = build_model()
    model.compile(
        optimizer='adam',
        loss='mae',
        metrics=[
            'mae',
            RootMeanSquaredError(name='rmse'),
            R2Score()
        ]
    )
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=[
            EarlyStopping(patience=3, monitor='val_loss'),
            ModelCheckpoint(
                "C:\\Users\\mjarb\\OneDrive\\Bureau\\Project 4\\MachineLearningProject\\outputs\\models\\best_model.h5",
                monitor='val_r2_score',
                mode='max',
                save_best_only=True
            )
        ]
    )
    
    # Calculate final R² score on validation set
    print("\nCalculating final validation metrics...")
    y_true, y_pred = [], []
    for _ in range(val_steps):
        x, y = next(val_gen)
        y_true.extend(y.flatten())
        y_pred.extend(model.predict(x).flatten())
    
    val_r2 = r2_score(y_true, y_pred)
    print(f"Final Validation R²: {val_r2:.4f}")
    
    return model, history

if __name__ == "__main__":
    model, history = train_model(batch_size=32)