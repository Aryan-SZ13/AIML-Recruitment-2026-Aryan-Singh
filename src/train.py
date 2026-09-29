"""
Model training execution, timing, and checkpoint saving.
"""
import time
import os
import random
import numpy as np
import tensorflow as tf

def set_seed(seed=42):
    """Set random seeds across Python, NumPy, and TensorFlow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_model(model, x_train, y_train, epochs=15, batch_size=128, validation_split=0.1, verbose=1):
    """
    Trains the Keras model while recording training history and wall-clock time.
    """
    start_time = time.time()
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose
    )
    training_time = time.time() - start_time
    
    history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
    return history_dict, training_time

def save_trained_model(model, filepath):
    """Persists model in Keras native format (.keras)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
