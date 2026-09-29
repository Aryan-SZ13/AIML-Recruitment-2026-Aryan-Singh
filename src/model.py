"""
Neural network architecture builders and parameter utilities.
"""
import tensorflow as tf

def build_model(hidden_units=128, input_shape=(28, 28), num_classes=10, learning_rate=0.001):
    """
    Builds a fully connected neural network with explicit Input, Flatten, Dense(ReLU), Dense(Softmax).
    Compiled with Adam optimizer and sparse_categorical_crossentropy loss for integer labels.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, name="input_image"),
        tf.keras.layers.Flatten(name="flatten_2d_to_1d"),
        tf.keras.layers.Dense(hidden_units, activation="relu", name=f"hidden_dense_{hidden_units}"),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output_probabilities")
    ], name=f"mnist_fcnn_{hidden_units}_units")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def count_model_parameters(model):
    """
    Returns total trainable and non-trainable parameter count.
    """
    return {
        "total_params": int(model.count_params()),
        "trainable_params": int(sum([tf.size(w).numpy() for w in model.trainable_weights]))
    }
