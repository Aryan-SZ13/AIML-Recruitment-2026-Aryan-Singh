"""
Neural network architecture builders and parameter utilities.
"""
import tensorflow as tf

def build_model(hidden_units=128, input_shape=(28, 28), num_classes=10, activation="relu", learning_rate=0.001):
    """
    Builds a fully connected neural network with explicit Input, Flatten, one or more Dense hidden layers,
    and Dense(Softmax) output.
    Compiled with Adam optimizer and sparse_categorical_crossentropy loss for integer labels.
    """
    if isinstance(hidden_units, int):
        units_list = [hidden_units]
    else:
        units_list = list(hidden_units)
        
    layers = [
        tf.keras.layers.Input(shape=input_shape, name="input_image"),
        tf.keras.layers.Flatten(name="flatten_2d_to_1d")
    ]
    
    for i, units in enumerate(units_list):
        layer_name = f"hidden_dense_{units}" if len(units_list) == 1 else f"hidden_dense_l{i+1}_{units}"
        layers.append(tf.keras.layers.Dense(units, activation=activation, name=layer_name))
        
    layers.append(tf.keras.layers.Dense(num_classes, activation="softmax", name="output_probabilities"))
    
    name_str = f"mnist_fcnn_{'_'.join(map(str, units_list))}_{activation}"
    model = tf.keras.Sequential(layers, name=name_str)
    
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
