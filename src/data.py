"""
Dataset loading, statistics calculation, and preprocessing.
Author: Aryan Singh
"""
import os
import numpy as np
import tensorflow as tf


def load_mnist(path="data/mnist.npz"):
    """
    Load raw MNIST dataset.
    Checks local file path relative to the project root first for offline and
    sandbox reproducibility; falls back to tf.keras.datasets.mnist.load_data().

    Args:
        path (str): Relative or absolute path to local mnist.npz archive.

    Returns:
        tuple: ((x_train, y_train), (x_test, y_test)) containing raw uint8 images
               and integer scalar labels (0-9).
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = path if os.path.isabs(path) else os.path.join(project_root, path)

    if os.path.exists(full_path):
        with np.load(full_path, allow_pickle=True) as data:
            x_train, y_train = data["x_train"], data["y_train"]
            x_test, y_test = data["x_test"], data["y_test"]
            return (x_train, y_train), (x_test, y_test)

    return tf.keras.datasets.mnist.load_data()


def preprocess_images(images):
    """
    Convert raw images to float32 and normalize pixel values from [0, 255] to [0.0, 1.0].
    Preserves spatial image dimensions as (..., 28, 28) for the model's Flatten layer.

    Args:
        images (np.ndarray): Array of grayscale images with shape (..., 28, 28).

    Returns:
        np.ndarray: Normalized float32 images in range [0.0, 1.0] with shape (..., 28, 28).
    """
    return images.astype(np.float32) / 255.0


def load_and_preprocess_data(path="data/mnist.npz"):
    """
    Loads raw MNIST and normalizes images while preserving integer class labels (0-9).
    No one-hot encoding is applied; compatible with sparse_categorical_crossentropy.

    Args:
        path (str): Relative or absolute path to local mnist.npz archive.

    Returns:
        tuple: ((x_train_norm, y_train), (x_test_norm, y_test))
    """
    (x_train, y_train), (x_test, y_test) = load_mnist(path=path)
    x_train_norm = preprocess_images(x_train)
    x_test_norm = preprocess_images(x_test)
    return (x_train_norm, y_train), (x_test_norm, y_test)


def get_dataset_stats(x_train, y_train, x_test, y_test):
    """
    Computes comprehensive dataset summary statistics.

    Args:
        x_train (np.ndarray): Training images.
        y_train (np.ndarray): Training integer labels.
        x_test (np.ndarray): Test images.
        y_test (np.ndarray): Test integer labels.

    Returns:
        dict: Summary statistics including dimensions, sample counts, pixel stats,
              and per-class sample frequencies.
    """
    class_counts_train = {int(i): int(np.sum(y_train == i)) for i in range(10)}
    class_counts_test = {int(i): int(np.sum(y_test == i)) for i in range(10)}

    stats = {
        "num_train_samples": int(len(x_train)),
        "num_test_samples": int(len(x_test)),
        "image_shape": list(x_train.shape[1:]),
        "num_classes": 10,
        "pixel_min_raw": float(np.min(x_train)),
        "pixel_max_raw": float(np.max(x_train)),
        "pixel_mean_raw": float(np.mean(x_train)),
        "pixel_std_raw": float(np.std(x_train)),
        "class_counts_train": class_counts_train,
        "class_counts_test": class_counts_test,
    }
    return stats
