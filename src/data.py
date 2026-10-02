"""
Dataset loading, statistics calculation, and preprocessing.
Author: Aryan Singh
"""
import os
import numpy as np
import tensorflow as tf


from src.mnist_parser import load_mnist_raw_from_idx
from src.mnist_download import download_mnist_raw

def load_mnist(raw_dir="data/raw"):
    """
    Load raw MNIST dataset from original IDX binary files.
    Ensures raw IDX files are downloaded from CVDFoundation mirror,
    and parses them using the custom project IDX parser.
    Falls back to legacy local archive if available.

    Args:
        raw_dir (str): Path to raw IDX directory.

    Returns:
        tuple: ((x_train, y_train), (x_test, y_test)) containing raw uint8 images
               and integer scalar labels (0-9).
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_raw_dir = raw_dir if os.path.isabs(raw_dir) else os.path.join(project_root, raw_dir)

    # 1. Primary path: Original IDX files via custom parser
    try:
        if not os.path.exists(os.path.join(full_raw_dir, "train-images-idx3-ubyte.gz")):
            download_mnist_raw(target_dir=full_raw_dir)
        return load_mnist_raw_from_idx(raw_dir=full_raw_dir)
    except Exception as e:
        print(f"Notice: Loading from raw IDX encountered {e}; trying local cache fallback...")

    # 2. Local npz cache fallback
    npz_path = os.path.join(project_root, "data/mnist.npz")
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=True) as data:
            return (data["x_train"], data["y_train"]), (data["x_test"], data["y_test"])

    raise FileNotFoundError("Could not locate or load raw MNIST IDX files.")


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


def load_and_preprocess_data(raw_dir="data/raw"):
    """
    Loads raw MNIST from original IDX files and normalizes images while preserving integer class labels (0-9).
    No one-hot encoding is applied; compatible with sparse_categorical_crossentropy.

    Args:
        raw_dir (str): Relative or absolute path to raw IDX directory.

    Returns:
        tuple: ((x_train_norm, y_train), (x_test_norm, y_test))
    """
    (x_train, y_train), (x_test, y_test) = load_mnist(raw_dir=raw_dir)
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
