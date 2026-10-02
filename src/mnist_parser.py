"""
Binary IDX format parser for MNIST handwritten digit database.
Parses both 2D image matrices and 1D label vectors directly from compressed or uncompressed IDX files.
Author: Aryan Singh
"""
import os
import gzip
import struct
import numpy as np

# Canonical IDX Magic Numbers
IMAGE_MAGIC_NUMBER = 2051  # 0x00000803: 08=unsigned byte, 03=3 dimensions (num_images, rows, cols)
LABEL_MAGIC_NUMBER = 2049  # 0x00000801: 08=unsigned byte, 01=1 dimension (num_items)


def _open_file(filepath):
    """Opens a file using gzip if .gz, otherwise standard open."""
    if filepath.endswith(".gz"):
        return gzip.open(filepath, "rb")
    return open(filepath, "rb")


def parse_idx_images(filepath):
    """
    Parses MNIST IDX image file (train-images or t10k-images).

    Format:
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803 (2051) magic number
        0004     32 bit integer  number of images
        0008     32 bit integer  number of rows (28)
        0012     32 bit integer  number of columns (28)
        0016     unsigned byte   pixel values

    Returns:
        np.ndarray: uint8 array of shape (N, 28, 28) with values in [0, 255].
    """
    with _open_file(filepath) as f:
        # Read header: 4 big-endian 32-bit integers (>IIII)
        header_bytes = f.read(16)
        if len(header_bytes) < 16:
            raise ValueError(f"File {filepath} is too small to contain a valid IDX image header.")

        magic, num_images, rows, cols = struct.unpack(">IIII", header_bytes)

        if magic != IMAGE_MAGIC_NUMBER:
            raise ValueError(
                f"Invalid image magic number: expected {IMAGE_MAGIC_NUMBER} (0x00000803), "
                f"got {magic} (0x{magic:08x}) in {filepath}"
            )
        if rows != 28 or cols != 28:
            raise ValueError(
                f"Invalid image dimensions: expected (28, 28), got ({rows}, {cols}) in {filepath}"
            )

        # Read remaining pixel data
        data_bytes = f.read()
        expected_bytes = num_images * rows * cols
        if len(data_bytes) != expected_bytes:
            raise ValueError(
                f"Corrupted image data: expected {expected_bytes} bytes for {num_images} images, "
                f"got {len(data_bytes)} bytes in {filepath}"
            )

        images = np.frombuffer(data_bytes, dtype=np.uint8).reshape((num_images, rows, cols))
        return images


def parse_idx_labels(filepath):
    """
    Parses MNIST IDX label file (train-labels or t10k-labels).

    Format:
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000801 (2049) magic number
        0004     32 bit integer  number of items
        0008     unsigned byte   label values (0-9)

    Returns:
        np.ndarray: uint8 array of shape (N,) containing integer labels 0-9.
    """
    with _open_file(filepath) as f:
        # Read header: 2 big-endian 32-bit integers (>II)
        header_bytes = f.read(8)
        if len(header_bytes) < 8:
            raise ValueError(f"File {filepath} is too small to contain a valid IDX label header.")

        magic, num_items = struct.unpack(">II", header_bytes)

        if magic != LABEL_MAGIC_NUMBER:
            raise ValueError(
                f"Invalid label magic number: expected {LABEL_MAGIC_NUMBER} (0x00000801), "
                f"got {magic} (0x{magic:08x}) in {filepath}"
            )

        data_bytes = f.read()
        if len(data_bytes) != num_items:
            raise ValueError(
                f"Corrupted label data: expected {num_items} bytes, got {len(data_bytes)} in {filepath}"
            )

        labels = np.frombuffer(data_bytes, dtype=np.uint8)
        return labels


def load_mnist_raw_from_idx(raw_dir="data/raw"):
    """
    Loads and validates all four raw MNIST IDX files from the raw data directory.

    Returns:
        tuple: ((x_train, y_train), (x_test, y_test)) as raw uint8 numpy arrays.
    """
    train_images_path = os.path.join(raw_dir, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(raw_dir, "train-labels-idx1-ubyte.gz")
    test_images_path = os.path.join(raw_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(raw_dir, "t10k-labels-idx1-ubyte.gz")

    for path in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required raw MNIST IDX archive missing: {path}. "
                f"Run `download_mnist_raw()` first."
            )

    x_train = parse_idx_images(train_images_path)
    y_train = parse_idx_labels(train_labels_path)
    x_test = parse_idx_images(test_images_path)
    y_test = parse_idx_labels(test_labels_path)

    # Validate structural consistency
    if len(x_train) != len(y_train):
        raise ValueError(f"Train sample mismatch: {len(x_train)} images vs {len(y_train)} labels.")
    if len(x_test) != len(y_test):
        raise ValueError(f"Test sample mismatch: {len(x_test)} images vs {len(y_test)} labels.")
    if len(x_train) != 60000:
        raise ValueError(f"Expected 60,000 training samples, got {len(x_train)}.")
    if len(x_test) != 10000:
        raise ValueError(f"Expected 10,000 test samples, got {len(x_test)}.")

    return (x_train, y_train), (x_test, y_test)
