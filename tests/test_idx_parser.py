"""
Unit tests for the custom binary IDX parser and raw canonical MNIST dataset.
Verifies all 12 dataset provenance, magic number, dimension, type, and range criteria.
Author: Aryan Singh
"""
import os
import struct
import numpy as np
import pytest

from src.mnist_parser import (
    parse_idx_images,
    parse_idx_labels,
    load_mnist_raw_from_idx,
    IMAGE_MAGIC_NUMBER,
    LABEL_MAGIC_NUMBER
)
from src.mnist_download import download_mnist_raw, MNIST_FILES

RAW_DIR = "data/raw"


@pytest.fixture(scope="module")
def raw_mnist_data():
    """Fixture ensuring raw IDX files are downloaded and parsed once for tests."""
    download_mnist_raw(target_dir=RAW_DIR)
    (x_train, y_train), (x_test, y_test) = load_mnist_raw_from_idx(raw_dir=RAW_DIR)
    return (x_train, y_train), (x_test, y_test)


# 1. Correct image magic number
def test_1_image_magic_number():
    assert IMAGE_MAGIC_NUMBER == 2051  # 0x00000803
    train_img_path = os.path.join(RAW_DIR, "train-images-idx3-ubyte.gz")
    import gzip
    with gzip.open(train_img_path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        assert magic == 2051


# 2. Correct label magic number
def test_2_label_magic_number():
    assert LABEL_MAGIC_NUMBER == 2049  # 0x00000801
    train_lbl_path = os.path.join(RAW_DIR, "train-labels-idx1-ubyte.gz")
    import gzip
    with gzip.open(train_lbl_path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        assert magic == 2049


# 3. Train image count == 60000
def test_3_train_image_count(raw_mnist_data):
    (x_train, _), _ = raw_mnist_data
    assert len(x_train) == 60000


# 4. Train label count == 60000
def test_4_train_label_count(raw_mnist_data):
    (_, y_train), _ = raw_mnist_data
    assert len(y_train) == 60000


# 5. Test image count == 10000
def test_5_test_image_count(raw_mnist_data):
    _, (x_test, _) = raw_mnist_data
    assert len(x_test) == 10000


# 6. Test label count == 10000
def test_6_test_label_count(raw_mnist_data):
    _, (_, y_test) = raw_mnist_data
    assert len(y_test) == 10000


# 7. Image shape == (28, 28)
def test_7_image_shape(raw_mnist_data):
    (x_train, _), (x_test, _) = raw_mnist_data
    assert x_train.shape[1:] == (28, 28)
    assert x_test.shape[1:] == (28, 28)


# 8. Raw dtype == uint8
def test_8_raw_dtype(raw_mnist_data):
    (x_train, y_train), (x_test, y_test) = raw_mnist_data
    assert x_train.dtype == np.uint8
    assert y_train.dtype == np.uint8
    assert x_test.dtype == np.uint8
    assert y_test.dtype == np.uint8


# 9. Raw pixel range == [0, 255]
def test_9_raw_pixel_range(raw_mnist_data):
    (x_train, _), (x_test, _) = raw_mnist_data
    assert x_train.min() == 0
    assert x_train.max() == 255
    assert x_test.min() == 0
    assert x_test.max() == 255


# 10. Labels are integers in [0, 9]
def test_10_labels_integer_range(raw_mnist_data):
    (_, y_train), (_, y_test) = raw_mnist_data
    assert np.issubdtype(y_train.dtype, np.integer)
    assert np.issubdtype(y_test.dtype, np.integer)
    assert set(np.unique(y_train)) == set(range(10))
    assert set(np.unique(y_test)) == set(range(10))


# 11. Train image/label counts match
def test_11_train_image_label_count_match(raw_mnist_data):
    (x_train, y_train), _ = raw_mnist_data
    assert len(x_train) == len(y_train)


# 12. Test image/label counts match
def test_12_test_image_label_count_match(raw_mnist_data):
    _, (x_test, y_test) = raw_mnist_data
    assert len(x_test) == len(y_test)
