import numpy as np
from src.data import load_mnist, preprocess_images, get_dataset_stats

def test_load_mnist_shapes():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    assert x_train.shape == (60000, 28, 28)
    assert y_train.shape == (60000,)
    assert x_test.shape == (10000, 28, 28)
    assert y_test.shape == (10000,)

def test_preprocess_range_and_dtype():
    sample = np.random.randint(0, 256, size=(10, 28, 28), dtype=np.uint8)
    norm = preprocess_images(sample)
    assert norm.dtype == np.float32
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
    assert norm.shape == (10, 28, 28)

def test_labels_are_integers():
    _, (_, y_test) = load_mnist()
    assert np.issubdtype(y_test.dtype, np.integer)
    assert set(np.unique(y_test)).issubset(set(range(10)))
