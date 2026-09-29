import numpy as np
import tensorflow as tf
from src.model import build_model, count_model_parameters

def test_model_architecture_and_output_shape():
    model = build_model(hidden_units=128)
    assert len(model.layers) == 3  # Flatten, Dense, Dense
    dummy_input = np.random.rand(5, 28, 28).astype(np.float32)
    output = model(dummy_input)
    assert output.shape == (5, 10)

def test_softmax_sum_to_one():
    model = build_model(hidden_units=128)
    dummy_input = np.random.rand(4, 28, 28).astype(np.float32)
    output = model(dummy_input).numpy()
    sums = np.sum(output, axis=1)
    np.testing.assert_allclose(sums, np.ones(4), atol=1e-5)

def test_parameter_counts():
    # 784 * 128 + 128 = 100,480 (hidden)
    # 128 * 10 + 10 = 1,290 (output)
    # Total = 101,770
    model_128 = build_model(hidden_units=128)
    params_128 = count_model_parameters(model_128)
    assert params_128["total_params"] == 101770
    
    # 784 * 256 + 256 = 200,960
    # 256 * 10 + 10 = 2,570
    # Total = 203,530
    model_256 = build_model(hidden_units=256)
    params_256 = count_model_parameters(model_256)
    assert params_256["total_params"] == 203530
