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

def test_multi_layer_architecture_and_parameters():
    # 2 hidden layers: [128, 64]
    # 784 * 128 + 128 = 100,480 (layer 1)
    # 128 * 64 + 64   =   8,256 (layer 2)
    # 64 * 10 + 10    =     650 (output)
    # Total           = 109,386
    model = build_model(hidden_units=[128, 64])
    assert len(model.layers) == 4  # Flatten, Dense, Dense, Dense
    params = count_model_parameters(model)
    assert params["total_params"] == 109386

def test_different_activations_and_learning_rate():
    model_sig = build_model(hidden_units=128, activation="sigmoid", learning_rate=0.01)
    assert model_sig.layers[1].activation.__name__ == "sigmoid"
    assert np.isclose(float(model_sig.optimizer.learning_rate.numpy()), 0.01)
    
    model_tanh = build_model(hidden_units=128, activation="tanh")
    assert model_tanh.layers[1].activation.__name__ == "tanh"

