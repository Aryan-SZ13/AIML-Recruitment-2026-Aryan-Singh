"""
Configuration settings, hyperparameters, seeds, and directory paths.
"""
import os

# Set cache directories to writable workspace paths before TF/Keras import
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["KERAS_HOME"] = os.path.join(BASE_DIR, ".keras_cache")
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, ".mpl_cache")
os.makedirs(os.environ["KERAS_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# Reproducibility seed
SEED = 42

# Training Hyperparameters
EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1

# Baseline Architecture & Training specifications
INPUT_SHAPE = (28, 28)
NUM_CLASSES = 10
BASELINE_UNITS = 128
EXPERIMENT_UNITS = 256
BASELINE_ACTIVATION = "relu"

# Comprehensive 6-Dimension Ablation Suite Specifications
EXPERIMENT_SPECS = {
    "exp1_depth": {
        "name": "Hidden Layers (Depth)",
        "dimension": "Architecture Depth",
        "variants": {
            "1_layer": {"hidden_units": [128], "description": "Single Hidden Layer (128 units)"},
            "2_layers": {"hidden_units": [128, 64], "description": "Two Hidden Layers (128 -> 64 units)"}
        }
    },
    "exp2_width": {
        "name": "Number of Neurons (Width)",
        "dimension": "Layer Width / Capacity",
        "variants": {
            "128_units": {"hidden_units": 128, "description": "128 Units (Baseline)"},
            "256_units": {"hidden_units": 256, "description": "256 Units (Expanded Capacity)"}
        }
    },
    "exp3_lr": {
        "name": "Learning Rate",
        "dimension": "Optimization Step Size",
        "variants": {
            "lr_0.001": {"learning_rate": 0.001, "description": "Standard Adam LR (0.001)"},
            "lr_0.01": {"learning_rate": 0.01, "description": "High / Aggressive LR (0.01)"},
            "lr_0.0001": {"learning_rate": 0.0001, "description": "Low / Conservative LR (0.0001)"}
        }
    },
    "exp4_batch_size": {
        "name": "Batch Size",
        "dimension": "Gradient Stochasticity",
        "variants": {
            "batch_32": {"batch_size": 32, "description": "Small Batch / High Noise (32)"},
            "batch_128": {"batch_size": 128, "description": "Medium Batch (128 Baseline)"},
            "batch_512": {"batch_size": 512, "description": "Large Batch / Smooth Gradient (512)"}
        }
    },
    "exp5_epochs": {
        "name": "Number of Epochs",
        "dimension": "Training Budget / Convergence Horizon",
        "variants": {
            "epochs_5": {"epochs": 5, "description": "Short Budget (5 Epochs)"},
            "epochs_15": {"epochs": 15, "description": "Standard Budget (15 Epochs Baseline)"},
            "epochs_30": {"epochs": 30, "description": "Extended Budget (30 Epochs)"}
        }
    },
    "exp6_activation": {
        "name": "Activation Function",
        "dimension": "Non-linearity & Gradient Flow",
        "variants": {
            "relu": {"activation": "relu", "description": "Rectified Linear Unit (ReLU)"},
            "sigmoid": {"activation": "sigmoid", "description": "Sigmoid (Saturating Gradient)"},
            "tanh": {"activation": "tanh", "description": "Hyperbolic Tangent (Zero-centered Tanh)"}
        }
    }
}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
MANIFEST_PATH = os.path.join(RESULTS_DIR, "experiment_config.json")
SUITE_MANIFEST_PATH = os.path.join(RESULTS_DIR, "experiment_suite_manifest.json")

def ensure_directories():
    """Ensure all required output directories exist."""
    for directory in [OUTPUT_DIR, FIGURES_DIR, RESULTS_DIR, MODELS_DIR]:
        os.makedirs(directory, exist_ok=True)
