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

# Architecture specifications
INPUT_SHAPE = (28, 28)
NUM_CLASSES = 10
BASELINE_UNITS = 128
EXPERIMENT_UNITS = 256

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
MANIFEST_PATH = os.path.join(RESULTS_DIR, "experiment_config.json")

def ensure_directories():
    """Ensure all required output directories exist."""
    for directory in [OUTPUT_DIR, FIGURES_DIR, RESULTS_DIR, MODELS_DIR]:
        os.makedirs(directory, exist_ok=True)
