# MNIST Handwritten Digit Classification (Fully Connected Neural Network)

**Author:** Aryan Singh  
**Task:** Task 2 — Neural Network (AIML Recruitment 2026)  
**Framework:** TensorFlow / Keras  

---

## 📌 Project Overview
This project implements, evaluates, and analyzes a fully connected neural network (Multi-Layer Perceptron) for handwritten digit classification on the classic MNIST dataset. It demonstrates core deep learning concepts from first principles without unnecessary model complexity:

- **Part A — Dataset Understanding:** Comprehensive inspection of MNIST dimensions, class distribution, pixel ranges, and visual samples.
- **Part B — Data Preprocessing:** Pixel normalization ($[0, 255] \to [0.0, 1.0]$), explicit architectural flattening ($28 \times 28 \to 784$), and integer label preservation.
- **Part C — Model Architecture:** Fully connected feedforward architecture using explicit `Input(shape=(28, 28))`, `Flatten()`, hidden `Dense(128, activation='relu')`, and output `Dense(10, activation='softmax')`.
- **Part D — Activation Functions:** Detailed theoretical and empirical rationale for ReLU in hidden representations and Softmax for categorical posterior probabilities.
- **Part E — Training Dynamics:** Loss and accuracy tracking over epochs with train vs. validation curve diagnostic plots.
- **Part F — Comprehensive Evaluation:** Accuracy, confusion matrix, per-class Precision, Recall, and F1-score, alongside automated data-driven confusion-pair discovery.
- **Part G — Controlled Experimentation:** Model capacity experiment comparing 128 hidden neurons (baseline) against 256 hidden neurons under identical training regimes, evaluated across 8 dimensions including an exploratory Parameter Efficiency metric.
- **Advanced Analysis:** Prediction confidence calibration, high-confidence error diagnosis, hardest / most uncertain test cases, and confusion-pair visualization.

---

## 📂 Project Structure

```
AIML-Recruitment-2026-Aryan/
├── README.md                          # Project documentation and reproduction guide
├── requirements.txt                   # Dependency specifications
├── .gitignore                         # Standard git ignore rules
├── main.py                            # Standalone end-to-end execution pipeline
│
├── src/                               # Modular Python package
│   ├── __init__.py
│   ├── config.py                      # Hyperparameters, seeds, and paths
│   ├── data.py                        # Dataset loading, stats, and preprocessing
│   ├── model.py                       # Model construction and parameter calculation
│   ├── train.py                       # Model training loop and model checkpointing
│   ├── evaluate.py                    # Metrics, confusion matrix, error analysis
│   ├── visualize.py                   # Plotting functions (saved to outputs/figures)
│   └── experiment.py                  # Controlled baseline vs. experiment workflow
│
├── notebooks/
│   └── mnist_neural_network.ipynb     # Demonstration and evaluation notebook
│
├── tests/                             # Smoke and unit tests
│   ├── __init__.py
│   ├── test_data.py                   # Preprocessing shape and range tests
│   ├── test_model.py                  # Model architecture and output shape tests
│   └── test_pipeline.py               # End-to-end pipeline smoke test
│
└── outputs/                           # Generated runtime outputs (gitignored)
    ├── figures/                       # Generated diagnostic plots
    ├── results/                       # Metrics JSON and experiment manifest
    └── models/                        # Serialized .keras model files
```

---

## ⚙️ Installation & Setup

1. **Activate the Virtual Environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Smoke Tests:**
   ```bash
   pytest tests/ -v
   ```

---

## 🚀 How to Run

### Option 1: Standalone Script Pipeline
To execute the complete data loading, model training, evaluation, experiment comparison, and artifact generation from scratch:
```bash
python main.py
```
This generates:
- Saved models in `outputs/models/` (`baseline_model.keras`, `experiment_model.keras`)
- Evaluation metrics and experiment manifest in `outputs/results/` (`experiment_config.json`, `baseline_metrics.json`, `experiment_metrics.json`, `comparison_summary.json`)
- High-resolution plots in `outputs/figures/`

### Option 2: Jupyter Demonstration Notebook
Open `notebooks/mnist_neural_network.ipynb` in VS Code or JupyterLab.
The notebook includes a `RUN_TRAINING` flag:
- `RUN_TRAINING = False`: Instantly loads previously generated models and metrics from `outputs/` for inspection and deep-dive analysis without retraining.
- `RUN_TRAINING = True`: Retrains both baseline and experimental models from scratch within the notebook.

To execute and verify the notebook headlessly from the CLI via Papermill:
```bash
papermill notebooks/mnist_neural_network.ipynb notebooks/mnist_neural_network_executed.ipynb
```
