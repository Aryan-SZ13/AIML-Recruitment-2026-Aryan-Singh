# AIML Recruitment 2026 — Assessment Submission

[![Architecture diagram](https://gitdiagram.com/diagram-badge.svg)](https://gitdiagram.com/aryan-sz13/aiml-recruitment-2026-aryan-singh?utm_source=readme&utm_medium=badge)

<p align="center">
  <a href="https://gitdiagram.com/aryan-sz13/aiml-recruitment-2026-aryan-singh?utm_source=readme&utm_medium=picture">
    <img src="https://gitdiagram.com/aryan-sz13/aiml-recruitment-2026-aryan-singh/diagram.png" alt="Architecture diagram of aryan-sz13/aiml-recruitment-2026-aryan-singh" width="550"/>
  </a>
</p>

**Candidate Details:**
- **Name:** Aryan Singh
- **Institution:** SRM Institute of Science and Technology
- **Degree / Specialization:** B.Tech in Computer Science and Engineering (Artificial Intelligence and Machine Learning)
- **Year / Batch:** Class of 2026

**Tasks Completed:**
- **Task 2:** Neural Network — MNIST Handwritten Digit Classification (TensorFlow / Keras)

---

## 1. Problem Statement
The objective of Task 2 is to design, implement, train, evaluate, and scientifically analyze a foundational **Multi-Layer Perceptron (fully connected feedforward neural network)** for classifying handwritten digits ($0$–$9$) from the standard MNIST benchmark database.

The implementation strictly avoids high-level convolutional shortcuts (CNNs) in order to inspect, diagnose, and explain foundational neural network mechanics:
- 2D image matrix flattening ($28 \times 28 \to 784$)
- Feature normalization and integer target representation
- Role of non-linear activations (ReLU vs. Softmax)
- Training convergence and generalization tracking
- Multi-metric classification diagnosis via confusion matrices and classification reports
- Controlled capacity experimentation (128 vs. 256 hidden units)

---

## 2. Technical Approach & Architecture

### A. Dataset Provenance & Ingestion
- **Primary Dataset Provenance:** MNIST Handwritten Digit Database
- **Original Source Attribution:** Yann LeCun, Corinna Cortes, and Christopher J.C. Burges ([https://yann.lecun.org/exdb/mnist/](https://yann.lecun.org/exdb/mnist/)).
- **Download Source:** Canonical CVDFoundation mirror ([https://github.com/cvdfoundation/mnist](https://github.com/cvdfoundation/mnist)).
- **Loading Method:** Custom binary IDX format parser (`src/mnist_parser.py`) implementing big-endian byte unpack of magic numbers (`2051` for images, `2049` for labels), sample counts ($60,000$ train / $10,000$ test), and pixel dimensions ($28 \times 28$).
- *Provenance Note:* This project uses the canonical MNIST Handwritten Digit Database referenced by the assignment. The original MNIST IDX files are obtained from the CVDFoundation mirror of the database and parsed locally using a project-specific IDX parser. The dataset provenance is attributed to the original MNIST database associated with Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.

### B. Preprocessing Pipeline
- **Raw Input:** $28 \times 28$ grayscale images with integer pixels in $[0, 255]$ ($60,000$ training, $10,000$ test, `uint8`).
- **Pixel Normalization:** Pixels are cast to `float32` and scaled to $[0.0, 1.0]$ by dividing by $255.0$. This keeps input magnitudes manageable and supports stable optimization.
- **Label Representation:** Preserved as scalar integer class IDs ($0, 1, \dots, 9$). One-hot encoding is avoided to minimize memory overhead and utilize TensorFlow's native `sparse_categorical_crossentropy`.
- **Architectural Flattening:** Images are passed as $(28, 28)$ tensors directly to the model's explicit `Flatten()` layer, keeping the spatial-to-vector transformation fully transparent within the computational graph.

### B. Baseline Architecture
```
Input(shape=(28, 28))
    │
    ▼
Flatten()                                  ──► Outputs vector of size 784
    │
    ▼
Dense(128, activation="relu")              ──► Affine transform + ReLU non-linearity (100,480 weights + 128 biases)
    │
    ▼
Dense(10, activation="softmax")            ──► 10 class probabilities (1,280 weights + 10 biases)
```
- **Total Parameters:** $101,770$ trainable weights and biases.
- **Optimization:** Adam optimizer ($\text{lr} = 0.001$), `sparse_categorical_crossentropy` loss, batch size $128$, validation split $10\%$ ($6,000$ validation samples, $54,000$ train samples), $15$ epochs, reproducible seed $42$.

---

## 3. Technologies Used
- **Core ML / Deep Learning:** TensorFlow 2.21.0, Keras 3.15.1
- **Numerical Computing:** NumPy 2.4.6
- **Evaluation & Metrics:** scikit-learn 1.9.1
- **Visualization:** Matplotlib 3.11.2, Seaborn 0.13.2
- **Testing & Orchestration:** pytest 9.1.1, ipykernel 7.3.0
- **Language / Runtime:** Python 3.11.9 (Apple Silicon arm64)

---

## 4. Controlled Experiment & Empirical Results
*(All results below reflect real, unmanipulated metrics recorded from execution of `main.py` and saved in `outputs/results/comparison_summary.json`)*

### 8-Dimension Model Comparison Table

| Evaluation Dimension / Metric | Baseline Model (128 Units) | Experimental Model (256 Units) |
|---|---|---|
| **Hidden Layer Capacity** | 128 neurons | 256 neurons |
| **Total Trainable Parameters** | **101,770** | 203,530 (+100.0%) |
| **Measured Training Time** | **10.70 seconds** | 13.54 seconds (+26.5%) |
| **Test Accuracy** | **97.69%** | 97.53% |
| **Test Loss** | **0.0803** | 0.0917 |
| **Macro F1-Score** | **0.9768** | 0.9751 |
| **Final Training Accuracy** | 99.84% | 99.80% |
| **Final Validation Accuracy** | 97.95% | 97.52% |
| **Train–Validation Accuracy Gap** | **1.89%** | 2.29% |
| **Top Observed Confusion Pair** | True **7** $\to$ Pred **2** (10 errors) | True **9** $\to$ Pred **8** (21 errors) |
| **Exploratory Accuracy Efficiency\*** | **9.60** | 4.79 (-50.1%) |

*\*Exploratory Metric Definition:*
$$\text{Accuracy Efficiency} = \frac{\text{Test Accuracy (\%)}}{\text{Total Parameters} / 10{,}000}$$
*Note: This is an exploratory efficiency measure, not an established standard benchmark. It evaluates how much accuracy the model delivers per 10k parameter slice.*

### Per-Class Performance Summary (Baseline Model)
From the test set evaluation ($10,000$ unseen samples):
- **Digit 0:** Precision: $0.9778$, Recall: $0.9878$, F1: $0.9827$ ($980$ samples)
- **Digit 1:** Precision: $0.9877$, Recall: $0.9921$, F1: $0.9899$ ($1,135$ samples)
- **Digit 2:** Precision: $0.9666$, Recall: $0.9816$, F1: $0.9740$ ($1,032$ samples)
- **Digit 3:** Precision: $0.9734$, Recall: $0.9772$, F1: $0.9753$ ($1,010$ samples)
- **Digit 4:** Precision: $0.9806$, Recall: $0.9786$, F1: $0.9796$ ($982$ samples)
- **Digit 5:** Precision: $0.9853$, Recall: $0.9753$, F1: $0.9803$ ($892$ samples)
- **Digit 6:** Precision: $0.9893$, Recall: $0.9676$, F1: $0.9784$ ($958$ samples)
- **Digit 7:** Precision: $0.9870$, Recall: $0.9621$, F1: $0.9744$ ($1,028$ samples)
- **Digit 8:** Precision: $0.9473$, Recall: $0.9774$, F1: $0.9621$ ($974$ samples)
- **Digit 9:** Precision: $0.9750$, Recall: $0.9673$, F1: $0.9711$ ($1,009$ samples)
- **Overall Test Accuracy:** **97.69%** | **Macro Average F1:** **0.9768**

### Controlled Comparison & Empirical Takeaway
In this controlled run, doubling the hidden-layer width did not improve test-set generalization ($-0.16$ percentage points in this run: $97.69\% \to 97.53\%$), while increasing parameter count by $100\%$ ($101,770 \to 203,530$) and measured training time by $26.5\%$ ($10.70\text{s} \to 13.54\text{s}$). Repeated-seed experiments would be required before drawing broader conclusions about model-width sensitivity.

---

## 5. Key Learnings
1. **Diminishing Returns of Raw Layer Width:** In this controlled run, doubling the hidden-layer width ($128 \to 256$ neurons) did not improve test-set generalization ($-0.16$ percentage points in this run), while increasing parameter count by $100\%$ and measured training time by $26.5\%$. Repeated-seed experiments would be required before drawing broader conclusions about model-width sensitivity.
2. **Structural Error Clustering via Confusion Analysis:** Evaluating the empirical confusion matrix demonstrated that misclassifications are not randomly distributed across digit classes; they concentrate predictably on topologically similar stroke patterns (such as $7 \leftrightarrow 2$, $4 \leftrightarrow 9$, and $9 \leftrightarrow 8$).
3. **Architectural Transparency:** Encapsulating the $28 \times 28 \to 784$ vector transformation within a Keras `Flatten()` layer rather than pre-flattening arrays in NumPy keeps the pipeline end-to-end differentiable and transparently defines where 2D spatial locality is discarded.

---

## 6. Challenges Faced & Solutions

| Challenge Encountered | Root Cause | Engineering Solution Implemented |
|---|---|---|
| **Keras 3 / Matplotlib Sandbox Permission Denial** | Under restricted or sandboxed environments, default user home paths (`~/.keras` and `~/.matplotlib`) can trigger `PermissionError: Operation not permitted`. | Dynamically redirected `KERAS_HOME` and `MPLCONFIGDIR` to local workspace cache directories (`.keras_cache` and `.mpl_cache`) in `src/config.py` and `tests/conftest.py` before third-party library imports. |
| **Canonical Dataset Acquisition & Provenance** | The historical Yann LeCun web page (`/exdb/mnist/`) is frequently unreachable or returns 404 in modern environments. | Built a dedicated data acquisition layer (`src/mnist_download.py`) downloading the four original IDX compressed archives directly from the canonical CVDFoundation mirror, accompanied by a custom binary parser (`src/mnist_parser.py`) that unpacks the big-endian IDX byte streams locally. |
| **Headless Notebook Execution without Interactive Server** | Standard Jupyter kernel discovery was blocked in sandboxed CLI mode when attempting headless execution. | Developed a robust programmatic runner (`execute_notebook.py`) using standard library tools that walks every code cell, records execution states, and embeds true output streams and base64 PNG charts directly into the notebook. |

---

## 7. Project Structure

```
AIML-Recruitment-2026-Aryan-Singh/
├── README.md                          # Comprehensive submission report
├── requirements.txt                   # Dependency definitions
├── .gitignore                         # Configured ignore patterns
├── main.py                            # Standalone end-to-end execution pipeline
│
├── docs/                              # Visual assets and documentation
│   └── images/
│       └── architecture_flowchart.png # Pipeline architecture flowchart
│
├── src/                               # Modular Python source package
│   ├── __init__.py
│   ├── config.py                      # Hyperparameters, seeds, and paths
│   ├── data.py                        # Dataset loading, stats, and normalization
│   ├── mnist_download.py              # Canonical mirror acquisition & local SHA-256 fingerprinting
│   ├── mnist_parser.py                # Binary IDX unpacker for images (2051) and labels (2049)
│   ├── model.py                       # Input -> Flatten -> Dense(ReLU) -> Dense(Softmax)
│   ├── train.py                       # Training loop, seed locking, wall-clock timing
│   ├── evaluate.py                    # Multi-metric evaluation and confusion diagnosis
│   ├── visualize.py                   # Plotting utilities for training curves and heatmaps
│   └── experiment.py                  # Controlled 128 vs 256 experiment & manifest generator
│
├── notebooks/
│   └── mnist_neural_network.ipynb     # Demonstration notebook with embedded outputs
│
├── tests/                             # Comprehensive test suite (19 passing tests)
│   ├── __init__.py
│   ├── conftest.py                    # Test harness cache environment setup
│   ├── test_data.py                   # Preprocessing shape and range tests
│   ├── test_idx_parser.py             # Binary IDX parser and canonical provenance tests
│   ├── test_model.py                  # Architecture and output dimension tests
│   └── test_pipeline.py               # End-to-end synthetic training and metric validation
│
└── outputs/                           # Generated runtime outputs
    ├── figures/                       # Generated diagnostic plots (curves, CM, etc.)
    ├── results/                       # JSON metrics and experiment manifest
    └── models/                        # Serialized .keras trained models
```

---

## 8. Setup and Reproduction Instructions

### 1. Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Smoke Tests
Verify preprocessing, model building, and pipeline integration:
```bash
pytest tests/ -v
```

### 3. Run the Standalone Pipeline
To train both models, calculate metrics, run the controlled experiment, and save all artifacts:
```bash
python main.py
```

### 4. Inspect the Notebook
Open `notebooks/mnist_neural_network.ipynb` in VS Code or Jupyter.
- By default, `RUN_TRAINING = False` loads all precomputed models and metrics instantly with full visual analysis.
- Set `RUN_TRAINING = True` to retrain the models directly inside the notebook.
