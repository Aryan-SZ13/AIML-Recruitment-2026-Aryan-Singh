"""
Data visualization, training curve plots, confusion matrices, and error inspections.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_sample_digits(x, y, n=10, save_path=None):
    """Plots a row of sample MNIST digits with their true labels."""
    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 2.0))
    for i in range(n):
        axes[i].imshow(x[i], cmap="gray")
        axes[i].set_title(f"Label: {y[i]}", fontsize=10)
        axes[i].axis("off")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    return fig

def plot_class_distributions(y_train, y_test, save_path=None):
    """Plots side-by-side bar plots of digit distributions in train and test splits."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    train_counts = np.bincount(y_train, minlength=10)
    test_counts = np.bincount(y_test, minlength=10)
    classes = np.arange(10)
    
    ax1.bar(classes, train_counts, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax1.set_title("Training Set Digit Distribution (N=60,000)")
    ax1.set_xlabel("Digit Class")
    ax1.set_ylabel("Count")
    ax1.set_xticks(classes)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    ax2.bar(classes, test_counts, color="#2ca02c", edgecolor="black", alpha=0.85)
    ax2.set_title("Test Set Digit Distribution (N=10,000)")
    ax2.set_xlabel("Digit Class")
    ax2.set_ylabel("Count")
    ax2.set_xticks(classes)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    return fig

def plot_training_curves(history, title_prefix="Baseline Model", save_path=None):
    """Plots training and validation loss and accuracy trajectories."""
    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    
    # Loss
    ax1.plot(epochs, history["loss"], "o-", label="Training Loss", color="#d62728")
    ax1.plot(epochs, history["val_loss"], "s--", label="Validation Loss", color="#ff7f0e")
    ax1.set_title(f"{title_prefix} — Loss vs. Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    
    # Accuracy
    ax2.plot(epochs, history["accuracy"], "o-", label="Training Accuracy", color="#1f77b4")
    ax2.plot(epochs, history["val_accuracy"], "s--", label="Validation Accuracy", color="#2ca02c")
    ax2.set_title(f"{title_prefix} — Accuracy vs. Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    return fig

def plot_confusion_matrix_heatmap(cm, title="Confusion Matrix", save_path=None):
    """Renders an annotated heatmap of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=range(10), yticklabels=range(10), ax=ax)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Predicted Digit", fontsize=10)
    ax.set_ylabel("True Digit", fontsize=10)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    return fig

def plot_sample_predictions_with_probs(x, y_true, y_pred, probs, indices, save_path=None):
    """
    Visualizes selected samples alongside a horizontal bar chart of predicted probabilities.
    """
    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(8, 2.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
        
    for i, idx in enumerate(indices):
        img = x[idx]
        t = y_true[idx]
        p = y_pred[idx]
        prob = probs[idx]
        
        # Image
        axes[i, 0].imshow(img, cmap="gray")
        color = "green" if t == p else "red"
        axes[i, 0].set_title(f"True: {t} | Pred: {p} ({prob[p]*100:.1f}%)", color=color, fontsize=10)
        axes[i, 0].axis("off")
        
        # Probability Bar Chart
        bar_colors = ["#1f77b4" if c != p else ("#2ca02c" if t == p else "#d62728") for c in range(10)]
        axes[i, 1].barh(range(10), prob, color=bar_colors, edgecolor="black", alpha=0.85)
        axes[i, 1].set_yticks(range(10))
        axes[i, 1].set_xlim(0, 1.0)
        axes[i, 1].set_xlabel("Probability")
        axes[i, 1].set_title(f"Predicted Class Probabilities", fontsize=9)
        axes[i, 1].grid(axis="x", linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    return fig

def plot_comparison_curves(hist_base, hist_exp, save_path=None):
    """Plots baseline (128 units) vs experimental (256 units) training & validation curves."""
    epochs = range(1, len(hist_base["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(epochs, hist_base["val_loss"], "o--", label="Baseline (128) Val Loss", color="#1f77b4")
    ax1.plot(epochs, hist_exp["val_loss"], "s--", label="Experiment (256) Val Loss", color="#ff7f0e")
    ax1.set_title("Validation Loss Comparison")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    
    # Accuracy
    ax2.plot(epochs, hist_base["val_accuracy"], "o--", label="Baseline (128) Val Acc", color="#1f77b4")
    ax2.plot(epochs, hist_exp["val_accuracy"], "s--", label="Experiment (256) Val Acc", color="#2ca02c")
    ax2.set_title("Validation Accuracy Comparison")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    return fig

def plot_per_class_f1_comparison(report_base, report_exp, save_path=None):
    """Grouped bar plot comparing per-class F1-scores between baseline and experiment."""
    f1_base = [report_base[str(i)]["f1-score"] for i in range(10)]
    f1_exp = [report_exp[str(i)]["f1-score"] for i in range(10)]
    
    x = np.arange(10)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width/2, f1_base, width, label="Baseline (128 units)", color="#1f77b4", alpha=0.85)
    ax.bar(x + width/2, f1_exp, width, label="Experiment (256 units)", color="#2ca02c", alpha=0.85)
    
    ax.set_ylabel("F1-Score")
    ax.set_xlabel("Digit Class")
    ax.set_title("Per-Class F1-Score: Baseline (128) vs Experiment (256)")
    ax.set_xticks(x)
    ax.set_xticklabels(range(10))
    ax.set_ylim(0.9, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    return fig

def plot_all_experiments_summary(suite_results, suite_histories, save_path=None):
    """
    Renders a 4-panel multi-experiment comparative figure:
      Panel 1: Test Accuracy across all 11 ablation variants (horizontal bar chart).
      Panel 2: Activation Function convergence (ReLU vs Sigmoid vs Tanh validation loss).
      Panel 3: Learning Rate optimization dynamics (lr=0.01 vs 0.001 vs 0.0001).
      Panel 4: Training Time vs Test Accuracy trade-off scatter plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ------------------ Panel 1: Test Accuracy Bar Chart ------------------
    ax1 = axes[0, 0]
    names = [r["name"] for r in suite_results]
    accs = [r["test_accuracy"] * 100 for r in suite_results]
    y_pos = np.arange(len(names))
    
    # Highlight baseline in blue, variants in green/coral
    colors = ["#1f77b4" if r["id"] == "baseline" else "#2ca02c" if r["test_accuracy"] >= suite_results[0]["test_accuracy"] else "#e377c2" for r in suite_results]
    bars = ax1.barh(y_pos, accs, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlim(min(accs) - 2.0, 100.0)
    ax1.set_xlabel("Test Accuracy (%)", fontsize=10)
    ax1.set_title("1. Test Accuracy Across All 11 Ablation Variants", fontsize=11, fontweight="bold")
    ax1.axvline(accs[0], color="#d62728", linestyle="--", linewidth=1.5, label=f"Baseline Reference ({accs[0]:.2f}%)")
    ax1.legend(loc="lower left", fontsize=9)
    ax1.grid(axis="x", linestyle="--", alpha=0.6)
    
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 0.1, bar.get_y() + bar.get_height()/2, f"{w:.2f}%", va="center", fontsize=8)
        
    # ------------------ Panel 2: Activation Functions (ReLU vs Sigmoid vs Tanh) ------------------
    ax2 = axes[0, 1]
    act_keys = [("baseline", "ReLU (Baseline)", "#1f77b4"),
                ("exp6_act_sigmoid", "Sigmoid (Saturating)", "#d62728"),
                ("exp6_act_tanh", "Tanh (Zero-Centered)", "#2ca02c")]
    for key, label, col in act_keys:
        if key in suite_histories:
            ep = range(1, len(suite_histories[key]["val_loss"]) + 1)
            ax2.plot(ep, suite_histories[key]["val_loss"], "o-", label=label, color=col, alpha=0.9)
    ax2.set_title("2. Activation Function Convergence (Validation Loss)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Validation Loss", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)
    
    # ------------------ Panel 3: Learning Rate Dynamics ------------------
    ax3 = axes[1, 0]
    lr_keys = [("baseline", "lr = 0.001 (Baseline)", "#1f77b4"),
               ("exp3_lr_high", "lr = 0.01 (High / Aggressive)", "#d62728"),
               ("exp3_lr_low", "lr = 0.0001 (Low / Conservative)", "#ff7f0e")]
    for key, label, col in lr_keys:
        if key in suite_histories:
            ep = range(1, len(suite_histories[key]["val_loss"]) + 1)
            ax3.plot(ep, suite_histories[key]["val_loss"], "s--", label=label, color=col, alpha=0.9)
    ax3.set_title("3. Learning Rate Optimization Stability (Validation Loss)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Epoch", fontsize=10)
    ax3.set_ylabel("Validation Loss", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(fontsize=9)
    
    # ------------------ Panel 4: Accuracy vs Computational Time ------------------
    ax4 = axes[1, 1]
    times = [r["training_time_sec"] for r in suite_results]
    for r in suite_results:
        c = "#1f77b4" if r["id"] == "baseline" else "#2ca02c" if r["test_accuracy"] >= suite_results[0]["test_accuracy"] else "#e377c2"
        ax4.scatter(r["training_time_sec"], r["test_accuracy"] * 100, s=r["total_params"] / 1500, color=c, edgecolors="black", alpha=0.8)
        ax4.annotate(r["id"].replace("exp", "E").replace("_", " "), (r["training_time_sec"] + 0.3, r["test_accuracy"] * 100), fontsize=7)
        
    ax4.set_title("4. Pareto Frontier: Training Time vs. Accuracy (Bubble Size = Params)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Wall-Clock Training Time (seconds)", fontsize=10)
    ax4.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    return fig

