"""
Stand-alone end-to-end execution pipeline.
Trains the comprehensive 6-dimensional ablation suite (11 variants), evaluates metrics,
and exports all diagnostic charts and results JSONs.
"""
import os
import json
from src import config
from src.experiment import run_all_experiments
from src.visualize import (
    plot_sample_digits,
    plot_class_distributions,
    plot_training_curves,
    plot_confusion_matrix_heatmap,
    plot_sample_predictions_with_probs,
    plot_comparison_curves,
    plot_per_class_f1_comparison,
    plot_all_experiments_summary
)
from src.evaluate import find_confident_errors, find_most_uncertain
from src.data import load_and_preprocess_data

def main():
    print("=" * 70)
    print("  TASK 2: MNIST NEURAL NETWORK PIPELINE (TENSORFLOW/KERAS)")
    print("  Author: Aryan Singh")
    print("=" * 70)
    
    config.ensure_directories()
    
    # 1. Dataset Loading & Exploration Visualizations
    print("\n[Step 1/4] Loading canonical MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    print("Generating dataset exploratory artifacts in outputs/figures/...")
    plot_sample_digits(x_train, y_train, n=10, save_path=os.path.join(config.FIGURES_DIR, "part_a_sample_digits.png"))
    plot_class_distributions(y_train, y_test, save_path=os.path.join(config.FIGURES_DIR, "part_a_class_distributions.png"))
    
    # 2. Execute Full 11-Variant 6-Dimensional Ablation Suite (Single Authoritative Runner)
    print("\n[Step 2/4] Executing 6-Dimensional Controlled Ablation Suite (11 Variants)...")
    suite_data = run_all_experiments(verbose=1)
    suite_res = suite_data["suite_results"]
    suite_hist = suite_data["suite_histories"]
    models_dict = suite_data["models_dict"]
    
    # Extract Baseline components
    base_bundle = models_dict["baseline"]
    base_preds = base_bundle["preds"]
    base_probs = base_bundle["probs"]
    base_eval = base_bundle["metrics"]
    base_hist = suite_hist["baseline"]
    
    # Extract 256-unit experiment components for Part G width comparison curves
    exp_width_bundle = models_dict["exp2_width_256"]
    exp_width_eval = exp_width_bundle["metrics"]
    exp_width_hist = suite_hist["exp2_width_256"]
    
    # 3. Generate Baseline & Comparative Diagnostic Visualizations
    print("\n[Step 3/4] Generating model evaluation diagnostics...")
    
    # Part E: Baseline training dynamics
    plot_training_curves(base_hist, title_prefix="Baseline (128 units)", save_path=os.path.join(config.FIGURES_DIR, "part_e_baseline_training_curves.png"))
    
    # Part F: Baseline Confusion Matrix
    cm_base = base_eval["confusion_matrix"]
    plot_confusion_matrix_heatmap(cm_base, title="Baseline (128 units) Confusion Matrix", save_path=os.path.join(config.FIGURES_DIR, "part_f_baseline_confusion_matrix.png"))
    
    # Sample predictions with probability distributions
    sample_indices = [0, 1, 2, 3, 4]
    plot_sample_predictions_with_probs(x_test, y_test, base_preds, base_probs, sample_indices, save_path=os.path.join(config.FIGURES_DIR, "advanced_sample_predictions.png"))
    
    # Confident Errors & Most Uncertain Cases
    conf_error_indices = find_confident_errors(y_test, base_preds, base_probs, threshold=0.90)
    if conf_error_indices:
        plot_sample_predictions_with_probs(x_test, y_test, base_preds, base_probs, conf_error_indices[:5], save_path=os.path.join(config.FIGURES_DIR, "advanced_high_confidence_errors.png"))
        
    uncertain_indices = find_most_uncertain(base_probs, n=5)
    plot_sample_predictions_with_probs(x_test, y_test, base_preds, base_probs, uncertain_indices, save_path=os.path.join(config.FIGURES_DIR, "advanced_most_uncertain_predictions.png"))
    
    # Width Capacity Comparison (Baseline vs 256)
    plot_comparison_curves(base_hist, exp_width_hist, save_path=os.path.join(config.FIGURES_DIR, "part_g_comparison_curves.png"))
    plot_per_class_f1_comparison(base_eval["classification_report"], exp_width_eval["classification_report"], save_path=os.path.join(config.FIGURES_DIR, "part_g_per_class_f1_comparison.png"))
    cm_exp = exp_width_eval["confusion_matrix"]
    plot_confusion_matrix_heatmap(cm_exp, title="Experiment (256 units) Confusion Matrix", save_path=os.path.join(config.FIGURES_DIR, "part_g_experiment_confusion_matrix.png"))
    
    # 4. Multi-Experiment Suite Visual Summary & Terminal Reporting
    print("\n[Step 4/4] Generating 6-dimensional ablation suite summary...")
    plot_all_experiments_summary(
        suite_res, suite_hist,
        save_path=os.path.join(config.FIGURES_DIR, "experiment_suite_comparison.png")
    )
    
    # Print Multi-Experiment Comparison Table
    print("\n" + "=" * 90)
    print("        COMPREHENSIVE 6-DIMENSIONAL ABLATION EXPERIMENTAL SUITE RESULTS")
    print("=" * 90)
    suite_headers = ["Ablation Dimension / Variant", "Params", "Time (s)", "Test Acc", "Test Loss", "Macro F1", "Train-Val Gap"]
    suite_fmt = "{:<40} | {:<8} | {:<8} | {:<9} | {:<9} | {:<8} | {:<12}"
    print(suite_fmt.format(*suite_headers))
    print("-" * 90)
    for r in suite_res:
        print(suite_fmt.format(
            r["name"],
            f"{r['total_params']:,}",
            f"{r['training_time_sec']:.1f}s",
            f"{r['test_accuracy']*100:.2f}%",
            f"{r['test_loss']:.4f}",
            f"{r['macro_f1']:.4f}",
            f"{r['train_val_gap']*100:.2f}%"
        ))
    print("=" * 90)
    
    print("\n✓ Pipeline execution complete. All 11 ablation models, figures, and manifests saved successfully.")

if __name__ == "__main__":
    main()
