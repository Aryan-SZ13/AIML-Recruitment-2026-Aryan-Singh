"""
Stand-alone end-to-end execution pipeline.
Trains models, evaluates metrics, runs controlled capacity experiment,
and exports all diagnostic charts and results JSONs.
"""
import os
import json
from src import config
from src.experiment import run_controlled_experiment, run_all_experiments
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
    
    # 1. Run controlled experiment
    results = run_controlled_experiment(verbose=1)
    
    # 2. Extract components
    comp = results["comparison"]
    b_info = comp["baseline"]
    e_info = comp["experiment"]
    x_test = results["x_test"]
    y_test = results["y_test"]
    base_preds = results["baseline_preds"]
    base_probs = results["baseline_probs"]
    
    # Load raw images for plotting distribution
    (x_train, y_train), _ = load_and_preprocess_data()
    
    # 3. Print Results Summary Table
    print("\n" + "=" * 70)
    print("                    CONTROLLED EXPERIMENT SUMMARY")
    print("=" * 70)
    headers = ["Metric / Dimension", f"Baseline ({b_info['hidden_units']} units)", f"Experiment ({e_info['hidden_units']} units)"]
    row_fmt = "{:<32} | {:<16} | {:<16}"
    print(row_fmt.format(*headers))
    print("-" * 70)
    print(row_fmt.format("Total Trainable Parameters", str(b_info["total_params"]), str(e_info["total_params"])))
    print(row_fmt.format("Training Time (seconds)", f"{b_info['training_time_sec']}s", f"{e_info['training_time_sec']}s"))
    print(row_fmt.format("Test Accuracy", f"{b_info['test_accuracy']*100:.2f}%", f"{e_info['test_accuracy']*100:.2f}%"))
    print(row_fmt.format("Test Loss", str(b_info["test_loss"]), str(e_info["test_loss"])))
    print(row_fmt.format("Macro F1-Score", str(b_info["macro_f1"]), str(e_info["macro_f1"])))
    print(row_fmt.format("Final Train Accuracy", f"{b_info['final_train_acc']*100:.2f}%", f"{e_info['final_train_acc']*100:.2f}%"))
    print(row_fmt.format("Final Val Accuracy", f"{b_info['final_val_acc']*100:.2f}%", f"{e_info['final_val_acc']*100:.2f}%"))
    print(row_fmt.format("Train-Val Accuracy Gap", f"{b_info['train_val_gap']*100:.2f}%", f"{e_info['train_val_gap']*100:.2f}%"))
    print(row_fmt.format("Accuracy Efficiency (Exploratory)*", str(b_info["accuracy_efficiency_exploratory"]), str(e_info["accuracy_efficiency_exploratory"])))
    print("-" * 70)
    print("*Exploratory measure defined as: Test Accuracy (%) / (Total Parameters / 10,000)")
    print(" Note: Higher means more accuracy per parameter slice; baseline has lower parameter overhead.")
    
    # 4. Generate and save all figures
    print("\nGenerating visual artifacts in outputs/figures/...")
    
    # Part A: Dataset visualization & class distribution
    plot_sample_digits(x_train, y_train, n=10, save_path=os.path.join(config.FIGURES_DIR, "part_a_sample_digits.png"))
    plot_class_distributions(y_train, y_test, save_path=os.path.join(config.FIGURES_DIR, "part_a_class_distributions.png"))
    
    # Part E: Baseline training dynamics
    plot_training_curves(results["baseline_hist"], title_prefix="Baseline (128 units)", save_path=os.path.join(config.FIGURES_DIR, "part_e_baseline_training_curves.png"))
    
    # Part F: Baseline Confusion Matrix
    cm_base = results["baseline_eval"]["confusion_matrix"]
    plot_confusion_matrix_heatmap(cm_base, title="Baseline (128 units) Confusion Matrix", save_path=os.path.join(config.FIGURES_DIR, "part_f_baseline_confusion_matrix.png"))
    
    # Advanced Analysis: Sample predictions with probability bars
    sample_indices = [0, 1, 2, 3, 4]
    plot_sample_predictions_with_probs(x_test, y_test, base_preds, base_probs, sample_indices, save_path=os.path.join(config.FIGURES_DIR, "advanced_sample_predictions.png"))
    
    # Advanced Analysis: Confident Errors
    conf_error_indices = find_confident_errors(y_test, base_preds, base_probs, threshold=0.90)
    if conf_error_indices:
        plot_sample_predictions_with_probs(x_test, y_test, base_preds, base_probs, conf_error_indices[:5], save_path=os.path.join(config.FIGURES_DIR, "advanced_high_confidence_errors.png"))
        
    # Advanced Analysis: Most Uncertain
    uncertain_indices = find_most_uncertain(base_probs, n=5)
    plot_sample_predictions_with_probs(x_test, y_test, base_preds, base_probs, uncertain_indices, save_path=os.path.join(config.FIGURES_DIR, "advanced_most_uncertain_predictions.png"))
    
    # Part G: Experiment comparisons
    plot_comparison_curves(results["baseline_hist"], results["experiment_hist"], save_path=os.path.join(config.FIGURES_DIR, "part_g_comparison_curves.png"))
    plot_per_class_f1_comparison(results["baseline_eval"]["classification_report"], results["experiment_eval"]["classification_report"], save_path=os.path.join(config.FIGURES_DIR, "part_g_per_class_f1_comparison.png"))
    cm_exp = results["experiment_eval"]["confusion_matrix"]
    plot_confusion_matrix_heatmap(cm_exp, title="Experiment (256 units) Confusion Matrix", save_path=os.path.join(config.FIGURES_DIR, "part_g_experiment_confusion_matrix.png"))
    
    # 5. Run Comprehensive 6-Dimensional Ablation Suite
    suite_data = run_all_experiments(verbose=1)
    suite_res = suite_data["suite_results"]
    suite_hist = suite_data["suite_histories"]
    
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
    
    # Plot Multi-Experiment Visual Summary
    plot_all_experiments_summary(
        suite_res, suite_hist,
        save_path=os.path.join(config.FIGURES_DIR, "experiment_suite_comparison.png")
    )
    
    print("\n✓ Pipeline execution complete. All models, figures, and manifests saved successfully.")

if __name__ == "__main__":
    main()
