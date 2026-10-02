"""
Controlled experiment comparing baseline (128 units) vs experiment (256 units).
Saves experiment manifest and structured comparative metrics.
"""
import os
import json
from src import config
from src.data import load_and_preprocess_data
from src.model import build_model, count_model_parameters
from src.train import set_seed, train_model, save_trained_model
from src.evaluate import (
    evaluate_model,
    get_predictions_and_probabilities,
    compute_metrics,
    find_top_confused_pairs
)

def run_controlled_experiment(verbose=1):
    """
    Executes a strictly controlled experiment comparing hidden units: 128 vs 256.
    All seeds, data splits, optimizers, learning rates, batch sizes, and epochs are held constant.
    """
    config.ensure_directories()
    
    # 1. Save experiment manifest
    manifest = {
        "author": "Aryan Singh",
        "task": "Task 2 — Neural Network (Controlled Experiment)",
        "seed": config.SEED,
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "validation_split": config.VALIDATION_SPLIT,
        "loss_function": "sparse_categorical_crossentropy",
        "optimizer": "Adam",
        "baseline_hidden_units": config.BASELINE_UNITS,
        "experiment_hidden_units": config.EXPERIMENT_UNITS,
        "input_shape": list(config.INPUT_SHAPE),
        "num_classes": config.NUM_CLASSES
    }
    with open(config.MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=4)
        
    # 2. Data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # 3. Baseline Model (128 units)
    if verbose:
        print(f"\n{'='*25} TRAINING BASELINE MODEL ({config.BASELINE_UNITS} UNITS) {'='*25}")
    set_seed(config.SEED)
    baseline_model = build_model(
        hidden_units=config.BASELINE_UNITS,
        input_shape=config.INPUT_SHAPE,
        num_classes=config.NUM_CLASSES,
        learning_rate=config.LEARNING_RATE
    )
    baseline_params = count_model_parameters(baseline_model)
    base_hist, base_time = train_model(
        baseline_model, x_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        verbose=verbose
    )
    base_loss, base_acc = evaluate_model(baseline_model, x_test, y_test)
    base_preds, base_probs = get_predictions_and_probabilities(baseline_model, x_test)
    base_eval = compute_metrics(y_test, base_preds)
    base_top_confused = find_top_confused_pairs(base_eval["confusion_matrix"], n=3)
    
    # Save baseline model
    base_model_path = os.path.join(config.MODELS_DIR, "baseline_model.keras")
    save_trained_model(baseline_model, base_model_path)
    
    # 4. Experimental Model (256 units)
    if verbose:
        print(f"\n{'='*25} TRAINING EXPERIMENTAL MODEL ({config.EXPERIMENT_UNITS} UNITS) {'='*25}")
    set_seed(config.SEED)
    exp_model = build_model(
        hidden_units=config.EXPERIMENT_UNITS,
        input_shape=config.INPUT_SHAPE,
        num_classes=config.NUM_CLASSES,
        learning_rate=config.LEARNING_RATE
    )
    exp_params = count_model_parameters(exp_model)
    exp_hist, exp_time = train_model(
        exp_model, x_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        verbose=verbose
    )
    exp_loss, exp_acc = evaluate_model(exp_model, x_test, y_test)
    exp_preds, exp_probs = get_predictions_and_probabilities(exp_model, x_test)
    exp_eval = compute_metrics(y_test, exp_preds)
    exp_top_confused = find_top_confused_pairs(exp_eval["confusion_matrix"], n=3)
    
    # Save experiment model
    exp_model_path = os.path.join(config.MODELS_DIR, "experiment_model.keras")
    save_trained_model(exp_model, exp_model_path)
    
    # 5. Exploratory Efficiency metric: Accuracy Efficiency = Test Accuracy (%) / (Total Parameters / 10,000)
    base_eff = (base_acc * 100.0) / (baseline_params["total_params"] / 10000.0)
    exp_eff = (exp_acc * 100.0) / (exp_params["total_params"] / 10000.0)
    
    # 6. Structured Comparison Summary
    comparison = {
        "manifest": manifest,
        "baseline": {
            "hidden_units": config.BASELINE_UNITS,
            "total_params": baseline_params["total_params"],
            "training_time_sec": round(base_time, 2),
            "test_loss": round(base_loss, 4),
            "test_accuracy": round(base_acc, 4),
            "macro_f1": round(base_eval["macro_f1"], 4),
            "final_train_acc": round(base_hist["accuracy"][-1], 4),
            "final_val_acc": round(base_hist["val_accuracy"][-1], 4),
            "train_val_gap": round(base_hist["accuracy"][-1] - base_hist["val_accuracy"][-1], 4),
            "accuracy_efficiency_exploratory": round(base_eff, 4),
            "top_confused_pairs": base_top_confused
        },
        "experiment": {
            "hidden_units": config.EXPERIMENT_UNITS,
            "total_params": exp_params["total_params"],
            "training_time_sec": round(exp_time, 2),
            "test_loss": round(exp_loss, 4),
            "test_accuracy": round(exp_acc, 4),
            "macro_f1": round(exp_eval["macro_f1"], 4),
            "final_train_acc": round(exp_hist["accuracy"][-1], 4),
            "final_val_acc": round(exp_hist["val_accuracy"][-1], 4),
            "train_val_gap": round(exp_hist["accuracy"][-1] - exp_hist["val_accuracy"][-1], 4),
            "accuracy_efficiency_exploratory": round(exp_eff, 4),
            "top_confused_pairs": exp_top_confused
        }
    }
    
    # Save results
    with open(os.path.join(config.RESULTS_DIR, "comparison_summary.json"), "w") as f:
        json.dump(comparison, f, indent=4)
        
    with open(os.path.join(config.RESULTS_DIR, "baseline_history.json"), "w") as f:
        json.dump(base_hist, f, indent=4)
        
    with open(os.path.join(config.RESULTS_DIR, "experiment_history.json"), "w") as f:
        json.dump(exp_hist, f, indent=4)
        
    with open(os.path.join(config.RESULTS_DIR, "baseline_eval.json"), "w") as f:
        json.dump(base_eval, f, indent=4)
        
    with open(os.path.join(config.RESULTS_DIR, "experiment_eval.json"), "w") as f:
        json.dump(exp_eval, f, indent=4)
        
    return {
        "baseline_model": baseline_model,
        "experiment_model": exp_model,
        "baseline_hist": base_hist,
        "experiment_hist": exp_hist,
        "baseline_eval": base_eval,
        "experiment_eval": exp_eval,
        "comparison": comparison,
        "x_test": x_test,
        "y_test": y_test,
        "baseline_preds": base_preds,
        "baseline_probs": base_probs,
        "experiment_preds": exp_preds,
        "experiment_probs": exp_probs
    }

def run_all_experiments(verbose=1):
    """
    Executes a comprehensive 6-dimensional ablation study where each experiment modifies
    exactly ONE aspect of the canonical baseline model or training configuration:
      1. Hidden Layers (Depth): 1 Layer (128) vs 2 Layers (128 -> 64)
      2. Number of Neurons (Width): 128 vs 256 units
      3. Learning Rate: 0.001 vs 0.01 vs 0.0001
      4. Batch Size: 128 vs 32 vs 512
      5. Number of Epochs: 15 vs 5 vs 30
      6. Activation Function: relu vs sigmoid vs tanh
    """
    config.ensure_directories()
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    variants = [
        {
            "id": "baseline",
            "dimension": "Baseline Control",
            "name": "Baseline (128 units, ReLU, lr=1e-3, B=128, E=15)",
            "hidden_units": 128, "activation": "relu", "lr": 0.001, "batch_size": 128, "epochs": 15
        },
        {
            "id": "exp1_depth_2layers",
            "dimension": "1. Hidden Layers (Depth)",
            "name": "2 Hidden Layers (128 -> 64)",
            "hidden_units": [128, 64], "activation": "relu", "lr": 0.001, "batch_size": 128, "epochs": 15
        },
        {
            "id": "exp2_width_256",
            "dimension": "2. Number of Neurons (Width)",
            "name": "256 Units (Expanded Capacity)",
            "hidden_units": 256, "activation": "relu", "lr": 0.001, "batch_size": 128, "epochs": 15
        },
        {
            "id": "exp3_lr_high",
            "dimension": "3. Learning Rate",
            "name": "Aggressive LR (0.01)",
            "hidden_units": 128, "activation": "relu", "lr": 0.01, "batch_size": 128, "epochs": 15
        },
        {
            "id": "exp3_lr_low",
            "dimension": "3. Learning Rate",
            "name": "Conservative LR (0.0001)",
            "hidden_units": 128, "activation": "relu", "lr": 0.0001, "batch_size": 128, "epochs": 15
        },
        {
            "id": "exp4_batch_32",
            "dimension": "4. Batch Size",
            "name": "Small Batch / High Noise (B=32)",
            "hidden_units": 128, "activation": "relu", "lr": 0.001, "batch_size": 32, "epochs": 15
        },
        {
            "id": "exp4_batch_512",
            "dimension": "4. Batch Size",
            "name": "Large Batch / Smooth Gradient (B=512)",
            "hidden_units": 128, "activation": "relu", "lr": 0.001, "batch_size": 512, "epochs": 15
        },
        {
            "id": "exp5_epochs_5",
            "dimension": "5. Number of Epochs",
            "name": "Short Budget (5 Epochs)",
            "hidden_units": 128, "activation": "relu", "lr": 0.001, "batch_size": 128, "epochs": 5
        },
        {
            "id": "exp5_epochs_30",
            "dimension": "5. Number of Epochs",
            "name": "Extended Budget (30 Epochs)",
            "hidden_units": 128, "activation": "relu", "lr": 0.001, "batch_size": 128, "epochs": 30
        },
        {
            "id": "exp6_act_sigmoid",
            "dimension": "6. Activation Function",
            "name": "Sigmoid (Saturating Gradient)",
            "hidden_units": 128, "activation": "sigmoid", "lr": 0.001, "batch_size": 128, "epochs": 15
        },
        {
            "id": "exp6_act_tanh",
            "dimension": "6. Activation Function",
            "name": "Tanh (Zero-Centered)",
            "hidden_units": 128, "activation": "tanh", "lr": 0.001, "batch_size": 128, "epochs": 15
        }
    ]
    
    suite_results = []
    suite_histories = {}
    models_dict = {}
    
    if verbose:
        print(f"\n{'='*70}")
        print("  EXECUTING 6-DIMENSIONAL ABLATION EXPERIMENTAL SUITE (11 VARIANTS)")
        print(f"{'='*70}")
        
    for idx, v in enumerate(variants, 1):
        if verbose:
            print(f"[{idx:02d}/11] Running {v['dimension']}: {v['name']}...")
            
        set_seed(config.SEED)
        model = build_model(
            hidden_units=v["hidden_units"],
            input_shape=config.INPUT_SHAPE,
            num_classes=config.NUM_CLASSES,
            activation=v["activation"],
            learning_rate=v["lr"]
        )
        params = count_model_parameters(model)
        
        hist, train_time = train_model(
            model, x_train, y_train,
            epochs=v["epochs"],
            batch_size=v["batch_size"],
            validation_split=config.VALIDATION_SPLIT,
            verbose=0
        )
        
        loss, acc = evaluate_model(model, x_test, y_test)
        preds, probs = get_predictions_and_probabilities(model, x_test)
        metrics = compute_metrics(y_test, preds)
        
        # Save key models
        if v["id"] in ["baseline", "exp1_depth_2layers", "exp2_width_256", "exp6_act_sigmoid"]:
            model_path = os.path.join(config.MODELS_DIR, f"{v['id']}_model.keras")
            save_trained_model(model, model_path)
            
        row = {
            "id": v["id"],
            "dimension": v["dimension"],
            "name": v["name"],
            "hidden_units": str(v["hidden_units"]),
            "activation": v["activation"],
            "learning_rate": v["lr"],
            "batch_size": v["batch_size"],
            "epochs": v["epochs"],
            "total_params": params["total_params"],
            "training_time_sec": round(train_time, 2),
            "test_loss": round(loss, 4),
            "test_accuracy": round(acc, 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "final_train_acc": round(hist["accuracy"][-1], 4),
            "final_val_acc": round(hist["val_accuracy"][-1], 4),
            "train_val_gap": round(hist["accuracy"][-1] - hist["val_accuracy"][-1], 4)
        }
        suite_results.append(row)
        suite_histories[v["id"]] = hist
        models_dict[v["id"]] = {
            "model": model,
            "metrics": metrics,
            "loss": loss,
            "accuracy": acc,
            "preds": preds,
            "probs": probs
        }
        
        if verbose:
            print(f"       -> Test Acc: {acc*100:.2f}% | Test Loss: {loss:.4f} | F1: {metrics['macro_f1']:.4f} | Time: {train_time:.1f}s")
            
    # Persist suite manifest
    with open(config.SUITE_MANIFEST_PATH, "w") as f:
        json.dump({
            "seed": config.SEED,
            "total_variants": len(suite_results),
            "results": suite_results
        }, f, indent=4)
        
    with open(os.path.join(config.RESULTS_DIR, "suite_histories.json"), "w") as f:
        json.dump(suite_histories, f, indent=4)
        
    return {
        "suite_results": suite_results,
        "suite_histories": suite_histories,
        "models_dict": models_dict,
        "x_test": x_test,
        "y_test": y_test
    }
