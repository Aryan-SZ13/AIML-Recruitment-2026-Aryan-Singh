"""
Model evaluation, classification metrics, confusion analysis, and error diagnosis.
"""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, x_test, y_test, verbose=0):
    """
    Computes test loss and test accuracy.
    """
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=verbose)
    return float(test_loss), float(test_acc)

def get_predictions_and_probabilities(model, x_test):
    """
    Generates predicted probabilities and predicted class IDs.
    """
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    return predictions, probabilities

def compute_metrics(y_true, y_pred, num_classes=10):
    """
    Computes full classification report and confusion matrix.
    Explicitly uses labels=list(range(num_classes)) to ensure complete 10x10 matrix
    and zero_division=0 to gracefully handle unpredicted classes.
    """
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)
    return {
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"])
    }

def find_top_confused_pairs(cm, n=5):
    """
    Finds the top n most confused off-diagonal pairs (True Class, Predicted Class, Count).
    Derived purely from observed confusion matrix.
    """
    cm_arr = np.array(cm)
    # Mask diagonal
    off_diag = cm_arr.copy()
    np.fill_diagonal(off_diag, 0)
    
    # Sort indices descending
    indices = np.argsort(off_diag.flatten())[::-1]
    top_pairs = []
    for idx in indices[:n]:
        true_c, pred_c = np.unravel_index(idx, off_diag.shape)
        count = int(off_diag[true_c, pred_c])
        if count > 0:
            top_pairs.append({
                "true_class": int(true_c),
                "pred_class": int(pred_c),
                "count": count
            })
    return top_pairs

def find_confident_errors(y_true, y_pred, probabilities, threshold=0.9):
    """
    Finds sample indices where prediction was incorrect but confidence >= threshold.
    """
    confidences = np.max(probabilities, axis=1)
    incorrect_mask = (y_true != y_pred)
    confident_mask = (confidences >= threshold)
    indices = np.where(incorrect_mask & confident_mask)[0]
    # Sort by descending confidence
    sorted_indices = indices[np.argsort(confidences[indices])[::-1]]
    return sorted_indices.tolist()

def find_most_uncertain(probabilities, n=10):
    """
    Finds indices where prediction confidence (max probability) is lowest.
    """
    confidences = np.max(probabilities, axis=1)
    indices = np.argsort(confidences)[:n]
    return indices.tolist()
