import numpy as np
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model, get_predictions_and_probabilities, compute_metrics, find_top_confused_pairs

def test_train_and_evaluate_smoke():
    # Small synthetic dummy data: 50 samples
    x_dummy = np.random.rand(50, 28, 28).astype(np.float32)
    y_dummy = np.random.randint(0, 10, size=(50,)).astype(np.int32)
    
    model = build_model(hidden_units=32)
    history, train_time = train_model(
        model, x_dummy, y_dummy,
        epochs=1, batch_size=16, validation_split=0.2, verbose=0
    )
    
    assert "loss" in history
    assert "val_loss" in history
    assert train_time > 0
    
    test_loss, test_acc = evaluate_model(model, x_dummy, y_dummy)
    assert isinstance(test_loss, float)
    assert isinstance(test_acc, float)
    
    preds, probs = get_predictions_and_probabilities(model, x_dummy)
    assert preds.shape == (50,)
    assert probs.shape == (50, 10)
    
    metrics = compute_metrics(y_dummy, preds)
    assert "macro_f1" in metrics
    assert len(metrics["confusion_matrix"]) == 10
    
    top_pairs = find_top_confused_pairs(metrics["confusion_matrix"], n=2)
    assert isinstance(top_pairs, list)
