import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix

LABELS = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    result = {
        "train_acc": accuracy_score(y_train, model.predict(X_train)),
        "val_acc": accuracy_score(y_val, model.predict(X_val)),
        "test_acc": accuracy_score(y_test, model.predict(X_test)),
    }
    try:
        result["test_logloss"] = log_loss(y_test, model.predict_proba(X_test), labels=list(range(len(LABELS))))
    except Exception:
        result["test_logloss"] = None
    result["overfit_gap_train_minus_test"] = result["train_acc"] - result["test_acc"]
    return result

def measure_inference_speed(model, X_test, repeats=30, warmup=5):
    for _ in range(warmup):
        model.predict(X_test)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(X_test)
        times.append(time.perf_counter() - start)
    avg = float(np.mean(times))
    return {
        "avg_total_ms": avg * 1000,
        "avg_per_sample_us": avg / len(X_test) * 1_000_000,
        "samples_per_second": len(X_test) / avg,
    }

def save_classification_files(model_name, model, X_test, y_test, output_dir):
    from pathlib import Path
    output_dir = Path(output_dir)
    y_pred = model.predict(X_test)
    text = classification_report(y_test, y_pred, target_names=LABELS, digits=4)
    (output_dir / f"{model_name}_classification_report.txt").write_text(text, encoding="utf-8")
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(LABELS))))
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
        output_dir / f"{model_name}_confusion_matrix.csv", encoding="utf-8-sig"
    )
