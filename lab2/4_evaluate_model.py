"""
Stage 4: Model Evaluation
Loads the saved model, runs predictions on the test set,
prints metrics and saves an evaluation report (JSON).
Pipeline fails if accuracy < THRESHOLD.
"""
import os
import json
import pickle
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

DATA_DIR  = "data"
MODEL_DIR = "model"
THRESHOLD = 0.78

def evaluate_model():
    print("[Stage 4] Loading model...")
    with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    print("[Stage 4] Loading test data...")
    with open(os.path.join(DATA_DIR, "test.pkl"), "rb") as f:
        X_test, y_test = pickle.load(f)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_proba)
    report   = classification_report(
        y_test, y_pred,
        target_names=["Not Survived", "Survived"],
        output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n[Stage 4] Test Accuracy : {accuracy:.4f}")
    print(f"[Stage 4] ROC-AUC Score : {roc_auc:.4f}")
    print("\n[Stage 4] Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Not Survived", "Survived"]
    ))
    print(f"[Stage 4] Confusion Matrix:\n{cm}")

    result = {
        "accuracy":              round(accuracy, 4),
        "roc_auc":               round(roc_auc, 4),
        "classification_report": report,
        "confusion_matrix":      cm
    }
    report_path = os.path.join(MODEL_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\n[Stage 4] Evaluation report saved to '{report_path}'")

    if accuracy < THRESHOLD:
        raise ValueError(
            f"[Stage 4] FAILED: accuracy {accuracy:.4f} < threshold {THRESHOLD}"
        )
    print(f"[Stage 4] Quality check PASSED (accuracy >= {THRESHOLD})")

if __name__ == "__main__":
    evaluate_model()
