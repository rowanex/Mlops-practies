"""
Stage 3: Model Training
Trains a GradientBoostingClassifier on the Titanic train set.
Saves the model in pickle format.
"""
import os
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

DATA_DIR  = "data"
MODEL_DIR = "model"

def train_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("[Stage 3] Loading training data...")
    with open(os.path.join(DATA_DIR, "train.pkl"), "rb") as f:
        X_train, y_train = pickle.load(f)

    with open(os.path.join(DATA_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)

    print("[Stage 3] Training GradientBoostingClassifier...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"[Stage 3] Train accuracy: {train_acc:.4f}")

    print("[Stage 3] Feature importances:")
    for feat, imp in sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    ):
        print(f"  {feat:20s}: {imp:.4f}")

    model_path = os.path.join(MODEL_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[Stage 3] Model saved to '{model_path}'")

if __name__ == "__main__":
    train_model()
