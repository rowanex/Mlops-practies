from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


LAB_DIR = Path(__file__).parent
R2_THRESHOLD = 0.85


def load_model():
    return joblib.load(LAB_DIR / "linear_model.joblib")


def load_dataset(filename):
    return pd.read_csv(LAB_DIR / filename)


def evaluate(model, data):
    X = data[["x"]]
    y_true = data["y"]
    y_pred = model.predict(X)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, r2


def test_model_quality_on_clean_1():
    model = load_model()
    data = load_dataset("clean_1.csv")
    mse, r2 = evaluate(model, data)

    assert r2 >= R2_THRESHOLD, f"Качество на clean_1 слишком низкое: R2={r2:.3f}, MSE={mse:.3f}"


def test_model_quality_on_clean_2():
    model = load_model()
    data = load_dataset("clean_2.csv")
    mse, r2 = evaluate(model, data)

    assert r2 >= R2_THRESHOLD, f"Качество на clean_2 слишком низкое: R2={r2:.3f}, MSE={mse:.3f}"


def test_model_quality_on_clean_3():
    model = load_model()
    data = load_dataset("clean_3.csv")
    mse, r2 = evaluate(model, data)

    assert r2 >= R2_THRESHOLD, f"Качество на clean_3 слишком низкое: R2={r2:.3f}, MSE={mse:.3f}"


def test_noisy_dataset_problem_is_detected():
    model = load_model()
    data = load_dataset("noisy.csv")
    mse, r2 = evaluate(model, data)

    assert r2 < R2_THRESHOLD, (
        f"Проблема с шумом не обнаружена: R2={r2:.3f}, MSE={mse:.3f}. "
        f"Ожидалось, что качество будет ниже порога {R2_THRESHOLD}."
    )
