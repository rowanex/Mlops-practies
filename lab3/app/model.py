from __future__ import annotations

from dataclasses import dataclass

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


@dataclass
class PredictionResult:
    predicted_class: str
    probabilities: dict[str, float]


class IrisModelService:
    def __init__(self) -> None:
        dataset = load_iris()
        self.feature_names = [str(name) for name in dataset.feature_names]
        self.class_names = [str(name) for name in dataset.target_names]

        self._model = LogisticRegression(max_iter=300)
        self._model.fit(dataset.data, dataset.target)

    def predict(self, features: list[float]) -> PredictionResult:
        probabilities = self._model.predict_proba([features])[0]
        predicted_index = int(self._model.predict([features])[0])

        return PredictionResult(
            predicted_class=str(self.class_names[predicted_index]),
            probabilities={
                str(class_name): round(float(probability), 4)
                for class_name, probability in zip(self.class_names, probabilities)
            },
        )
