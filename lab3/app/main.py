from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.model import IrisModelService


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, example=5.1)
    sepal_width: float = Field(..., gt=0, example=3.5)
    petal_length: float = Field(..., gt=0, example=1.4)
    petal_width: float = Field(..., gt=0, example=0.2)


model_service = IrisModelService()
app = FastAPI(
    title="Iris ML Microservice",
    description="Микросервис с ML-моделью для классификации ирисов.",
    version="1.0.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Iris ML microservice is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, list[str] | str]:
    return {
        "model": "LogisticRegression",
        "dataset": "Iris",
        "features": model_service.feature_names,
        "classes": model_service.class_names,
    }


@app.post("/predict")
def predict(payload: IrisFeatures) -> dict[str, object]:
    features = [
        payload.sepal_length,
        payload.sepal_width,
        payload.petal_length,
        payload.petal_width,
    ]
    result = model_service.predict(features)

    return {
        "input_features": payload.model_dump(),
        "predicted_class": result.predicted_class,
        "probabilities": result.probabilities,
    }
