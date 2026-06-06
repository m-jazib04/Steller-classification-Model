"""ML model loading and prediction service."""

import logging
from pathlib import Path

import joblib
import numpy as np

from app.config import settings
from app.schemas import PredictRequest, PredictResponse, ProbabilityItem

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self) -> None:
        self.model = None
        self.scaler = None
        self.feature_columns: list[str] = []
        self.classes: list[str] = []
        self.accuracy: float | None = None
        self.model_type: str = ""

    def load(self) -> None:
        model_dir = Path(settings.model_dir)
        model_path = model_dir / "stellar_classifier.pkl"
        scaler_path = model_dir / "scaler.pkl"
        metadata_path = model_dir / "model_metadata.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        metadata = joblib.load(metadata_path)

        self.feature_columns = metadata["feature_columns"]
        self.classes = metadata["classes"]
        self.accuracy = metadata.get("accuracy")
        self.model_type = metadata.get("model_type", "RandomForestClassifier")
        logger.info("Model loaded from %s (accuracy: %s)", model_dir, self.accuracy)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None

    def predict(self, payload: PredictRequest) -> PredictResponse:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        features = np.array(
            [[getattr(payload, col) for col in self.feature_columns]],
            dtype=np.float64,
        )
        scaled = self.scaler.transform(features)
        prediction = self.model.predict(scaled)[0]
        probabilities = self.model.predict_proba(scaled)[0]
        confidence = float(max(probabilities))

        prob_items = [
            ProbabilityItem(label=str(label), probability=float(prob))
            for label, prob in zip(self.model.classes_, probabilities)
        ]
        prob_items.sort(key=lambda item: item.probability, reverse=True)

        return PredictResponse(
            prediction=prediction,
            confidence=round(confidence, 4),
            probabilities=prob_items,
        )


model_service = ModelService()
