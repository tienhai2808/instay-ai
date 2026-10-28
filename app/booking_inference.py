from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Any

import joblib
import pandas as pd
from catboost import CatBoostClassifier

from app.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class BookingPrediction:
    cancellation_probability: float
    prediction_label: int
    will_cancel: bool
    risk_level: str


class BookingInferenceEngine:
    def __init__(
        self,
        catboost_model_path: str,
        calibrator_path: str,
        meta_path: str,
        threshold: float,
    ) -> None:
        self._lock = threading.Lock()
        self._threshold = min(1.0, max(0.0, threshold))

        self._model_path = Path(catboost_model_path)
        self._calibrator_path = Path(calibrator_path)
        self._meta_path = Path(meta_path)

        self._meta = self._load_meta(self._meta_path)
        self._feature_columns = self._meta["feature_columns"]
        self._categorical_columns = self._meta["categorical_columns"]
        self._numerical_columns = self._meta["numerical_columns"]
        self._numeric_medians = self._meta["numeric_medians"]

        self._base_model = CatBoostClassifier()
        self._base_model.load_model(str(self._model_path))
        self._calibrator = joblib.load(self._calibrator_path)

        logger.info(
            "Booking model ready | model=%s | calibrator=%s | features=%d",
            self._model_path,
            self._calibrator_path,
            len(self._feature_columns),
        )

    def predict(self, feature_values: dict[str, Any]) -> BookingPrediction:
        row = pd.DataFrame([feature_values])
        row_processed = self._preprocess(row)

        with self._lock:
            probability = float(self._calibrator.predict_proba(row_processed)[:, 1][0])

        prediction_label = int(probability >= self._threshold)
        return BookingPrediction(
            cancellation_probability=probability,
            prediction_label=prediction_label,
            will_cancel=bool(prediction_label == 1),
            risk_level=self._risk_level(probability),
        )

    def _preprocess(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column in self._feature_columns:
            if column not in out.columns:
                out[column] = pd.NA
        out = out[self._feature_columns]

        for column in self._categorical_columns:
            out[column] = out[column].astype("string").fillna("missing")

        for column in self._numerical_columns:
            median = self._numeric_medians.get(column, 0.0)
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(median)

        return out

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability >= 0.7:
            return "high"
        if probability >= 0.4:
            return "medium"
        return "low"

    @staticmethod
    def _load_meta(meta_path: Path) -> dict[str, Any]:
        if not meta_path.exists():
            raise RuntimeError(f"Booking meta file not found: {meta_path}")

        with meta_path.open("r", encoding="utf-8") as file:
            raw_meta = json.load(file)

        required_keys = (
            "feature_columns",
            "categorical_columns",
            "numerical_columns",
            "numeric_medians",
        )
        for key in required_keys:
            if key not in raw_meta:
                raise RuntimeError(f"Booking meta missing key: {key}")

        numeric_medians = {
            str(key): float(value) for key, value in raw_meta["numeric_medians"].items()
        }
        return {
            "feature_columns": [str(item) for item in raw_meta["feature_columns"]],
            "categorical_columns": [
                str(item) for item in raw_meta["categorical_columns"]
            ],
            "numerical_columns": [str(item) for item in raw_meta["numerical_columns"]],
            "numeric_medians": numeric_medians,
        }
