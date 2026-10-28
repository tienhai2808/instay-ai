import json
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.booking_inference import BookingInferenceEngine


class FakeCalibrator:
    def __init__(self, probability: float) -> None:
        self._probability = probability
        self.seen_frame: pd.DataFrame | None = None

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        self.seen_frame = frame
        return np.array([[1.0 - self._probability, self._probability]])


class BookingInferenceTest(unittest.TestCase):
    def test_risk_level_thresholds(self) -> None:
        self.assertEqual(BookingInferenceEngine._risk_level(0.39), "low")
        self.assertEqual(BookingInferenceEngine._risk_level(0.40), "medium")
        self.assertEqual(BookingInferenceEngine._risk_level(0.70), "high")

    def test_load_meta_missing_file_raises(self) -> None:
        with self.assertRaises(RuntimeError) as exc:
            BookingInferenceEngine._load_meta(Path("/tmp/not-found-meta.json"))
        self.assertIn("Booking meta file not found", str(exc.exception))

    def test_load_meta_validates_required_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            meta_path = Path(temp_dir) / "meta.json"
            meta_path.write_text(json.dumps({"feature_columns": []}), encoding="utf-8")

            with self.assertRaises(RuntimeError) as exc:
                BookingInferenceEngine._load_meta(meta_path)

        self.assertIn("Booking meta missing key", str(exc.exception))

    def test_preprocess_fills_missing_values_and_orders_columns(self) -> None:
        engine = BookingInferenceEngine.__new__(BookingInferenceEngine)
        engine._feature_columns = ["hotel", "lead_time", "adr"]
        engine._categorical_columns = ["hotel"]
        engine._numerical_columns = ["lead_time", "adr"]
        engine._numeric_medians = {"lead_time": 12.0, "adr": 99.5}

        frame = pd.DataFrame([{"hotel": None, "lead_time": "invalid"}])
        processed = engine._preprocess(frame)

        self.assertEqual(list(processed.columns), ["hotel", "lead_time", "adr"])
        self.assertEqual(processed.loc[0, "hotel"], "missing")
        self.assertEqual(float(processed.loc[0, "lead_time"]), 12.0)
        self.assertEqual(float(processed.loc[0, "adr"]), 99.5)

    def test_predict_maps_probability_to_prediction_fields(self) -> None:
        engine = BookingInferenceEngine.__new__(BookingInferenceEngine)
        engine._lock = threading.Lock()
        engine._threshold = 0.5
        engine._preprocess = lambda frame: frame
        engine._calibrator = FakeCalibrator(probability=0.82)

        prediction = engine.predict({"hotel": "City Hotel", "adults": 2})

        self.assertAlmostEqual(prediction.cancellation_probability, 0.82, places=6)
        self.assertEqual(prediction.prediction_label, 1)
        self.assertTrue(prediction.will_cancel)
        self.assertEqual(prediction.risk_level, "high")


if __name__ == "__main__":
    unittest.main()
