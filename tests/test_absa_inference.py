import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from app.absa_inference import ABSAInferenceEngine, DEFAULT_ASPECT_LABELS, ReviewPrediction


class ABSAInferenceTest(unittest.TestCase):
    def test_unique_preserve_order_and_weight_detection(self) -> None:
        self.assertEqual(
            ABSAInferenceEngine._unique_preserve_order(["a", "", "a", "b", "b"]),
            ["a", "b"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_dir = Path(temp_dir)
            self.assertFalse(ABSAInferenceEngine._has_model_weights(snapshot_dir))
            (snapshot_dir / "model.safetensors").touch()
            self.assertTrue(ABSAInferenceEngine._has_model_weights(snapshot_dir))

    def test_label_normalization_helpers(self) -> None:
        self.assertEqual(ABSAInferenceEngine._normalize_sentiment_label(" Pos "), "positive")
        self.assertEqual(ABSAInferenceEngine._normalize_sentiment_label("NEG"), "negative")
        self.assertEqual(ABSAInferenceEngine._normalize_sentiment_label("n/a"), "none")
        self.assertTrue(ABSAInferenceEngine._has_only_generic_labels(["label_0", "label_1"]))
        self.assertFalse(ABSAInferenceEngine._has_only_generic_labels(["positive", "label_1"]))

    def test_default_and_built_class_labels(self) -> None:
        self.assertEqual(
            ABSAInferenceEngine._default_sentiment_labels(3),
            ["positive", "negative", "neutral", "none"],
        )
        self.assertEqual(
            ABSAInferenceEngine._default_sentiment_labels("100"),
            [
                "positive",
                "negative",
                "neutral",
                "sentiment_3",
                "sentiment_4",
                "sentiment_5",
                "sentiment_6",
                "sentiment_7",
                "none",
            ],
        )

        engine = ABSAInferenceEngine.__new__(ABSAInferenceEngine)
        engine._sentiment_labels = ["positive", "negative", "none"]
        self.assertEqual(engine._build_class_labels(2), ["positive", "negative"])
        self.assertEqual(
            engine._build_class_labels(5),
            ["positive", "negative", "none", "class_3", "class_4"],
        )
        self.assertEqual(ABSAInferenceEngine._resolve_none_index(["positive", "none"], 2), 1)
        self.assertEqual(ABSAInferenceEngine._resolve_none_index(["positive"], 1), 0)

    def test_extract_logits_from_supported_output_shapes(self) -> None:
        tensor = torch.randn(1, 3, 4)
        self.assertTrue(
            torch.equal(
                ABSAInferenceEngine._extract_logits(SimpleNamespace(logits=tensor)),
                tensor,
            )
        )
        self.assertTrue(torch.equal(ABSAInferenceEngine._extract_logits((tensor,)), tensor))

        with self.assertRaises(RuntimeError):
            ABSAInferenceEngine._extract_logits(SimpleNamespace())

    def test_resolve_label_sources(self) -> None:
        engine = ABSAInferenceEngine.__new__(ABSAInferenceEngine)
        engine._model = SimpleNamespace(
            aspect_labels=["A#ONE", "B#TWO"],
            config=SimpleNamespace(
                num_sentiments=4,
                id2label={"2": "Neu", "0": "Pos", "1": "Neg", "3": "None"},
            ),
        )
        self.assertEqual(engine._resolve_aspect_labels(), ["A#ONE", "B#TWO"])
        self.assertEqual(
            engine._resolve_sentiment_labels(),
            ["positive", "negative", "neutral", "none"],
        )

        fallback_engine = ABSAInferenceEngine.__new__(ABSAInferenceEngine)
        fallback_engine._model = SimpleNamespace(
            config=SimpleNamespace(num_sentiments=None, id2label={"0": "label_0", "1": "label_1"})
        )
        self.assertEqual(fallback_engine._resolve_aspect_labels(), list(DEFAULT_ASPECT_LABELS))
        self.assertEqual(
            fallback_engine._resolve_sentiment_labels(),
            ["positive", "negative", "neutral", "none"],
        )

    def test_predict_handles_empty_and_batches_with_normalized_text(self) -> None:
        engine = ABSAInferenceEngine.__new__(ABSAInferenceEngine)
        engine._default_threshold = 0.6
        empty_out = engine.predict(texts=[])
        self.assertEqual(empty_out.predictions, [])
        self.assertEqual(empty_out.elapsed_ms, 0.0)
        self.assertEqual(empty_out.applied_confidence_threshold, 0.6)

        engine = ABSAInferenceEngine.__new__(ABSAInferenceEngine)
        engine._default_batch_size = 2
        engine._default_threshold = 0.4
        engine._lock = threading.Lock()

        calls: list[tuple[list[str], float, bool]] = []

        def fake_predict_batch(
            batch_texts: list[str],
            confidence_threshold: float,
            include_all_aspects: bool,
        ) -> list[ReviewPrediction]:
            calls.append((batch_texts, confidence_threshold, include_all_aspects))
            return [ReviewPrediction(text=text, aspects=[]) for text in batch_texts]

        engine._predict_batch = fake_predict_batch
        out = engine.predict(texts=["first", 123, "third"], include_all_aspects=True)

        self.assertEqual([prediction.text for prediction in out.predictions], ["first", "123", "third"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], ["first", "123"])
        self.assertEqual(calls[1][0], ["third"])
        self.assertEqual(calls[0][1], 0.4)
        self.assertTrue(calls[0][2])


if __name__ == "__main__":
    unittest.main()
