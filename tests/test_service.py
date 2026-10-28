import unittest
from unittest.mock import MagicMock, patch

import grpc

from app.absa_inference import AspectPrediction, PredictOutput, ReviewPrediction
from app.booking_inference import BookingPrediction
from app.service import AIService
from proto.gen.ai import ai_pb2


class AbortError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        super().__init__()
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


class FakeContext:
    def abort(self, code: grpc.StatusCode, details: str) -> None:
        raise AbortError(code, details)


class AIServiceTest(unittest.TestCase):
    def _make_service(self) -> AIService:
        with patch.object(AIService, "_preload_absa_engine", return_value=None):
            with patch("app.service.BookingInferenceEngine", autospec=True):
                return AIService()

    def test_analyze_review_empty_text_aborts_invalid_argument(self) -> None:
        service = self._make_service()
        context = FakeContext()
        request = ai_pb2.AnalyzeReviewRequest(text="   ")

        with self.assertRaises(AbortError) as exc:
            service.AnalyzeReview(request, context)

        self.assertEqual(exc.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("must not be empty", exc.exception.details())

    def test_analyze_review_maps_sentiment_response_field(self) -> None:
        service = self._make_service()
        service._absa_engine = MagicMock()
        service._absa_engine.predict.return_value = PredictOutput(
            predictions=[
                ReviewPrediction(
                    text="Great room",
                    aspects=[
                        AspectPrediction(
                            aspect="ROOMS#COMFORT",
                            sentiment="positive",
                            sentiment_id=0,
                            score=0.93,
                            is_detected=True,
                        )
                    ],
                )
            ],
            elapsed_ms=12.0,
            applied_confidence_threshold=0.5,
        )

        response = service.AnalyzeReview(
            ai_pb2.AnalyzeReviewRequest(text="Great room"),
            FakeContext(),
        )

        self.assertEqual(len(response.aspects), 1)
        self.assertEqual(response.aspects[0].aspect, "ROOMS#COMFORT")
        self.assertEqual(response.aspects[0].sentiment, "positive")
        self.assertAlmostEqual(response.aspects[0].score, 0.93, places=6)

    def test_extract_booking_features_only_returns_present_fields(self) -> None:
        request = ai_pb2.PredictBookingRequest(
            id=123,
            features=ai_pb2.BookingFeatures(
                hotel="City Hotel",
                adults=2,
                country="IDN",
            ),
        )

        result = AIService._extract_booking_features(request)

        self.assertEqual(result, {"hotel": "City Hotel", "adults": 2, "country": "IDN"})

    def test_predict_booking_maps_engine_output(self) -> None:
        service = self._make_service()
        service._booking_engine.predict = MagicMock(
            return_value=BookingPrediction(
                cancellation_probability=0.83,
                prediction_label=1,
                will_cancel=True,
                risk_level="high",
            )
        )

        request = ai_pb2.PredictBookingRequest(
            id=99,
            features=ai_pb2.BookingFeatures(adults=2, country="IDN"),
        )

        response = service.PredictBooking(request, FakeContext())

        self.assertEqual(response.id, 99)
        self.assertAlmostEqual(response.cancellation_probability, 0.83, places=6)
        self.assertEqual(response.prediction_label, 1)
        self.assertTrue(response.will_cancel)
        self.assertEqual(response.risk_level, "high")

    def test_preload_absa_engine_failure_raises_runtime_error(self) -> None:
        service = self._make_service()
        service._get_absa_engine = MagicMock(side_effect=RuntimeError("boom"))

        with self.assertRaises(RuntimeError) as exc:
            service._preload_absa_engine()

        self.assertIn("Cannot start server because ABSA model preload failed", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
