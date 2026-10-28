import grpc
import threading

from app.absa_inference import ABSAInferenceEngine
from app.booking_inference import BookingInferenceEngine
from app.config import config
from app.logger import get_logger
from proto.gen.ai import ai_pb2, ai_pb2_grpc

logger = get_logger(__name__)


class AIService(ai_pb2_grpc.AIServiceServicer):
    def __init__(self) -> None:
        self._absa_lock = threading.Lock()
        self._absa_engine: ABSAInferenceEngine | None = None
        self._absa_error: RuntimeError | None = None
        self._booking_engine = BookingInferenceEngine(
            catboost_model_path=config.booking_model.catboost_model_path,
            calibrator_path=config.booking_model.calibrator_path,
            meta_path=config.booking_model.meta_path,
            threshold=config.booking_model.threshold,
        )
        self._preload_absa_engine()

    def Ping(
        self,
        request: ai_pb2.PingRequest,
        context: grpc.ServicerContext,
    ) -> ai_pb2.PingResponse:
        del request
        del context
        return ai_pb2.PingResponse()

    def AnalyzeReview(
        self,
        request: ai_pb2.AnalyzeReviewRequest,
        context: grpc.ServicerContext,
    ) -> ai_pb2.AnalyzeReviewResponse:
        try:
            text = request.text.strip()
            if not text:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "`text` must not be empty.",
                )

            inference_output = self._get_absa_engine().predict(
                texts=[text],
            )

            prediction = inference_output.predictions[0]
            aspect_messages = [
                ai_pb2.AspectResponse(
                    aspect=aspect.aspect,
                    sentiment=aspect.sentiment,
                    score=aspect.score,
                )
                for aspect in prediction.aspects
            ]

            return ai_pb2.AnalyzeReviewResponse(
                aspects=aspect_messages,
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except grpc.RpcError:
            raise
        except RuntimeError as exc:
            logger.exception("ABSA runtime error")
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        except Exception:
            logger.exception("Unexpected error while processing AnalyzeReview")
            context.abort(grpc.StatusCode.INTERNAL, "Unexpected inference error")

    def PredictBooking(
        self,
        request: ai_pb2.PredictBookingRequest,
        context: grpc.ServicerContext,
    ) -> ai_pb2.PredictBookingResponse:
        try:
            features = self._extract_booking_features(request)
            prediction = self._booking_engine.predict(features)

            return ai_pb2.PredictBookingResponse(
                id=request.id,
                cancellation_probability=prediction.cancellation_probability,
                prediction_label=prediction.prediction_label,
                will_cancel=prediction.will_cancel,
                risk_level=prediction.risk_level,
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except grpc.RpcError:
            raise
        except RuntimeError as exc:
            logger.exception("Booking runtime error")
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        except Exception:
            logger.exception("Unexpected error while processing PredictBooking")
            context.abort(
                grpc.StatusCode.INTERNAL, "Unexpected booking inference error"
            )

    @staticmethod
    def _extract_booking_features(
        request: ai_pb2.PredictBookingRequest,
    ) -> dict[str, object]:
        if not request.HasField("features"):
            return {}

        features = request.features
        return {
            field_descriptor.name: value
            for field_descriptor, value in features.ListFields()
        }

    def _get_absa_engine(self) -> ABSAInferenceEngine:
        if self._absa_engine is not None:
            return self._absa_engine
        if self._absa_error is not None:
            raise self._absa_error

        with self._absa_lock:
            if self._absa_engine is not None:
                return self._absa_engine
            if self._absa_error is not None:
                raise self._absa_error
            try:
                self._absa_engine = ABSAInferenceEngine(
                    repo_id=config.model.repo_id,
                    fallback_repo_id=config.model.fallback_repo_id,
                    repo_revision=config.model.repo_revision,
                    fallback_repo_revision=config.model.fallback_repo_revision,
                    tokenizer_fallback_revision=config.model.tokenizer_fallback_revision,
                    max_length=config.model.max_length,
                    batch_size=config.model.batch_size,
                    confidence_threshold=config.model.confidence_threshold,
                    device=config.model.device,
                    torch_num_threads=config.model.torch_num_threads,
                )
                return self._absa_engine
            except RuntimeError as exc:
                logger.exception("Cannot initialize ABSA model")
                self._absa_error = exc
                raise

    def _preload_absa_engine(self) -> None:
        logger.info("Preloading ABSA model during server startup...")
        try:
            engine = self._get_absa_engine()
            engine.predict(texts=["Warmup review text."])
            logger.info("ABSA model preload completed.")
        except Exception as exc:
            logger.exception("ABSA preload failed. Server startup aborted.")
            raise RuntimeError(
                "Cannot start server because ABSA model preload failed."
            ) from exc
