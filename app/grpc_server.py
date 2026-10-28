from concurrent import futures
import grpc

from app.config import config
from app.logger import get_logger
from app.service import AIService
from proto.gen.ai import ai_pb2_grpc

logger = get_logger(__name__)

_KEEPALIVE_OPTIONS = [
    ("grpc.keepalive_time_ms", 5 * 60 * 1000),
    ("grpc.keepalive_timeout_ms", 20 * 1000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.min_recv_ping_interval_without_data_ms", 60 * 1000),
    (
        "grpc.max_receive_message_length",
        config.server.max_receive_message_mb * 1024 * 1024,
    ),
    (
        "grpc.max_send_message_length",
        config.server.max_send_message_mb * 1024 * 1024,
    ),
]


class GrpcServer:
    def __init__(self) -> None:
        self.server = self._create_server()
        self._setup_services()

    def _create_server(self) -> grpc.Server:
        return grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=_KEEPALIVE_OPTIONS,
        )

    def _setup_services(self) -> None:
        ai_service = AIService()
        ai_pb2_grpc.add_AIServiceServicer_to_server(ai_service, self.server)

    def start(self) -> None:
        address = f"{config.server.host}:{config.server.port}"
        self.server.add_insecure_port(address)
        self.server.start()
        logger.info(f"gRPC Server successfully started on {address}")

    def stop(self, grace_period_seconds: int = 5) -> None:
        if self.server:
            logger.info(
                f"Initiating graceful shutdown (Grace period: {grace_period_seconds}s)..."
            )
            shutdown_event = self.server.stop(grace_period_seconds)
            shutdown_event.wait()
            logger.info("gRPC Server has been cleanly stopped.")
