import signal
import sys
import time
from types import FrameType

from app.grpc_server import GrpcServer
from .logger import get_logger

logger = get_logger("main")


def main() -> None:
    grpc_server = GrpcServer()

    def handle_shutdown_signal(signum: int, frame: FrameType | None) -> None:
        del frame
        signal_name = signal.Signals(signum).name
        logger.info(f"Received signal {signal_name}. Starting shutdown process...")
        grpc_server.stop(grace_period_seconds=10)
        logger.info("Application exited.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    grpc_server.start()

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
