import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class ServerConfig:
    host: str = "localhost"
    port: int = 50051
    max_receive_message_mb: int = 16
    max_send_message_mb: int = 16


@dataclass
class ModelConfig:
    repo_id: str = "visolex/xlm-roberta-absa-hotel"
    fallback_repo_id: str = "visolex/xlm-r-absa-hotel"
    repo_revision: str = "main"
    fallback_repo_revision: str = "main"
    tokenizer_fallback_revision: str = "main"
    max_length: int = 256
    batch_size: int = 16
    confidence_threshold: float = 0.5
    device: str = "cpu"
    torch_num_threads: int = 2


@dataclass
class BookingModelConfig:
    catboost_model_path: str = "models/booking_cancel_catboost.cbm"
    calibrator_path: str = "models/booking_cancel_calibrator.pkl"
    meta_path: str = "models/booking_cancel_meta.json"
    threshold: float = 0.5


@dataclass
class Config:
    server: ServerConfig
    model: ModelConfig
    booking_model: BookingModelConfig


def _pick_env(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_config(env_file: Path = Path(".env")) -> Config:
    load_dotenv(dotenv_path=env_file, override=False)

    host = _pick_env("SV_HOST", "SERVER_HOST") or "localhost"
    port = _to_int(_pick_env("SV_PORT", "SERVER_PORT"), 50051)
    max_receive_message_mb = max(
        1,
        _to_int(_pick_env("SV_MAX_RECV_MB", "SERVER_MAX_RECV_MB"), 16),
    )
    max_send_message_mb = max(
        1,
        _to_int(_pick_env("SV_MAX_SEND_MB", "SERVER_MAX_SEND_MB"), 16),
    )

    model_repo_id = _pick_env("ABSA_MODEL_ID") or "visolex/xlm-roberta-absa-hotel"
    fallback_repo_id = _pick_env("ABSA_FALLBACK_MODEL_ID") or "visolex/xlm-r-absa-hotel"
    model_repo_revision = _pick_env("ABSA_MODEL_REVISION") or "main"
    fallback_repo_revision = _pick_env("ABSA_FALLBACK_MODEL_REVISION") or "main"
    tokenizer_fallback_revision = (
        _pick_env("ABSA_TOKENIZER_FALLBACK_REVISION") or "main"
    )
    model_max_length = max(
        32,
        _to_int(_pick_env("ABSA_MAX_LENGTH"), 256),
    )
    model_batch_size = max(
        1,
        _to_int(_pick_env("ABSA_BATCH_SIZE"), 16),
    )
    model_confidence_threshold = min(
        1.0,
        max(
            0.0,
            _to_float(_pick_env("ABSA_CONFIDENCE_THRESHOLD"), 0.5),
        ),
    )
    model_device = (_pick_env("ABSA_DEVICE") or "cpu").lower()
    torch_num_threads = max(
        1,
        _to_int(_pick_env("ABSA_TORCH_NUM_THREADS"), 2),
    )
    booking_threshold = min(
        1.0,
        max(
            0.0,
            _to_float(_pick_env("BOOKING_CANCEL_THRESHOLD"), 0.5),
        ),
    )
    booking_model_path = (
        _pick_env("BOOKING_MODEL_PATH") or "models/booking_cancel_catboost.cbm"
    )
    booking_calibrator_path = (
        _pick_env("BOOKING_CALIBRATOR_PATH") or "models/booking_cancel_calibrator.pkl"
    )
    booking_meta_path = (
        _pick_env("BOOKING_META_PATH") or "models/booking_cancel_meta.json"
    )

    return Config(
        server=ServerConfig(
            host=host,
            port=port,
            max_receive_message_mb=max_receive_message_mb,
            max_send_message_mb=max_send_message_mb,
        ),
        model=ModelConfig(
            repo_id=model_repo_id,
            fallback_repo_id=fallback_repo_id,
            repo_revision=model_repo_revision,
            fallback_repo_revision=fallback_repo_revision,
            tokenizer_fallback_revision=tokenizer_fallback_revision,
            max_length=model_max_length,
            batch_size=model_batch_size,
            confidence_threshold=model_confidence_threshold,
            device=model_device,
            torch_num_threads=torch_num_threads,
        ),
        booking_model=BookingModelConfig(
            catboost_model_path=booking_model_path,
            calibrator_path=booking_calibrator_path,
            meta_path=booking_meta_path,
            threshold=booking_threshold,
        ),
    )


config = load_config()
