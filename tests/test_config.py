import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import _pick_env, _to_float, _to_int, load_config


class ConfigTest(unittest.TestCase):
    def test_pick_env_returns_first_existing_key(self) -> None:
        with patch.dict(os.environ, {"SECOND": "value-2"}, clear=True):
            self.assertEqual(_pick_env("FIRST", "SECOND"), "value-2")
            self.assertIsNone(_pick_env("THIRD", "FOURTH"))

    def test_to_int_and_to_float_fallback_to_default(self) -> None:
        self.assertEqual(_to_int("123", 0), 123)
        self.assertEqual(_to_int("invalid", 7), 7)
        self.assertEqual(_to_float("0.25", 0.0), 0.25)
        self.assertEqual(_to_float("invalid", 0.5), 0.5)

    def test_load_config_applies_env_and_clamps_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "SV_HOST": "0.0.0.0",
                    "SV_PORT": "6000",
                    "SV_MAX_RECV_MB": "0",
                    "SV_MAX_SEND_MB": "-10",
                    "ABSA_MODEL_REVISION": "abc123",
                    "ABSA_FALLBACK_MODEL_REVISION": "def456",
                    "ABSA_TOKENIZER_FALLBACK_REVISION": "ghi789",
                    "ABSA_MAX_LENGTH": "16",
                    "ABSA_BATCH_SIZE": "0",
                    "ABSA_CONFIDENCE_THRESHOLD": "9",
                    "ABSA_DEVICE": "CUDA",
                    "ABSA_TORCH_NUM_THREADS": "0",
                    "BOOKING_CANCEL_THRESHOLD": "-1",
                    "BOOKING_MODEL_PATH": "m.cbm",
                    "BOOKING_CALIBRATOR_PATH": "c.pkl",
                    "BOOKING_META_PATH": "meta.json",
                },
                clear=True,
            ):
                loaded = load_config(env_file=env_file)

        self.assertEqual(loaded.server.host, "0.0.0.0")
        self.assertEqual(loaded.server.port, 6000)
        self.assertEqual(loaded.server.max_receive_message_mb, 1)
        self.assertEqual(loaded.server.max_send_message_mb, 1)

        self.assertEqual(loaded.model.max_length, 32)
        self.assertEqual(loaded.model.batch_size, 1)
        self.assertEqual(loaded.model.confidence_threshold, 1.0)
        self.assertEqual(loaded.model.device, "cuda")
        self.assertEqual(loaded.model.torch_num_threads, 1)
        self.assertEqual(loaded.model.repo_revision, "abc123")
        self.assertEqual(loaded.model.fallback_repo_revision, "def456")
        self.assertEqual(loaded.model.tokenizer_fallback_revision, "ghi789")

        self.assertEqual(loaded.booking_model.threshold, 0.0)
        self.assertEqual(loaded.booking_model.catboost_model_path, "m.cbm")
        self.assertEqual(loaded.booking_model.calibrator_path, "c.pkl")
        self.assertEqual(loaded.booking_model.meta_path, "meta.json")


if __name__ == "__main__":
    unittest.main()
