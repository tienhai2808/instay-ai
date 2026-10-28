from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import threading
import tempfile
import time
from typing import Any

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel, AutoTokenizer

from app.logger import get_logger

logger = get_logger(__name__)

MODEL_WEIGHT_FILES: tuple[str, ...] = (
    "pytorch_model.bin",
    "model.safetensors",
    "tf_model.h5",
    "flax_model.msgpack",
)

DEFAULT_ASPECT_LABELS: tuple[str, ...] = (
    "FACILITIES#CLEANLINESS",
    "FACILITIES#COMFORT",
    "FACILITIES#DESIGN&FEATURES",
    "FACILITIES#GENERAL",
    "FACILITIES#MISCELLANEOUS",
    "FACILITIES#PRICES",
    "FACILITIES#QUALITY",
    "FOOD&DRINKS#MISCELLANEOUS",
    "FOOD&DRINKS#PRICES",
    "FOOD&DRINKS#QUALITY",
    "FOOD&DRINKS#STYLE&OPTIONS",
    "HOTEL#CLEANLINESS",
    "HOTEL#COMFORT",
    "HOTEL#DESIGN&FEATURES",
    "HOTEL#GENERAL",
    "HOTEL#MISCELLANEOUS",
    "HOTEL#PRICES",
    "HOTEL#QUALITY",
    "LOCATION#GENERAL",
    "ROOMS#CLEANLINESS",
    "ROOMS#COMFORT",
    "ROOMS#DESIGN&FEATURES",
    "ROOMS#GENERAL",
    "ROOMS#MISCELLANEOUS",
    "ROOMS#PRICES",
    "ROOMS#QUALITY",
    "ROOM_AMENITIES#CLEANLINESS",
    "ROOM_AMENITIES#COMFORT",
    "ROOM_AMENITIES#DESIGN&FEATURES",
    "ROOM_AMENITIES#GENERAL",
    "ROOM_AMENITIES#MISCELLANEOUS",
    "ROOM_AMENITIES#PRICES",
    "ROOM_AMENITIES#QUALITY",
    "SERVICE#GENERAL",
)


@dataclass(slots=True)
class AspectPrediction:
    aspect: str
    sentiment: str
    sentiment_id: int
    score: float
    is_detected: bool


@dataclass(slots=True)
class ReviewPrediction:
    text: str
    aspects: list[AspectPrediction]


@dataclass(slots=True)
class PredictOutput:
    predictions: list[ReviewPrediction]
    elapsed_ms: float
    applied_confidence_threshold: float


class ABSAInferenceEngine:
    def __init__(
        self,
        repo_id: str,
        fallback_repo_id: str,
        repo_revision: str,
        fallback_repo_revision: str,
        tokenizer_fallback_revision: str,
        max_length: int,
        batch_size: int,
        confidence_threshold: float,
        device: str,
        torch_num_threads: int,
    ) -> None:
        self._lock = threading.Lock()
        self._max_length = max_length
        self._default_batch_size = max(1, batch_size)
        self._default_threshold = min(1.0, max(0.0, confidence_threshold))

        if torch_num_threads > 0:
            torch.set_num_threads(torch_num_threads)

        self._device = self._resolve_device(device)
        (
            self._model_name,
            self._tokenizer,
            self._model,
        ) = self._load_assets(
            repo_id=repo_id,
            fallback_repo_id=fallback_repo_id,
            repo_revision=repo_revision,
            fallback_repo_revision=fallback_repo_revision,
            tokenizer_fallback_revision=tokenizer_fallback_revision,
        )
        self._model.to(self._device)
        self._model.eval()
        self._sentiment_labels = self._resolve_sentiment_labels()
        self._aspect_labels = self._resolve_aspect_labels()

        logger.info(
            "ABSA engine ready | model=%s | device=%s | aspects=%d | sentiments=%s",
            self._model_name,
            self._device,
            len(self._aspect_labels),
            ",".join(self._sentiment_labels),
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def predict(
        self,
        texts: list[str],
        batch_size: int | None = None,
        confidence_threshold: float | None = None,
        include_all_aspects: bool = False,
    ) -> PredictOutput:
        if not texts:
            return PredictOutput(
                predictions=[],
                elapsed_ms=0.0,
                applied_confidence_threshold=self._default_threshold,
            )

        effective_batch_size = (
            self._default_batch_size if not batch_size else max(1, batch_size)
        )
        effective_threshold = (
            self._default_threshold
            if confidence_threshold is None
            else min(1.0, max(0.0, confidence_threshold))
        )
        normalized_texts = [
            text if isinstance(text, str) else str(text) for text in texts
        ]

        start = time.perf_counter()
        predictions: list[ReviewPrediction] = []
        with self._lock:
            for i in range(0, len(normalized_texts), effective_batch_size):
                batch_texts = normalized_texts[i : i + effective_batch_size]
                predictions.extend(
                    self._predict_batch(
                        batch_texts,
                        confidence_threshold=effective_threshold,
                        include_all_aspects=include_all_aspects,
                    )
                )

        elapsed_ms = (time.perf_counter() - start) * 1000
        return PredictOutput(
            predictions=predictions,
            elapsed_ms=elapsed_ms,
            applied_confidence_threshold=effective_threshold,
        )

    def _predict_batch(
        self,
        batch_texts: list[str],
        confidence_threshold: float,
        include_all_aspects: bool,
    ) -> list[ReviewPrediction]:
        encoded = self._tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        encoded.pop("token_type_ids", None)
        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = self._model(**encoded)
            logits = self._extract_logits(outputs)
            probs = torch.softmax(logits, dim=-1).detach().cpu()

        class_labels = self._build_class_labels(probs.size(-1))
        none_index = self._resolve_none_index(class_labels, probs.size(-1))

        batch_predictions: list[ReviewPrediction] = []
        for row_index, text in enumerate(batch_texts):
            row_probs = probs[row_index]
            aspects: list[AspectPrediction] = []
            for aspect_index in range(row_probs.size(0)):
                scores = row_probs[aspect_index]
                predicted_index = int(scores.argmax().item())
                predicted_score = float(scores[predicted_index].item())
                sentiment = class_labels[predicted_index]
                is_detected = (
                    predicted_index != none_index
                    and predicted_score >= confidence_threshold
                )

                if not include_all_aspects and not is_detected:
                    continue

                aspect_name = (
                    self._aspect_labels[aspect_index]
                    if aspect_index < len(self._aspect_labels)
                    else f"ASPECT_{aspect_index}"
                )
                aspects.append(
                    AspectPrediction(
                        aspect=aspect_name,
                        sentiment=sentiment,
                        sentiment_id=predicted_index,
                        score=predicted_score,
                        is_detected=is_detected,
                    )
                )
            batch_predictions.append(ReviewPrediction(text=text, aspects=aspects))
        return batch_predictions

    def _load_assets(
        self,
        repo_id: str,
        fallback_repo_id: str,
        repo_revision: str,
        fallback_repo_revision: str,
        tokenizer_fallback_revision: str,
    ) -> tuple[str, Any, Any]:
        code_repo_candidates = self._unique_repo_candidates(
            [
                (repo_id, repo_revision),
                (fallback_repo_id, fallback_repo_revision),
            ]
        )
        tokenizer_repo_candidates = self._unique_repo_candidates(
            [
                (repo_id, repo_revision),
                (fallback_repo_id, fallback_repo_revision),
                ("xlm-roberta-base", tokenizer_fallback_revision),
            ]
        )
        model_repo_candidates = self._unique_repo_candidates(
            [
                (repo_id, repo_revision),
                (fallback_repo_id, fallback_repo_revision),
            ]
        )

        tokenizer = self._load_tokenizer(tokenizer_repo_candidates)

        last_error: Exception | None = None
        for model_repo, model_revision in model_repo_candidates:
            try:
                model_source, display_name = self._prepare_model_source(
                    model_repo=model_repo,
                    model_revision=model_revision,
                    code_repo_candidates=code_repo_candidates,
                )
                logger.info(
                    "Loading ABSA model from `%s` (revision=%s)",
                    display_name,
                    model_revision,
                )
                model = AutoModel.from_pretrained(
                    model_source,
                    revision=model_revision,
                    trust_remote_code=True,
                )
                return display_name, tokenizer, model
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Failed to load model `%s` (revision=%s): %s",
                    model_repo,
                    model_revision,
                    exc,
                )

        raise RuntimeError(
            "Cannot load ABSA model from configured repositories"
        ) from last_error

    def _load_tokenizer(self, candidates: list[tuple[str, str]]) -> Any:
        last_error: Exception | None = None
        for candidate, revision in candidates:
            try:
                logger.info(
                    "Loading tokenizer from `%s` (revision=%s)", candidate, revision
                )
                return AutoTokenizer.from_pretrained(
                    candidate,
                    revision=revision,
                    trust_remote_code=True,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Failed to load tokenizer from `%s` (revision=%s): %s",
                    candidate,
                    revision,
                    exc,
                )

        raise RuntimeError(
            "Cannot load tokenizer from configured repositories"
        ) from last_error

    def _prepare_model_source(
        self,
        model_repo: str,
        model_revision: str,
        code_repo_candidates: list[tuple[str, str]],
    ) -> tuple[str, str]:
        snapshot_dir = Path(snapshot_download(model_repo, revision=model_revision))
        if not self._has_model_weights(snapshot_dir):
            raise RuntimeError(f"Model repo `{model_repo}` has no weight file")

        models_file = snapshot_dir / "models.py"
        if models_file.exists():
            return str(snapshot_dir), model_repo

        for code_repo, code_repo_revision in code_repo_candidates:
            try:
                code_file_path = hf_hub_download(
                    code_repo,
                    "models.py",
                    revision=code_repo_revision,
                )
                runtime_dir = Path(tempfile.mkdtemp(prefix="absa_model_"))
                shutil.copytree(snapshot_dir, runtime_dir, dirs_exist_ok=True)
                shutil.copy2(code_file_path, runtime_dir / "models.py")
                display_name = (
                    f"{model_repo}@{model_revision} + models.py from "
                    f"{code_repo}@{code_repo_revision}"
                )
                return str(runtime_dir), display_name
            except Exception as exc:
                logger.warning(
                    "Cannot reuse models.py from `%s` (revision=%s): %s",
                    code_repo,
                    code_repo_revision,
                    exc,
                )

        raise RuntimeError(
            f"`models.py` is missing in `{model_repo}` and no fallback code repo is available."
        )

    @staticmethod
    def _has_model_weights(snapshot_dir: Path) -> bool:
        return any(
            (snapshot_dir / weight_file).exists() for weight_file in MODEL_WEIGHT_FILES
        )

    @staticmethod
    def _unique_preserve_order(values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            if not value:
                continue
            if value not in deduped:
                deduped.append(value)
        return deduped

    @staticmethod
    def _unique_repo_candidates(values: list[tuple[str, str]]) -> list[tuple[str, str]]:
        deduped: list[tuple[str, str]] = []
        for repo, revision in values:
            if not repo:
                continue
            normalized_revision = revision or "main"
            candidate = (repo, normalized_revision)
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _resolve_aspect_labels(self) -> list[str]:
        for holder in (self._model, self._model.config):
            for attr in ("aspect_labels", "aspects", "aspect_names"):
                labels = getattr(holder, attr, None)
                if isinstance(labels, (list, tuple)) and labels:
                    return [str(item) for item in labels]
        return list(DEFAULT_ASPECT_LABELS)

    def _resolve_sentiment_labels(self) -> list[str]:
        config_num_sentiments = getattr(self._model.config, "num_sentiments", None)
        default_labels = self._default_sentiment_labels(config_num_sentiments)
        id2label = getattr(self._model.config, "id2label", None)
        if isinstance(id2label, dict) and id2label:
            sorted_items = sorted(id2label.items(), key=self._id2label_sort_key)
            labels = [
                self._normalize_sentiment_label(label) for _, label in sorted_items
            ]
            if labels and not self._has_only_generic_labels(labels):
                return labels
        return default_labels

    @staticmethod
    def _id2label_sort_key(item: tuple[Any, Any]) -> tuple[int, str]:
        key, _ = item
        key_str = str(key)
        if key_str.isdigit():
            return int(key_str), key_str
        return 10_000, key_str

    @staticmethod
    def _normalize_sentiment_label(label: Any) -> str:
        value = str(label).strip().lower().replace(" ", "_")
        alias_map = {
            "pos": "positive",
            "neg": "negative",
            "neu": "neutral",
            "null": "none",
            "n/a": "none",
            "na": "none",
            "other": "none",
        }
        return alias_map.get(value, value)

    @staticmethod
    def _has_only_generic_labels(labels: list[str]) -> bool:
        return all(re.fullmatch(r"label_\d+", label) for label in labels)

    @staticmethod
    def _default_sentiment_labels(config_num_sentiments: Any) -> list[str]:
        try:
            num_sentiments = int(config_num_sentiments)
        except (TypeError, ValueError):
            num_sentiments = 3
        num_sentiments = max(1, min(num_sentiments, 8))

        base_labels = ["positive", "negative", "neutral"]
        labels = base_labels[:num_sentiments]
        if len(labels) < num_sentiments:
            labels.extend(
                [f"sentiment_{index}" for index in range(len(labels), num_sentiments)]
            )
        labels.append("none")
        return labels

    def _build_class_labels(self, num_classes: int) -> list[str]:
        labels = list(self._sentiment_labels)
        if len(labels) >= num_classes:
            return labels[:num_classes]

        generated = [f"class_{idx}" for idx in range(len(labels), num_classes)]
        return labels + generated

    @staticmethod
    def _resolve_none_index(class_labels: list[str], num_classes: int) -> int:
        for index, label in enumerate(class_labels):
            if label == "none":
                return index
        return num_classes - 1

    @staticmethod
    def _extract_logits(outputs: Any) -> torch.Tensor:
        logits = getattr(outputs, "logits", None)
        if isinstance(logits, torch.Tensor):
            return logits
        if isinstance(outputs, (list, tuple)) and outputs:
            if isinstance(outputs[0], torch.Tensor):
                return outputs[0]
        raise RuntimeError("Model output does not contain logits")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
