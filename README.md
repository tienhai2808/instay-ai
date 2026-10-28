# Instay AI

Instay AI is a gRPC service that provides:

- Aspect-based sentiment analysis for hotel reviews (`AnalyzeReview`).
- Booking cancellation risk prediction (`PredictBooking`).

## Features

- gRPC server with 3 RPC methods: `Ping`, `AnalyzeReview`, `PredictBooking`.
- ABSA model is preloaded at startup to avoid first-request cold start failures.
- Booking model uses CatBoost + calibrator to return probability and risk level.
- Distroless Docker runtime for production.

## Project Structure

```text
.
├── app/                  # service logic, config, inference engines
├── proto/                # ai.proto + generated Python stubs
├── models/               # booking model artifacts (.cbm/.pkl/.json)
├── tests/                # unit tests
├── scripts/run-app.sh    # local run entrypoint
├── Dockerfile
├── Makefile
├── pyproject.toml
└── .github/workflows/ci.yml
```

## Requirements

- Python `3.12`
- Poetry `2.x`
- Required model files under `models/`:
- `models/booking_cancel_catboost.cbm`
- `models/booking_cancel_calibrator.pkl`
- `models/booking_cancel_meta.json`

## Local Setup

```bash
poetry install --with dev --no-root
```

Note: this repository uses Poetry for dependency management in app/service mode, so `--no-root` is expected.

## Run Service

```bash
make app
```

Or:

```bash
poetry run python -m app.main
```

By default, the server listens on `localhost:50051`.

## Development Commands

- Run app locally: `make app`
- Generate gRPC stubs from `proto/ai.proto`: `make proto-gen`
- Run tests: `poetry run pytest -q`

## gRPC API

Proto definition: `proto/ai.proto`

### Ping

- Request: `PingRequest {}`
- Response: `PingResponse {}`

### AnalyzeReview

- Request field: `text` (string), review content.
- Response field: `aspects[]` (list of detected aspects).
- Each aspect item contains `aspect` (string), `sentiment` (string), and `score` (float).

### PredictBooking

- Request fields: `id` (int64), `features` (`BookingFeatures`).
- Response fields: `id` (int64), `cancellation_probability` (float), `prediction_label` (int32), `will_cancel` (bool), `risk_level` (string).

## Environment Variables

### Server

- `SV_HOST` (default: `localhost`)
- `SV_PORT` (default: `50051`)
- `SV_MAX_RECV_MB` (default: `16`)
- `SV_MAX_SEND_MB` (default: `16`)

### ABSA Model

- `ABSA_MODEL_ID` (default: `visolex/xlm-roberta-absa-hotel`)
- `ABSA_FALLBACK_MODEL_ID` (default: `visolex/xlm-r-absa-hotel`)
- `ABSA_MODEL_REVISION` (default: `main`)
- `ABSA_FALLBACK_MODEL_REVISION` (default: `main`)
- `ABSA_TOKENIZER_FALLBACK_REVISION` (default: `main`)
- `ABSA_MAX_LENGTH` (default: `256`)
- `ABSA_BATCH_SIZE` (default: `16`)
- `ABSA_CONFIDENCE_THRESHOLD` (default: `0.5`)
- `ABSA_DEVICE` (`cpu` or `cuda`, default: `cpu`)
- `ABSA_TORCH_NUM_THREADS` (default: `2`)

### Booking Model

- `BOOKING_CANCEL_THRESHOLD` (default: `0.5`)
- `BOOKING_MODEL_PATH` (default: `models/booking_cancel_catboost.cbm`)
- `BOOKING_CALIBRATOR_PATH` (default: `models/booking_cancel_calibrator.pkl`)
- `BOOKING_META_PATH` (default: `models/booking_cancel_meta.json`)

## Docker

Build image:

```bash
docker build -t instay-ai:local .
```

Run container:

```bash
docker run --rm -p 50051:50051 instay-ai:local
```

Inside the container, `SV_HOST` is set to `0.0.0.0`.

## CI

Workflow: `.github/workflows/ci.yml`, using GitHub-hosted `ubuntu-24.04` runners.

Pushes to `main`, pull requests targeting `main` and manual runs execute the checks. Container scan and publish run only on a push to `main`.

1. `pre-checks`: gitleaks, Poetry lock consistency, Ruff and compile checks.
2. `sast` and `sca` run in parallel: Bandit + Semgrep, and pip-audit from the lockfile export.
3. `build-test`: pytest with the existing `40%` coverage threshold.
4. `container-scan`: Docker build → SPDX SBOM generation → Trivy SBOM scan, matching frontend/backend ordering.
5. `publish`: push `ghcr.io/<owner>/<repo>:<commit-sha>`, sign the image and attach the SBOM.

Image and SBOM are passed between jobs as workflow artifacts. Only the commit SHA image tag is published; deployment selects the desired SHA. `make github CM="message"` requires the current branch to be `main` and pushes it via `git push origin main`.
