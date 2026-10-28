ARG PYTHON_VERSION=3.13
ARG POETRY_VERSION=2.3.4
ARG DEBIAN_VERSION=13

FROM python:${PYTHON_VERSION}-slim AS builder

ARG POETRY_VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv --copies /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml poetry.lock ./

RUN pip install --upgrade pip setuptools wheel && \
    pip install "poetry==${POETRY_VERSION}" poetry-plugin-export && \
    poetry export --only main --without-hashes -f requirements.txt -o requirements.txt && \
    /opt/venv/bin/pip install -r requirements.txt

FROM gcr.io/distroless/python3-debian${DEBIAN_VERSION}:nonroot AS production

ENV PYTHONPATH="/opt/venv/lib/python3.13/site-packages:/app" \
    PYTHONUNBUFFERED=1 \
    SV_HOST=0.0.0.0

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/lib/x86_64-linux-gnu/libgomp.so.1*  /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libstdc++.so.6* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /lib/x86_64-linux-gnu/libgcc_s.so.1       /lib/x86_64-linux-gnu/
COPY --from=builder /lib/x86_64-linux-gnu/libz.so.1*          /lib/x86_64-linux-gnu/

COPY --chown=nonroot:nonroot app/ app/
COPY --chown=nonroot:nonroot proto/ proto/
COPY --chown=nonroot:nonroot models/ models

EXPOSE 50051

ENTRYPOINT ["python3", "-m", "app.main"]