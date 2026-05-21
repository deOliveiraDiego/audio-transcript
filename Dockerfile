FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PARAKEET_MODEL_DIR=/var/lib/audio-transcript/models/sherpa-onnx-parakeet-v3-int8 \
    AUDIO_TRANSCRIPT_DATA_DIR=/var/lib/audio-transcript/data \
    AUDIO_TRANSCRIPT_HOST=0.0.0.0 \
    AUDIO_TRANSCRIPT_PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.32" \
        "python-multipart>=0.0.12" \
        "sherpa-onnx>=1.13" \
        "soundfile>=0.12" \
        "sse-starlette>=2.1" \
        "httpx>=0.27"

COPY audio_transcript/ ./audio_transcript/
COPY scripts/ ./scripts/

RUN pip install --no-deps -e .

RUN mkdir -p /var/lib/audio-transcript/models /var/lib/audio-transcript/data

EXPOSE 8000

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["audio-transcript"]
