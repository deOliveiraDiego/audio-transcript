"""Central configuration. All env-var reads happen here."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL_DIR = Path(
    "~/Library/Application Support/audio-transcript/models/sherpa-onnx-parakeet-v3-int8"
).expanduser()

MODEL_DIR = Path(
    os.environ.get("PARAKEET_MODEL_DIR", str(DEFAULT_MODEL_DIR))
).expanduser()

DATA_DIR = Path(
    os.environ.get("AUDIO_TRANSCRIPT_DATA_DIR", str(Path.cwd() / "data"))
).expanduser()

UPLOADS_DIR = DATA_DIR / "uploads"

MAX_UPLOAD_BYTES = int(
    os.environ.get("AUDIO_TRANSCRIPT_MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024))
)

HOST = os.environ.get("AUDIO_TRANSCRIPT_HOST", "127.0.0.1")
PORT = int(os.environ.get("AUDIO_TRANSCRIPT_PORT", "8000"))

SILENCE_NOISE_DB = float(os.environ.get("AUDIO_TRANSCRIPT_SILENCE_DB", "-30"))
SILENCE_MIN_DURATION_S = float(os.environ.get("AUDIO_TRANSCRIPT_SILENCE_DUR", "1.5"))

MAX_CHUNK_SECONDS = float(os.environ.get("AUDIO_TRANSCRIPT_MAX_CHUNK_S", "90"))

TRANSCRIBE_PARALLELISM = int(
    os.environ.get("AUDIO_TRANSCRIPT_PARALLELISM", "4")
)

RETENTION_DAYS = int(
    os.environ.get("AUDIO_TRANSCRIPT_RETENTION_DAYS", "7")
)

HCAPTCHA_SITEKEY = os.environ.get("HCAPTCHA_SITEKEY", "")
HCAPTCHA_SECRET = os.environ.get("HCAPTCHA_SECRET", "")

ADSENSE_CLIENT_ID = os.environ.get("ADSENSE_CLIENT_ID", "")
