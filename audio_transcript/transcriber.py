"""Wrap sherpa-onnx Parakeet TDT inference behind a small synchronous API."""
from __future__ import annotations

from pathlib import Path

import sherpa_onnx
import soundfile as sf

from audio_transcript import config


def _build_recognizer(model_dir: Path) -> sherpa_onnx.OfflineRecognizer:
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        joiner=str(model_dir / "joiner.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=2,
        sample_rate=16000,
        feature_dim=128,
        model_type="nemo_transducer",
    )


class Transcriber:
    """Single-instance wrapper. Create once at app boot; call transcribe per chunk."""

    def __init__(self, recognizer: sherpa_onnx.OfflineRecognizer):
        self._recognizer = recognizer

    @classmethod
    def from_default_config(cls) -> "Transcriber":
        if not config.MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Parakeet model directory not found: {config.MODEL_DIR}. "
                "Download the model from "
                "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 "
                "(encoder.int8.onnx, decoder.int8.onnx, joiner.int8.onnx, tokens.txt) "
                "or set PARAKEET_MODEL_DIR to a directory containing those files."
            )
        return cls(recognizer=_build_recognizer(config.MODEL_DIR))

    def transcribe(self, wav_path: Path) -> str:
        samples, sample_rate = sf.read(str(wav_path), dtype="float32")
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self._recognizer.decode_stream(stream)
        return stream.result.text
