"""Shared pytest fixtures."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _ffmpeg_concat(*segments: str, output: Path) -> None:
    """Build a WAV by concatenating ffmpeg lavfi segments."""
    inputs = []
    filter_parts = []
    for i, seg in enumerate(segments):
        inputs.extend(["-f", "lavfi", "-i", seg])
        filter_parts.append(f"[{i}:a]")
    filter_complex = "".join(filter_parts) + f"concat=n={len(segments)}:v=0:a=1[out]"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-ac", "1", "-ar", "16000",
            str(output),
        ],
        check=True,
    )


@pytest.fixture
def two_tones_wav(tmp_path: Path) -> Path:
    """10s WAV: 3s tone, 4s silence, 3s tone. mono 16kHz."""
    out = tmp_path / "two_tones.wav"
    _ffmpeg_concat(
        "sine=frequency=440:duration=3",
        "anullsrc=channel_layout=mono:sample_rate=16000:d=4",
        "sine=frequency=880:duration=3",
        output=out,
    )
    return out


@pytest.fixture
def short_speech_m4a(tmp_path: Path) -> Path:
    """3s of synthesized PT-BR speech encoded as m4a, for normalize tests."""
    aiff = tmp_path / "speech.aiff"
    m4a = tmp_path / "speech.m4a"
    subprocess.run(
        ["say", "-v", "Luciana", "-o", str(aiff),
         "Senhor, abençoe nosso pequeno grupo."],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
         "-c:a", "aac", str(m4a)],
        check=True,
    )
    return m4a
