import subprocess
from pathlib import Path

import pytest
import soundfile as sf

from audio_transcript.chunker import (
    detect_silences,
    normalize,
    split,
)


def _duration_seconds(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate


def test_normalize_converts_to_mono_16khz_pcm(
    short_speech_m4a: Path, tmp_path: Path
):
    out = normalize(short_speech_m4a, tmp_path / "normalized.wav")
    info = sf.info(str(out))
    assert info.channels == 1
    assert info.samplerate == 16000
    assert info.format == "WAV"
    assert info.subtype == "PCM_16"


def test_detect_silences_finds_long_silence(two_tones_wav: Path):
    silences = detect_silences(
        two_tones_wav, noise_db=-30.0, min_duration_s=1.5
    )
    # There should be exactly one silence interval (the 4s gap)
    assert len(silences) == 1
    start, end = silences[0]
    # Silence starts somewhere around t=3.0 and ends around t=7.0
    assert 2.5 <= start <= 3.5
    assert 6.5 <= end <= 7.5


def test_split_produces_two_chunks_covering_full_audio(
    two_tones_wav: Path, tmp_path: Path
):
    silences = detect_silences(
        two_tones_wav, noise_db=-30.0, min_duration_s=1.5
    )
    chunks = split(two_tones_wav, silences, tmp_path / "chunks", max_chunk_s=90.0)
    assert len(chunks) == 2
    total = sum(_duration_seconds(c) for c in chunks)
    # Together the chunks cover ~10s (the silence is split between them)
    assert 9.5 <= total <= 10.5


def test_split_forces_cut_when_no_silence_long_enough(tmp_path: Path):
    # 5-minute tone with no silence
    long_wav = tmp_path / "long.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=300",
         "-ac", "1", "-ar", "16000",
         str(long_wav)],
        check=True,
    )
    chunks = split(long_wav, silences=[], out_dir=tmp_path / "chunks", max_chunk_s=90.0)
    # 300s / 90s = 4 chunks (3 of 90s + 1 of 30s)
    assert len(chunks) == 4


def test_detect_silences_on_audio_with_no_long_pause_returns_empty(
    tmp_path: Path,
):
    short = tmp_path / "noisy.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
         "-ac", "1", "-ar", "16000",
         str(short)],
        check=True,
    )
    silences = detect_silences(short, noise_db=-30.0, min_duration_s=1.5)
    assert silences == []
