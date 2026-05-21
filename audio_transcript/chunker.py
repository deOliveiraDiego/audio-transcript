"""ffmpeg-backed audio preprocessing: normalize, detect silences, split."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable

import soundfile as sf


def normalize(src: Path, dest: Path) -> Path:
    """Convert any ffmpeg-readable file into mono 16kHz PCM 16-bit WAV."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(dest),
        ],
        check=True,
    )
    return dest


_SILENCE_START_RE = re.compile(r"silence_start: ([\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end: ([\d.]+)")


def detect_silences(
    wav_path: Path, noise_db: float, min_duration_s: float
) -> list[tuple[float, float]]:
    """Return list of (start_s, end_s) tuples for each detected silence."""
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(wav_path),
            "-af", f"silencedetect=noise={noise_db}dB:duration={min_duration_s}",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    stderr = result.stderr
    starts = [float(m) for m in _SILENCE_START_RE.findall(stderr)]
    ends = [float(m) for m in _SILENCE_END_RE.findall(stderr)]
    # Pair them up; ffmpeg may report a start without an end if file ends in silence
    pairs: list[tuple[float, float]] = []
    for i, s in enumerate(starts):
        if i < len(ends):
            pairs.append((s, ends[i]))
    return pairs


def _audio_duration(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate


def _compute_cut_points(
    duration: float,
    silences: Iterable[tuple[float, float]],
    max_chunk_s: float,
) -> list[float]:
    """Return interior cut times. Cuts at midpoints of silences. Forces cuts
    when a chunk would exceed max_chunk_s without any silence inside it."""
    silence_mids = sorted((s + e) / 2 for s, e in silences)
    cuts: list[float] = []
    last_cut = 0.0
    i = 0
    while i < len(silence_mids):
        mid = silence_mids[i]
        if mid - last_cut > max_chunk_s:
            # Force a cut at last_cut + max_chunk_s before reaching this silence
            cuts.append(last_cut + max_chunk_s)
            last_cut = last_cut + max_chunk_s
            continue
        cuts.append(mid)
        last_cut = mid
        i += 1
    # Handle the tail: from last_cut to duration may also need forced cuts
    while duration - last_cut > max_chunk_s:
        cuts.append(last_cut + max_chunk_s)
        last_cut = last_cut + max_chunk_s
    return cuts


def split(
    wav_path: Path,
    silences: list[tuple[float, float]],
    out_dir: Path,
    max_chunk_s: float,
) -> list[Path]:
    """Split wav into chunks at silence midpoints (with forced cuts past
    max_chunk_s). Returns ordered list of chunk file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = _audio_duration(wav_path)
    cuts = _compute_cut_points(duration, silences, max_chunk_s)
    boundaries = [0.0, *cuts, duration]

    chunk_paths: list[Path] = []
    for idx in range(len(boundaries) - 1):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        chunk = out_dir / f"chunk_{idx + 1:03d}.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(wav_path),
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-c:a", "pcm_s16le",
                "-ac", "1", "-ar", "16000",
                str(chunk),
            ],
            check=True,
        )
        chunk_paths.append(chunk)
    return chunk_paths
