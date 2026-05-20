# Audio Transcript MVP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local web app that transcribes long audio recordings by reusing the Parakeet TDT model already installed on disk by the Handy app, chunking audio at silence boundaries, and producing a paragraph-formatted `.txt`.

**Architecture:** Single-process Python + FastAPI server on `localhost:8000`. Pipeline: upload → ffmpeg normalize → silence-based chunking → sequential sherpa-onnx inference per chunk → assemble into paragraphs. Progress streamed to browser via SSE. State persisted per-job as JSON on disk.

**Tech Stack:** Python 3.11+, FastAPI, sherpa-onnx (Python bindings), ffmpeg (subprocess), HTML/vanilla JS, pytest.

**Spec:** `docs/superpowers/specs/2026-05-20-audio-transcript-design.md`

---

## File Structure

Files created over the course of this plan:

```
audio-transcript/
├── pyproject.toml                       # Task 2
├── README.md                            # Task 12
├── docs/superpowers/spike/
│   └── 2026-05-20-onnx-compat-result.md # Task 1 (decision doc)
├── audio_transcript/
│   ├── __init__.py                      # Task 2
│   ├── config.py                        # Task 2 (env vars, paths)
│   ├── job.py                           # Task 3
│   ├── assembler.py                     # Task 4
│   ├── chunker.py                       # Task 5
│   ├── transcriber.py                   # Task 6
│   ├── pipeline.py                      # Task 8 (orchestrator)
│   ├── app.py                           # Tasks 7, 8, 9
│   └── static/
│       └── index.html                   # Task 10
└── tests/
    ├── __init__.py                      # Task 2
    ├── conftest.py                      # Task 5 (fixtures)
    ├── test_assembler.py                # Task 4
    ├── test_chunker.py                  # Task 5
    ├── test_transcriber.py              # Task 6
    ├── test_pipeline.py                 # Task 8
    └── test_app.py                      # Task 11
```

**Module responsibilities:**

- `config.py` — central place for env-var-driven settings (model dir, data dir, upload limit). Avoids scattering `os.environ.get` calls.
- `job.py` — `JobState` dataclass + JSON persistence (`save()`, `load()`).
- `assembler.py` — pure text combination logic (no IO, no deps).
- `chunker.py` — three pure functions wrapping ffmpeg: `normalize`, `detect_silences`, `split`.
- `transcriber.py` — singleton wrapper around sherpa-onnx; one method `transcribe(wav_path) -> str`.
- `pipeline.py` — async orchestrator that runs the full normalize→chunk→transcribe→assemble flow and publishes progress events.
- `app.py` — FastAPI app with endpoints and SSE wiring.

---

## Task 1: Spike — validate Handy's ONNX files with sherpa-onnx

**Goal:** Determine the exact model loading configuration to use in `transcriber.py`. Three possible outcomes, in preference order: (a) use Handy's files directly, (b) use Handy's files with renamed/symlinked paths, (c) download the official sherpa-onnx Parakeet TDT v3 package.

**Files:**
- Create: `docs/superpowers/spike/2026-05-20-onnx-compat-result.md`

This task is exploratory — no test, no commit until findings are written down.

- [ ] **Step 1: Create a throwaway venv for the spike**

```bash
cd /Users/deoliveiradiego/Projects/deoliveiratech/audio-transcript
python3.11 -m venv .spike-venv
source .spike-venv/bin/activate
pip install --quiet sherpa-onnx soundfile
```

Expected: installs successfully. Note any errors — sherpa-onnx wheels for arm64 macOS should exist.

- [ ] **Step 2: Inspect Handy's model directory**

```bash
ls -la "$HOME/Library/Application Support/com.pais.handy/models/parakeet-tdt-0.6b-v3-int8/"
```

Expected: see `encoder-model.int8.onnx`, `decoder_joint-model.int8.onnx`, `nemo128.onnx`. Note: there should also be a `tokens.txt` or similar — if missing, that's already a strong signal we need the fallback.

- [ ] **Step 3: Generate a short test wav with PT-BR speech**

Use macOS `say` to create a deterministic test audio:

```bash
mkdir -p /tmp/spike-audio
say -v Luciana -o /tmp/spike-audio/hello.aiff "Senhor, abençoe nosso pequeno grupo nesta noite."
ffmpeg -y -i /tmp/spike-audio/hello.aiff -ac 1 -ar 16000 -c:a pcm_s16le /tmp/spike-audio/hello.wav
ls -la /tmp/spike-audio/hello.wav
```

Expected: a ~3s 16kHz mono WAV file.

- [ ] **Step 4: Try Approach (a) — use Handy's files directly via sherpa-onnx Python API**

Create a throwaway script `/tmp/spike-audio/try_direct.py`:

```python
import os
import sherpa_onnx
import soundfile as sf

MODEL_DIR = os.path.expanduser(
    "~/Library/Application Support/com.pais.handy/models/parakeet-tdt-0.6b-v3-int8"
)

# Discover what API surface sherpa_onnx offers for Parakeet TDT
print("sherpa_onnx version:", sherpa_onnx.__version__)
print("OfflineRecognizer factory methods:")
for attr in dir(sherpa_onnx.OfflineRecognizer):
    if attr.startswith("from_"):
        print("  -", attr)

# Most likely: from_nemo_parakeet_tdt or from_transducer
try:
    recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_parakeet_tdt(
        encoder=f"{MODEL_DIR}/encoder-model.int8.onnx",
        decoder=f"{MODEL_DIR}/decoder_joint-model.int8.onnx",
        joiner="",  # Handy bundles decoder+joiner in one file
        tokens=f"{MODEL_DIR}/tokens.txt",
        num_threads=2,
        sample_rate=16000,
        feature_dim=128,
    )
    print("APPROACH (a) loaded model OK")

    stream = recognizer.create_stream()
    samples, sr = sf.read("/tmp/spike-audio/hello.wav", dtype="float32")
    stream.accept_waveform(sr, samples)
    recognizer.decode_stream(stream)
    print("APPROACH (a) transcription:", repr(stream.result.text))
except Exception as e:
    print("APPROACH (a) FAILED:", type(e).__name__, str(e))
```

Run it:

```bash
python /tmp/spike-audio/try_direct.py
```

Record outcome:
- If transcription produces recognizable PT-BR text → **Approach (a) works**, proceed to Step 7.
- If error is "file not found" for `tokens.txt` → try Step 5.
- If error mentions decoder/joiner separation → try Step 5.
- Any other error → record it verbatim and try Step 5.

- [ ] **Step 5: Try Approach (b) — symlink/rename Handy's files to sherpa-onnx conventions**

The sherpa-onnx team's official Parakeet TDT package typically uses names like `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `tokens.txt`. Try mapping:

```bash
mkdir -p /tmp/spike-audio/model-handy-mapped
HANDY_DIR="$HOME/Library/Application Support/com.pais.handy/models/parakeet-tdt-0.6b-v3-int8"
ln -sf "$HANDY_DIR/encoder-model.int8.onnx" /tmp/spike-audio/model-handy-mapped/encoder.onnx
ln -sf "$HANDY_DIR/decoder_joint-model.int8.onnx" /tmp/spike-audio/model-handy-mapped/decoder.onnx
ln -sf "$HANDY_DIR/nemo128.onnx" /tmp/spike-audio/model-handy-mapped/feature.onnx
ls -la /tmp/spike-audio/model-handy-mapped/
find "$HANDY_DIR" -name "*.txt" -o -name "*.json"
```

If a `tokens.txt` or `vocab.txt` is found in Handy's dir, symlink it too. If no tokens file exists at all in Handy's models dir, Approach (b) is dead and we must use Approach (c).

Re-run the Python script from Step 4 pointing at `/tmp/spike-audio/model-handy-mapped` instead. Record outcome.

- [ ] **Step 6: Try Approach (c) — download official sherpa-onnx Parakeet TDT v3 (fallback)**

```bash
mkdir -p "$HOME/Library/Application Support/audio-transcript/models"
cd "$HOME/Library/Application Support/audio-transcript/models"
# Find the latest official Parakeet TDT v3 package URL by searching:
#   https://github.com/k2-fsa/sherpa-onnx/releases
# Look for: sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2 (or similar)
# Download with curl, extract:
curl -L -o parakeet.tar.bz2 \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
tar -xjf parakeet.tar.bz2
ls -la sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/
```

If the exact filename has changed, browse https://github.com/k2-fsa/sherpa-onnx/releases manually. Adjust the URL.

Re-run the Python script from Step 4 pointing at the downloaded directory. Record outcome.

- [ ] **Step 7: Write the spike decision doc**

Create `docs/superpowers/spike/2026-05-20-onnx-compat-result.md` with this template, filled in based on what actually worked:

```markdown
# ONNX Compatibility Spike — Result

**Date:** 2026-05-20
**Outcome:** [Approach A / B / C] — [one line summary]

## What worked

[Concrete summary: which factory method, which file paths, what arguments]

## Final model configuration

- Model directory: `<absolute path>`
- Files used:
  - encoder: `<filename>`
  - decoder/joiner: `<filename(s)>`
  - tokens: `<filename>`
- sherpa-onnx factory method: `OfflineRecognizer.<method>(...)`
- Required arguments (full code snippet):

```python
recognizer = sherpa_onnx.OfflineRecognizer.<method>(
    ...exact args here...
)
```

## What did not work

[List each failed approach with the exact error message]

## Implications for `transcriber.py`

- Env var `HANDY_MODELS_DIR` should default to: `<the path that worked>`
- Loader function signature: `load_model(model_dir: Path) -> OfflineRecognizer`
- Tokens file lookup: `<filename pattern>`

## Test verification

- Test audio: `/tmp/spike-audio/hello.wav` (PT-BR, "Senhor, abençoe nosso pequeno grupo nesta noite.")
- Transcribed output: `<exact text returned>`
- Quality assessment: [acceptable / needs investigation]
```

- [ ] **Step 8: Clean up spike venv and commit decision doc**

```bash
deactivate 2>/dev/null || true
rm -rf .spike-venv /tmp/spike-audio
git add docs/superpowers/spike/2026-05-20-onnx-compat-result.md
git commit -m "spike: document ONNX compatibility result for Parakeet TDT"
```

Expected: commit succeeds, repo clean.

**Important:** Every subsequent task references the configuration locked down in this decision doc. If Approach (a) won, `transcriber.py` uses Handy's path. If (c) won, `transcriber.py` uses the downloaded path and `pyproject.toml` includes a download helper. The `config.py` module in Task 2 reflects whichever path won.

---

## Task 2: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `audio_transcript/__init__.py`
- Create: `audio_transcript/config.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "audio-transcript"
version = "0.1.0"
description = "Local web app to transcribe long audio recordings using Parakeet TDT."
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "python-multipart>=0.0.12",
    "sherpa-onnx>=1.10",
    "soundfile>=0.12",
    "sse-starlette>=2.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "httpx>=0.27",
]

[project.scripts]
audio-transcript = "audio_transcript.app:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["audio_transcript*"]

[tool.setuptools.package-data]
audio_transcript = ["static/*"]

[tool.pytest.ini_options]
markers = [
    "local: tests that require local model files and skip in CI",
]
asyncio_mode = "auto"
```

- [ ] **Step 2: Create empty package files**

```bash
mkdir -p audio_transcript/static tests
touch audio_transcript/__init__.py tests/__init__.py
```

- [ ] **Step 3: Write `audio_transcript/config.py`**

Use the model path locked in by the spike decision doc. The example below assumes Approach (a) won; substitute the actual path from the spike result if a different approach was chosen.

```python
"""Central configuration. All env-var reads happen here."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HANDY_MODEL_DIR = Path(
    "~/Library/Application Support/com.pais.handy/models/parakeet-tdt-0.6b-v3-int8"
).expanduser()

MODEL_DIR = Path(
    os.environ.get("HANDY_MODELS_DIR", str(DEFAULT_HANDY_MODEL_DIR))
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

# Silence detection
SILENCE_NOISE_DB = float(os.environ.get("AUDIO_TRANSCRIPT_SILENCE_DB", "-30"))
SILENCE_MIN_DURATION_S = float(os.environ.get("AUDIO_TRANSCRIPT_SILENCE_DUR", "1.5"))

# Max chunk length before forcing a cut even without silence
MAX_CHUNK_SECONDS = float(os.environ.get("AUDIO_TRANSCRIPT_MAX_CHUNK_S", "90"))
```

- [ ] **Step 4: Verify install works**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --quiet -e ".[dev]"
python -c "from audio_transcript import config; print(config.MODEL_DIR)"
```

Expected: prints the model directory path with no errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml audio_transcript/ tests/
git commit -m "scaffold: project layout, pyproject.toml, config module"
```

---

## Task 3: JobState module

**Files:**
- Create: `audio_transcript/job.py`
- Create: `tests/test_job.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_job.py`:

```python
from pathlib import Path

from audio_transcript.job import JobState, JobStatus


def test_jobstate_round_trip(tmp_path: Path):
    job = JobState(
        id="abc-123",
        status=JobStatus.PENDING,
        source_path=tmp_path / "source.m4a",
        total_chunks=0,
        processed_chunks=0,
        stage="pending",
    )
    job.save(tmp_path)
    loaded = JobState.load(tmp_path)

    assert loaded.id == "abc-123"
    assert loaded.status == JobStatus.PENDING
    assert loaded.source_path == tmp_path / "source.m4a"
    assert loaded.total_chunks == 0
    assert loaded.processed_chunks == 0


def test_jobstate_update_progress(tmp_path: Path):
    job = JobState(
        id="abc-123",
        status=JobStatus.TRANSCRIBING,
        source_path=tmp_path / "source.m4a",
        total_chunks=5,
        processed_chunks=2,
        stage="transcribing",
    )
    job.save(tmp_path)
    loaded = JobState.load(tmp_path)
    loaded.processed_chunks = 3
    loaded.save(tmp_path)
    reloaded = JobState.load(tmp_path)
    assert reloaded.processed_chunks == 3


def test_jobstate_failure_message(tmp_path: Path):
    job = JobState(
        id="abc",
        status=JobStatus.FAILED,
        source_path=tmp_path / "source.m4a",
        error_message="ffmpeg failed",
    )
    job.save(tmp_path)
    loaded = JobState.load(tmp_path)
    assert loaded.status == JobStatus.FAILED
    assert loaded.error_message == "ffmpeg failed"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_job.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'audio_transcript.job'`.

- [ ] **Step 3: Implement `audio_transcript/job.py`**

```python
"""Job state with JSON persistence on disk."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    NORMALIZING = "normalizing"
    CHUNKING = "chunking"
    TRANSCRIBING = "transcribing"
    ASSEMBLING = "assembling"
    DONE = "done"
    FAILED = "failed"


@dataclass
class JobState:
    id: str
    status: JobStatus
    source_path: Path
    stage: str = ""
    total_chunks: int = 0
    processed_chunks: int = 0
    result_path: Optional[Path] = None
    error_message: Optional[str] = None

    JSON_FILENAME = "job.json"

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["source_path"] = str(self.source_path)
        payload["result_path"] = str(self.result_path) if self.result_path else None
        (dir_path / self.JSON_FILENAME).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, dir_path: Path) -> "JobState":
        data = json.loads((dir_path / cls.JSON_FILENAME).read_text())
        return cls(
            id=data["id"],
            status=JobStatus(data["status"]),
            source_path=Path(data["source_path"]),
            stage=data.get("stage", ""),
            total_chunks=data.get("total_chunks", 0),
            processed_chunks=data.get("processed_chunks", 0),
            result_path=Path(data["result_path"]) if data.get("result_path") else None,
            error_message=data.get("error_message"),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_job.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add audio_transcript/job.py tests/test_job.py
git commit -m "feat: JobState dataclass with JSON persistence"
```

---

## Task 4: Assembler module (pure text combination)

**Files:**
- Create: `audio_transcript/assembler.py`
- Create: `tests/test_assembler.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_assembler.py`:

```python
from audio_transcript.assembler import combine


def test_combine_three_chunks_separated_by_double_newline():
    chunks = ["Senhor, abençoe.", "Ore por minha mãe.", "Obrigado a todos."]
    result = combine(chunks)
    assert result == "Senhor, abençoe.\n\nOre por minha mãe.\n\nObrigado a todos."


def test_combine_trims_whitespace_in_each_chunk():
    chunks = ["  hello  ", "\nworld\n"]
    result = combine(chunks)
    assert result == "hello\n\nworld"


def test_combine_drops_empty_chunks():
    chunks = ["one", "", "  ", "two"]
    result = combine(chunks)
    assert result == "one\n\ntwo"


def test_combine_single_chunk_no_trailing_separator():
    result = combine(["only one"])
    assert result == "only one"


def test_combine_empty_input_returns_empty_string():
    assert combine([]) == ""
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_assembler.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'audio_transcript.assembler'`.

- [ ] **Step 3: Implement `audio_transcript/assembler.py`**

```python
"""Combine transcribed chunk texts into a paragraph-formatted document."""
from __future__ import annotations

from collections.abc import Iterable


def combine(chunks: Iterable[str]) -> str:
    """Join chunk texts with blank lines between them.

    Each chunk becomes one paragraph in the output. Empty or whitespace-only
    chunks are dropped. Per-chunk leading/trailing whitespace is trimmed.
    """
    cleaned = [c.strip() for c in chunks]
    paragraphs = [c for c in cleaned if c]
    return "\n\n".join(paragraphs)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_assembler.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add audio_transcript/assembler.py tests/test_assembler.py
git commit -m "feat: assembler that joins chunk texts with paragraph breaks"
```

---

## Task 5: Chunker module (ffmpeg normalize + silence detect + split)

**Files:**
- Create: `audio_transcript/chunker.py`
- Create: `tests/conftest.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write the shared fixture in `tests/conftest.py`**

This fixture programmatically builds a 10-second test WAV with two short tones separated by a long silence, so chunker tests are reproducible without checking in binary files.

```python
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
```

- [ ] **Step 2: Write the failing chunker tests**

Create `tests/test_chunker.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_chunker.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'audio_transcript.chunker'`.

- [ ] **Step 4: Implement `audio_transcript/chunker.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_chunker.py -v
```

Expected: 5 passed. If `test_normalize_converts_to_mono_16khz_pcm` fails because the `say` command is unavailable, that fixture is mac-only — mark the test with `@pytest.mark.skipif(not shutil.which("say"), reason="macOS only")` and re-run.

- [ ] **Step 6: Commit**

```bash
git add audio_transcript/chunker.py tests/conftest.py tests/test_chunker.py
git commit -m "feat: chunker — ffmpeg normalize, silence detection, split"
```

---

## Task 6: Transcriber module (sherpa-onnx wrapper)

**Files:**
- Create: `audio_transcript/transcriber.py`
- Create: `tests/test_transcriber.py`

This task uses the model configuration locked in by Task 1's spike result. The code below assumes Approach (a) succeeded; if a different approach won, adjust the `_build_recognizer` body to match exactly what the spike doc says worked.

- [ ] **Step 1: Write the failing test**

Create `tests/test_transcriber.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from audio_transcript.transcriber import Transcriber


def test_transcriber_uses_injected_recognizer(tmp_path: Path):
    """The Transcriber must accept a pre-built recognizer for testability."""
    mock_stream = MagicMock()
    mock_stream.result.text = "olá mundo"
    mock_recognizer = MagicMock()
    mock_recognizer.create_stream.return_value = mock_stream

    fake_wav = tmp_path / "chunk.wav"
    # Write a tiny valid WAV header so soundfile can read it
    import soundfile as sf
    import numpy as np
    sf.write(str(fake_wav), np.zeros(16000, dtype="float32"), 16000)

    transcriber = Transcriber(recognizer=mock_recognizer)
    text = transcriber.transcribe(fake_wav)

    assert text == "olá mundo"
    mock_recognizer.create_stream.assert_called_once()
    mock_recognizer.decode_stream.assert_called_once_with(mock_stream)


@pytest.mark.local
def test_transcriber_real_model_on_pt_speech(tmp_path: Path):
    """Integration test — runs the actual Parakeet model on synthesized speech."""
    import subprocess
    aiff = tmp_path / "speech.aiff"
    wav = tmp_path / "speech.wav"
    subprocess.run(
        ["say", "-v", "Luciana", "-o", str(aiff),
         "Senhor, abençoe nosso pequeno grupo."],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        check=True,
    )

    transcriber = Transcriber.from_default_config()
    text = transcriber.transcribe(wav).lower()
    # Loose match — just expect SOME content with a key word
    assert any(word in text for word in ["senhor", "grupo", "abençoe", "abencoe"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_transcriber.py -v -m "not local"
```

Expected: FAIL with `ModuleNotFoundError: No module named 'audio_transcript.transcriber'`.

- [ ] **Step 3: Implement `audio_transcript/transcriber.py`**

The exact `_build_recognizer` body below assumes the spike confirmed Approach (a). If the spike concluded with a different config (e.g., `from_transducer` with mapped filenames, or the downloaded official model), copy the working `OfflineRecognizer.<method>(...)` call from `docs/superpowers/spike/2026-05-20-onnx-compat-result.md` and substitute it inside `_build_recognizer`.

```python
"""Wrap sherpa-onnx Parakeet TDT inference behind a small synchronous API."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import sherpa_onnx
import soundfile as sf

from audio_transcript import config


def _build_recognizer(model_dir: Path) -> sherpa_onnx.OfflineRecognizer:
    """Build a Parakeet TDT recognizer from the model directory.

    Configuration locked in by spike: docs/superpowers/spike/2026-05-20-onnx-compat-result.md
    """
    return sherpa_onnx.OfflineRecognizer.from_nemo_parakeet_tdt(
        encoder=str(model_dir / "encoder-model.int8.onnx"),
        decoder=str(model_dir / "decoder_joint-model.int8.onnx"),
        joiner="",
        tokens=str(model_dir / "tokens.txt"),
        num_threads=2,
        sample_rate=16000,
        feature_dim=128,
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
                "Install Handy (it downloads this model) or set HANDY_MODELS_DIR."
            )
        return cls(recognizer=_build_recognizer(config.MODEL_DIR))

    def transcribe(self, wav_path: Path) -> str:
        samples, sample_rate = sf.read(str(wav_path), dtype="float32")
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self._recognizer.decode_stream(stream)
        return stream.result.text
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_transcriber.py -v -m "not local"
```

Expected: 1 passed (the unit test). The `@pytest.mark.local` integration test is skipped.

- [ ] **Step 5: Run the local integration test to verify against the real model**

```bash
pytest tests/test_transcriber.py -v -m local
```

Expected: PASS (transcribes synthesized speech and finds a key word). If it fails, the spike's locked-in config is wrong — revisit Task 1's decision doc and fix `_build_recognizer`.

- [ ] **Step 6: Commit**

```bash
git add audio_transcript/transcriber.py tests/test_transcriber.py
git commit -m "feat: transcriber wrapping sherpa-onnx Parakeet TDT"
```

---

## Task 7: FastAPI app skeleton + upload endpoint + result endpoint

**Files:**
- Create: `audio_transcript/app.py`

The pipeline orchestration and SSE come in Task 8. This task only stands up the app, upload, and result download — the upload endpoint creates the job and returns the id; the actual processing is wired in Task 8.

- [ ] **Step 1: Write `audio_transcript/app.py` (initial version)**

```python
"""FastAPI app: upload, result download, and (in Task 8) SSE streaming."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from audio_transcript import config
from audio_transcript.job import JobState, JobStatus

app = FastAPI(title="audio-transcript")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        # Placeholder until Task 10 lands index.html
        return HTMLResponse("<h1>audio-transcript</h1><p>UI not built yet.</p>")
    return HTMLResponse(index_path.read_text())


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    job_id = str(uuid.uuid4())
    job_dir = config.UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "audio").suffix or ".bin"
    source_path = job_dir / f"source{suffix}"

    total_bytes = 0
    with source_path.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > config.MAX_UPLOAD_BYTES:
                out.close()
                source_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds limit ({config.MAX_UPLOAD_BYTES} bytes).",
                )
            out.write(chunk)

    job = JobState(
        id=job_id,
        status=JobStatus.PENDING,
        source_path=source_path,
        stage="pending",
    )
    job.save(job_dir)

    # Task 8 will dispatch the pipeline here.
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job_dir = config.UPLOADS_DIR / job_id
    if not (job_dir / JobState.JSON_FILENAME).exists():
        raise HTTPException(status_code=404, detail="Job not found")
    job = JobState.load(job_dir)
    return {
        "id": job.id,
        "status": job.status.value,
        "stage": job.stage,
        "total_chunks": job.total_chunks,
        "processed_chunks": job.processed_chunks,
        "error_message": job.error_message,
    }


@app.get("/jobs/{job_id}/result")
async def get_result(job_id: str) -> FileResponse:
    job_dir = config.UPLOADS_DIR / job_id
    if not (job_dir / JobState.JSON_FILENAME).exists():
        raise HTTPException(status_code=404, detail="Job not found")
    job = JobState.load(job_dir)
    if job.status != JobStatus.DONE or not job.result_path:
        raise HTTPException(status_code=409, detail="Job not done yet")
    if not job.result_path.exists():
        raise HTTPException(status_code=410, detail="Result file missing")
    return FileResponse(
        str(job.result_path),
        media_type="text/plain",
        filename="transcript.txt",
    )


def main() -> None:
    """Entry point for the `audio-transcript` console script."""
    import uvicorn
    uvicorn.run(
        "audio_transcript.app:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the app boots**

```bash
python -c "from audio_transcript.app import app; print(app.title)"
```

Expected: prints `audio-transcript` with no import error.

- [ ] **Step 3: Manually verify upload via curl (transient — no test commit yet)**

```bash
audio-transcript &
APP_PID=$!
sleep 2
# Generate a tiny test file
ffmpeg -y -loglevel error -f lavfi -i sine=frequency=440:duration=1 -ac 1 -ar 16000 /tmp/tiny.wav
curl -s -X POST -F "file=@/tmp/tiny.wav" http://127.0.0.1:8000/upload
echo
kill $APP_PID
wait 2>/dev/null
```

Expected: response is `{"job_id": "<uuid>"}`. Inspect `data/uploads/<uuid>/`: should contain `source.wav` and `job.json` with status `pending`.

- [ ] **Step 4: Commit**

```bash
git add audio_transcript/app.py
git commit -m "feat: FastAPI app skeleton with upload and result endpoints"
```

---

## Task 8: Pipeline orchestrator + SSE progress streaming

**Files:**
- Create: `audio_transcript/pipeline.py`
- Modify: `audio_transcript/app.py` (add SSE endpoint, wire pipeline into upload)
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing pipeline test**

Create `tests/test_pipeline.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from audio_transcript.job import JobState, JobStatus
from audio_transcript.pipeline import run_pipeline


@pytest.mark.asyncio
async def test_run_pipeline_happy_path(tmp_path: Path, monkeypatch):
    # Build a tiny test wav as source
    import subprocess
    source = tmp_path / "source.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000:d=2",
         "-f", "lavfi", "-i", "sine=frequency=880:duration=3",
         "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]",
         "-map", "[out]", "-ac", "1", "-ar", "16000",
         str(source)],
        check=True,
    )

    job = JobState(
        id="test-job",
        status=JobStatus.PENDING,
        source_path=source,
        stage="pending",
    )
    job.save(tmp_path)

    fake_transcriber = MagicMock()
    fake_transcriber.transcribe.side_effect = ["first chunk", "second chunk"]

    events: list[dict] = []
    async def collect(event: dict) -> None:
        events.append(event)

    await run_pipeline(
        job_dir=tmp_path,
        transcriber=fake_transcriber,
        publish=collect,
        silence_noise_db=-30.0,
        silence_min_duration_s=1.5,
        max_chunk_s=90.0,
    )

    final = JobState.load(tmp_path)
    assert final.status == JobStatus.DONE
    assert final.result_path is not None
    assert final.result_path.exists()
    transcript = final.result_path.read_text()
    assert "first chunk" in transcript
    assert "second chunk" in transcript
    # Events should include at least one progress event and a done event
    stages = [e.get("event") for e in events]
    assert "done" in stages


@pytest.mark.asyncio
async def test_run_pipeline_chunk_failure_inserts_placeholder(
    tmp_path: Path,
):
    import subprocess
    source = tmp_path / "source.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000:d=2",
         "-f", "lavfi", "-i", "sine=frequency=880:duration=3",
         "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]",
         "-map", "[out]", "-ac", "1", "-ar", "16000",
         str(source)],
        check=True,
    )

    job = JobState(
        id="test-job",
        status=JobStatus.PENDING,
        source_path=source,
        stage="pending",
    )
    job.save(tmp_path)

    flaky_transcriber = MagicMock()
    flaky_transcriber.transcribe.side_effect = [
        "ok first",
        RuntimeError("model boom"),
    ]

    async def noop(event: dict) -> None:
        pass

    await run_pipeline(
        job_dir=tmp_path,
        transcriber=flaky_transcriber,
        publish=noop,
        silence_noise_db=-30.0,
        silence_min_duration_s=1.5,
        max_chunk_s=90.0,
    )

    final = JobState.load(tmp_path)
    assert final.status == JobStatus.DONE
    text = final.result_path.read_text()
    assert "ok first" in text
    assert "TRECHO NÃO TRANSCRITO" in text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'audio_transcript.pipeline'`.

- [ ] **Step 3: Implement `audio_transcript/pipeline.py`**

```python
"""Async orchestrator: normalize → chunk → transcribe → assemble.

`publish` is an async callable taking a dict and forwarding it to any
listening SSE clients. Events use the shape:
    {"event": "<stage>", "data": {...}}
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable

from audio_transcript import chunker
from audio_transcript.assembler import combine
from audio_transcript.job import JobState, JobStatus

log = logging.getLogger(__name__)

PublishFn = Callable[[dict], Awaitable[None]]


def _format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


async def run_pipeline(
    *,
    job_dir: Path,
    transcriber,
    publish: PublishFn,
    silence_noise_db: float,
    silence_min_duration_s: float,
    max_chunk_s: float,
) -> None:
    """Run the full pipeline. Updates job.json and publishes events."""
    job = JobState.load(job_dir)
    try:
        await _set_stage(job, job_dir, JobStatus.NORMALIZING, "normalizing", publish)
        normalized = await asyncio.to_thread(
            chunker.normalize, job.source_path, job_dir / "normalized.wav"
        )

        await _set_stage(job, job_dir, JobStatus.CHUNKING, "chunking", publish)
        silences = await asyncio.to_thread(
            chunker.detect_silences,
            normalized, silence_noise_db, silence_min_duration_s,
        )
        chunks = await asyncio.to_thread(
            chunker.split, normalized, silences, job_dir / "chunks", max_chunk_s,
        )
        if not chunks:
            raise RuntimeError("No audio chunks produced — input may be empty.")

        job.total_chunks = len(chunks)
        job.processed_chunks = 0
        await _set_stage(job, job_dir, JobStatus.TRANSCRIBING, "transcribing", publish)

        texts: list[str] = []
        # Compute approximate chunk boundary times for placeholder messages
        boundaries = [0.0]
        for c in chunks:
            import soundfile as sf
            boundaries.append(boundaries[-1] + sf.info(str(c)).frames / sf.info(str(c)).samplerate)

        for i, chunk_path in enumerate(chunks):
            try:
                text = await asyncio.to_thread(transcriber.transcribe, chunk_path)
            except Exception as exc:
                log.warning("chunk %s failed: %s", chunk_path, exc)
                start = _format_time(boundaries[i])
                end = _format_time(boundaries[i + 1])
                text = f"[TRECHO NÃO TRANSCRITO: {start}–{end}]"
            texts.append(text)
            job.processed_chunks = i + 1
            job.save(job_dir)
            await publish({
                "event": "progress",
                "data": {
                    "processed": job.processed_chunks,
                    "total": job.total_chunks,
                    "stage": "transcribing",
                },
            })

        await _set_stage(job, job_dir, JobStatus.ASSEMBLING, "assembling", publish)
        final_text = combine(texts)
        result_path = job_dir / "transcript.txt"
        result_path.write_text(final_text, encoding="utf-8")

        job.result_path = result_path
        job.status = JobStatus.DONE
        job.stage = "done"
        job.save(job_dir)
        await publish({"event": "done", "data": {"job_id": job.id}})

    except Exception as exc:
        log.exception("pipeline failed for job %s", job.id)
        job.status = JobStatus.FAILED
        job.stage = "failed"
        job.error_message = str(exc)
        job.save(job_dir)
        await publish({"event": "error", "data": {"message": str(exc)}})


async def _set_stage(
    job: JobState, job_dir: Path, status: JobStatus, stage: str, publish: PublishFn
) -> None:
    job.status = status
    job.stage = stage
    job.save(job_dir)
    await publish({"event": "progress", "data": {"stage": stage}})
```

- [ ] **Step 4: Run pipeline tests to verify they pass**

```bash
pytest tests/test_pipeline.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Wire pipeline into `app.py` and add SSE endpoint**

Replace the contents of `audio_transcript/app.py` with the version below. The changes from Task 7: add a process-wide `Transcriber` singleton, add an in-memory per-job event broker, dispatch `run_pipeline` from `/upload`, and add the `/jobs/{id}/stream` SSE endpoint.

```python
"""FastAPI app: upload, SSE progress, result download."""
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from audio_transcript import config
from audio_transcript.job import JobState, JobStatus
from audio_transcript.pipeline import run_pipeline
from audio_transcript.transcriber import Transcriber

STATIC_DIR = Path(__file__).parent / "static"


class JobBroker:
    """Per-job in-memory pub/sub for SSE clients."""

    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue]] = {}

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        if job_id in self._queues:
            self._queues[job_id] = [x for x in self._queues[job_id] if x is not q]
            if not self._queues[job_id]:
                del self._queues[job_id]

    async def publish(self, job_id: str, event: dict) -> None:
        for q in self._queues.get(job_id, []):
            await q.put(event)


broker = JobBroker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.transcriber = Transcriber.from_default_config()
    yield


app = FastAPI(title="audio-transcript", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>audio-transcript</h1><p>UI not built yet.</p>")
    return HTMLResponse(index_path.read_text())


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    job_id = str(uuid.uuid4())
    job_dir = config.UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "audio").suffix or ".bin"
    source_path = job_dir / f"source{suffix}"

    total_bytes = 0
    with source_path.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > config.MAX_UPLOAD_BYTES:
                out.close()
                source_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds limit ({config.MAX_UPLOAD_BYTES} bytes).",
                )
            out.write(chunk)

    JobState(
        id=job_id,
        status=JobStatus.PENDING,
        source_path=source_path,
        stage="pending",
    ).save(job_dir)

    async def _publish(event: dict) -> None:
        await broker.publish(job_id, event)

    asyncio.create_task(
        run_pipeline(
            job_dir=job_dir,
            transcriber=app.state.transcriber,
            publish=_publish,
            silence_noise_db=config.SILENCE_NOISE_DB,
            silence_min_duration_s=config.SILENCE_MIN_DURATION_S,
            max_chunk_s=config.MAX_CHUNK_SECONDS,
        )
    )

    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job_dir = config.UPLOADS_DIR / job_id
    if not (job_dir / JobState.JSON_FILENAME).exists():
        raise HTTPException(status_code=404, detail="Job not found")
    job = JobState.load(job_dir)
    return {
        "id": job.id,
        "status": job.status.value,
        "stage": job.stage,
        "total_chunks": job.total_chunks,
        "processed_chunks": job.processed_chunks,
        "error_message": job.error_message,
    }


@app.get("/jobs/{job_id}/stream")
async def stream(job_id: str) -> EventSourceResponse:
    job_dir = config.UPLOADS_DIR / job_id
    if not (job_dir / JobState.JSON_FILENAME).exists():
        raise HTTPException(status_code=404, detail="Job not found")

    queue = broker.subscribe(job_id)

    async def event_generator():
        # Send current snapshot first so reconnecting clients catch up
        job = JobState.load(job_dir)
        yield {
            "event": "progress",
            "data": (
                f'{{"stage": "{job.stage}", "processed": {job.processed_chunks}, '
                f'"total": {job.total_chunks}, "status": "{job.status.value}"}}'
            ),
        }
        if job.status == JobStatus.DONE:
            yield {"event": "done", "data": f'{{"job_id": "{job_id}"}}'}
            return
        if job.status == JobStatus.FAILED:
            yield {
                "event": "error",
                "data": f'{{"message": {job.error_message!r}}}',
            }
            return

        try:
            while True:
                event = await queue.get()
                import json
                yield {"event": event["event"], "data": json.dumps(event["data"])}
                if event["event"] in {"done", "error"}:
                    return
        finally:
            broker.unsubscribe(job_id, queue)

    return EventSourceResponse(event_generator())


@app.get("/jobs/{job_id}/result")
async def get_result(job_id: str) -> FileResponse:
    job_dir = config.UPLOADS_DIR / job_id
    if not (job_dir / JobState.JSON_FILENAME).exists():
        raise HTTPException(status_code=404, detail="Job not found")
    job = JobState.load(job_dir)
    if job.status != JobStatus.DONE or not job.result_path:
        raise HTTPException(status_code=409, detail="Job not done yet")
    if not job.result_path.exists():
        raise HTTPException(status_code=410, detail="Result file missing")
    return FileResponse(
        str(job.result_path),
        media_type="text/plain",
        filename="transcript.txt",
    )


def main() -> None:
    import uvicorn
    uvicorn.run(
        "audio_transcript.app:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add audio_transcript/pipeline.py audio_transcript/app.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestrator + SSE progress streaming"
```

---

## Task 9: Frontend HTML/JS

**Files:**
- Create: `audio_transcript/static/index.html`

- [ ] **Step 1: Write `audio_transcript/static/index.html`**

```html
<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>audio-transcript</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 4rem auto; padding: 0 1rem; color: #1a1a1a; }
    h1 { font-weight: 600; }
    .drop {
      border: 2px dashed #999; border-radius: 12px; padding: 4rem 2rem;
      text-align: center; cursor: pointer; transition: background 0.15s;
    }
    .drop.dragover { background: #f0f7ff; border-color: #4a90e2; }
    .progress { margin-top: 2rem; }
    .bar { height: 12px; background: #eee; border-radius: 6px; overflow: hidden; }
    .bar > div { height: 100%; background: #4a90e2; transition: width 0.2s; }
    .stage { margin-top: 0.5rem; color: #555; font-size: 0.9rem; }
    .error { background: #fde7e7; border: 1px solid #e74c3c; padding: 1rem; border-radius: 8px; color: #b00020; }
    button { font: inherit; padding: 0.75rem 1.5rem; border-radius: 8px; border: 0; background: #4a90e2; color: white; cursor: pointer; }
    pre { background: #f6f6f6; padding: 1rem; border-radius: 6px; max-height: 200px; overflow: auto; white-space: pre-wrap; }
    .hidden { display: none; }
  </style>
</head>
<body>
  <h1>Transcrição de áudio</h1>

  <div id="drop" class="drop">
    Arraste um arquivo de áudio aqui ou clique para escolher.
    <input id="file" type="file" accept="audio/*" hidden />
  </div>

  <div id="progress" class="progress hidden">
    <div class="bar"><div id="fill" style="width: 0%"></div></div>
    <div id="stage" class="stage">Iniciando…</div>
  </div>

  <div id="done" class="hidden">
    <h2>Pronto!</h2>
    <a id="download" href="#"><button>Baixar transcrição (.txt)</button></a>
    <h3>Prévia</h3>
    <pre id="preview"></pre>
    <p><a href="/">Nova transcrição</a></p>
  </div>

  <div id="error" class="error hidden"></div>

<script>
const drop = document.getElementById("drop");
const fileInput = document.getElementById("file");
const progress = document.getElementById("progress");
const fill = document.getElementById("fill");
const stage = document.getElementById("stage");
const doneBox = document.getElementById("done");
const errorBox = document.getElementById("error");

drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragover"); });
drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("dragover");
  if (e.dataTransfer.files.length) upload(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) upload(fileInput.files[0]);
});

async function upload(file) {
  drop.classList.add("hidden");
  progress.classList.remove("hidden");
  stage.textContent = "Enviando arquivo…";

  const form = new FormData();
  form.append("file", file);
  let res;
  try {
    res = await fetch("/upload", { method: "POST", body: form });
  } catch (err) {
    return showError("Falha ao enviar o arquivo: " + err.message);
  }
  if (!res.ok) {
    const body = await res.text();
    return showError("Upload falhou (" + res.status + "): " + body);
  }
  const { job_id } = await res.json();
  watch(job_id);
}

function watch(jobId) {
  const es = new EventSource(`/jobs/${jobId}/stream`);
  es.addEventListener("progress", (e) => {
    const d = JSON.parse(e.data);
    if (typeof d.processed === "number" && typeof d.total === "number" && d.total > 0) {
      fill.style.width = ((d.processed / d.total) * 100).toFixed(1) + "%";
      stage.textContent = `Transcrevendo ${d.processed} de ${d.total} trechos…`;
    } else if (d.stage) {
      const labels = {
        normalizing: "Normalizando áudio…",
        chunking: "Dividindo em trechos…",
        transcribing: "Transcrevendo…",
        assembling: "Montando texto final…",
      };
      stage.textContent = labels[d.stage] || d.stage;
    }
  });
  es.addEventListener("done", (e) => {
    es.close();
    const { job_id } = JSON.parse(e.data);
    showDone(job_id);
  });
  es.addEventListener("error", (e) => {
    es.close();
    let msg = "Erro desconhecido.";
    try { msg = JSON.parse(e.data).message; } catch (_) {}
    showError(msg);
  });
}

async function showDone(jobId) {
  progress.classList.add("hidden");
  doneBox.classList.remove("hidden");
  document.getElementById("download").href = `/jobs/${jobId}/result`;
  const text = await fetch(`/jobs/${jobId}/result`).then((r) => r.text());
  document.getElementById("preview").textContent = text.slice(0, 500);
}

function showError(msg) {
  progress.classList.add("hidden");
  errorBox.classList.remove("hidden");
  errorBox.textContent = msg;
}
</script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add audio_transcript/static/index.html
git commit -m "feat: drag-and-drop UI with SSE progress and result preview"
```

---

## Task 10: End-to-end integration test

**Files:**
- Create: `tests/test_app.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_app.py`:

```python
import asyncio
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app_with_fake_transcriber(tmp_path, monkeypatch):
    """Build a fresh FastAPI app instance with the transcriber stubbed
    and the data dir redirected to tmp_path."""
    monkeypatch.setenv("AUDIO_TRANSCRIPT_DATA_DIR", str(tmp_path / "data"))

    import importlib
    from audio_transcript import config
    importlib.reload(config)
    from audio_transcript import app as app_module
    importlib.reload(app_module)

    fake = MagicMock()
    fake.transcribe.side_effect = lambda p: f"text-of-{Path(p).stem}"

    with patch(
        "audio_transcript.app.Transcriber.from_default_config",
        return_value=fake,
    ):
        yield app_module.app


@pytest.mark.asyncio
async def test_upload_then_stream_then_result(
    app_with_fake_transcriber, tmp_path: Path
):
    # Build a small 10s audio file with one obvious silence
    audio = tmp_path / "in.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000:d=2",
         "-f", "lavfi", "-i", "sine=frequency=880:duration=3",
         "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]",
         "-map", "[out]", "-ac", "1", "-ar", "16000",
         str(audio)],
        check=True,
    )

    transport = ASGITransport(app=app_with_fake_transcriber)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Manually trigger app lifespan to instantiate the transcriber
        async with app_with_fake_transcriber.router.lifespan_context(
            app_with_fake_transcriber
        ):
            with audio.open("rb") as f:
                res = await client.post(
                    "/upload",
                    files={"file": ("in.wav", f, "audio/wav")},
                )
            assert res.status_code == 200
            job_id = res.json()["job_id"]

            # Poll job status until done (or fail after 30s)
            for _ in range(60):
                status_res = await client.get(f"/jobs/{job_id}")
                assert status_res.status_code == 200
                status = status_res.json()["status"]
                if status == "done":
                    break
                if status == "failed":
                    pytest.fail(f"job failed: {status_res.json()}")
                await asyncio.sleep(0.5)
            else:
                pytest.fail("job did not finish in 30s")

            result = await client.get(f"/jobs/{job_id}/result")
            assert result.status_code == 200
            assert "text-of-chunk_001" in result.text
            assert "text-of-chunk_002" in result.text
            assert result.text.count("\n\n") >= 1


@pytest.mark.asyncio
async def test_upload_rejects_huge_file(app_with_fake_transcriber, monkeypatch):
    monkeypatch.setenv("AUDIO_TRANSCRIPT_MAX_UPLOAD_BYTES", "1024")
    import importlib
    from audio_transcript import config
    importlib.reload(config)
    from audio_transcript import app as app_module
    importlib.reload(app_module)

    transport = ASGITransport(app=app_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with patch(
            "audio_transcript.app.Transcriber.from_default_config",
            return_value=MagicMock(),
        ):
            async with app_module.app.router.lifespan_context(app_module.app):
                res = await client.post(
                    "/upload",
                    files={"file": ("big.bin", b"x" * 2048, "application/octet-stream")},
                )
                assert res.status_code == 413
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_app.py -v
```

Expected: at least the first test runs (may pass if everything from earlier tasks is correct). If it fails, fix what's broken in `app.py`/`pipeline.py` and re-run.

- [ ] **Step 3: Make tests pass (fixing any wiring issues exposed)**

Iterate on `app.py` / `pipeline.py` until both integration tests pass.

```bash
pytest tests/test_app.py -v
```

Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_app.py
git commit -m "test: end-to-end upload/stream/result integration"
```

---

## Task 11: Manual smoke test with real audio

This task has no code changes — it's a manual verification using a real recording, in line with the project guideline "for UI or frontend changes, start the dev server and use the feature in a browser before reporting the task as complete."

- [ ] **Step 1: Start the server**

```bash
source .venv/bin/activate
audio-transcript
```

Expected: server starts on `http://127.0.0.1:8000`, model loads without errors.

- [ ] **Step 2: Open the UI in a browser**

```bash
open http://127.0.0.1:8000
```

Expected: drag-and-drop UI loads.

- [ ] **Step 3: Generate a 60-second multi-segment test audio**

```bash
TMP=/tmp/audio-transcript-smoke
mkdir -p $TMP
say -v Luciana -o $TMP/p1.aiff "Senhor, eu peço por minha mãe que está doente."
say -v Luciana -o $TMP/p2.aiff "Quero agradecer pela bênção do trabalho novo."
say -v Luciana -o $TMP/p3.aiff "Ore por meu casamento, está passando por um momento difícil."

ffmpeg -y -loglevel error \
  -i $TMP/p1.aiff \
  -f lavfi -i anullsrc=channel_layout=mono:sample_rate=16000:d=3 \
  -i $TMP/p2.aiff \
  -f lavfi -i anullsrc=channel_layout=mono:sample_rate=16000:d=3 \
  -i $TMP/p3.aiff \
  -filter_complex "[0:a][1:a][2:a][3:a][4:a]concat=n=5:v=0:a=1[out]" \
  -map "[out]" -ac 1 -ar 16000 \
  $TMP/smoke.m4a
ls -la $TMP/smoke.m4a
```

Expected: `smoke.m4a` exists, ~30-40s long, three speech segments separated by 3s silences.

- [ ] **Step 4: Upload via the browser UI**

Drag `$TMP/smoke.m4a` onto the drop zone. Observe:

- Stage labels progress: "Enviando arquivo…" → "Normalizando áudio…" → "Dividindo em trechos…" → "Transcrevendo X de 3 trechos…" → "Montando texto final…"
- Progress bar advances to 100%.
- "Pronto!" appears with a "Baixar transcrição (.txt)" button.
- Preview shows 3 paragraphs, one per prayer request.

- [ ] **Step 5: Download and inspect the transcript**

Click the download button. Open the file.

Expected: 3 paragraphs separated by blank lines, each paragraph containing the words from one of the three sentences (allowing for ASR errors). If any paragraph is `[TRECHO NÃO TRANSCRITO: …]`, that's a model-quality issue worth recording but not a code bug.

- [ ] **Step 6: Stop the server and clean up**

```bash
# Stop the server (Ctrl+C in the terminal where it's running)
rm -rf $TMP
```

No commit for this task (no code changed). If anything failed, file the issue in the README's troubleshooting section in Task 12.

---

## Task 12: README + final polish

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# audio-transcript

Transcribes long audio recordings into a paragraph-formatted `.txt`, using the Parakeet TDT 0.6B v3 model that the [Handy](https://handy.computer) app already installs locally. Runs as a local web app — no API calls, no recurring cost, works offline.

## Requirements

- macOS (the default model path is `~/Library/Application Support/com.pais.handy/...`; on Linux, set `HANDY_MODELS_DIR`)
- Python 3.11+
- [ffmpeg](https://ffmpeg.org/) on `$PATH`
- [Handy](https://handy.computer) installed (so the Parakeet model is on disk) — or download the model manually and set `HANDY_MODELS_DIR`

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
audio-transcript
# server starts at http://127.0.0.1:8000
```

Open the URL, drag a recording onto the page, wait for the progress bar to fill, download the `.txt`.

## How it works

1. Upload writes the file to `data/uploads/<uuid>/source.<ext>`.
2. ffmpeg normalizes it to mono 16kHz PCM.
3. ffmpeg's `silencedetect` finds pauses ≥ 1.5s (configurable).
4. The audio is split into chunks at those pauses (forcing a cut after 90s if no pause was found).
5. Each chunk is transcribed by sherpa-onnx (Parakeet TDT). If a chunk fails, the pipeline inserts `[TRECHO NÃO TRANSCRITO: hh:mm:ss–hh:mm:ss]` and continues.
6. Chunk texts are joined with blank lines — each chunk becomes one paragraph.

## Configuration

All settings come from env vars (see `audio_transcript/config.py`).

| Variable | Default | Purpose |
|---|---|---|
| `HANDY_MODELS_DIR` | `~/Library/Application Support/com.pais.handy/models/parakeet-tdt-0.6b-v3-int8` | Where to find Parakeet ONNX files |
| `AUDIO_TRANSCRIPT_DATA_DIR` | `./data` | Where uploads and outputs live |
| `AUDIO_TRANSCRIPT_MAX_UPLOAD_BYTES` | `1073741824` (1 GiB) | Reject uploads larger than this |
| `AUDIO_TRANSCRIPT_HOST` | `127.0.0.1` | Bind host |
| `AUDIO_TRANSCRIPT_PORT` | `8000` | Bind port |
| `AUDIO_TRANSCRIPT_SILENCE_DB` | `-30` | dB threshold below which counts as silence |
| `AUDIO_TRANSCRIPT_SILENCE_DUR` | `1.5` | Minimum silence duration (s) to count as a paragraph break |
| `AUDIO_TRANSCRIPT_MAX_CHUNK_S` | `90` | Force a cut after this many seconds even without silence |

## Tests

```bash
pytest -m "not local"   # CI-safe (no real model)
pytest -m local         # Runs against real Parakeet (requires HANDY_MODELS_DIR populated)
```

## Troubleshooting

- **`Modelo Parakeet não encontrado`** at boot: install Handy or set `HANDY_MODELS_DIR` to a directory containing the Parakeet ONNX files.
- **Upload hangs or times out**: check `ffmpeg` is on `$PATH` (`which ffmpeg`).
- **Empty transcript**: input audio may be entirely below the silence threshold. Try lowering `AUDIO_TRANSCRIPT_SILENCE_DB` (e.g., `-40`).
- **Chunks coming out as `[TRECHO NÃO TRANSCRITO]`**: a model inference error on that chunk. Re-running the same file usually succeeds — Parakeet inference is non-deterministic at the edges.

## Out of scope (deliberately)

- Speaker diarization
- Timestamps inside the transcript
- LLM-based cleanup / summarization
- Hosted deployment
- History UI of past jobs

See `docs/superpowers/specs/2026-05-20-audio-transcript-design.md` for the full design.
```

- [ ] **Step 2: Run all tests one final time**

```bash
pytest -v -m "not local"
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with install, config, troubleshooting"
```

---

## Self-Review

**Spec coverage check** (skimmed each section of the spec):

| Spec section | Implementing task(s) |
|---|---|
| Decisões de produto (Parakeet, PT-BR, web local, texto cru) | Tasks 1, 2, 6, 9, 10 |
| Stack (FastAPI + sherpa-onnx + ffmpeg + HTML vanilla) | Tasks 2, 7, 8, 9 |
| Arquitetura (single-process, localhost:8000, JSON state) | Tasks 2, 3, 7, 8 |
| Endpoints (`/`, `/upload`, `/jobs/{id}/stream`, `/jobs/{id}/result`, `/jobs/{id}`) | Tasks 7, 8 |
| Componentes (5 modules + static) | Tasks 2-9 (one per module) |
| Fluxo de dados (upload → normalize → detect → split → transcribe → assemble → done) | Task 8 (pipeline) |
| UI states (vazio / progresso / concluído / erro) | Task 9 |
| Tratamento de erros (7 cases in the spec table) | Tasks 7 (413), 8 (most), 6 (missing model), 9 (display) |
| Testes TDD (assembler, chunker, transcriber, app) | Tasks 3, 4, 5, 6, 8, 10 |
| Layout do repositório | Tasks 2, 3-10 |
| Risco aberto (ONNX compat + fallbacks) | Task 1 (spike) |
| Critérios de aceitação do MVP (5 items) | Task 11 (manual smoke test) |

No gaps identified.

**Placeholder scan:** Reviewed all tasks. Every code step has executable code. Every test step has the failing test followed by the implementation. No "TBD" / "add appropriate error handling" / "similar to Task N" patterns. The only conditional language is in Task 6 ("if the spike concluded with a different config, substitute…") which is explicit and references the concrete decision doc from Task 1.

**Type consistency:** Verified that:
- `JobState` fields used in Task 3 match what `pipeline.py` (Task 8) and `app.py` (Tasks 7, 8) reference (`id`, `status`, `source_path`, `stage`, `total_chunks`, `processed_chunks`, `result_path`, `error_message`).
- `JobStatus` enum values used consistently (`PENDING`, `NORMALIZING`, `CHUNKING`, `TRANSCRIBING`, `ASSEMBLING`, `DONE`, `FAILED`).
- `Transcriber.transcribe(wav_path)` returns `str` in Task 6 and is consumed as `str` in Task 8.
- `chunker.normalize(src, dest)`, `chunker.detect_silences(wav, noise_db, min_duration_s)`, `chunker.split(wav, silences, out_dir, max_chunk_s)` signatures match between Task 5's definitions and Task 8's calls.
- `assembler.combine(chunks)` signature matches between Task 4 and Task 8.
- `config` module names (`MODEL_DIR`, `DATA_DIR`, `UPLOADS_DIR`, `MAX_UPLOAD_BYTES`, `HOST`, `PORT`, `SILENCE_NOISE_DB`, `SILENCE_MIN_DURATION_S`, `MAX_CHUNK_SECONDS`) match between Task 2's definitions and uses in Tasks 7, 8.
