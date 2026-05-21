# audio-transcript

Transcribes long audio recordings into a paragraph-formatted `.txt` using NVIDIA's Parakeet TDT 0.6B v3 model running locally via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Runs as a local web app — no API calls, no recurring cost, works offline once the model is downloaded.

## Requirements

- macOS or Linux on x86_64 / arm64
- Python 3.11+ (tested on 3.13)
- [ffmpeg](https://ffmpeg.org/) on `$PATH`
- ~670 MB of disk for the Parakeet model (one-time download)

## Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## First-run setup: download the model

The app expects four files under `~/Library/Application Support/audio-transcript/models/sherpa-onnx-parakeet-v3-int8/` (or wherever `PARAKEET_MODEL_DIR` points). Download them once:

```bash
MODEL_DIR="$HOME/Library/Application Support/audio-transcript/models/sherpa-onnx-parakeet-v3-int8"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"
BASE="https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main"
curl -L -o encoder.int8.onnx "$BASE/encoder.int8.onnx"
curl -L -o decoder.int8.onnx "$BASE/decoder.int8.onnx"
curl -L -o joiner.int8.onnx  "$BASE/joiner.int8.onnx"
curl -L -o tokens.txt        "$BASE/tokens.txt"
```

## Run

```bash
audio-transcript
# server starts at http://127.0.0.1:8000
```

Open the URL in a browser, drag a recording onto the page, wait for the progress bar to fill, download the `.txt`.

## How it works

1. Upload writes the file to `data/uploads/<uuid>/source.<ext>`.
2. `ffmpeg` normalizes it to mono 16 kHz PCM.
3. `ffmpeg`'s `silencedetect` filter finds pauses ≥ 1.5 s (configurable).
4. The audio is split into chunks at those pauses (forcing a cut after 90 s if no pause was found).
5. Each chunk is transcribed by sherpa-onnx (Parakeet TDT). If a chunk fails, the pipeline inserts `[TRECHO NÃO TRANSCRITO: hh:mm:ss–hh:mm:ss]` and continues.
6. Chunk texts are joined with blank lines — each chunk becomes one paragraph.

## Configuration

All settings come from env vars (see `audio_transcript/config.py`).

| Variable | Default | Purpose |
|---|---|---|
| `PARAKEET_MODEL_DIR` | `~/Library/Application Support/audio-transcript/models/sherpa-onnx-parakeet-v3-int8` | Where to find Parakeet ONNX files |
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
pytest -m local         # Runs against real Parakeet (requires PARAKEET_MODEL_DIR populated)
```

## Troubleshooting

- **`Parakeet model directory not found`** at boot: run the first-run download in the section above, or set `PARAKEET_MODEL_DIR` to a directory containing the four model files.
- **Upload hangs or times out**: check `ffmpeg` is on `$PATH` (`which ffmpeg`).
- **Empty transcript**: input audio may be entirely below the silence threshold. Try lowering `AUDIO_TRANSCRIPT_SILENCE_DB` (e.g., `-40`).
- **Chunks coming out as `[TRECHO NÃO TRANSCRITO]`**: a model inference error on that chunk. Re-running the same file usually succeeds.

## Out of scope (deliberately)

- Speaker diarization
- Timestamps inside the transcript
- LLM-based cleanup / summarization
- Hosted deployment
- History UI of past jobs

See `docs/superpowers/specs/2026-05-20-audio-transcript-design.md` for the full design and `docs/superpowers/spike/2026-05-20-onnx-compat-result.md` for the model compatibility investigation.
