# ONNX Compatibility Spike — Result

**Date:** 2026-05-20
**Outcome:** **Approach (c) — download official sherpa-onnx Parakeet TDT v3 int8 model from Hugging Face.** Reusing Handy's ONNX files directly is not viable; sherpa-onnx 1.13.2 cannot consume them.

## What worked

Downloaded the official `csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` model from Hugging Face (~670MB across 4 files) and loaded it via `sherpa_onnx.OfflineRecognizer.from_transducer(...)` with `model_type="nemo_transducer"`. First-try transcription of a 3.6s PT-BR test wav produced exact match including punctuation:

- Input speech: "Senhor, abençoe nosso pequeno grupo nesta noite." (macOS `say -v Luciana`)
- Transcribed output: "Senhor, abençoe nosso pequeno grupo nesta noite."

## Final model configuration

- **Model directory:** `~/Library/Application Support/audio-transcript/models/sherpa-onnx-parakeet-v3-int8/`
- **Files used:**
  - encoder: `encoder.int8.onnx` (622 MB)
  - decoder: `decoder.int8.onnx` (12 MB)
  - joiner: `joiner.int8.onnx` (7 MB)
  - tokens: `tokens.txt` (92 KB)
- **Source repo:** [csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8)
- **sherpa-onnx factory method:** `OfflineRecognizer.from_transducer`
- **Required arguments (full code snippet):**

```python
import sherpa_onnx

recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=f"{MODEL_DIR}/encoder.int8.onnx",
    decoder=f"{MODEL_DIR}/decoder.int8.onnx",
    joiner=f"{MODEL_DIR}/joiner.int8.onnx",
    tokens=f"{MODEL_DIR}/tokens.txt",
    num_threads=2,
    sample_rate=16000,
    feature_dim=128,
    model_type="nemo_transducer",
)

stream = recognizer.create_stream()
samples, sr = sf.read("audio.wav", dtype="float32")
stream.accept_waveform(sr, samples)
recognizer.decode_stream(stream)
text = stream.result.text
```

## What did not work

### Approach (a): Use Handy's ONNX files directly via `from_nemo_parakeet_tdt`

- Result: `AttributeError: type object 'OfflineRecognizer' has no attribute 'from_nemo_parakeet_tdt'`
- This method does not exist in sherpa-onnx 1.13.2. Plan had assumed it did.

### Approach (a'): Use Handy's ONNX files via `from_transducer`

- Result: `sherpa-onnx/csrc/offline-transducer-nemo-model.cc:InitEncoder:205 'vocab_size' does not exist in the metadata`
- The Handy ONNX export does not include the `vocab_size` model metadata key that sherpa-onnx requires.
- Inspection (`onnxruntime`) confirmed that Handy's encoder has only one `custom_metadata_map` entry (`onnx.infer: onnxruntime.quant`), no model-config metadata.

### Approach (b): Symlink Handy's files to sherpa-onnx naming conventions

- Skipped. The metadata problem from Approach (a') is structural and not solvable by renaming files.

### Notes on Handy's export

Handy's model directory contains:

| File | Size | Purpose |
|---|---|---|
| `encoder-model.int8.onnx` | 622 MB | Encoder, but no `vocab_size`/`model_type` metadata |
| `decoder_joint-model.int8.onnx` | 17 MB | Decoder + joiner combined (sherpa-onnx expects them separate) |
| `nemo128.onnx` | 137 KB | Standalone log-mel feature extractor (sherpa-onnx bundles features into the encoder) |
| `vocab.txt` | 92 KB | 8193 tokens, includes special tokens like `<unk>`, `<\|nospeech\|>`, etc. |
| `config.json` | 97 B | `{"model_type": "nemo-conformer-tdt", "features_size": 128, "subsampling_factor": 8}` |

The decoder_joint output dimension is 8198 = 8193 vocab + 5 TDT duration tokens.

The Handy export pipeline appears bespoke (combined decoder_joint, separated feature extractor, missing model metadata). Making it work with sherpa-onnx would require either:
- A custom Python inference loop using `onnxruntime` directly (~200 LOC + TDT decoder loop)
- Re-exporting from the original NeMo checkpoint to sherpa-onnx's expected format

Neither is worth doing when the official sherpa-onnx package exists and works first-try.

## Implications for `transcriber.py` and `config.py`

The plan as written needs the following adjustments:

1. **`config.py` env var name and default.** Rename `HANDY_MODELS_DIR` → `PARAKEET_MODEL_DIR`. Default to `~/Library/Application Support/audio-transcript/models/sherpa-onnx-parakeet-v3-int8/`. The "Handy" branding was misleading — we don't actually reuse Handy's files.

2. **`transcriber.py` model loader.** Use `from_transducer` (not `from_nemo_parakeet_tdt`). File names: `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt`. Pass `model_type="nemo_transducer"`.

3. **First-run download.** Since the model is no longer guaranteed to be on disk (Handy isn't doing the work), the app needs a way to bootstrap the model. Options for Task 2:
   - **Option A:** App fails at boot with a clear message and shell instructions to download. Simple, manual.
   - **Option B (Recommended):** Ship a `audio-transcript download-model` console command that fetches the 4 files into `PARAKEET_MODEL_DIR`. Single command, idempotent, friendlier for first run.

4. **Python version.** Plan said 3.11+; the user's machine has 3.13 and 3.14. sherpa-onnx 1.13.2 has wheels for 3.13. Update `pyproject.toml` `requires-python` to `>=3.11` (still works) and verify CI uses 3.13.

5. **README.** Add a "first-run setup" section explaining the one-time model download.

6. **Spike test wav.** The `tests/conftest.py` `short_speech_m4a` fixture using macOS `say` works as a reproducible PT-BR source. Keep as-is for the transcriber's local integration test.

## Test verification

- **Test audio:** `/tmp/spike-audio/hello.wav` (3.6s, 16kHz mono PCM, generated via `say -v Luciana` + ffmpeg conversion)
- **Test phrase:** "Senhor, abençoe nosso pequeno grupo nesta noite."
- **Transcribed output:** "Senhor, abençoe nosso pequeno grupo nesta noite." (exact match)
- **Quality assessment:** Excellent on this clean synthesized sample. Real-world quality on phone-recorded group conversations is still untested — that's what Task 11 (manual smoke test) is for.

## Environment

- Platform: macOS, arm64 (Apple Silicon)
- Python: 3.13.12 (from Homebrew; spike venv at `.spike-venv/`, removed after spike)
- sherpa-onnx: 1.13.2
- soundfile: 0.13.1
- onnxruntime: 1.x (installed only for metadata inspection)
