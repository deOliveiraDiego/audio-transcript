"""Spike: verify shared OfflineRecognizer is safe across concurrent threads.

Compares sequential vs parallel transcription of 4 real chunks from the
2h25min job, sharing a single recognizer. Asserts outputs match.

Run from project root:
    source .venv/bin/activate
    python docs/superpowers/spike/parallel_thread_safety.py
"""
import asyncio
import os
import time
from pathlib import Path

import sherpa_onnx
import soundfile as sf

MODEL_DIR = Path(os.path.expanduser(
    "~/Library/Application Support/audio-transcript/models/sherpa-onnx-parakeet-v3-int8"
))
CHUNKS_DIR = Path(
    "data/uploads/39243f17-08b9-4ceb-8d1e-b8013bfaa30f/chunks"
)


def make_recognizer():
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(MODEL_DIR / "encoder.int8.onnx"),
        decoder=str(MODEL_DIR / "decoder.int8.onnx"),
        joiner=str(MODEL_DIR / "joiner.int8.onnx"),
        tokens=str(MODEL_DIR / "tokens.txt"),
        num_threads=2,
        sample_rate=16000,
        feature_dim=128,
        model_type="nemo_transducer",
    )


def transcribe(recognizer, wav_path):
    stream = recognizer.create_stream()
    samples, sr = sf.read(str(wav_path), dtype="float32")
    stream.accept_waveform(sr, samples)
    recognizer.decode_stream(stream)
    return stream.result.text


def main() -> None:
    chunks = sorted(CHUNKS_DIR.glob("chunk_0[0-9][0-9].wav"))[:4]
    print(f"Chunks: {[c.name for c in chunks]}")

    recognizer = make_recognizer()

    print("\n--- SEQUENTIAL ---")
    t0 = time.time()
    sequential = [transcribe(recognizer, c) for c in chunks]
    seq_time = time.time() - t0
    print(f"Time: {seq_time:.2f}s")
    for i, t in enumerate(sequential):
        print(f"  chunk_{i+1}: {t[:80]!r}")

    print("\n--- PARALLEL (shared recognizer, 4 workers) ---")

    async def run_parallel():
        sem = asyncio.Semaphore(4)
        async def one(c):
            async with sem:
                return await asyncio.to_thread(transcribe, recognizer, c)
        return await asyncio.gather(*[one(c) for c in chunks])

    t0 = time.time()
    parallel = asyncio.run(run_parallel())
    par_time = time.time() - t0
    print(f"Time: {par_time:.2f}s")
    for i, t in enumerate(parallel):
        print(f"  chunk_{i+1}: {t[:80]!r}")

    print("\n--- COMPARISON ---")
    print(f"Sequential: {seq_time:.2f}s")
    print(f"Parallel:   {par_time:.2f}s")
    print(f"Speedup:    {seq_time / par_time:.2f}x")
    all_match = all(s == p for s, p in zip(sequential, parallel))
    print(f"Outputs match exactly: {all_match}")
    if not all_match:
        for i, (s, p) in enumerate(zip(sequential, parallel)):
            if s != p:
                print(f"  DIFF chunk_{i+1}:")
                print(f"    seq:  {s[:120]!r}")
                print(f"    par:  {p[:120]!r}")


if __name__ == "__main__":
    main()
