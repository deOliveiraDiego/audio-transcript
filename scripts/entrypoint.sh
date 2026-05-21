#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${PARAKEET_MODEL_DIR:-/var/lib/audio-transcript/models/sherpa-onnx-parakeet-v3-int8}"
BASE_URL="https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main"

FILES=("encoder.int8.onnx" "decoder.int8.onnx" "joiner.int8.onnx" "tokens.txt")

need_download=0
for f in "${FILES[@]}"; do
    if [ ! -s "$MODEL_DIR/$f" ]; then
        need_download=1
        break
    fi
done

if [ "$need_download" = "1" ]; then
    echo "[entrypoint] Parakeet model not found at $MODEL_DIR — downloading (~670MB, one-time)"
    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"
    for f in "${FILES[@]}"; do
        echo "[entrypoint]   $f"
        curl -fL --retry 3 --retry-delay 5 -o "$f" "$BASE_URL/$f"
    done
    echo "[entrypoint] Model download complete"
else
    echo "[entrypoint] Model already present at $MODEL_DIR"
fi

exec "$@"
