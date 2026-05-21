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
    assert any(word in text for word in ["senhor", "grupo", "abençoe", "abencoe"])
