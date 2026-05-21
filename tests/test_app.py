import asyncio
import importlib
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


def _reload_app_with_env(monkeypatch, **env: str):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import audio_transcript.config as cfg
    importlib.reload(cfg)
    import audio_transcript.app as app_mod
    importlib.reload(app_mod)
    return app_mod


@pytest.mark.asyncio
async def test_upload_then_poll_then_result(tmp_path, monkeypatch):
    app_mod = _reload_app_with_env(
        monkeypatch, AUDIO_TRANSCRIPT_DATA_DIR=str(tmp_path / "data")
    )

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

    fake = MagicMock()
    fake.transcribe.side_effect = lambda p: f"text-of-{Path(p).stem}"

    with patch(
        "audio_transcript.app.Transcriber.from_default_config",
        return_value=fake,
    ):
        transport = ASGITransport(app=app_mod.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            async with app_mod.app.router.lifespan_context(app_mod.app):
                with audio.open("rb") as f:
                    res = await client.post(
                        "/upload",
                        files={"file": ("in.wav", f, "audio/wav")},
                    )
                assert res.status_code == 200
                job_id = res.json()["job_id"]

                for _ in range(60):
                    status_res = await client.get(f"/jobs/{job_id}")
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
async def test_upload_rejects_huge_file(tmp_path, monkeypatch):
    app_mod = _reload_app_with_env(
        monkeypatch,
        AUDIO_TRANSCRIPT_DATA_DIR=str(tmp_path / "data"),
        AUDIO_TRANSCRIPT_MAX_UPLOAD_BYTES="1024",
    )

    with patch(
        "audio_transcript.app.Transcriber.from_default_config",
        return_value=MagicMock(),
    ):
        transport = ASGITransport(app=app_mod.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            async with app_mod.app.router.lifespan_context(app_mod.app):
                res = await client.post(
                    "/upload",
                    files={"file": ("big.bin", b"x" * 2048, "application/octet-stream")},
                )
                assert res.status_code == 413
