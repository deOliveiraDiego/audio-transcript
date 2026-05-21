from pathlib import Path
from unittest.mock import MagicMock

import pytest

from audio_transcript.job import JobState, JobStatus
from audio_transcript.pipeline import run_pipeline


@pytest.mark.asyncio
async def test_run_pipeline_happy_path(tmp_path: Path, monkeypatch):
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
