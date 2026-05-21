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
