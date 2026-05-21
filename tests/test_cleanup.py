import os
import shutil
import time
from pathlib import Path

from audio_transcript.cleanup import sweep_once


def _make_job(uploads_dir: Path, name: str, age_days: float) -> Path:
    job_dir = uploads_dir / name
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text("{}")
    new_mtime = time.time() - age_days * 86400
    os.utime(job_dir, (new_mtime, new_mtime))
    return job_dir


def test_sweep_deletes_old_jobs_only(tmp_path: Path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    young = _make_job(uploads, "young", age_days=1)
    old = _make_job(uploads, "old", age_days=10)

    deleted = sweep_once(uploads, retention_days=7)
    assert deleted == 1
    assert young.exists()
    assert not old.exists()


def test_sweep_disabled_when_retention_zero(tmp_path: Path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    old = _make_job(uploads, "old", age_days=100)

    deleted = sweep_once(uploads, retention_days=0)
    assert deleted == 0
    assert old.exists()


def test_sweep_ignores_files(tmp_path: Path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    stray = uploads / "stray.txt"
    stray.write_text("hello")
    new_mtime = time.time() - 100 * 86400
    os.utime(stray, (new_mtime, new_mtime))

    deleted = sweep_once(uploads, retention_days=7)
    assert deleted == 0
    assert stray.exists()


def test_sweep_noop_on_missing_dir(tmp_path: Path):
    deleted = sweep_once(tmp_path / "does-not-exist", retention_days=7)
    assert deleted == 0
