"""Periodic cleanup of stale job directories."""
from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _seconds_since(path: Path) -> float:
    return time.time() - path.stat().st_mtime


def sweep_once(uploads_dir: Path, retention_days: int) -> int:
    """Delete job dirs older than retention_days. Returns count deleted."""
    if retention_days <= 0 or not uploads_dir.exists():
        return 0
    cutoff_seconds = retention_days * 86400
    deleted = 0
    for entry in uploads_dir.iterdir():
        if not entry.is_dir():
            continue
        if _seconds_since(entry) <= cutoff_seconds:
            continue
        try:
            shutil.rmtree(entry)
            deleted += 1
            log.info("cleanup: removed %s", entry.name)
        except Exception as exc:
            log.warning("cleanup: failed to remove %s: %s", entry.name, exc)
    return deleted


async def run_periodic_cleanup(
    uploads_dir: Path,
    retention_days: int,
    interval_seconds: int = 3600,
) -> None:
    """Run sweep_once forever at the given interval. Cancelable."""
    if retention_days <= 0:
        log.info("cleanup: disabled (RETENTION_DAYS=0)")
        return
    while True:
        try:
            deleted = await asyncio.to_thread(
                sweep_once, uploads_dir, retention_days
            )
            if deleted:
                log.info("cleanup: swept %d job dir(s)", deleted)
        except Exception:
            log.exception("cleanup: sweep failed")
        await asyncio.sleep(interval_seconds)
