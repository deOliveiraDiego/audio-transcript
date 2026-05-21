"""Async orchestrator: normalize → chunk → transcribe → assemble."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable

import soundfile as sf

from audio_transcript import chunker
from audio_transcript.assembler import combine
from audio_transcript.job import JobState, JobStatus

log = logging.getLogger(__name__)

PublishFn = Callable[[dict], Awaitable[None]]


def _format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


async def _set_stage(
    job: JobState, job_dir: Path, status: JobStatus, stage: str, publish: PublishFn
) -> None:
    job.status = status
    job.stage = stage
    job.save(job_dir)
    await publish({"event": "progress", "data": {"stage": stage}})


async def run_pipeline(
    *,
    job_dir: Path,
    transcriber,
    publish: PublishFn,
    silence_noise_db: float,
    silence_min_duration_s: float,
    max_chunk_s: float,
    parallelism: int = 1,
) -> None:
    job = JobState.load(job_dir)
    try:
        await _set_stage(job, job_dir, JobStatus.NORMALIZING, "normalizing", publish)
        normalized = await asyncio.to_thread(
            chunker.normalize, job.source_path, job_dir / "normalized.wav"
        )

        await _set_stage(job, job_dir, JobStatus.CHUNKING, "chunking", publish)
        silences = await asyncio.to_thread(
            chunker.detect_silences,
            normalized, silence_noise_db, silence_min_duration_s,
        )
        chunks = await asyncio.to_thread(
            chunker.split, normalized, silences, job_dir / "chunks", max_chunk_s,
        )
        if not chunks:
            raise RuntimeError("No audio chunks produced — input may be empty.")

        job.total_chunks = len(chunks)
        job.processed_chunks = 0
        await _set_stage(job, job_dir, JobStatus.TRANSCRIBING, "transcribing", publish)

        boundaries = [0.0]
        for c in chunks:
            info = sf.info(str(c))
            boundaries.append(boundaries[-1] + info.frames / info.samplerate)

        texts: list[str | None] = [None] * len(chunks)
        chunk_texts_dir = job_dir / "chunk_texts"
        chunk_texts_dir.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(max(1, parallelism))
        progress_lock = asyncio.Lock()

        async def process_chunk(i: int, chunk_path: Path) -> None:
            async with semaphore:
                try:
                    text = await asyncio.to_thread(transcriber.transcribe, chunk_path)
                except Exception as exc:
                    log.warning("chunk %s failed: %s", chunk_path, exc)
                    start = _format_time(boundaries[i])
                    end = _format_time(boundaries[i + 1])
                    text = f"[TRECHO NÃO TRANSCRITO: {start}–{end}]"
                texts[i] = text
                (chunk_texts_dir / f"chunk_{i + 1:03d}.txt").write_text(
                    text, encoding="utf-8"
                )
                async with progress_lock:
                    job.processed_chunks += 1
                    job.save(job_dir)
                    await publish({
                        "event": "progress",
                        "data": {
                            "processed": job.processed_chunks,
                            "total": job.total_chunks,
                            "stage": "transcribing",
                        },
                    })

        await asyncio.gather(
            *[process_chunk(i, c) for i, c in enumerate(chunks)]
        )

        await _set_stage(job, job_dir, JobStatus.ASSEMBLING, "assembling", publish)
        final_text = combine([t for t in texts if t is not None])
        result_path = job_dir / "transcript.txt"
        result_path.write_text(final_text, encoding="utf-8")

        job.result_path = result_path
        job.status = JobStatus.DONE
        job.stage = "done"
        job.save(job_dir)
        await publish({"event": "done", "data": {"job_id": job.id}})

    except Exception as exc:
        log.exception("pipeline failed for job %s", job.id)
        job.status = JobStatus.FAILED
        job.stage = "failed"
        job.error_message = str(exc)
        job.save(job_dir)
        await publish({"event": "error", "data": {"message": str(exc)}})
