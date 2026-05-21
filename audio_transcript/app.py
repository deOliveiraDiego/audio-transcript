"""FastAPI app: upload, SSE progress, result download."""
from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from audio_transcript import config
from audio_transcript.cleanup import run_periodic_cleanup
from audio_transcript.job import JobState, JobStatus
from audio_transcript.pipeline import run_pipeline
from audio_transcript.transcriber import Transcriber

STATIC_DIR = Path(__file__).parent / "static"


class JobBroker:
    """Per-job in-memory pub/sub for SSE clients."""

    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue]] = {}

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        if job_id in self._queues:
            self._queues[job_id] = [x for x in self._queues[job_id] if x is not q]
            if not self._queues[job_id]:
                del self._queues[job_id]

    async def publish(self, job_id: str, event: dict) -> None:
        for q in self._queues.get(job_id, []):
            await q.put(event)


broker = JobBroker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.transcriber = Transcriber.from_default_config()
    cleanup_task = asyncio.create_task(
        run_periodic_cleanup(
            uploads_dir=config.UPLOADS_DIR,
            retention_days=config.RETENTION_DAYS,
        )
    )
    try:
        yield
    finally:
        cleanup_task.cancel()


app = FastAPI(title="audio-transcript", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _ad_head() -> str:
    if not config.ADSENSE_CLIENT_ID:
        return ""
    return (
        f'<script async '
        f'src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js'
        f'?client={config.ADSENSE_CLIENT_ID}" crossorigin="anonymous"></script>'
    )


def _ad_slot() -> str:
    if not config.ADSENSE_CLIENT_ID:
        return '<div class="ad-slot ad-placeholder">Espaço para anúncio</div>'
    return (
        '<div class="ad-slot">'
        '<ins class="adsbygoogle" style="display:block" '
        f'data-ad-client="{config.ADSENSE_CLIENT_ID}" '
        'data-ad-format="auto" data-full-width-responsive="true"></ins>'
        '<script>(adsbygoogle = window.adsbygoogle || []).push({});</script>'
        '</div>'
    )


def _render_template(path: Path) -> str:
    html = path.read_text()
    return (
        html.replace("{{ADSENSE_HEAD}}", _ad_head())
            .replace("{{ADSENSE_SLOT_TOP}}", _ad_slot())
            .replace("{{ADSENSE_SLOT_BOTTOM}}", _ad_slot())
            .replace("{{ADSENSE_CLIENT_ID}}", config.ADSENSE_CLIENT_ID)
            .replace("{{HCAPTCHA_SITEKEY}}", config.HCAPTCHA_SITEKEY)
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>audio-transcript</h1><p>UI not built yet.</p>")
    return HTMLResponse(_render_template(index_path))


@app.get("/privacy", response_class=HTMLResponse)
async def privacy() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "privacy.html").read_text())


@app.get("/terms", response_class=HTMLResponse)
async def terms() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "terms.html").read_text())


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    job_id = str(uuid.uuid4())
    job_dir = config.UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "audio").suffix or ".bin"
    source_path = job_dir / f"source{suffix}"

    total_bytes = 0
    with source_path.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > config.MAX_UPLOAD_BYTES:
                out.close()
                source_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds limit ({config.MAX_UPLOAD_BYTES} bytes).",
                )
            out.write(chunk)

    JobState(
        id=job_id,
        status=JobStatus.PENDING,
        source_path=source_path,
        stage="pending",
    ).save(job_dir)

    async def _publish(event: dict) -> None:
        await broker.publish(job_id, event)

    asyncio.create_task(
        run_pipeline(
            job_dir=job_dir,
            transcriber=app.state.transcriber,
            publish=_publish,
            silence_noise_db=config.SILENCE_NOISE_DB,
            silence_min_duration_s=config.SILENCE_MIN_DURATION_S,
            max_chunk_s=config.MAX_CHUNK_SECONDS,
            parallelism=config.TRANSCRIBE_PARALLELISM,
        )
    )

    return {"job_id": job_id}


@app.get("/jobs")
async def list_jobs() -> list[dict]:
    if not config.UPLOADS_DIR.exists():
        return []
    entries: list[tuple[float, dict]] = []
    for job_dir in config.UPLOADS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        if not (job_dir / JobState.JSON_FILENAME).exists():
            continue
        if not (job_dir / "transcript.txt").exists():
            continue
        try:
            job = JobState.load(job_dir)
        except Exception:
            continue
        if job.status != JobStatus.DONE:
            continue
        mtime = job_dir.stat().st_mtime
        entries.append((mtime, {
            "id": job.id,
            "filename": job.source_path.name,
            "modified": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
        }))
    entries.sort(key=lambda e: e[0], reverse=True)
    return [e[1] for e in entries]


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str) -> Response:
    job_dir = config.UPLOADS_DIR / job_id
    if not (job_dir / JobState.JSON_FILENAME).exists():
        raise HTTPException(status_code=404, detail="Job not found")
    shutil.rmtree(job_dir)
    return Response(status_code=204)


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job_dir = config.UPLOADS_DIR / job_id
    if not (job_dir / JobState.JSON_FILENAME).exists():
        raise HTTPException(status_code=404, detail="Job not found")
    job = JobState.load(job_dir)
    return {
        "id": job.id,
        "status": job.status.value,
        "stage": job.stage,
        "total_chunks": job.total_chunks,
        "processed_chunks": job.processed_chunks,
        "error_message": job.error_message,
    }


@app.get("/jobs/{job_id}/stream")
async def stream(job_id: str) -> EventSourceResponse:
    job_dir = config.UPLOADS_DIR / job_id
    if not (job_dir / JobState.JSON_FILENAME).exists():
        raise HTTPException(status_code=404, detail="Job not found")

    queue = broker.subscribe(job_id)

    async def event_generator():
        job = JobState.load(job_dir)
        yield {
            "event": "progress",
            "data": json.dumps({
                "stage": job.stage,
                "processed": job.processed_chunks,
                "total": job.total_chunks,
                "status": job.status.value,
            }),
        }
        if job.status == JobStatus.DONE:
            yield {"event": "done", "data": json.dumps({"job_id": job_id})}
            return
        if job.status == JobStatus.FAILED:
            yield {
                "event": "error",
                "data": json.dumps({"message": job.error_message or "unknown"}),
            }
            return

        try:
            while True:
                event = await queue.get()
                yield {"event": event["event"], "data": json.dumps(event["data"])}
                if event["event"] in {"done", "error"}:
                    return
        finally:
            broker.unsubscribe(job_id, queue)

    return EventSourceResponse(event_generator())


@app.get("/jobs/{job_id}/transcript-partial")
async def get_partial(job_id: str) -> dict:
    job_dir = config.UPLOADS_DIR / job_id
    if not (job_dir / JobState.JSON_FILENAME).exists():
        raise HTTPException(status_code=404, detail="Job not found")
    texts_dir = job_dir / "chunk_texts"
    if not texts_dir.exists():
        return {"text": "", "processed": 0}
    from audio_transcript.assembler import combine
    paths = sorted(texts_dir.glob("chunk_*.txt"))
    texts = [p.read_text(encoding="utf-8") for p in paths]
    return {"text": combine(texts), "processed": len(texts)}


@app.get("/jobs/{job_id}/result")
async def get_result(job_id: str) -> FileResponse:
    job_dir = config.UPLOADS_DIR / job_id
    if not (job_dir / JobState.JSON_FILENAME).exists():
        raise HTTPException(status_code=404, detail="Job not found")
    job = JobState.load(job_dir)
    if job.status != JobStatus.DONE or not job.result_path:
        raise HTTPException(status_code=409, detail="Job not done yet")
    if not job.result_path.exists():
        raise HTTPException(status_code=410, detail="Result file missing")
    return FileResponse(
        str(job.result_path),
        media_type="text/plain",
        filename="transcript.txt",
    )


def main() -> None:
    import uvicorn
    uvicorn.run(
        "audio_transcript.app:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
