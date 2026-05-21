"""FastAPI app: upload, result download, and (in Task 8) SSE streaming."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from audio_transcript import config
from audio_transcript.job import JobState, JobStatus

app = FastAPI(title="audio-transcript")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>audio-transcript</h1><p>UI not built yet.</p>")
    return HTMLResponse(index_path.read_text())


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

    job = JobState(
        id=job_id,
        status=JobStatus.PENDING,
        source_path=source_path,
        stage="pending",
    )
    job.save(job_dir)

    return {"job_id": job_id}


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
