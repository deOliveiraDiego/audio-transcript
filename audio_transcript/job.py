"""Job state with JSON persistence on disk."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    NORMALIZING = "normalizing"
    CHUNKING = "chunking"
    TRANSCRIBING = "transcribing"
    ASSEMBLING = "assembling"
    DONE = "done"
    FAILED = "failed"


@dataclass
class JobState:
    id: str
    status: JobStatus
    source_path: Path
    stage: str = ""
    total_chunks: int = 0
    processed_chunks: int = 0
    result_path: Optional[Path] = None
    error_message: Optional[str] = None

    JSON_FILENAME = "job.json"

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["source_path"] = str(self.source_path)
        payload["result_path"] = str(self.result_path) if self.result_path else None
        (dir_path / self.JSON_FILENAME).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, dir_path: Path) -> "JobState":
        data = json.loads((dir_path / cls.JSON_FILENAME).read_text())
        return cls(
            id=data["id"],
            status=JobStatus(data["status"]),
            source_path=Path(data["source_path"]),
            stage=data.get("stage", ""),
            total_chunks=data.get("total_chunks", 0),
            processed_chunks=data.get("processed_chunks", 0),
            result_path=Path(data["result_path"]) if data.get("result_path") else None,
            error_message=data.get("error_message"),
        )
