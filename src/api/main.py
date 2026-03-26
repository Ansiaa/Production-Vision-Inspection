from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

from src.api.config import UPLOAD_DIR
from src.api.schemas import InspectResponse
from src.api.service import inspect_one

app = FastAPI(title="Vision Inspection API", version="1.3.0")


def _next_available_path(base_dir: Path, original_name: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)

    original = Path(original_name)
    stem = original.stem
    suffix = original.suffix if original.suffix else ".jpg"

    candidate = base_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate

    idx = 2
    while True:
        candidate = base_dir / f"{stem}_{idx:02d}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/inspect", response_model=InspectResponse)
async def inspect_api(
    cls: str = Query(..., pattern="^(pcb4|cashew)$"),
    file: UploadFile = File(...),
    gt_label: str = Form("UNKNOWN"),
    source_path: str | None = Form(None),
):
    try:
        original_filename = file.filename or "upload.jpg"
        save_path = _next_available_path(UPLOAD_DIR, original_filename)
        with save_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        result = inspect_one(
            image_path=str(save_path),
            cls=cls,
            gt_label=gt_label,
            source_path=source_path,
            original_filename=original_filename,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
