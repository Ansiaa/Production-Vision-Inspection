from __future__ import annotations

from pydantic import BaseModel


class InspectResponse(BaseModel):
    cls: str
    gt_label: str
    pred_label: str
    is_correct: bool | None
    pred_score: float
    decision_thr: float
    decision_reason: str
    overlay_path: str
    heatmap_path: str
    image_path: str
    processing_ms: int
    roi_path: str
    roi_mask_path: str
    preprocessed_path: str
