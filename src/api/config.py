from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "runs" / "api"
VISA_ROOT = PROJECT_ROOT / "data" / "VisA"

UPLOAD_DIR = ARTIFACT_ROOT / "uploads"
ROI_DIR = ARTIFACT_ROOT / "roi"
ROI_MASK_DIR = ARTIFACT_ROOT / "roi_mask"
PREPROC_DIR = ARTIFACT_ROOT / "preprocessed"
OVERLAY_DIR = ARTIFACT_ROOT / "overlay"
HEATMAP_DIR = ARTIFACT_ROOT / "heatmap"

for _d in [UPLOAD_DIR, ROI_DIR, ROI_MASK_DIR, PREPROC_DIR, OVERLAY_DIR, HEATMAP_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

THRESHOLD_CONFIG_PATH = CONFIG_DIR / "thresholds_final_operating.json"

DEFAULT_THRESHOLDS = {
    "pcb4": {
        "threshold": 0.0750242769,
        "reason": "thresholds_final_operating.json",
        "variant": "clahe_median3",
        "model_dir": "artifacts/runs/extension_filters/pcb4/clahe_median3/train",
    },
    "cashew": {
        "threshold": 0.0498208888,
        "reason": "thresholds_final_operating.json",
        "variant": "roi_median5",
        "model_dir": "artifacts/runs/extension_filters_round3/cashew/roi_median5/train",
    },
}

if THRESHOLD_CONFIG_PATH.exists():
    try:
        THRESHOLDS = json.loads(THRESHOLD_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        THRESHOLDS = DEFAULT_THRESHOLDS
else:
    THRESHOLDS = DEFAULT_THRESHOLDS
