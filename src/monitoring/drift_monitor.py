from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


LABEL_CANDIDATES = ["y_true", "label", "target", "gt", "true_label"]
PRED_CANDIDATES = ["y_pred", "pred", "prediction", "is_anomaly_pred"]
SCORE_CANDIDATES = ["score", "anomaly_score", "image_score"]
IMAGE_CANDIDATES = ["image_path", "relative_path", "path", "file_path", "input_path"]


def find_col(fieldnames: List[str], candidates: List[str]):
    lower_map = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def safe_int(x) -> int:
    s = str(x).strip().lower()
    if s in {"1", "true", "anomaly", "defect", "ng"}:
        return 1
    try:
        return int(float(s))
    except Exception:
        return 0


def load_predictions(csv_path: Path, threshold: float | None = None) -> List[Dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        score_col = find_col(fieldnames, SCORE_CANDIDATES)
        image_col = find_col(fieldnames, IMAGE_CANDIDATES)
        pred_col = find_col(fieldnames, PRED_CANDIDATES)

        if score_col is None:
            raise RuntimeError(f"Missing score column in {csv_path}")
        if image_col is None:
            raise RuntimeError(f"Missing image path column in {csv_path}")

        rows = []
        for row in reader:
            score = safe_float(row.get(score_col, "0"))
            if pred_col is not None and row.get(pred_col, "") != "":
                pred = safe_int(row[pred_col])
            elif threshold is not None:
                pred = 1 if score >= threshold else 0
            else:
                pred = 0

            rows.append({
                "image_path": row[image_col],
                "score": score,
                "pred": pred,
            })
    return rows


def resolve_input_path(input_root: Path, image_path: str) -> Path:
    p = Path(image_path)
    if p.is_absolute():
        return p
    return input_root / p


def image_stats(img_path: Path) -> Dict[str, float]:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return {
            "gray_mean": 0.0,
            "gray_std": 0.0,
            "grad_mean": 0.0,
            "grad_std": 0.0,
            "valid": 0.0,
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "grad_mean": float(grad.mean()),
        "grad_std": float(grad.std()),
        "valid": 1.0,
    }


def summarize_scores(rows: List[Dict]) -> Dict[str, float]:
    scores = np.array([r["score"] for r in rows], dtype=np.float32)
    preds = np.array([r["pred"] for r in rows], dtype=np.float32)

    return {
        "n_samples": int(len(rows)),
        "score_mean": float(scores.mean()) if len(scores) else 0.0,
        "score_std": float(scores.std()) if len(scores) else 0.0,
        "score_p95": float(np.percentile(scores, 95)) if len(scores) else 0.0,
        "score_p99": float(np.percentile(scores, 99)) if len(scores) else 0.0,
        "predicted_anomaly_ratio": float(preds.mean()) if len(preds) else 0.0,
    }


def summarize_inputs(rows: List[Dict], input_root: Path) -> Dict[str, float]:
    stats = []
    for r in rows:
        p = resolve_input_path(input_root, r["image_path"])
        stats.append(image_stats(p))

    valid = [s for s in stats if s["valid"] > 0]
    if not valid:
        return {
            "gray_mean_mean": 0.0,
            "gray_std_mean": 0.0,
            "grad_mean_mean": 0.0,
            "grad_std_mean": 0.0,
            "valid_images": 0,
        }

    return {
        "gray_mean_mean": float(np.mean([s["gray_mean"] for s in valid])),
        "gray_std_mean": float(np.mean([s["gray_std"] for s in valid])),
        "grad_mean_mean": float(np.mean([s["grad_mean"] for s in valid])),
        "grad_std_mean": float(np.mean([s["grad_std"] for s in valid])),
        "valid_images": int(len(valid)),
    }


def abs_ratio_change(cur: float, ref: float) -> float:
    denom = max(abs(ref), 1e-6)
    return abs(cur - ref) / denom


def classify_drift(reference: Dict[str, float], current: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    deltas = {
        "score_mean_delta_ratio": abs_ratio_change(current["score_mean"], reference["score_mean"]),
        "score_p95_delta_ratio": abs_ratio_change(current["score_p95"], reference["score_p95"]),
        "anomaly_ratio_delta_abs": abs(current["predicted_anomaly_ratio"] - reference["predicted_anomaly_ratio"]),
        "gray_mean_delta_ratio": abs_ratio_change(current["gray_mean_mean"], reference["gray_mean_mean"]),
        "grad_mean_delta_ratio": abs_ratio_change(current["grad_mean_mean"], reference["grad_mean_mean"]),
    }

    severe_count = 0
    moderate_count = 0

    if deltas["score_mean_delta_ratio"] >= 0.30:
        severe_count += 1
    elif deltas["score_mean_delta_ratio"] >= 0.15:
        moderate_count += 1

    if deltas["score_p95_delta_ratio"] >= 0.30:
        severe_count += 1
    elif deltas["score_p95_delta_ratio"] >= 0.15:
        moderate_count += 1

    if deltas["anomaly_ratio_delta_abs"] >= 0.20:
        severe_count += 1
    elif deltas["anomaly_ratio_delta_abs"] >= 0.10:
        moderate_count += 1

    if deltas["gray_mean_delta_ratio"] >= 0.20:
        moderate_count += 1

    if deltas["grad_mean_delta_ratio"] >= 0.25:
        moderate_count += 1

    if severe_count >= 2:
        level = "severe"
    elif severe_count >= 1 or moderate_count >= 2:
        level = "moderate"
    else:
        level = "weak"

    return level, deltas


def build_reference_or_compare(
    predictions_csv: Path,
    input_root: Path,
    threshold: float,
    out_json: Path,
    reference_json: Path | None = None,
) -> None:
    rows = load_predictions(predictions_csv, threshold=threshold)
    score_summary = summarize_scores(rows)
    input_summary = summarize_inputs(rows, input_root)

    current = {**score_summary, **input_summary}

    payload = {
        "predictions_csv": str(predictions_csv),
        "input_root": str(input_root),
        "threshold": threshold,
        "current": current,
    }

    if reference_json is not None and reference_json.exists():
        ref = json.loads(reference_json.read_text(encoding="utf-8"))
        reference = ref["current"]
        level, deltas = classify_drift(reference, current)
        payload["reference_json"] = str(reference_json)
        payload["reference"] = reference
        payload["drift_level"] = level
        payload["drift_metrics"] = deltas

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] {out_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--reference-json", default="")
    args = parser.parse_args()

    reference_json = Path(args.reference_json) if args.reference_json else None

    build_reference_or_compare(
        predictions_csv=Path(args.predictions_csv),
        input_root=Path(args.input_root),
        threshold=args.threshold,
        out_json=Path(args.out_json),
        reference_json=reference_json,
    )


if __name__ == "__main__":
    main()
