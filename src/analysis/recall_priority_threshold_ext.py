from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(r"D:\project1")

EXPERIMENTS = [
    {
        "name": "pcb4_clahe_median3",
        "category": "pcb4",
        "version": "clahe_median3",
        "csv_path": PROJECT_ROOT / r"artifacts\runs\extension_filters\pcb4\clahe_median3\infer\predictions.csv",
        "target_recalls": [0.85, 0.90, 0.95],
    },
    {
        "name": "cashew_roi_gaussian3",
        "category": "cashew",
        "version": "roi_gaussian3",
        "csv_path": PROJECT_ROOT / r"artifacts\runs\extension_filters_round3\cashew\roi_gaussian3\infer\predictions.csv",
        "target_recalls": [0.85, 0.90, 0.95],
    },
    {
        "name": "cashew_roi_median5",
        "category": "cashew",
        "version": "roi_median5",
        "csv_path": PROJECT_ROOT / r"artifacts\runs\extension_filters_round3\cashew\roi_median5\infer\predictions.csv",
        "target_recalls": [0.90, 0.95, 0.97],
    },
]

OUT_DIR = PROJECT_ROOT / r"docs\tables\recall_priority_threshold_ext"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_CANDIDATES = ["y_true", "label", "target", "gt", "true_label"]
SCORE_CANDIDATES = ["score", "anomaly_score", "image_score"]


def find_first_existing(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    fieldset = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c.lower() in fieldset:
            return fieldset[c.lower()]
    return None


def safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def safe_int(x: str) -> int:
    try:
        return int(float(x))
    except Exception:
        s = str(x).strip().lower()
        if s in {"true", "anomaly", "ng", "defect", "1"}:
            return 1
        return 0


@dataclass
class SweepRow:
    threshold: float
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    accuracy: float


def calc_metrics(labels: List[int], scores: List[float], threshold: float) -> SweepRow:
    tp = tn = fp = fn = 0

    for y, s in zip(labels, scores):
        pred = 1 if s >= threshold else 0
        if y == 1 and pred == 1:
            tp += 1
        elif y == 0 and pred == 0:
            tn += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 1 and pred == 0:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))

    return SweepRow(
        threshold=threshold,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
    )


def load_predictions(csv_path: Path) -> Tuple[List[int], List[float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise RuntimeError(f"No header found: {csv_path}")

        label_col = find_first_existing(fieldnames, LABEL_CANDIDATES)
        score_col = find_first_existing(fieldnames, SCORE_CANDIDATES)

        if label_col is None:
            raise RuntimeError(f"Could not find label column in: {csv_path}")
        if score_col is None:
            raise RuntimeError(f"Could not find score column in: {csv_path}")

        labels: List[int] = []
        scores: List[float] = []

        for row in reader:
            y = safe_int(row[label_col])
            s = safe_float(row[score_col])
            if math.isnan(s):
                continue
            labels.append(y)
            scores.append(s)

    if not labels:
        raise RuntimeError(f"No valid rows in: {csv_path}")

    return labels, scores


def make_threshold_candidates(scores: List[float]) -> List[float]:
    uniq = sorted(set(scores))
    if not uniq:
        return [0.5]
    eps = 1e-12
    candidates = [uniq[0] - eps] + uniq + [uniq[-1] + eps]
    return sorted(set(candidates), reverse=True)


def run_sweep(labels: List[int], scores: List[float]) -> List[SweepRow]:
    thresholds = make_threshold_candidates(scores)
    return [calc_metrics(labels, scores, t) for t in thresholds]


def pick_best_f1(sweep_rows: List[SweepRow]) -> SweepRow:
    return max(sweep_rows, key=lambda r: (r.f1, r.recall, r.precision, r.threshold))


def pick_recall_priority(sweep_rows: List[SweepRow], target_recall: float) -> Optional[SweepRow]:
    candidates = [r for r in sweep_rows if r.recall >= target_recall]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r.threshold, r.f1, r.precision))


def write_sweep_csv(out_csv: Path, sweep_rows: List[SweepRow]) -> None:
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["threshold", "tp", "tn", "fp", "fn", "precision", "recall", "f1", "accuracy"],
        )
        writer.writeheader()
        for r in sweep_rows:
            writer.writerow({
                "threshold": f"{r.threshold:.10f}",
                "tp": r.tp,
                "tn": r.tn,
                "fp": r.fp,
                "fn": r.fn,
                "precision": f"{r.precision:.6f}",
                "recall": f"{r.recall:.6f}",
                "f1": f"{r.f1:.6f}",
                "accuracy": f"{r.accuracy:.6f}",
            })


def main() -> None:
    summary_rows: List[Dict[str, str]] = []

    for exp in EXPERIMENTS:
        labels, scores = load_predictions(Path(exp["csv_path"]))
        sweep_rows = run_sweep(labels, scores)
        best_f1 = pick_best_f1(sweep_rows)

        out_sweep_csv = OUT_DIR / f'{exp["name"]}_threshold_sweep.csv'
        out_recommend_json = OUT_DIR / f'{exp["name"]}_recommended_thresholds.json'
        write_sweep_csv(out_sweep_csv, sweep_rows)

        recommendations = {
            "experiment": exp["name"],
            "category": exp["category"],
            "version": exp["version"],
            "csv_path": str(exp["csv_path"]),
            "best_f1": asdict(best_f1),
            "recall_priority": {},
        }

        summary_rows.append({
            "experiment": exp["name"],
            "mode": "best_f1",
            "target_recall": "",
            "threshold": f"{best_f1.threshold:.10f}",
            "precision": f"{best_f1.precision:.6f}",
            "recall": f"{best_f1.recall:.6f}",
            "f1": f"{best_f1.f1:.6f}",
            "tp": str(best_f1.tp),
            "tn": str(best_f1.tn),
            "fp": str(best_f1.fp),
            "fn": str(best_f1.fn),
        })

        for target in exp["target_recalls"]:
            row = pick_recall_priority(sweep_rows, target)
            key = f"recall_at_least_{target:.2f}"

            if row is None:
                recommendations["recall_priority"][key] = None
                summary_rows.append({
                    "experiment": exp["name"],
                    "mode": "recall_priority",
                    "target_recall": f"{target:.2f}",
                    "threshold": "",
                    "precision": "",
                    "recall": "",
                    "f1": "",
                    "tp": "",
                    "tn": "",
                    "fp": "",
                    "fn": "",
                })
                continue

            recommendations["recall_priority"][key] = asdict(row)
            summary_rows.append({
                "experiment": exp["name"],
                "mode": "recall_priority",
                "target_recall": f"{target:.2f}",
                "threshold": f"{row.threshold:.10f}",
                "precision": f"{row.precision:.6f}",
                "recall": f"{row.recall:.6f}",
                "f1": f"{row.f1:.6f}",
                "tp": str(row.tp),
                "tn": str(row.tn),
                "fp": str(row.fp),
                "fn": str(row.fn),
            })

        with out_recommend_json.open("w", encoding="utf-8") as f:
            json.dump(recommendations, f, ensure_ascii=False, indent=2)

    summary_csv = OUT_DIR / "recall_priority_threshold_ext_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "mode", "target_recall", "threshold", "precision", "recall", "f1", "tp", "tn", "fp", "fn"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[DONE] {summary_csv}")


if __name__ == "__main__":
    main()