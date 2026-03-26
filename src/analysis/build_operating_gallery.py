from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(r"D:\project1")

EXPERIMENTS = [
    {
        "name": "pcb4_clahe_recall_090",
        "category": "pcb4",
        "version": "clahe_recall_090",
        "run_root": PROJECT_ROOT / r"artifacts\runs\operating_threshold\pcb4\clahe_recall_090",
        "input_root": PROJECT_ROOT / r"data\processed_clahe\pcb4",
        "threshold": 0.074116,
    },
    {
        "name": "cashew_roi_only_recall_095",
        "category": "cashew",
        "version": "roi_only_recall_095",
        "run_root": PROJECT_ROOT / r"artifacts\runs\operating_threshold\cashew\roi_only_recall_095",
        "input_root": PROJECT_ROOT / r"data\processed\cashew",
        "threshold": 0.051222,
    },
]

OUT_ROOT = PROJECT_ROOT / r"docs\operating_threshold"
REP_DIR = OUT_ROOT / "representative"
TOP5_DIR = OUT_ROOT / "fp_fn_top5"
GALLERY_DIR = OUT_ROOT / "gallery"
TABLE_DIR = OUT_ROOT / "tables"

for d in [REP_DIR, TOP5_DIR, GALLERY_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LABEL_CANDIDATES = ["y_true", "label", "target", "gt", "true_label"]
PRED_CANDIDATES = ["y_pred", "pred", "prediction", "is_anomaly_pred"]
SCORE_CANDIDATES = ["score", "anomaly_score", "image_score"]
IMAGE_CANDIDATES = ["image_path", "relative_path", "path", "file_path", "input_path"]
OVERLAY_CANDIDATES = ["overlay_path"]
HEATMAP_CANDIDATES = ["heatmap_path"]


def find_col(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def safe_int(x: str) -> int:
    s = str(x).strip().lower()
    if s in {"1", "true", "anomaly", "defect", "ng"}:
        return 1
    try:
        return int(float(s))
    except Exception:
        return 0


def safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def infer_label_from_path(image_path: str) -> int:
    s = str(image_path).replace("/", "\\").lower()
    if "\\anomaly\\" in s:
        return 1
    if "\\normal\\" in s:
        return 0
    return 0


def image_to_png_name(image_path: str) -> str:
    return image_path.replace("\\", "__").replace("/", "__") + ".png"


def resolve_asset_path(
    row: Dict[str, str],
    explicit_col: Optional[str],
    fallback_dir: Path,
    image_col_key: str,
) -> Path:
    if explicit_col and row.get(explicit_col):
        p = Path(row[explicit_col])
        if p.exists():
            return p
    return fallback_dir / image_to_png_name(row[image_col_key])


def resolve_input_path(input_root: Path, image_path: str) -> Path:
    rel_path = Path(image_path)
    if rel_path.is_absolute():
        return rel_path
    return input_root / rel_path


def load_predictions(csv_path: Path, threshold: float) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise RuntimeError(f"Empty header: {csv_path}")

        print(f"[DEBUG] columns in {csv_path.name}: {fieldnames}")

        label_col = find_col(fieldnames, LABEL_CANDIDATES)
        pred_col = find_col(fieldnames, PRED_CANDIDATES)
        score_col = find_col(fieldnames, SCORE_CANDIDATES)
        image_col = find_col(fieldnames, IMAGE_CANDIDATES)
        overlay_col = find_col(fieldnames, OVERLAY_CANDIDATES)
        heatmap_col = find_col(fieldnames, HEATMAP_CANDIDATES)

        if score_col is None:
            raise RuntimeError(
                f"Missing score column in {csv_path}. "
                f"Found columns={fieldnames}"
            )
        if image_col is None:
            raise RuntimeError(
                f"Missing image path column in {csv_path}. "
                f"Found columns={fieldnames}"
            )

        rows = []
        for row in reader:
            image_path = row.get(image_col, "")
            score = safe_float(row.get(score_col, ""))

            if label_col is not None and row.get(label_col, "") != "":
                label = safe_int(row[label_col])
            else:
                label = infer_label_from_path(image_path)

            if pred_col is not None and row.get(pred_col, "") != "":
                pred = safe_int(row[pred_col])
            else:
                pred = 1 if score >= threshold else 0

            rows.append({
                "image_path": image_path,
                "label": label,
                "pred": pred,
                "score": score,
                "overlay_path": row.get(overlay_col, "") if overlay_col else "",
                "heatmap_path": row.get(heatmap_col, "") if heatmap_col else "",
            })

        if not rows:
            raise RuntimeError(f"No valid rows found in {csv_path}")

        return rows


def pick_top5(rows: List[Dict[str, str]], mode: str) -> List[Dict[str, str]]:
    if mode == "fp":
        target = [r for r in rows if r["label"] == 0 and r["pred"] == 1]
        return sorted(target, key=lambda r: r["score"], reverse=True)[:5]

    if mode == "fn":
        target = [r for r in rows if r["label"] == 1 and r["pred"] == 0]
        return sorted(target, key=lambda r: r["score"])[:5]

    raise ValueError(f"Unknown mode: {mode}")


def copy_file(src: Path, dst: Path) -> str:
    if not src.exists():
        return f"[WARN] missing: {src}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return f"[OK] {src} -> {dst}"


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        print(f"[WARN] no rows to write: {path}")
        return

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_gallery_assets(
    exp: Dict[str, str],
    rows: List[Dict[str, str]],
    mode: str,
) -> List[Dict[str, str]]:
    run_root = exp["run_root"]
    input_root = exp["input_root"]
    overlays_dir = run_root / "overlays"
    heatmaps_dir = run_root / "heatmaps"

    copied = []

    for idx, row in enumerate(rows, start=1):
        image_path = row["image_path"]
        overlay_src = resolve_asset_path(row, "overlay_path", overlays_dir, "image_path")
        heatmap_src = resolve_asset_path(row, "heatmap_path", heatmaps_dir, "image_path")
        input_src = resolve_input_path(input_root, image_path)

        input_suffix = input_src.suffix.lower() if input_src.suffix else ".jpg"
        base = f'{exp["category"]}_{exp["version"]}_{mode}_{idx:02d}'

        input_dst = GALLERY_DIR / f"{base}_input{input_suffix}"
        overlay_dst = GALLERY_DIR / f"{base}_overlay.png"
        heatmap_dst = GALLERY_DIR / f"{base}_heatmap.png"

        msg1 = copy_file(input_src, input_dst)
        msg2 = copy_file(overlay_src, overlay_dst)
        msg3 = copy_file(heatmap_src, heatmap_dst)

        print(msg1)
        print(msg2)
        print(msg3)

        copied.append({
            "experiment": exp["name"],
            "mode": mode,
            "rank": idx,
            "image_path": image_path,
            "score": row["score"],
            "input_src": str(input_src),
            "overlay_src": str(overlay_src),
            "heatmap_src": str(heatmap_src),
            "input_dst": str(input_dst),
            "overlay_dst": str(overlay_dst),
            "heatmap_dst": str(heatmap_dst),
        })

    return copied


def choose_representative(
    fp_rows: List[Dict[str, str]],
    fn_rows: List[Dict[str, str]],
):
    rep_failure = fp_rows[0] if fp_rows else None
    rep_miss = fn_rows[0] if fn_rows else None
    return rep_failure, rep_miss


def copy_representative(exp: Dict[str, str], row: Dict[str, str], tag: str) -> Dict[str, str]:
    run_root = exp["run_root"]
    input_root = exp["input_root"]
    overlays_dir = run_root / "overlays"
    heatmaps_dir = run_root / "heatmaps"

    overlay_src = resolve_asset_path(row, "overlay_path", overlays_dir, "image_path")
    heatmap_src = resolve_asset_path(row, "heatmap_path", heatmaps_dir, "image_path")
    input_src = resolve_input_path(input_root, row["image_path"])

    input_suffix = input_src.suffix.lower() if input_src.suffix else ".jpg"
    input_dst = REP_DIR / f"{tag}_input{input_suffix}"
    overlay_dst = REP_DIR / f"{tag}_overlay.png"
    heatmap_dst = REP_DIR / f"{tag}_heatmap.png"

    msg1 = copy_file(input_src, input_dst)
    msg2 = copy_file(overlay_src, overlay_dst)
    msg3 = copy_file(heatmap_src, heatmap_dst)

    print(msg1)
    print(msg2)
    print(msg3)

    return {
        "tag": tag,
        "image_path": row["image_path"],
        "score": row["score"],
        "input_src": str(input_src),
        "overlay_src": str(overlay_src),
        "heatmap_src": str(heatmap_src),
        "input_dst": str(input_dst),
        "overlay_dst": str(overlay_dst),
        "heatmap_dst": str(heatmap_dst),
    }


def main():
    copied_index = []
    representative_index = []

    for exp in EXPERIMENTS:
        print("=" * 80)
        print(f"[RUN] {exp['name']}")

        csv_path = exp["run_root"] / "predictions.csv"
        rows = load_predictions(csv_path, exp["threshold"])

        fp_top5 = pick_top5(rows, "fp")
        fn_top5 = pick_top5(rows, "fn")

        fp_csv = TOP5_DIR / f'{exp["name"]}_fp_top5.csv'
        fn_csv = TOP5_DIR / f'{exp["name"]}_fn_top5.csv'

        write_csv(fp_csv, fp_top5)
        write_csv(fn_csv, fn_top5)

        print(f"[SAVE] {fp_csv}")
        print(f"[SAVE] {fn_csv}")

        copied_index.extend(save_gallery_assets(exp, fp_top5, "fp"))
        copied_index.extend(save_gallery_assets(exp, fn_top5, "fn"))

        rep_failure, rep_miss = choose_representative(fp_top5, fn_top5)

        if rep_failure:
            representative_index.append(copy_representative(
                exp,
                rep_failure,
                f'{exp["category"]}_{exp["version"]}_representative_fp'
            ))

        if rep_miss:
            representative_index.append(copy_representative(
                exp,
                rep_miss,
                f'{exp["category"]}_{exp["version"]}_representative_fn'
            ))

    gallery_index_csv = TABLE_DIR / "operating_gallery_copied_index.csv"
    representative_index_csv = TABLE_DIR / "operating_representative_index.csv"

    write_csv(gallery_index_csv, copied_index)
    write_csv(representative_index_csv, representative_index)

    print("=" * 80)
    print("[DONE] operating gallery assets prepared.")
    print(f"[SAVE] {gallery_index_csv}")
    print(f"[SAVE] {representative_index_csv}")


if __name__ == "__main__":
    main()