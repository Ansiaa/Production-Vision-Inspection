from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NORMAL_DIR_NAMES = {"good", "normal", "ok", "pass"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lightweight PatchCore-style inference (dependency-safe version)")
    p.add_argument("--data-root", type=str, required=True, help="Dataset root or category root")
    p.add_argument("--category", type=str, default="", help="Category name, e.g. pcb4")
    p.add_argument("--model-dir", type=str, required=True, help="Directory from patchcore_train")
    p.add_argument("--output-dir", type=str, required=True, help="Inference output directory")
    p.add_argument("--threshold", type=float, default=None,
                   help="Optional fixed threshold. If omitted, best F1 threshold is searched when labels are available.")
    p.add_argument("--topk", type=float, default=0.05,
                   help="Top-k patch ratio for image score aggregation. 0.05 means top 5%% mean.")
    p.add_argument("--save-heatmaps", action="store_true")
    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--save-npy", action="store_true", help="Save raw anomaly maps as .npy")
    return p.parse_args()


def resolve_category_root(data_root: Path, category: str) -> Path:
    if category:
        if data_root.name == category:
            return data_root
        candidate = data_root / category
        if candidate.exists():
            return candidate
    return data_root


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if is_image_file(p)])


def list_images_strict(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return list_images(root)


def discover_test_images(category_root: Path) -> List[Tuple[Path, int, str]]:
    """
    Safety-first discovery.

    Priority:
    1) current project format:
       - <category>/normal/images/*   -> label 0
       - <category>/anomaly/images/*  -> label 1
    2) mvtec-like fallback:
       - <category>/test/<good|normal|ok|pass>/* -> label 0
       - <category>/test/<other defect name>/*   -> label 1

    Explicitly avoids category_root-wide rglob so masks/, meta/, debug_roi/ do not leak in.
    """
    items: List[Tuple[Path, int, str]] = []

    normal_imgs = list_images_strict(category_root / "normal" / "images")
    anomaly_imgs = list_images_strict(category_root / "anomaly" / "images")
    if normal_imgs or anomaly_imgs:
        items.extend((p, 0, "normal") for p in normal_imgs)
        items.extend((p, 1, "anomaly") for p in anomaly_imgs)
        return items

    test_dir = category_root / "test"
    if test_dir.exists():
        for sub in sorted([p for p in test_dir.iterdir() if p.is_dir()]):
            label_name = sub.name
            label = 0 if label_name.lower() in NORMAL_DIR_NAMES else 1
            for img_path in list_images(sub):
                items.append((img_path, label, label_name))
        if items:
            return items

    raise RuntimeError(
        "No valid inference images found. Expected one of:\n"
        f"  - {category_root / 'normal' / 'images'}\n"
        f"  - {category_root / 'anomaly' / 'images'}\n"
        f"  - {category_root / 'test'} / <good|defect_name>"
    )


def load_rgb(path: Path, image_size: int) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img_bgr = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def patch_feature_vector(patch_rgb: np.ndarray) -> np.ndarray:
    patch = patch_rgb.astype(np.float32) / 255.0
    patch_lab = cv2.cvtColor((patch * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    gray = cv2.cvtColor((patch * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    feats = []
    feats.extend(patch.reshape(-1, 3).mean(axis=0).tolist())
    feats.extend(patch.reshape(-1, 3).std(axis=0).tolist())
    feats.extend(patch_lab.reshape(-1, 3).mean(axis=0).tolist())
    feats.extend(patch_lab.reshape(-1, 3).std(axis=0).tolist())
    feats.append(float(gray.mean()))
    feats.append(float(gray.std()))
    feats.append(float(grad.mean()))
    feats.append(float(grad.std()))
    return np.asarray(feats, dtype=np.float32)


def extract_patch_embeddings(image_rgb: np.ndarray, patch_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = image_rgb.shape[:2]
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"image_size must be divisible by patch_size. got {(h, w)} and patch_size={patch_size}")
    rows = h // patch_size
    cols = w // patch_size
    feats = []
    for r in range(rows):
        for c in range(cols):
            patch = image_rgb[r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size]
            feats.append(patch_feature_vector(patch))
    return np.stack(feats, axis=0), (rows, cols)


def min_distances_chunked(patches: np.ndarray, memory_bank: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    mins = []
    for start in range(0, len(patches), chunk_size):
        chunk = patches[start:start + chunk_size]
        dists = np.sqrt(np.sum((chunk[:, None, :] - memory_bank[None, :, :]) ** 2, axis=2))
        mins.append(dists.min(axis=1))
    return np.concatenate(mins, axis=0)


def aggregate_image_score(patch_scores: np.ndarray, topk_ratio: float) -> float:
    topk_ratio = float(np.clip(topk_ratio, 0.001, 1.0))
    n = len(patch_scores)
    k = max(1, int(n * topk_ratio))
    idx = np.argpartition(patch_scores, -k)[-k:]
    return float(np.mean(patch_scores[idx]))


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    amap = anomaly_map.astype(np.float32)
    amin, amax = float(amap.min()), float(amap.max())
    if amax - amin < 1e-8:
        return np.zeros_like(amap, dtype=np.float32)
    return (amap - amin) / (amax - amin)


def make_heatmap_overlay(rgb_image: np.ndarray, anomaly_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    amap = normalize_map(anomaly_map)
    amap_u8 = (amap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(amap_u8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb_image, 0.65, heatmap, 0.35, 0.0)
    return heatmap, overlay


def compute_best_threshold(labels: Sequence[int], scores: Sequence[float]) -> Tuple[float, Dict[str, float]]:
    labels_arr = np.asarray(labels, dtype=np.int32)
    scores_arr = np.asarray(scores, dtype=np.float32)
    unique_scores = np.unique(scores_arr)
    best = {"threshold": float(unique_scores[0]), "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for thr in unique_scores:
        preds = (scores_arr >= thr).astype(np.int32)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_arr, preds, average="binary", zero_division=0
        )
        if f1 > best["f1"]:
            best = {
                "threshold": float(thr),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
    threshold = float(best.pop("threshold"))
    return threshold, best


def save_image(path: Path, array_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(array_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    memory_bank = np.load(model_dir / "memory_bank.npz")["memory_bank"].astype(np.float32)
    image_size = int(meta["image_size"])
    patch_size = int(meta["patch_size"])

    category_root = resolve_category_root(data_root, args.category)
    if not category_root.exists():
        raise FileNotFoundError(f"Category root not found: {category_root}")

    test_items = discover_test_images(category_root)
    if not test_items:
        raise RuntimeError(f"No test images found under: {category_root}")

    heatmap_dir = output_dir / "heatmaps"
    overlay_dir = output_dir / "overlays"
    npy_dir = output_dir / "anomaly_maps"
    if args.save_heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.save_npy:
        npy_dir.mkdir(parents=True, exist_ok=True)

    n_normal = sum(1 for _, label, _ in test_items if label == 0)
    n_anomaly = sum(1 for _, label, _ in test_items if label == 1)
    print(f"[info] category_root={category_root}")
    print(f"[info] n_test_images={len(test_items)} (normal={n_normal}, anomaly={n_anomaly})")
    print(f"[info] memory_bank shape={memory_bank.shape}")
    print(f"[info] image_size={image_size}, patch_size={patch_size}")

    records: List[Dict[str, object]] = []
    for idx, (path, label, label_name) in enumerate(test_items, start=1):
        img_rgb = load_rgb(path, image_size)
        patch_embeds, (rows, cols) = extract_patch_embeddings(img_rgb, patch_size)
        patch_scores = min_distances_chunked(patch_embeds, memory_bank)
        anomaly_map_small = patch_scores.reshape(rows, cols)
        anomaly_map = cv2.resize(anomaly_map_small, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        image_score = aggregate_image_score(patch_scores, args.topk)

        try:
            rel = path.relative_to(category_root)
        except ValueError:
            rel = Path(path.name)
        stem_safe = str(rel).replace("\\", "__").replace("/", "__")

        heatmap_path = ""
        overlay_path = ""
        npy_path = ""
        if args.save_heatmaps or args.save_overlays:
            heatmap, overlay = make_heatmap_overlay(img_rgb, anomaly_map)
            if args.save_heatmaps:
                heatmap_path = str(heatmap_dir / f"{stem_safe}.png")
                save_image(Path(heatmap_path), heatmap)
            if args.save_overlays:
                overlay_path = str(overlay_dir / f"{stem_safe}.png")
                save_image(Path(overlay_path), overlay)
        if args.save_npy:
            npy_path = str(npy_dir / f"{stem_safe}.npy")
            np.save(npy_path, anomaly_map.astype(np.float32))

        records.append(
            {
                "image_path": str(path),
                "relative_path": str(rel),
                "label": int(label),
                "label_name": label_name,
                "score": float(image_score),
                "heatmap_path": heatmap_path,
                "overlay_path": overlay_path,
                "anomaly_map_path": npy_path,
            }
        )

        if idx % 10 == 0 or idx == len(test_items):
            print(f"[info] processed test image {idx}/{len(test_items)}")

    records = sorted(records, key=lambda x: float(x["score"]), reverse=True)
    labels = [int(r["label"]) for r in records]
    scores = [float(r["score"]) for r in records]

    metrics: Dict[str, float] = {}
    if args.threshold is not None:
        threshold = float(args.threshold)
        preds = [1 if s >= threshold else 0 for s in scores]
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        metrics.update({"precision": float(precision), "recall": float(recall), "f1": float(f1)})
    else:
        if len(set(labels)) >= 2:
            threshold, m = compute_best_threshold(labels, scores)
            metrics.update(m)
        else:
            threshold = float(np.median(scores)) if scores else 0.0

    for r in records:
        r["pred_label"] = 1 if float(r["score"]) >= threshold else 0

    fieldnames = [
        "image_path", "relative_path", "label", "label_name", "score",
        "pred_label", "heatmap_path", "overlay_path", "anomaly_map_path"
    ]
    write_csv(output_dir / "predictions.csv", records, fieldnames)

    tp = sum(1 for r in records if int(r["label"]) == 1 and int(r["pred_label"]) == 1)
    tn = sum(1 for r in records if int(r["label"]) == 0 and int(r["pred_label"]) == 0)
    fp = sum(1 for r in records if int(r["label"]) == 0 and int(r["pred_label"]) == 1)
    fn = sum(1 for r in records if int(r["label"]) == 1 and int(r["pred_label"]) == 0)

    if len(set(labels)) >= 2:
        try:
            metrics["auroc"] = float(roc_auc_score(labels, scores))
        except Exception:
            metrics["auroc"] = float("nan")
    else:
        metrics["auroc"] = float("nan")

    summary = {
        "category": args.category or category_root.name,
        "category_root": str(category_root),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "n_images": int(len(records)),
        "n_normal": int(n_normal),
        "n_anomaly": int(n_anomaly),
        "threshold": float(threshold),
        "topk": float(args.topk),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        **metrics,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fp_rows = [r for r in records if int(r["label"]) == 0 and int(r["pred_label"]) == 1]
    fp_rows = sorted(fp_rows, key=lambda x: float(x["score"]), reverse=True)
    fn_rows = [r for r in records if int(r["label"]) == 1 and int(r["pred_label"]) == 0]
    fn_rows = sorted(fn_rows, key=lambda x: float(x["score"]))
    write_csv(output_dir / "fp_top30.csv", fp_rows[:30], fieldnames)
    write_csv(output_dir / "fn_top30.csv", fn_rows[:30], fieldnames)

    print(f"[done] saved predictions: {output_dir / 'predictions.csv'}")
    print(f"[done] saved summary    : {output_dir / 'summary.json'}")
    print(f"[done] threshold        : {threshold:.6f}")
    print(f"[done] confusion        : TP={tp} TN={tn} FP={fp} FN={fn}")
    if any("masks" in str(r["relative_path"]).lower() or "meta" in str(r["relative_path"]).lower() for r in records):
        print("[warn] unexpected masks/meta path detected in predictions. check dataset layout.")


if __name__ == "__main__":
    main()
