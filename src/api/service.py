from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from src.api.config import (
    PROJECT_ROOT,
    THRESHOLDS,
    ROI_DIR,
    ROI_MASK_DIR,
    PREPROC_DIR,
    OVERLAY_DIR,
    HEATMAP_DIR,
    VISA_ROOT,
)
from src.api.schemas import InspectResponse


def _norm_path(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _load_model_info(cls: str) -> Tuple[np.ndarray, Dict]:
    cfg = THRESHOLDS[cls]
    model_dir = PROJECT_ROOT / Path(cfg["model_dir"])
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    meta_path = model_dir / "metadata.json"
    bank_path = model_dir / "memory_bank.npz"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")
    if not bank_path.exists():
        raise FileNotFoundError(f"memory_bank.npz not found: {bank_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    memory_bank = np.load(bank_path)["memory_bank"].astype(np.float32)
    return memory_bank, meta


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    areas[0] = 0
    best = int(np.argmax(areas))

    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out


def _estimate_foreground_mask(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    bw = max(3, min(h, w) // 25)
    border_pixels = np.concatenate(
        [
            lab[:bw, :, :].reshape(-1, 3),
            lab[-bw:, :, :].reshape(-1, 3),
            lab[:, :bw, :].reshape(-1, 3),
            lab[:, -bw:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    border_med = np.median(border_pixels, axis=0)

    diff = lab - border_med.reshape(1, 1, 3)
    dist = np.sqrt((diff**2).sum(axis=2))

    border_dist = np.sqrt(((border_pixels - border_med.reshape(1, 3)) ** 2).sum(axis=1))
    thr = max(float(np.percentile(border_dist, 98) * 1.25), 8.0)

    fg = (dist > thr).astype(np.uint8) * 255
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)

    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel3)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel5)
    fg = _largest_connected_component(fg)
    fg = cv2.dilate(fg, kernel3, iterations=1)
    return fg


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    h, w = mask.shape[:2]
    pad_x = max(2, int((x2 - x1 + 1) * 0.03))
    pad_y = max(2, int((y2 - y1 + 1) * 0.03))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)

    return x1, y1, x2 + 1, y2 + 1


def _extract_roi(raw_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fg = _estimate_foreground_mask(raw_bgr)
    x1, y1, x2, y2 = _bbox_from_mask(fg)
    roi_bgr = raw_bgr[y1:y2, x1:x2].copy()
    roi_mask = fg[y1:y2, x1:x2].copy()
    return roi_bgr, roi_mask


def _apply_variant_preproc(roi_bgr: np.ndarray, cls: str) -> np.ndarray:
    cfg = THRESHOLDS[cls]
    variant = str(cfg.get("variant", "")).lower()

    out = roi_bgr.copy()

    if cls == "pcb4" and "clahe" in variant:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if "median5" in variant:
        out = cv2.medianBlur(out, 5)
    elif "median3" in variant:
        out = cv2.medianBlur(out, 3)
    elif "gaussian3" in variant:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    elif "bilateral" in variant:
        out = cv2.bilateralFilter(out, d=5, sigmaColor=30, sigmaSpace=30)

    return out


def _patch_feature_vector(patch_rgb: np.ndarray) -> np.ndarray:
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


def _extract_patch_embeddings(image_rgb: np.ndarray, patch_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = image_rgb.shape[:2]
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"image_size must be divisible by patch_size. got {(h, w)} / patch_size={patch_size}")

    rows = h // patch_size
    cols = w // patch_size
    feats = []

    for r in range(rows):
        for c in range(cols):
            patch = image_rgb[r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size]
            feats.append(_patch_feature_vector(patch))

    return np.stack(feats, axis=0), (rows, cols)


def _min_distances_chunked(patches: np.ndarray, memory_bank: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    mins = []
    for start in range(0, len(patches), chunk_size):
        chunk = patches[start:start + chunk_size]
        dists = np.sqrt(np.sum((chunk[:, None, :] - memory_bank[None, :, :]) ** 2, axis=2))
        mins.append(dists.min(axis=1))
    return np.concatenate(mins, axis=0)


def _aggregate_image_score(patch_scores: np.ndarray, topk_ratio: float = 0.05) -> float:
    n = len(patch_scores)
    k = max(1, int(n * topk_ratio))
    idx = np.argpartition(patch_scores, -k)[-k:]
    return float(np.mean(patch_scores[idx]))


def _normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    amap = anomaly_map.astype(np.float32)
    amin, amax = float(amap.min()), float(amap.max())
    if amax - amin < 1e-8:
        return np.zeros_like(amap, dtype=np.float32)
    return (amap - amin) / (amax - amin)


def _make_heatmap_overlay(rgb_image: np.ndarray, anomaly_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    amap = _normalize_map(anomaly_map)
    amap_u8 = (amap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(amap_u8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb_image, 0.65, heatmap, 0.35, 0.0)
    return heatmap, overlay


def _save_rgb(path: Path, arr_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))


def _save_bgr(path: Path, arr_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr_bgr)


def _stem_for_assets(image_path: Path, cls: str) -> str:
    return f"{image_path.stem}_{cls}"


def _infer_gt_label_from_paths(*texts: str | None) -> str:
    for text in texts:
        if not text:
            continue
        lowered = text.replace("\\", "/").lower()
        parts = [p for p in lowered.split("/") if p]

        if "normal" in parts or "ok" in parts:
            return "OK"
        if "anomaly" in parts or "ng" in parts:
            return "NG"

    return "UNKNOWN"


def _infer_gt_label_from_dataset_lookup(cls: str, original_filename: str | None) -> str:
    if not original_filename:
        return "UNKNOWN"

    name = Path(original_filename).name
    image_root = VISA_ROOT / cls / "Data" / "Images"
    normal_dir = image_root / "Normal"
    anomaly_dir = image_root / "Anomaly"

    normal_exists = (normal_dir / name).exists()
    anomaly_exists = (anomaly_dir / name).exists()

    if normal_exists and not anomaly_exists:
        return "OK"
    if anomaly_exists and not normal_exists:
        return "NG"

    return "UNKNOWN"


def inspect_one(
    image_path: str,
    cls: str,
    gt_label: str = "UNKNOWN",
    source_path: str | None = None,
    original_filename: str | None = None,
) -> InspectResponse:
    if cls not in THRESHOLDS:
        raise ValueError(f"Unsupported cls: {cls}")

    start = time.perf_counter()

    img_path = Path(image_path)
    raw_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if raw_bgr is None:
        raise RuntimeError(f"Failed to read uploaded image: {img_path}")

    memory_bank, meta = _load_model_info(cls)
    image_size = int(meta["image_size"])
    patch_size = int(meta["patch_size"])

    roi_bgr, roi_mask = _extract_roi(raw_bgr)
    if roi_bgr.size == 0:
        roi_bgr = raw_bgr.copy()
        roi_mask = np.full(raw_bgr.shape[:2], 255, dtype=np.uint8)

    preproc_bgr = _apply_variant_preproc(roi_bgr, cls)
    proc_bgr = cv2.resize(preproc_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)

    patch_embeds, (rows, cols) = _extract_patch_embeddings(proc_rgb, patch_size)
    patch_scores = _min_distances_chunked(patch_embeds, memory_bank)
    anomaly_map_small = patch_scores.reshape(rows, cols)
    anomaly_map = cv2.resize(
        anomaly_map_small,
        (proc_rgb.shape[1], proc_rgb.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    pred_score = _aggregate_image_score(patch_scores, topk_ratio=0.05)

    heatmap_rgb, overlay_rgb = _make_heatmap_overlay(proc_rgb, anomaly_map)

    cfg = THRESHOLDS[cls]
    decision_thr = float(cfg["threshold"])
    decision_reason = str(cfg.get("reason", "thresholds_final_operating.json"))

    pred_label = "NG" if float(pred_score) >= decision_thr else "OK"

    resolved_gt_label = (gt_label or "UNKNOWN").strip().upper()
    if resolved_gt_label not in {"OK", "NG"}:
        resolved_gt_label = _infer_gt_label_from_paths(source_path, original_filename)
    if resolved_gt_label == "UNKNOWN":
        resolved_gt_label = _infer_gt_label_from_dataset_lookup(cls, original_filename)

    if resolved_gt_label == "UNKNOWN":
        is_correct = None
    else:
        is_correct = pred_label.upper() == resolved_gt_label.upper()

    stem = _stem_for_assets(img_path, cls)

    roi_path = ROI_DIR / f"{stem}_roi.png"
    roi_mask_path = ROI_MASK_DIR / f"{stem}_roi_mask.png"
    preproc_path = PREPROC_DIR / f"{stem}_preproc.png"
    overlay_path = OVERLAY_DIR / f"{stem}.png"
    heatmap_path = HEATMAP_DIR / f"{stem}.png"

    _save_bgr(roi_path, roi_bgr)
    _save_bgr(roi_mask_path, roi_mask)
    _save_bgr(preproc_path, proc_bgr)
    _save_rgb(overlay_path, overlay_rgb)
    _save_rgb(heatmap_path, heatmap_rgb)

    processing_ms = int((time.perf_counter() - start) * 1000)

    return InspectResponse(
        cls=cls,
        gt_label=resolved_gt_label,
        pred_label=pred_label,
        is_correct=is_correct,
        pred_score=round(float(pred_score), 4),
        decision_thr=round(float(decision_thr), 10),
        decision_reason=decision_reason,
        overlay_path=_norm_path(overlay_path),
        heatmap_path=_norm_path(heatmap_path),
        image_path=_norm_path(img_path),
        processing_ms=processing_ms,
        roi_path=_norm_path(roi_path),
        roi_mask_path=_norm_path(roi_mask_path),
        preprocessed_path=_norm_path(preproc_path),
    )
