from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessParams:
    # 최종 bbox margin
    margin_ratio: float = 0.02

    # 나중 CLAHE 비교용으로만 유지
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    # border background estimation
    border_width_ratio: float = 0.06
    min_border_px: int = 12
    max_border_px: int = 96

    # distance map smoothing / cleanup
    dist_blur_kernel: int = 5
    region_kernel: int = 15
    small_blob_min_area_ratio: float = 0.00003

    # AB 우선 / LAB fallback 선택 score
    score_center_weight: float = 3.0
    score_border_weight: float = 4.0
    score_area_weight: float = 2.0
    score_target_area_ratio: float = 0.16
    score_min_area_ratio: float = 0.01
    score_max_area_ratio: float = 0.80
    ab_preference_bonus: float = 0.15

    # LAB fallback에서 L 영향 과도함 방지
    lab_l_weight: float = 0.35

    # anomaly mask union
    raw_mask_dilate_kernel: int = 9

    # final validation
    min_area_ratio: float = 0.02
    max_area_ratio: float = 0.95
    min_box_size: int = 96
    min_box_w_ratio: float = 0.10
    min_box_h_ratio: float = 0.10

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["clahe_tile_grid_size"] = list(self.clahe_tile_grid_size)
        return data


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _normalize_u8(img: np.ndarray) -> np.ndarray:
    out = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return out.astype(np.uint8)


def _estimate_border_width(h: int, w: int, params: PreprocessParams) -> int:
    bw = int(round(min(h, w) * params.border_width_ratio))
    bw = max(params.min_border_px, bw)
    bw = min(params.max_border_px, bw)
    return max(1, bw)


def _estimate_background_lab(image_bgr: np.ndarray, params: PreprocessParams) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    bw = _estimate_border_width(h, w, params)

    top = lab[:bw, :, :].reshape(-1, 3)
    bottom = lab[h - bw:, :, :].reshape(-1, 3)
    left = lab[:, :bw, :].reshape(-1, 3)
    right = lab[:, w - bw:, :].reshape(-1, 3)

    border_pixels = np.concatenate([top, bottom, left, right], axis=0)
    bg_color = np.median(border_pixels, axis=0).astype(np.float32)
    return bg_color


def _background_distance_map_ab(image_bgr: np.ndarray, params: PreprocessParams) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = _estimate_background_lab(image_bgr, params)

    diff_ab = lab[:, :, 1:3] - bg[1:3].reshape(1, 1, 2)
    dist = np.sqrt(np.sum(diff_ab * diff_ab, axis=2))

    dist_u8 = _normalize_u8(dist)
    k = _ensure_odd(params.dist_blur_kernel)
    dist_u8 = cv2.GaussianBlur(dist_u8, (k, k), 0)
    return dist_u8


def _background_distance_map_lab(image_bgr: np.ndarray, params: PreprocessParams) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = _estimate_background_lab(image_bgr, params)

    diff = lab - bg.reshape(1, 1, 3)
    diff[:, :, 0] *= params.lab_l_weight
    dist = np.sqrt(np.sum(diff * diff, axis=2))

    dist_u8 = _normalize_u8(dist)
    k = _ensure_odd(params.dist_blur_kernel)
    dist_u8 = cv2.GaussianBlur(dist_u8, (k, k), 0)
    return dist_u8


def _otsu_binary(img: np.ndarray) -> np.ndarray:
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def _clear_border_connected(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    border_labels = set()
    border_labels.update(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[h - 1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, w - 1]).tolist())

    out = np.zeros_like(mask)
    for label_id in range(1, num_labels):
        if label_id in border_labels:
            continue
        out[labels == label_id] = 255
    return out


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == label_id] = 255

    return out


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    best_label = 0
    best_area = 0
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best_label = label_id

    out = np.zeros_like(mask)
    if best_label > 0:
        out[labels == best_label] = 255
    return out


def _fill_holes(bin_mask: np.ndarray) -> np.ndarray:
    h, w = bin_mask.shape[:2]
    flood = bin_mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    inv_flood = cv2.bitwise_not(flood)
    return cv2.bitwise_or(bin_mask, inv_flood)


def _region_clean(mask: np.ndarray, kernel_size: int, min_area: int) -> np.ndarray:
    k = _ensure_odd(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    out = _clear_border_connected(out)
    out = _remove_small_components(out, min_area)
    out = _keep_largest_component(out)
    out = _fill_holes(out)
    out = _keep_largest_component(out)
    return out


def _center_ratio(mask: np.ndarray, center_ratio: float = 0.5) -> float:
    h, w = mask.shape[:2]
    ch = max(1, int(round(h * center_ratio)))
    cw = max(1, int(round(w * center_ratio)))
    y1 = max(0, (h - ch) // 2)
    x1 = max(0, (w - cw) // 2)
    patch = mask[y1:y1 + ch, x1:x1 + cw]
    return float((patch > 0).sum()) / float(max(1, patch.size))


def _border_ratio(mask: np.ndarray, border_ratio: float = 0.08) -> float:
    h, w = mask.shape[:2]
    by = max(1, int(round(h * border_ratio)))
    bx = max(1, int(round(w * border_ratio)))

    border = np.zeros_like(mask, dtype=np.uint8)
    border[:by, :] = 1
    border[h - by:, :] = 1
    border[:, :bx] = 1
    border[:, w - bx:] = 1

    border_pixels = int(border.sum())
    if border_pixels == 0:
        return 0.0

    fg_border = int(((mask > 0) & (border > 0)).sum())
    return float(fg_border) / float(border_pixels)


def _score_candidate(
    mask: np.ndarray,
    source_name: str,
    params: PreprocessParams,
    img_area: float,
) -> Tuple[float, bool]:
    area_ratio = float((mask > 0).sum()) / img_area
    if area_ratio < params.score_min_area_ratio or area_ratio > params.score_max_area_ratio:
        return -1e9, False

    center = _center_ratio(mask, 0.5)
    border = _border_ratio(mask, 0.08)
    area_penalty = abs(area_ratio - params.score_target_area_ratio)

    score = (
        params.score_center_weight * center
        - params.score_border_weight * border
        - params.score_area_weight * area_penalty
    )

    if source_name == "ab_dist":
        score += params.ab_preference_bonus

    return score, True


def _get_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return x1, y1, max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)


def _expand_bbox(
    bbox: Tuple[int, int, int, int],
    expand_ratio: float,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    mx = int(round(w * expand_ratio))
    my = int(round(h * expand_ratio))

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(img_w, x + w + mx)
    y2 = min(img_h, y + h + my)

    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _union_bbox(
    b1: Tuple[int, int, int, int],
    b2: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    xa = min(x1, x2)
    ya = min(y1, y2)
    xb = max(x1 + w1, x2 + w2)
    yb = max(y1 + h1, y2 + h2)

    xa = max(0, xa)
    ya = max(0, ya)
    xb = min(img_w, xb)
    yb = min(img_h, yb)

    return xa, ya, max(1, xb - xa), max(1, yb - ya)


def detect_object_roi(
    image: np.ndarray,
    params: PreprocessParams,
    raw_mask: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]:
    """
    foreground mask 기반 ROI-only
    - AB distance 우선
    - LAB distance fallback
    - 최종 ROI는 boundingRect
    """
    if image is None or image.size == 0:
        raise ValueError("빈 이미지가 들어왔습니다.")

    img_h, img_w = image.shape[:2]
    img_area = float(img_w * img_h)
    min_area = max(20, int(img_area * params.small_blob_min_area_ratio))

    dist_ab = _background_distance_map_ab(image, params)
    dist_lab = _background_distance_map_lab(image, params)

    candidate_defs = [
        ("ab_dist", dist_ab),
        ("lab_dist", dist_lab),
    ]

    candidates = []
    for name, dist_map in candidate_defs:
        raw_mask_bin = _otsu_binary(dist_map)
        cleaned = _region_clean(raw_mask_bin, params.region_kernel, min_area)
        score, valid = _score_candidate(cleaned, name, params, img_area)
        if valid:
            candidates.append((score, name, cleaned))

    if not candidates:
        debug = {
            "dist_ab": dist_ab,
            "dist_lab": dist_lab,
            "foreground_mask": np.zeros((img_h, img_w), dtype=np.uint8),
        }
        roi_info = {"crop_mode": "bbox", "bbox": (0, 0, img_w, img_h)}
        meta = {
            "roi_status": "fallback_full_image",
            "roi_source": "fallback_full_image",
            "roi_method": "ab_priority_lab_fallback_bbox_roi_only",
            "roi_mask_source": "none",
            "roi_x": 0,
            "roi_y": 0,
            "roi_w": img_w,
            "roi_h": img_h,
        }
        return roi_info, meta, debug

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, chosen_source, foreground_mask = candidates[0]

    if raw_mask is not None:
        if raw_mask.ndim == 3:
            raw_mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)
        raw_mask_bin = np.where(raw_mask > 0, 255, 0).astype(np.uint8)
        if int((raw_mask_bin > 0).sum()) > 0:
            rk = _ensure_odd(params.raw_mask_dilate_kernel)
            raw_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rk, rk))
            raw_mask_bin = cv2.dilate(raw_mask_bin, raw_kernel, iterations=1)
            foreground_mask = cv2.bitwise_or(foreground_mask, raw_mask_bin)
            foreground_mask = _fill_holes(foreground_mask)
            foreground_mask = _keep_largest_component(foreground_mask)

    bbox = _get_bbox_from_mask(foreground_mask)
    if bbox is None:
        debug = {
            "dist_ab": dist_ab,
            "dist_lab": dist_lab,
            "foreground_mask": foreground_mask,
        }
        roi_info = {"crop_mode": "bbox", "bbox": (0, 0, img_w, img_h)}
        meta = {
            "roi_status": "fallback_full_image",
            "roi_source": "fallback_full_image",
            "roi_method": "ab_priority_lab_fallback_bbox_roi_only",
            "roi_mask_source": chosen_source,
            "roi_x": 0,
            "roi_y": 0,
            "roi_w": img_w,
            "roi_h": img_h,
        }
        return roi_info, meta, debug

    x, y, w, h = bbox
    area_ratio = float((foreground_mask > 0).sum()) / img_area

    valid = True
    if area_ratio < params.min_area_ratio or area_ratio > params.max_area_ratio:
        valid = False
    if w < params.min_box_size or h < params.min_box_size:
        valid = False
    if (w / img_w) < params.min_box_w_ratio:
        valid = False
    if (h / img_h) < params.min_box_h_ratio:
        valid = False

    if not valid:
        debug = {
            "dist_ab": dist_ab,
            "dist_lab": dist_lab,
            "foreground_mask": foreground_mask,
        }
        roi_info = {"crop_mode": "bbox", "bbox": (0, 0, img_w, img_h)}
        meta = {
            "roi_status": "fallback_full_image",
            "roi_source": "fallback_full_image",
            "roi_method": "ab_priority_lab_fallback_bbox_roi_only",
            "roi_mask_source": chosen_source,
            "roi_x": 0,
            "roi_y": 0,
            "roi_w": img_w,
            "roi_h": img_h,
        }
        return roi_info, meta, debug

    bbox = _expand_bbox(bbox, params.margin_ratio, img_w, img_h)
    bx, by, bw, bh = bbox

    if raw_mask is not None:
        ys, xs = np.where(raw_mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            mx1, my1 = int(xs.min()), int(ys.min())
            mx2, my2 = int(xs.max()), int(ys.max())
            mask_bbox = (mx1, my1, max(1, mx2 - mx1 + 1), max(1, my2 - my1 + 1))
            bbox = _union_bbox(bbox, mask_bbox, img_w, img_h)
            bx, by, bw, bh = bbox

    roi_info = {"crop_mode": "bbox", "bbox": bbox}
    meta = {
        "roi_status": "detected",
        "roi_source": "ab_priority_lab_fallback_bbox_roi_only",
        "roi_method": "ab_priority_lab_fallback_bbox_roi_only",
        "roi_mask_source": chosen_source,
        "roi_x": bx,
        "roi_y": by,
        "roi_w": bw,
        "roi_h": bh,
    }
    debug = {
        "dist_ab": dist_ab,
        "dist_lab": dist_lab,
        "foreground_mask": foreground_mask,
    }
    return roi_info, meta, debug


def crop_by_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return image[y:y + h, x:x + w]


def apply_clahe_lab(image: np.ndarray, params: PreprocessParams) -> np.ndarray:
    """
    ROI-only 단계에서는 호출하지 않음.
    나중 ROI+CLAHE 비교용으로만 유지.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("CLAHE 대상 이미지는 3채널 컬러 이미지여야 합니다.")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=params.clahe_clip_limit,
        tileGridSize=params.clahe_tile_grid_size,
    )
    l2 = clahe.apply(l)

    merged = cv2.merge([l2, a, b])
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return out


def process_image_and_mask(
    image: np.ndarray,
    params: PreprocessParams,
    raw_mask: Optional[np.ndarray] = None,
    roi_info: Optional[Dict[str, Any]] = None,
    roi_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    ROI-only
    - CLAHE 미적용
    - boundingRect crop만 수행
    """
    if roi_info is None or roi_meta is None:
        roi_info, roi_meta, _ = detect_object_roi(image, params, raw_mask=raw_mask)

    bbox = roi_info["bbox"]
    cropped_img = crop_by_bbox(image, bbox)

    if raw_mask is None:
        cropped_mask = np.zeros((cropped_img.shape[0], cropped_img.shape[1]), dtype=np.uint8)
    else:
        if raw_mask.ndim == 3:
            raw_mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)
        cropped_mask = crop_by_bbox(raw_mask, bbox)

    processed_img = cropped_img.copy()
    processed_mask = np.where(cropped_mask > 127, 255, 0).astype(np.uint8)

    meta = {
        **roi_meta,
        "processed_width": processed_img.shape[1],
        "processed_height": processed_img.shape[0],
        "preprocessing_stage": "roi_only",
        "clahe_applied": False,
        "margin_ratio": params.margin_ratio,
    }
    return processed_img, processed_mask, meta
