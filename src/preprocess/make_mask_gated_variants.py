from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def should_process(rel_path: Path) -> bool:
    parts = [p.lower() for p in rel_path.parts]
    return len(parts) >= 2 and parts[1] == "images"


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    areas = stats[:, cv2.CC_STAT_AREA].copy()
    areas[0] = 0  # background 제외
    best = int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out


def estimate_foreground_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    GT mask 사용 금지.
    이미지 border의 대표 색과의 LAB 거리로 foreground 추정.
    """
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    bw = max(2, min(h, w) // 20)  # border width
    border_pixels = np.concatenate(
        [
            lab[:bw, :, :].reshape(-1, 3),
            lab[-bw:, :, :].reshape(-1, 3),
            lab[:, :bw, :].reshape(-1, 3),
            lab[:, -bw:, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)

    border_med = np.median(border_pixels, axis=0)

    diff = lab.astype(np.float32) - border_med.reshape(1, 1, 3)
    dist = np.sqrt((diff ** 2).sum(axis=2))

    border_dist = np.sqrt(((border_pixels - border_med.reshape(1, 3)) ** 2).sum(axis=1))
    thr = max(float(np.percentile(border_dist, 98) * 1.25), 8.0)

    fg = (dist > thr).astype(np.uint8) * 255

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)

    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel3)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel5)
    fg = largest_connected_component(fg)

    # 가장자리 잘림 방지용 소폭 팽창
    fg = cv2.dilate(fg, kernel3, iterations=1)

    return fg


def apply_mask_gating(img_bgr: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    foreground 바깥은 검정으로 두지 않고,
    border 대표색으로 채워서 과도한 인공 경계를 줄임.
    """
    h, w = img_bgr.shape[:2]
    bw = max(2, min(h, w) // 20)

    border_pixels = np.concatenate(
        [
            img_bgr[:bw, :, :].reshape(-1, 3),
            img_bgr[-bw:, :, :].reshape(-1, 3),
            img_bgr[:, :bw, :].reshape(-1, 3),
            img_bgr[:, -bw:, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.uint8)

    border_med = np.median(border_pixels, axis=0).astype(np.uint8)
    bg = np.full_like(img_bgr, border_med.reshape(1, 1, 3))

    mask_f = (fg_mask.astype(np.float32) / 255.0)[..., None]
    out = (img_bgr.astype(np.float32) * mask_f + bg.astype(np.float32) * (1.0 - mask_f)).astype(np.uint8)
    return out


def process_dataset(src_root: Path, dst_root: Path, save_debug_mask: bool = False) -> None:
    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    for src_path in src_root.rglob("*"):
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel

        if src_path.is_dir():
            ensure_dir(dst_path)
            continue

        ensure_dir(dst_path.parent)

        if src_path.suffix.lower() in VALID_EXTS and should_process(rel):
            img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] failed to read image: {src_path}")
                shutil.copy2(src_path, dst_path)
                continue

            fg = estimate_foreground_mask(img)
            out = apply_mask_gating(img, fg)

            ext = src_path.suffix.lower()
            if ext in {".jpg", ".jpeg"}:
                cv2.imwrite(str(dst_path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            elif ext == ".png":
                cv2.imwrite(str(dst_path), out, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            else:
                cv2.imwrite(str(dst_path), out)

            if save_debug_mask:
                dbg_dir = dst_root / "meta" / "debug_fg_mask"
                ensure_dir(dbg_dir)
                dbg_name = "__".join(rel.parts) + ".png"
                cv2.imwrite(str(dbg_dir / dbg_name), fg)

        else:
            shutil.copy2(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--save-debug-mask", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    print(f"[INFO] src_root = {src_root}")
    print(f"[INFO] dst_root = {dst_root}")

    if dst_root.exists():
        print(f"[INFO] removing existing dst_root: {dst_root}")
        shutil.rmtree(dst_root)

    ensure_dir(dst_root)
    process_dataset(src_root, dst_root, save_debug_mask=args.save_debug_mask)

    print(f"[DONE] generated mask-gated dataset: {dst_root}")


if __name__ == "__main__":
    main()
