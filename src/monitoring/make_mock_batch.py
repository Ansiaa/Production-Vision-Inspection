from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(root: Path):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in VALID_EXTS])


def apply_synthetic_shift(img, mode: str):
    if mode == "none":
        return img

    if mode == "mild":
        out = cv2.convertScaleAbs(img, alpha=1.08, beta=8)
        return out

    if mode == "moderate":
        out = cv2.convertScaleAbs(img, alpha=1.15, beta=15)
        out = cv2.GaussianBlur(out, (3, 3), 0)
        return out

    if mode == "strong":
        out = cv2.convertScaleAbs(img, alpha=1.25, beta=25)
        out = cv2.GaussianBlur(out, (5, 5), 0)
        return out

    raise ValueError(f"Unknown mode: {mode}")


def copy_or_transform(src: Path, dst: Path, drift_mode: str):
    ensure_dir(dst.parent)
    if drift_mode == "none":
        shutil.copy2(src, dst)
        return

    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        shutil.copy2(src, dst)
        return

    out = apply_synthetic_shift(img, drift_mode)

    if src.suffix.lower() in {".jpg", ".jpeg"}:
        cv2.imwrite(str(dst), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    elif src.suffix.lower() == ".png":
        cv2.imwrite(str(dst), out, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        cv2.imwrite(str(dst), out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--n-normal", type=int, default=80)
    parser.add_argument("--n-anomaly", type=int, default=20)
    parser.add_argument("--drift-mode", choices=["none", "mild", "moderate", "strong"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    if dst_root.exists():
        shutil.rmtree(dst_root)

    normal_src = src_root / "normal" / "images"
    anomaly_src = src_root / "anomaly" / "images"

    normal_imgs = list_images(normal_src)
    anomaly_imgs = list_images(anomaly_src)

    if len(normal_imgs) < args.n_normal:
        raise RuntimeError(f"Not enough normal images: {len(normal_imgs)} < {args.n_normal}")
    if len(anomaly_imgs) < args.n_anomaly:
        raise RuntimeError(f"Not enough anomaly images: {len(anomaly_imgs)} < {args.n_anomaly}")

    chosen_normal = random.sample(normal_imgs, args.n_normal)
    chosen_anomaly = random.sample(anomaly_imgs, args.n_anomaly)

    for src in chosen_normal:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        copy_or_transform(src, dst, args.drift_mode)

    for src in chosen_anomaly:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        copy_or_transform(src, dst, args.drift_mode)

    print(f"[DONE] mock batch created: {dst_root}")
    print(f"[INFO] drift_mode={args.drift_mode}, n_normal={args.n_normal}, n_anomaly={args.n_anomaly}")


if __name__ == "__main__":
    main()
