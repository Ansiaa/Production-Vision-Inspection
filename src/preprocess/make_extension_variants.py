from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def apply_filter(img, filter_name: str):
    if filter_name == "median3":
        return cv2.medianBlur(img, 3)
    if filter_name == "median5":
        return cv2.medianBlur(img, 5)
    if filter_name == "gaussian3":
        return cv2.GaussianBlur(img, (3, 3), 0)
    if filter_name == "bilateral":
        return cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)
    raise ValueError(f"Unknown filter: {filter_name}")


def should_filter(rel_path: Path) -> bool:
    parts = [p.lower() for p in rel_path.parts]
    return len(parts) >= 2 and parts[1] == "images"


def process_dataset(src_root: Path, dst_root: Path, filter_name: str) -> None:
    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    for src_path in src_root.rglob("*"):
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel

        if src_path.is_dir():
            ensure_dir(dst_path)
            continue

        ensure_dir(dst_path.parent)

        if src_path.suffix.lower() in VALID_EXTS and should_filter(rel):
            img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] failed to read image: {src_path}")
                shutil.copy2(src_path, dst_path)
                continue

            out = apply_filter(img, filter_name)

            ext = src_path.suffix.lower()
            if ext in {".jpg", ".jpeg"}:
                cv2.imwrite(str(dst_path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            elif ext == ".png":
                cv2.imwrite(str(dst_path), out, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            else:
                cv2.imwrite(str(dst_path), out)

        else:
            shutil.copy2(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--filter", required=True, choices=["median3", "median5", "gaussian3", "bilateral"])
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    print(f"[INFO] src_root = {src_root}")
    print(f"[INFO] dst_root = {dst_root}")
    print(f"[INFO] filter   = {args.filter}")

    if dst_root.exists():
        print(f"[INFO] removing existing dst_root: {dst_root}")
        shutil.rmtree(dst_root)

    ensure_dir(dst_root)
    process_dataset(src_root, dst_root, args.filter)

    print(f"[DONE] generated variant dataset: {dst_root}")


if __name__ == "__main__":
    main()
