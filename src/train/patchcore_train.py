from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NORMAL_DIR_NAMES = {"good", "normal", "ok", "pass"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lightweight PatchCore-style trainer (dependency-safe version)")
    p.add_argument("--data-root", type=str, required=True, help="Dataset root or category root")
    p.add_argument("--category", type=str, default="", help="Category name, e.g. pcb4")
    p.add_argument("--output-dir", type=str, required=True, help="Output model directory")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=16, help="Patch size on resized image")
    p.add_argument("--coreset-ratio", type=float, default=0.2,
                   help="Random subsampling ratio for memory bank patch embeddings")
    p.add_argument("--max-train-images", type=int, default=0, help="Optional cap on train images. 0=all")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def discover_train_normal_images(category_root: Path) -> List[Path]:
    """
    Safety-first discovery.

    Priority:
    1) current project format:   <category>/normal/images/*
    2) mvtec-like train format:  <category>/train/<good|normal|ok|pass>/*

    Anything under masks/, meta/, debug_roi/, ground_truth/ is excluded by construction
    because we only read the explicit image directories above.
    """
    # Project-specific layout
    preferred = list_images_strict(category_root / "normal" / "images")
    if preferred:
        return preferred

    # MVTec-style fallback
    train_dir = category_root / "train"
    if train_dir.exists():
        candidates: List[Path] = []
        for sub in sorted([p for p in train_dir.iterdir() if p.is_dir()]):
            if sub.name.lower() in NORMAL_DIR_NAMES:
                candidates.extend(list_images(sub))
        if candidates:
            return sorted(candidates)

    raise RuntimeError(
        "No valid training images found. Expected one of:\n"
        f"  - {category_root / 'normal' / 'images'}\n"
        f"  - {category_root / 'train' / '<good|normal|ok|pass>'}"
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


def subsample_memory_bank(memory_bank: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    if ratio <= 0.0:
        raise ValueError("--coreset-ratio must be > 0")
    if ratio >= 1.0 or len(memory_bank) <= 1:
        return memory_bank
    n_select = max(1, int(len(memory_bank) * ratio))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(memory_bank), size=n_select, replace=False)
    return memory_bank[np.sort(idx)]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    category_root = resolve_category_root(data_root, args.category)
    if not category_root.exists():
        raise FileNotFoundError(f"Category root not found: {category_root}")

    train_images = discover_train_normal_images(category_root)
    if args.max_train_images > 0:
        train_images = train_images[:args.max_train_images]

    print(f"[info] category_root={category_root}")
    print(f"[info] train_source={category_root / 'normal' / 'images'} (preferred)")
    print(f"[info] n_train_images={len(train_images)}")
    print(f"[info] image_size={args.image_size}, patch_size={args.patch_size}")

    all_embeddings = []
    grid_shape = None
    for idx, path in enumerate(train_images, start=1):
        img_rgb = load_rgb(path, args.image_size)
        patch_embeds, rc = extract_patch_embeddings(img_rgb, args.patch_size)
        grid_shape = rc
        all_embeddings.append(patch_embeds)
        if idx % 10 == 0 or idx == len(train_images):
            print(f"[info] processed train image {idx}/{len(train_images)}")

    memory_bank = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    memory_bank = subsample_memory_bank(memory_bank, args.coreset_ratio, args.seed)

    meta = {
        "category": args.category or category_root.name,
        "category_root": str(category_root),
        "image_size": int(args.image_size),
        "patch_size": int(args.patch_size),
        "grid_shape": list(grid_shape),
        "feature_dim": int(memory_bank.shape[1]),
        "n_train_images": int(len(train_images)),
        "n_memory_patches": int(len(memory_bank)),
        "method": "patch_memory_bank_knn_handcrafted",
        "seed": int(args.seed),
        "train_discovery": "strict: normal/images only, with mvtec-like train/good fallback",
    }

    np.savez_compressed(output_dir / "memory_bank.npz", memory_bank=memory_bank)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(output_dir / "train_images.txt", "w", encoding="utf-8") as f:
        for p in train_images:
            f.write(str(p) + "\n")

    print(f"[done] saved model artifacts to: {output_dir}")
    print(f"[done] memory_bank shape = {memory_bank.shape}")


if __name__ == "__main__":
    main()
