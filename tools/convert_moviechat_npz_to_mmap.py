#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def convert_one(src_path: Path, src_root: Path, dst_root: Path, overwrite: bool) -> bool:
    rel_parent = src_path.parent.relative_to(src_root)
    dst_dir = dst_root / rel_parent / src_path.stem
    tokens_path = dst_dir / "memory_tokens.npy"
    masks_path = dst_dir / "memory_masks.npy"
    frame_indices_path = dst_dir / "frame_indices.npy"

    if (
        not overwrite
        and tokens_path.exists()
        and masks_path.exists()
        and frame_indices_path.exists()
    ):
        return False

    dst_dir.mkdir(parents=True, exist_ok=True)
    with np.load(src_path, allow_pickle=False) as data:
        np.save(tokens_path, data["memory_tokens"])
        np.save(masks_path, data["memory_masks"])
        np.save(frame_indices_path, data["frame_indices"])
        if "relevance" in data:
            np.save(dst_dir / "relevance.npy", data["relevance"])
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MovieChat per-episode .npz sidecars into mmap-friendly .npy directories."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme/robomme_data_h5/memory/siglip_moviechat_v1"),
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path("/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme/robomme_data_h5/memory/siglip_moviechat_v1_mmap"),
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_files = sorted(args.src_root.glob("*/*.npz"))
    if args.max_files is not None:
        src_files = src_files[: args.max_files]
    if not src_files:
        raise FileNotFoundError(f"No .npz sidecars found under {args.src_root}")

    converted = 0
    for i, src_path in enumerate(src_files, start=1):
        if convert_one(src_path, args.src_root, args.dst_root, args.overwrite):
            converted += 1
        if i == 1 or i % 25 == 0 or i == len(src_files):
            print(f"[convert] {i}/{len(src_files)} scanned, {converted} converted")

    print(f"[done] scanned={len(src_files)} converted={converted} dst={args.dst_root}")


if __name__ == "__main__":
    main()
