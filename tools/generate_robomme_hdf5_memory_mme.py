#!/usr/bin/env python
"""Generate RoboMME MovieChat sidecars with the same code path used by MME eval.

This intentionally does not import FluxVLA. It uses:
  - mme_vla_suite.shared.fluxvla_siglip_encoder.FluxVLASigLIPEncoder
  - mme_vla_suite.shared.mem_buffer.MemoryBufferMovieChat

The output layout matches the existing training loader:
  <raw-root>/memory/<memory-name>/<h5-stem>/episode_<id>.npz
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mme_vla_suite.shared.fluxvla_siglip_encoder import FluxVLASigLIPEncoder
from mme_vla_suite.shared.mem_buffer import MemoryBufferMovieChat


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate RoboMME MovieChat memory sidecars using the MME eval implementation."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme/robomme_data_h5"),
        help="RoboMME HDF5 root containing record_dataset_*.h5 files and fluxvla_hdf5_index.json.",
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path("/n/netscratch/hankyang_lab/Lab/felix/dataset/robomme/robomme_preprocessed_data"),
        help="Fallback RoboMME preprocessed root containing data/*.pkl.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "h5", "pkl"],
        default="auto",
        help="Input source. auto uses h5 if present, otherwise preprocessed pkl.",
    )
    parser.add_argument(
        "--memory-name",
        default="siglip_moviechat_mme_v1",
        help="Sidecar name under <raw-root>/memory/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional explicit output root. Default: <raw-root>/memory/<memory-name>.",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=Path("/n/netscratch/hankyang_lab/Lab/felix/ckpts/fluxvla/pi0_base/model.safetensors"),
        help="FluxVLA pi0/pi0.5 SigLIP checkpoint used by eval.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-memory-tokens", type=int, default=128)
    parser.add_argument("--frame-memory-tokens", type=int, default=32)
    parser.add_argument("--short-memory-size", type=int, default=18)
    parser.add_argument("--short-memory-merge", type=int, default=2)
    parser.add_argument("--high-relevance-keep-frames", type=int, default=3)
    parser.add_argument("--low-relevance-keep-frames", type=int, default=1)
    parser.add_argument("--relevance-threshold", type=float, default=0.25)
    parser.add_argument("--long-memory-size", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-episodes-per-file", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--max-verify-files", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def _decode_goal(episode) -> str:
    value = episode["setup"]["task_goal"][()].reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).lower()


def _step_ids(episode) -> list[int]:
    return sorted(
        int(key.rsplit("_", 1)[1])
        for key in episode.keys()
        if key.startswith("timestep_")
    )


def _sidecar_output_from_index(output_root: Path, episode_index: list[dict], epis_idx: int) -> Path:
    episode = episode_index[epis_idx]
    raw_stem = Path(episode["path"]).with_suffix("").name
    episode_id = int(episode["episode_id"])
    return output_root / raw_stem / f"episode_{episode_id}.npz"


def _verify_npz(path: Path) -> None:
    with np.load(path, allow_pickle=False) as data:
        required = {"memory_tokens", "memory_masks", "relevance", "frame_indices"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        tokens = data["memory_tokens"]
        masks = data["memory_masks"]
        frame_indices = data["frame_indices"]
        if tokens.ndim != 3:
            raise ValueError(f"{path}: memory_tokens must be rank 3, got {tokens.shape}")
        if masks.shape != tokens.shape[:2]:
            raise ValueError(f"{path}: mask shape {masks.shape} does not match tokens {tokens.shape}")
        if frame_indices.shape[0] != tokens.shape[0]:
            raise ValueError(f"{path}: frame_indices length does not match tokens")


def _atomic_savez_compressed(output: Path, **arrays) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with tmp_output.open("wb") as file:
            np.savez_compressed(file, **arrays)
        _verify_npz(tmp_output)
        if output.exists():
            _verify_npz(output)
            tmp_output.unlink()
            return False
        tmp_output.rename(output)
        _verify_npz(output)
        return True
    except Exception:
        tmp_output.unlink(missing_ok=True)
        raise


class MMEMovieChatSidecarEncoder:
    def __init__(self, args) -> None:
        self.max_memory_tokens = args.max_memory_tokens
        self.vision_encoder = FluxVLASigLIPEncoder(
            args.encoder_checkpoint,
            dtype=args.dtype,
            frame_memory_tokens=args.frame_memory_tokens,
        )
        self.mem_buffer = MemoryBufferMovieChat(
            num_views=1,
            img_emb_dim=1152,
            pos_emb_dim=768,
            state_emb_dim=8,
            short_memory_size=args.short_memory_size,
            short_memory_merge=args.short_memory_merge,
            long_memory_size=None if args.long_memory_size == 0 else args.long_memory_size,
            high_relevance_keep_frames=args.high_relevance_keep_frames,
            low_relevance_keep_frames=args.low_relevance_keep_frames,
            relevance_threshold=args.relevance_threshold,
            frame_memory_tokens=args.frame_memory_tokens,
            prepare_buffer=True,
            vision_enc_fn=self.vision_encoder,
        )

    def encode_episode(self, episode) -> dict[str, np.ndarray]:
        self.mem_buffer.clear()
        tokens_out = []
        masks_out = []
        frame_indices = _step_ids(episode)

        for step_id in frame_indices:
            obs = episode[f"timestep_{step_id}"]["obs"]
            images = np.stack(
                [
                    np.asarray(obs["front_rgb"][()], dtype=np.uint8),
                    np.asarray(obs["wrist_rgb"][()], dtype=np.uint8),
                ],
                axis=0,
            )[None]
            states = np.zeros((1, 8), dtype=np.float32)
            self.mem_buffer.add_buffer(images, states, [step_id])
            image_emb, _, _, mask = self.mem_buffer.prepare_moviechat_memory(
                step_id,
                self.max_memory_tokens,
                self.vision_encoder.frame_memory_tokens,
                self.mem_buffer.default_history_feats_gather_fn,
            )
            tokens_out.append(image_emb.astype(np.float16))
            masks_out.append(mask.astype(np.bool_))

        return {
            "memory_tokens": np.stack(tokens_out),
            "memory_masks": np.stack(masks_out),
            "relevance": np.full((len(frame_indices),), np.nan, dtype=np.float32),
            "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        }

    def encode_preprocessed_episode(self, step_paths: list[tuple[int, Path]]) -> dict[str, np.ndarray]:
        self.mem_buffer.clear()
        tokens_out = []
        masks_out = []
        frame_indices = []

        for step_id, path in sorted(step_paths, key=lambda item: item[0]):
            with path.open("rb") as file:
                item = pickle.load(file)
            images = np.stack(
                [
                    np.asarray(item["image"], dtype=np.uint8),
                    np.asarray(item["wrist_image"], dtype=np.uint8),
                ],
                axis=0,
            )[None]
            state = np.asarray(item.get("state", np.zeros((8,), dtype=np.float32)), dtype=np.float32)
            self.mem_buffer.add_buffer(images, state[None], [step_id])
            image_emb, _, _, mask = self.mem_buffer.prepare_moviechat_memory(
                step_id,
                self.max_memory_tokens,
                self.vision_encoder.frame_memory_tokens,
                self.mem_buffer.default_history_feats_gather_fn,
            )
            tokens_out.append(image_emb.astype(np.float16))
            masks_out.append(mask.astype(np.bool_))
            frame_indices.append(step_id)

        return {
            "memory_tokens": np.stack(tokens_out),
            "memory_masks": np.stack(masks_out),
            "relevance": np.full((len(frame_indices),), np.nan, dtype=np.float32),
            "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        }


def _generate_from_h5(args, output_root: Path, encoder: MMEMovieChatSidecarEncoder) -> tuple[int, int]:
    files = sorted(args.raw_root.rglob("*.h5"))
    if args.max_files is not None:
        files = files[:args.max_files]
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {args.raw_root}")

    episodes_written = 0
    episodes_seen = 0

    for file_index, path in enumerate(files, start=1):
        relative = path.relative_to(args.raw_root).with_suffix("")
        with h5py.File(path, "r") as h5_file:
            episode_ids = sorted(
                int(key.split("_", 1)[1])
                for key in h5_file.keys()
                if key.startswith("episode_")
            )
            if args.max_episodes_per_file is not None:
                episode_ids = episode_ids[:args.max_episodes_per_file]

            for episode_id in episode_ids:
                if args.max_episodes is not None and episodes_seen >= args.max_episodes:
                    return episodes_seen, episodes_written
                episodes_seen += 1
                output = output_root / relative / f"episode_{episode_id}.npz"
                if output.exists() and not args.overwrite:
                    _verify_npz(output)
                    continue
                episode = h5_file[f"episode_{episode_id}"]
                _ = _decode_goal(episode)
                encoded = encoder.encode_episode(episode)
                if _atomic_savez_compressed(output, **encoded):
                    episodes_written += 1

        if args.log_every > 0 and (file_index % args.log_every == 0 or file_index == len(files)):
            print(
                f"[h5 {file_index}/{len(files)}] {path.name} "
                f"seen={episodes_seen} wrote={episodes_written}",
                flush=True,
            )

    return episodes_seen, episodes_written


def _read_episode_index(raw_root: Path) -> list[dict]:
    index_path = raw_root / "fluxvla_hdf5_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing episode index: {index_path}")
    with index_path.open("r") as file:
        return json.load(file)["episodes"]


def _generate_from_preprocessed_pkl(
    args,
    output_root: Path,
    encoder: MMEMovieChatSidecarEncoder,
) -> tuple[int, int]:
    data_root = args.preprocessed_root / "data"
    paths = sorted(data_root.glob("*.pkl"), key=lambda p: int(p.stem))
    if not paths:
        raise FileNotFoundError(f"No .pkl files found under {data_root}")

    episode_index = _read_episode_index(args.raw_root)
    episodes_written = 0
    episodes_seen = 0

    def flush_episode(epis_idx: int | None, step_paths: list[tuple[int, Path]]) -> bool:
        nonlocal episodes_seen, episodes_written
        if epis_idx is None or not step_paths:
            return False
        if args.max_episodes is not None and episodes_seen >= args.max_episodes:
            return True
        episodes_seen += 1
        output = _sidecar_output_from_index(output_root, episode_index, epis_idx)
        if output.exists() and not args.overwrite:
            _verify_npz(output)
        else:
            encoded = encoder.encode_preprocessed_episode(step_paths)
            if _atomic_savez_compressed(output, **encoded):
                episodes_written += 1
        if args.log_every > 0 and (episodes_seen % args.log_every == 0 or episodes_seen == 1):
            print(
                f"[pkl {episodes_seen}] epis_idx={epis_idx} "
                f"steps={len(step_paths)} wrote={episodes_written}",
                flush=True,
            )
        return args.max_episodes is not None and episodes_seen >= args.max_episodes

    current_epis_idx = None
    current_steps: list[tuple[int, Path]] = []
    for i, path in enumerate(paths, start=1):
        with path.open("rb") as file:
            item = pickle.load(file)
        epis_idx = int(np.asarray(item["epis_idx"]).reshape(-1)[0])
        step_idx = int(np.asarray(item["step_idx"]).reshape(-1)[0])

        if current_epis_idx is None:
            current_epis_idx = epis_idx
        if epis_idx != current_epis_idx:
            if flush_episode(current_epis_idx, current_steps):
                return episodes_seen, episodes_written
            current_epis_idx = epis_idx
            current_steps = []

        current_steps.append((step_idx, path))
        if args.log_every > 0 and i % 10000 == 0:
            print(
                f"[pkl-scan] scanned {i}/{len(paths)} files, "
                f"completed_episodes={episodes_seen}, current_epis_idx={current_epis_idx}",
                flush=True,
            )

    flush_episode(current_epis_idx, current_steps)

    return episodes_seen, episodes_written


def main() -> None:
    args = parse_args()
    output_root = args.output_root or args.raw_root / "memory" / args.memory_name

    if args.verify_only:
        paths = sorted(output_root.rglob("episode_*.npz"))
        if args.max_verify_files is not None:
            paths = paths[:args.max_verify_files]
        for path in paths:
            _verify_npz(path)
        print(f"[OK] verified {len(paths)} sidecars under {output_root}")
        return

    encoder = MMEMovieChatSidecarEncoder(args)
    h5_files = sorted(args.raw_root.rglob("*.h5"))
    if args.source == "h5" or (args.source == "auto" and h5_files):
        source = "h5"
        episodes_seen, episodes_written = _generate_from_h5(args, output_root, encoder)
    else:
        source = "pkl"
        episodes_seen, episodes_written = _generate_from_preprocessed_pkl(args, output_root, encoder)

    print(
        f"[OK] source={source} seen={episodes_seen} wrote={episodes_written} "
        f"RoboMME MME MovieChat sidecars to {output_root}"
    )


if __name__ == "__main__":
    main()
