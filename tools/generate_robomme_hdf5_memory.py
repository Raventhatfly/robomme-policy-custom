#!/usr/bin/env python
# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Generate causal MovieChat sidecars for official RoboMME HDF5 files."""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

from generate_robomemarena_hdf5_memory import (
    HDF5MovieChatEncoder,
    verify_file,
)
from generate_memory_sidecar import _pad_or_truncate_visual_tokens


def _atomic_savez_compressed(output: Path, **arrays):
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(
        f'.{output.name}.tmp.{os.getpid()}')
    try:
        with tmp_output.open('wb') as file:
            np.savez_compressed(file, **arrays)
        verify_file(tmp_output)
        if output.exists():
            verify_file(output)
            tmp_output.unlink()
            return False
        tmp_output.rename(output)
        verify_file(output)
        return True
    except Exception:
        tmp_output.unlink(missing_ok=True)
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--memory-name', default='siglip_moviechat_v1')
    parser.add_argument('--output-root', type=Path, default=None)
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(
            'configs/pi0/pi0_paligemma_libero_all_full_finetune.py'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    parser.add_argument('--max-memory-tokens', type=int, default=128)
    parser.add_argument('--frame-memory-tokens', type=int, default=32)
    parser.add_argument('--short-memory-size', type=int, default=18)
    parser.add_argument('--high-relevance-keep-frames', type=int, default=3)
    parser.add_argument('--low-relevance-keep-frames', type=int, default=1)
    parser.add_argument('--relevance-threshold', type=float, default=0.25)
    parser.add_argument('--clip-relevance', action='store_true')
    parser.add_argument('--clip-model', default='ViT-B/32')
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--max-episodes-per-file', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--max-verify-files', type=int, default=None)
    return parser.parse_args()


def _decode_goal(episode):
    value = episode['setup']['task_goal'][()].reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return str(value).lower()


def _step_ids(episode):
    return sorted(
        int(key.rsplit('_', 1)[1])
        for key in episode.keys()
        if key.startswith('timestep_'))


def _encode_episode(encoder, episode, prompt):
    encoder.reset()
    query_tokens = encoder.query_tokens(prompt)
    tokens_out = []
    masks_out = []
    relevance_out = []
    frame_indices = _step_ids(episode)
    for step_id in frame_indices:
        obs = episode[f'timestep_{step_id}']['obs']
        images = [
            np.asarray(obs['front_rgb'][()], dtype=np.uint8),
            np.asarray(obs['wrist_rgb'][()], dtype=np.uint8),
        ]
        relevance_score = encoder.clip_relevance_score(images, prompt)
        visual_tokens = encoder.visual_tokens(images)
        encoder.memory.update(
            visual_tokens,
            query_tokens=query_tokens,
            relevance_score=relevance_score,
        )
        output = encoder.memory.read(mode='breakpoint')
        tokens, masks = _pad_or_truncate_visual_tokens(
            output['tokens'], encoder.max_memory_tokens, encoder.F)
        tokens_out.append(tokens)
        masks_out.append(masks)
        info = output.get('info') or {}
        relevance_out.append(float(info.get('relevance', np.nan)))
    return {
        'memory_tokens': np.stack(tokens_out),
        'memory_masks': np.stack(masks_out),
        'relevance': np.asarray(relevance_out, dtype=np.float32),
        'frame_indices': np.asarray(frame_indices, dtype=np.int64),
    }


def main():
    args = parse_args()
    args.image_keys = ['front_rgb', 'wrist_rgb']
    args.task = None
    output_root = (
        args.output_root or args.raw_root / 'memory' / args.memory_name)

    if args.verify_only:
        paths = sorted(output_root.rglob('episode_*.npz'))
        if args.max_verify_files is not None:
            paths = paths[:args.max_verify_files]
        for path in paths:
            verify_file(path)
        print(f'[OK] verified {len(paths)} sidecars under {output_root}')
        return

    files = sorted(args.raw_root.rglob('*.h5'))
    if args.max_files is not None:
        files = files[:args.max_files]
    if not files:
        raise FileNotFoundError(f'No .h5 files found under {args.raw_root}')

    encoder = HDF5MovieChatEncoder(args)
    episodes_done = 0
    for file_index, path in enumerate(files, start=1):
        relative = path.relative_to(args.raw_root).with_suffix('')
        with h5py.File(path, 'r') as h5_file:
            episode_ids = sorted(
                int(key.split('_', 1)[1])
                for key in h5_file.keys()
                if key.startswith('episode_'))
            if args.max_episodes_per_file is not None:
                episode_ids = episode_ids[:args.max_episodes_per_file]
            for episode_id in episode_ids:
                output = output_root / relative / f'episode_{episode_id}.npz'
                if output.exists() and not args.overwrite:
                    verify_file(output)
                    continue
                episode = h5_file[f'episode_{episode_id}']
                encoded = _encode_episode(
                    encoder, episode, _decode_goal(episode))
                if _atomic_savez_compressed(output, **encoded):
                    episodes_done += 1
        print(f'[{file_index}/{len(files)}] {path.name}')
    print(f'[OK] wrote {episodes_done} RoboMME memory sidecars to {output_root}')


if __name__ == '__main__':
    main()
