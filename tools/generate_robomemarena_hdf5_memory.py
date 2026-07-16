# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate MovieChat memory sidecars directly for RoboMemArena HDF5 files.

The output mirrors the raw HDF5 tree under:

    <raw_root>/memory/<memory_name>/<relative_hdf5_path>.npz

This intentionally avoids converting RoboMemArena to LeRobot first.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
from PIL import Image

from generate_memory_sidecar import (_pad_or_truncate_visual_tokens,
                                     _pool_frame_tokens,
                                     _require_torch_stack,
                                     _resize_and_normalize_images)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate MovieChat memory for RoboMemArena HDF5 files.')
    parser.add_argument(
        '--raw-root',
        type=Path,
        required=True,
        help='RoboMemArena root containing .hdf5 files.')
    parser.add_argument(
        '--memory-name',
        type=str,
        default='siglip_moviechat_v1',
        help='Sidecar name under <raw_root>/memory/.')
    parser.add_argument(
        '--output-root',
        type=Path,
        default=None,
        help='Optional memory root. Default: <raw_root>/memory/<memory-name>.')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/pi0/pi0_paligemma_libero_all_full_finetune.py'),
        help='PI0 config used to build the frozen SigLIP encoder.')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Torch device.')
    parser.add_argument(
        '--dtype',
        choices=['bf16', 'fp16', 'fp32'],
        default='bf16',
        help='Encoder dtype.')
    parser.add_argument(
        '--image-keys',
        nargs='+',
        default=['agentview_rgb', 'eye_in_hand_rgb'],
        help='RoboMemArena obs image keys.')
    parser.add_argument(
        '--max-memory-tokens',
        type=int,
        default=128,
        help='Fixed number of memory tokens saved per frame.')
    parser.add_argument(
        '--frame-memory-tokens',
        type=int,
        default=32,
        help='Pooled visual tokens per current frame.')
    parser.add_argument(
        '--short-memory-size',
        type=int,
        default=18,
        help='MovieChat short-memory length.')
    parser.add_argument(
        '--high-relevance-keep-frames',
        type=int,
        default=3,
        help='Frames kept when segment is relevant.')
    parser.add_argument(
        '--low-relevance-keep-frames',
        type=int,
        default=1,
        help='Frames kept when segment is weakly relevant.')
    parser.add_argument(
        '--relevance-threshold',
        type=float,
        default=0.25,
        help='MovieChat relevance threshold.')
    parser.add_argument(
        '--clip-relevance',
        action='store_true',
        help='Use CLIP text-image relevance.')
    parser.add_argument(
        '--clip-model',
        type=str,
        default='ViT-B/32',
        help='OpenAI CLIP model.')
    parser.add_argument(
        '--task',
        type=str,
        default=None,
        help='Override task prompt for all files.')
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Optional smoke-test limit.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing sidecar files.')
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing sidecars.')
    parser.add_argument(
        '--log-every',
        type=int,
        default=25,
        help='Print progress every N files.')
    return parser.parse_args()


def infer_task_from_path(path: Path) -> str:
    dataset_dir = next(
        (parent for parent in path.parents if parent.name.endswith('_dataset')),
        path.parent,
    )
    name = re.sub(r'^\d+_', '', dataset_dir.name)
    name = re.sub(r'_dataset$', '', name)
    return f'transfer {name.replace("_", " ")}'


class HDF5MovieChatEncoder:

    def __init__(self, args) -> None:
        stack = _require_torch_stack()
        self.torch = stack['torch']
        self.F = stack['F']
        Config = stack['Config']
        build_memory_from_cfg = stack['build_memory_from_cfg']
        build_tokenizer_from_cfg = stack['build_tokenizer_from_cfg']
        build_vla_from_cfg = stack['build_vla_from_cfg']

        self.device = self.torch.device(args.device)
        self.dtype = {
            'bf16': self.torch.bfloat16,
            'fp16': self.torch.float16,
            'fp32': self.torch.float32,
        }[args.dtype]
        self.max_memory_tokens = args.max_memory_tokens
        self.frame_memory_tokens = args.frame_memory_tokens
        self.image_keys = args.image_keys

        cfg = Config.fromfile(str(args.config))
        self.model = build_vla_from_cfg(cfg.model)
        self.model.from_pretrained()
        self.model.eval().to(device=self.device, dtype=self.dtype)
        self.tokenizer = build_tokenizer_from_cfg(
            getattr(cfg.runner, 'tokenizer', dict(type='PaligemmaTokenizer')))

        self.memory = build_memory_from_cfg(
            dict(
                type='MovieChatMemory',
                short_memory_size=args.short_memory_size,
                high_relevance_keep_frames=args.high_relevance_keep_frames,
                low_relevance_keep_frames=args.low_relevance_keep_frames,
                question_similarity_threshold=args.relevance_threshold,
                long_memory_size=None,
                carry_consolidated_to_short=False,
                adjacent_similarity='token_dot_mean',
                keep_recent_short_for_breakpoint=True,
                detach_memory=True,
            ))

        self.clip = None
        self.clip_model = None
        self.clip_preprocess = None
        if args.clip_relevance:
            try:
                import clip
            except ImportError as exc:
                raise ImportError(
                    'clip_relevance=True requires OpenAI CLIP. Install it '
                    'with `pip install git+https://github.com/openai/CLIP.git`'
                    '.') from exc
            self.clip = clip
            self.clip_model, self.clip_preprocess = clip.load(
                args.clip_model, device=str(self.device))
            self.clip_model.eval()

    def reset(self):
        self.memory.reset()

    def query_tokens(self, prompt: str):
        tokenized = self.tokenizer(prompt)
        input_ids = self.torch.tensor(
            tokenized['input_ids'], dtype=self.torch.long,
            device=self.device)[None]
        with self.torch.no_grad():
            query_tokens = self.model.llm_backbone.embed_tokens(input_ids)
            query_tokens = query_tokens * math.sqrt(query_tokens.shape[-1])
        return query_tokens

    def clip_relevance_score(self, images_hwc: List[np.ndarray],
                             prompt: str) -> Optional[float]:
        if self.clip_model is None:
            return None
        clip_images = []
        for image in images_hwc:
            pil_image = Image.fromarray(image.astype(np.uint8))
            clip_images.append(self.clip_preprocess(pil_image))
        image_tensor = self.torch.stack(clip_images).to(self.device)
        text_tensor = self.clip.tokenize([prompt]).to(self.device)
        with self.torch.no_grad():
            _, logits_per_text = self.clip_model(image_tensor, text_tensor)
            probs = logits_per_text.softmax(dim=-1)
        return float(probs.mean().detach().cpu().item())

    def visual_tokens(self, images_hwc: List[np.ndarray]):
        images_chw = [
            np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
            for image in images_hwc
        ]
        images_tensor = _resize_and_normalize_images(
            images_chw, self.torch, self.F, self.device, self.dtype)
        with self.torch.no_grad():
            tokens = self.model.vision_backbone(images_tensor)
            tokens = _pool_frame_tokens(tokens, self.frame_memory_tokens,
                                        self.F)
        return tokens[:, None]

    def encode_demo(self, demo, prompt: str) -> Dict[str, np.ndarray]:
        self.reset()
        obs = demo['obs']
        num_frames = int(demo['actions'].shape[0])
        query_tokens = self.query_tokens(prompt)
        memory_tokens = []
        memory_masks = []
        relevance = []

        for frame_idx in range(num_frames):
            images = [
                np.asarray(obs[image_key][frame_idx], dtype=np.uint8)
                for image_key in self.image_keys
            ]
            relevance_score = self.clip_relevance_score(images, prompt)
            visual_tokens = self.visual_tokens(images)
            self.memory.update(
                visual_tokens,
                query_tokens=query_tokens,
                relevance_score=relevance_score,
            )
            output = self.memory.read(mode='breakpoint')
            tokens, mask = _pad_or_truncate_visual_tokens(
                output['tokens'], self.max_memory_tokens, self.F)
            memory_tokens.append(tokens)
            memory_masks.append(mask)
            info = output.get('info') or {}
            relevance.append(float(info.get('relevance', np.nan)))

        return dict(
            memory_tokens=np.stack(memory_tokens),
            memory_masks=np.stack(memory_masks),
            relevance=np.asarray(relevance, dtype=np.float32),
            frame_indices=np.arange(num_frames, dtype=np.int64),
        )


def hdf5_files(raw_root: Path, max_files: Optional[int]) -> List[Path]:
    files = sorted(raw_root.rglob('*.hdf5'))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f'No .hdf5 files found under {raw_root}')
    return files


def output_path_for(memory_root: Path, raw_root: Path, hdf5_path: Path,
                    demo_key: str, num_demos: int) -> Path:
    rel = hdf5_path.relative_to(raw_root)
    out = memory_root / rel
    if num_demos == 1:
        return out.with_suffix('.npz')
    return out.with_name(f'{out.stem}_{demo_key}.npz')


def verify_file(npz_path: Path):
    with np.load(npz_path, allow_pickle=False) as data:
        tokens = data['memory_tokens']
        masks = data['memory_masks']
        frame_indices = data['frame_indices']
        if tokens.ndim != 3:
            raise ValueError(f'{npz_path}: memory_tokens must be rank 3.')
        if masks.shape != tokens.shape[:2]:
            raise ValueError(f'{npz_path}: mask/token shape mismatch.')
        if len(frame_indices) != tokens.shape[0]:
            raise ValueError(f'{npz_path}: frame index length mismatch.')


def main():
    args = parse_args()
    raw_root = args.raw_root.resolve()
    memory_root = (
        args.output_root.resolve() if args.output_root is not None else
        raw_root / 'memory' / args.memory_name)
    files = hdf5_files(raw_root, args.max_files)
    memory_root.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        checked = 0
        for npz_path in sorted(memory_root.rglob('*.npz')):
            verify_file(npz_path)
            checked += 1
        print(f'[OK] verified {checked} memory files under {memory_root}')
        return

    print(f'Raw root:    {raw_root}')
    print(f'Memory root: {memory_root}')
    print(f'Files:       {len(files)}')
    print(f'Memory name: {args.memory_name}')

    encoder = HDF5MovieChatEncoder(args)
    manifest = []
    total_frames = 0
    for file_idx, hdf5_path in enumerate(files, start=1):
        prompt = args.task or infer_task_from_path(hdf5_path)
        with h5py.File(hdf5_path, 'r') as h5_file:
            data = h5_file['data']
            demo_keys = sorted(data.keys())
            for demo_key in demo_keys:
                out_path = output_path_for(
                    memory_root, raw_root, hdf5_path, demo_key,
                    len(demo_keys))
                if out_path.exists() and not args.overwrite:
                    verify_file(out_path)
                    with np.load(out_path, allow_pickle=False) as existing:
                        num_frames = int(existing['memory_tokens'].shape[0])
                else:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    encoded = encoder.encode_demo(data[demo_key], prompt)
                    num_frames = int(encoded['memory_tokens'].shape[0])
                    np.savez_compressed(
                        out_path,
                        **encoded,
                        hdf5_path=str(hdf5_path),
                        demo_key=demo_key,
                        task=prompt,
                        image_keys=np.asarray(args.image_keys),
                        encoder='pi0-siglip-moviechat',
                    )
                manifest.append(
                    dict(
                        hdf5_path=str(hdf5_path.relative_to(raw_root)),
                        memory_path=str(out_path.relative_to(memory_root)),
                        demo_key=demo_key,
                        task=prompt,
                        num_frames=num_frames,
                    ))
                total_frames += num_frames

        if file_idx % args.log_every == 0 or file_idx == len(files):
            print(f'[{file_idx}/{len(files)}] frames={total_frames}')

    with open(memory_root / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(
            dict(
                raw_root=str(raw_root),
                memory_name=args.memory_name,
                encoder='pi0-siglip-moviechat',
                max_memory_tokens=args.max_memory_tokens,
                frame_memory_tokens=args.frame_memory_tokens,
                image_keys=args.image_keys,
                num_files=len(files),
                num_entries=len(manifest),
                total_frames=total_frames,
                entries=manifest,
            ),
            f,
            indent=2,
        )

    print(f'[OK] wrote {len(manifest)} entries, {total_frames} frames to '
          f'{memory_root}')


if __name__ == '__main__':
    main()
