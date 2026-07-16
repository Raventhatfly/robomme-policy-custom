"""Generate causal memory sidecar files for LeRobot-style parquet datasets.

This script keeps the original dataset files untouched. It writes per-episode
memory arrays under:

    <dataset_root>/memory/<memory_name>/episode_XXXXXX.npz

The first implementation provides a lightweight ``raw-state`` encoder so the
sidecar data flow can be generated and validated before plugging in a heavier
vision-token encoder.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            'pyarrow is required to read parquet files. Install it in the '
            'FluxVLA environment, e.g. `pip install pyarrow`.') from exc
    return pq


def _require_torch_stack():
    try:
        import torch
        import torch.nn.functional as F
        import torchvision
        from mmengine import Config

        original_get_device_capability = torch.cuda.get_device_capability
        if not torch.cuda.is_available():
            torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 0)
        try:
            import fluxvla  # noqa: F401
            from fluxvla.engines import (build_memory_from_cfg,
                                         build_tokenizer_from_cfg,
                                         build_vla_from_cfg)
            from fluxvla.transforms.transform_inputs import \
                ProcessParquetInputs
        finally:
            torch.cuda.get_device_capability = original_get_device_capability
    except ImportError as exc:
        raise ImportError(
            'The pi0-siglip-moviechat encoder requires torch, torchvision, '
            'mmengine, and the FluxVLA package installed in the active '
            'environment.') from exc
    return {
        'torch': torch,
        'F': F,
        'torchvision': torchvision,
        'Config': Config,
        'build_memory_from_cfg': build_memory_from_cfg,
        'build_tokenizer_from_cfg': build_tokenizer_from_cfg,
        'build_vla_from_cfg': build_vla_from_cfg,
        'ProcessParquetInputs': ProcessParquetInputs,
    }


def _episode_id_from_path(path: Path) -> int:
    stem = path.stem
    if stem.startswith('episode_'):
        return int(stem.split('_')[-1])
    raise ValueError(f'Cannot infer episode id from parquet file: {path}')


def _iter_episode_parquets(dataset_root: Path) -> Iterable[Path]:
    data_root = dataset_root / 'data'
    if not data_root.exists():
        raise FileNotFoundError(f'Missing data directory: {data_root}')
    yield from sorted(data_root.glob('**/episode_*.parquet'))


def _read_jsonl(path: Path) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def _read_dataset_meta(dataset_root: Path) -> Tuple[Dict, List[Dict]]:
    meta_root = dataset_root / 'meta'
    with open(meta_root / 'info.json', 'r', encoding='utf-8') as f:
        info = json.load(f)
    tasks = _read_jsonl(meta_root / 'tasks.jsonl')
    return info, tasks


def _resize_vector(vector: np.ndarray, target_dim: int) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size == target_dim:
        return vector
    if vector.size > target_dim:
        return vector[:target_dim]
    out = np.zeros((target_dim, ), dtype=np.float32)
    out[:vector.size] = vector
    return out


def _raw_state_frame_tokens(row: Dict, memory_dim: int) -> np.ndarray:
    """Create one deterministic token from proprio/state fields."""
    if 'observation.state' in row:
        state = row['observation.state']
    elif 'states' in row:
        state = row['states']
    else:
        raise KeyError('Expected `observation.state` in parquet row.')
    state = np.asarray(state, dtype=np.float32)
    return _resize_vector(state, memory_dim)[None]


def _pad_or_truncate_tokens(tokens: np.ndarray,
                            max_memory_tokens: int,
                            memory_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    tokens = np.asarray(tokens, dtype=np.float32).reshape(-1, memory_dim)
    out = np.zeros((max_memory_tokens, memory_dim), dtype=np.float16)
    mask = np.zeros((max_memory_tokens, ), dtype=np.bool_)
    keep = min(tokens.shape[0], max_memory_tokens)
    if keep > 0:
        out[:keep] = tokens[-keep:].astype(np.float16)
        mask[:keep] = True
    return out, mask


def _causal_raw_state_memory(rows: List[Dict], max_memory_tokens: int,
                             memory_dim: int,
                             stride: int) -> Tuple[np.ndarray, np.ndarray]:
    memory_tokens = []
    memory_masks = []
    history = []

    for frame_idx, row in enumerate(rows):
        token = _raw_state_frame_tokens(row, memory_dim)
        history.append(token)

        # Keep a causal, downsampled history ending at the current frame.
        sampled = history[::stride]
        if sampled[-1] is not history[-1]:
            sampled.append(history[-1])
        tokens = np.concatenate(sampled, axis=0)
        padded, mask = _pad_or_truncate_tokens(tokens, max_memory_tokens,
                                               memory_dim)
        memory_tokens.append(padded)
        memory_masks.append(mask)

    return np.stack(memory_tokens), np.stack(memory_masks)


def _read_episode_rows(parquet_path: Path) -> List[Dict]:
    pq = _require_pyarrow()
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    rows.sort(key=lambda row: row.get('frame_index', row.get('index', 0)))
    return rows


def _frame_indices(rows: List[Dict]) -> np.ndarray:
    if not rows:
        return np.zeros((0, ), dtype=np.int64)
    if 'frame_index' in rows[0]:
        return np.asarray([row['frame_index'] for row in rows],
                          dtype=np.int64)
    return np.arange(len(rows), dtype=np.int64)


def _prompt_from_row(row: Dict, tasks: List[Dict]) -> str:
    task_index = int(row['task_index'])
    return tasks[task_index]['task']


def _video_path(dataset_root: Path, info: Dict, row: Dict,
                video_key: str) -> Path:
    episode_index = int(row['episode_index'])
    episode_chunk = episode_index // int(info['chunks_size'])
    return dataset_root / info['video_path'].format(
        episode_chunk=episode_chunk,
        video_key=video_key,
        episode_index=episode_index,
    )


def _decode_video_frame(decoder, video_path: Path, timestamp: float):
    return decoder.decode_video_frames_torchvision(
        video_path=video_path,
        timestamps=[float(timestamp)],
        tolerance_s=0.1,
    )[0].numpy()


def _resize_and_normalize_images(images: List[np.ndarray], torch, F, device,
                                 dtype):
    tensor_images = []
    means = torch.tensor(
        [[123.515625, 116.04492188, 103.59375]],
        dtype=torch.float32,
        device=device)[:, :, None, None]
    stds = torch.tensor(
        [[58.27148438, 57.02636719, 57.27539062]],
        dtype=torch.float32,
        device=device)[:, :, None, None]
    for image in images:
        image_tensor = torch.from_numpy(image).to(device=device)
        image_tensor = image_tensor.to(dtype=torch.float32)
        image_tensor = F.interpolate(
            image_tensor[None], size=(224, 224), mode='bilinear',
            align_corners=False)[0]
        image_tensor = (image_tensor - means[0]) / stds[0]
        tensor_images.append(image_tensor)
    return torch.cat(tensor_images, dim=0).to(dtype=dtype)[None]


def _pool_frame_tokens(tokens, frame_memory_tokens: int, F):
    if tokens.size(1) <= frame_memory_tokens:
        return tokens
    tokens = tokens.transpose(1, 2)
    tokens = F.adaptive_avg_pool1d(tokens, frame_memory_tokens)
    return tokens.transpose(1, 2)


def _pad_or_truncate_visual_tokens(tokens, max_memory_tokens: int, F):
    tokens = tokens.detach().float().reshape(-1, tokens.shape[-1])
    memory_dim = int(tokens.shape[-1])
    if tokens.size(0) > max_memory_tokens:
        tokens = tokens.transpose(0, 1)[None]
        tokens = F.adaptive_avg_pool1d(tokens, max_memory_tokens)
        tokens = tokens[0].transpose(0, 1)
    tokens = tokens.cpu().numpy()
    out = np.zeros((max_memory_tokens, memory_dim), dtype=np.float16)
    mask = np.zeros((max_memory_tokens, ), dtype=np.bool_)
    keep = min(tokens.shape[0], max_memory_tokens)
    if keep > 0:
        out[:keep] = tokens[-keep:].astype(np.float16)
        mask[:keep] = True
    return out, mask


class PI0SigLIPMovieChatEncoder:
    """PI0 SigLIP native visual tokens plus MovieChat-style causal memory."""

    def __init__(self, config_path: Path, device: str, dtype: str,
                 max_memory_tokens: int, frame_memory_tokens: int,
                 short_memory_size: int, high_relevance_keep_frames: int,
                 low_relevance_keep_frames: int,
                 relevance_threshold: float,
                 video_keys: List[str],
                 clip_relevance: bool,
                 clip_model: str) -> None:
        stack = _require_torch_stack()
        self.torch = stack['torch']
        self.F = stack['F']
        Config = stack['Config']
        build_memory_from_cfg = stack['build_memory_from_cfg']
        build_tokenizer_from_cfg = stack['build_tokenizer_from_cfg']
        build_vla_from_cfg = stack['build_vla_from_cfg']
        ProcessParquetInputs = stack['ProcessParquetInputs']

        self.device = self.torch.device(device)
        self.dtype = {
            'bf16': self.torch.bfloat16,
            'fp16': self.torch.float16,
            'fp32': self.torch.float32,
        }[dtype]
        self.max_memory_tokens = max_memory_tokens
        self.frame_memory_tokens = frame_memory_tokens
        self.video_keys = video_keys
        self.decoder = ProcessParquetInputs(
            parquet_keys=[],
            video_keys=[],
        )

        cfg = Config.fromfile(str(config_path))
        self.model = build_vla_from_cfg(cfg.model)
        self.model.from_pretrained()
        self.model.eval().to(device=self.device, dtype=self.dtype)
        self.tokenizer = build_tokenizer_from_cfg(
            getattr(cfg.runner, 'tokenizer', dict(type='PaligemmaTokenizer')))

        self.memory_cfg = dict(
            type='MovieChatMemory',
            short_memory_size=short_memory_size,
            high_relevance_keep_frames=high_relevance_keep_frames,
            low_relevance_keep_frames=low_relevance_keep_frames,
            question_similarity_threshold=relevance_threshold,
            long_memory_size=None,
            carry_consolidated_to_short=False,
            adjacent_similarity='token_dot_mean',
            keep_recent_short_for_breakpoint=True,
            detach_memory=True,
        )
        self.memory = build_memory_from_cfg(self.memory_cfg)
        self.clip = None
        self.clip_preprocess = None
        if clip_relevance:
            try:
                import clip
            except ImportError as exc:
                raise ImportError(
                    'clip_relevance=True requires OpenAI CLIP. Install it '
                    'with `pip install git+https://github.com/openai/CLIP.git`'
                    '.') from exc
            self.clip = clip
            self.clip_model, self.clip_preprocess = clip.load(
                clip_model, device=str(self.device))
            self.clip_model.eval()

    def reset(self):
        self.memory.reset()

    def _query_tokens(self, prompt: str):
        tokenized = self.tokenizer(prompt)
        input_ids = self.torch.tensor(
            tokenized['input_ids'], dtype=self.torch.long,
            device=self.device)[None]
        with self.torch.no_grad():
            query_tokens = self.model.llm_backbone.embed_tokens(input_ids)
            query_tokens = query_tokens * math.sqrt(query_tokens.shape[-1])
        return query_tokens

    def _visual_tokens(self, dataset_root: Path, info: Dict, row: Dict):
        images = []
        for video_key in self.video_keys:
            path = _video_path(dataset_root, info, row, video_key)
            images.append(_decode_video_frame(self.decoder, path,
                                              row['timestamp']))
        images = _resize_and_normalize_images(
            images, self.torch, self.F, self.device, self.dtype)
        with self.torch.no_grad():
            tokens = self.model.vision_backbone(images)
            tokens = _pool_frame_tokens(tokens, self.frame_memory_tokens,
                                        self.F)
        return tokens[:, None]

    def _clip_relevance_score(self, images: List[np.ndarray],
                              prompt: str) -> Optional[float]:
        if self.clip is None:
            return None
        clip_images = []
        for image in images:
            image_hwc = image.transpose(1, 2, 0)
            pil_image = Image.fromarray(image_hwc.astype(np.uint8))
            clip_images.append(self.clip_preprocess(pil_image))
        image_tensor = self.torch.stack(clip_images).to(self.device)
        text_tensor = self.clip.tokenize([prompt]).to(self.device)
        with self.torch.no_grad():
            _, logits_per_text = self.clip_model(image_tensor, text_tensor)
            probs = logits_per_text.softmax(dim=-1)
        return float(probs.mean().detach().cpu().item())

    def _frame_inputs(self, dataset_root: Path, info: Dict, row: Dict):
        images = []
        for video_key in self.video_keys:
            path = _video_path(dataset_root, info, row, video_key)
            images.append(_decode_video_frame(self.decoder, path,
                                              row['timestamp']))
        return images

    def encode_episode(self, dataset_root: Path, info: Dict, tasks: List[Dict],
                       rows: List[Dict]):
        self.reset()
        memory_tokens = []
        memory_masks = []
        relevance = []

        current_task = None
        query_tokens = None
        for row in rows:
            prompt = _prompt_from_row(row, tasks)
            if prompt != current_task:
                current_task = prompt
                query_tokens = self._query_tokens(prompt)

            images = self._frame_inputs(dataset_root, info, row)
            relevance_score = self._clip_relevance_score(images, prompt)
            images_tensor = _resize_and_normalize_images(
                images, self.torch, self.F, self.device, self.dtype)
            with self.torch.no_grad():
                visual_tokens = self.model.vision_backbone(images_tensor)
                visual_tokens = _pool_frame_tokens(
                    visual_tokens, self.frame_memory_tokens, self.F)
            visual_tokens = visual_tokens[:, None]
            self.memory.update(
                visual_tokens,
                query_tokens=query_tokens,
                relevance_score=relevance_score)
            output = self.memory.read(mode='breakpoint')
            tokens, mask = _pad_or_truncate_visual_tokens(
                output['tokens'], self.max_memory_tokens, self.F)
            memory_tokens.append(tokens)
            memory_masks.append(mask)
            info_dict = output.get('info') or {}
            relevance.append(float(info_dict.get('relevance', np.nan)))

        return (np.stack(memory_tokens), np.stack(memory_masks),
                np.asarray(relevance, dtype=np.float32))


def generate_dataset_sidecar(dataset_root: Path, memory_name: str,
                             max_memory_tokens: int, memory_dim: int,
                             stride: int, overwrite: bool,
                             encoder_name: str,
                             encoder: Optional[PI0SigLIPMovieChatEncoder]
                             = None,
                             log_every: int = 10) -> Dict:
    sidecar_root = dataset_root / 'memory' / memory_name
    sidecar_root.mkdir(parents=True, exist_ok=True)

    parquet_files = list(_iter_episode_parquets(dataset_root))
    if not parquet_files:
        raise FileNotFoundError(f'No episode parquet files found in '
                                f'{dataset_root / "data"}')

    dataset_info, tasks = _read_dataset_meta(dataset_root)
    episodes = []
    total_frames = 0
    saved_memory_dim = memory_dim
    for parquet_path in parquet_files:
        episode_id = _episode_id_from_path(parquet_path)
        out_path = sidecar_root / f'episode_{episode_id:06d}.npz'
        if out_path.exists() and not overwrite:
            rows = _read_episode_rows(parquet_path)
            frame_indices = _frame_indices(rows)
            episodes.append({
                'episode_index': episode_id,
                'num_frames': int(len(rows)),
                'file': out_path.name,
            })
            total_frames += len(rows)
            if len(frame_indices) > 0:
                continue

        rows = _read_episode_rows(parquet_path)
        if encoder_name == 'raw-state':
            memory_tokens, memory_masks = _causal_raw_state_memory(
                rows, max_memory_tokens, memory_dim, stride)
            relevance = np.full((len(rows), ), np.nan, dtype=np.float32)
            saved_memory_dim = memory_dim
        elif encoder_name == 'pi0-siglip-moviechat':
            assert encoder is not None
            memory_tokens, memory_masks, relevance = encoder.encode_episode(
                dataset_root, dataset_info, tasks, rows)
            saved_memory_dim = int(memory_tokens.shape[-1])
        else:
            raise ValueError(f'Unsupported encoder: {encoder_name}')
        frame_indices = _frame_indices(rows)
        np.savez_compressed(
            out_path,
            memory_tokens=memory_tokens,
            memory_masks=memory_masks,
            relevance=relevance,
            frame_index=frame_indices,
            episode_index=np.asarray(episode_id, dtype=np.int64),
        )
        episodes.append({
            'episode_index': episode_id,
            'num_frames': int(len(rows)),
            'file': out_path.name,
        })
        total_frames += len(rows)
        if log_every > 0 and len(episodes) % log_every == 0:
            print(f'  processed {len(episodes)}/{len(parquet_files)} '
                  f'episodes from {dataset_root.name}')

    info = {
        'format': 'fluxvla_memory_sidecar_v1',
        'encoder': encoder_name,
        'memory_name': memory_name,
        'dataset_root': str(dataset_root),
        'num_episodes': len(episodes),
        'num_frames': total_frames,
        'max_memory_tokens': max_memory_tokens,
        'memory_dim': saved_memory_dim if total_frames > 0 else memory_dim,
        'stride': stride,
        'dtype': 'float16',
        'causal': True,
        'episodes': episodes,
    }
    with open(sidecar_root / 'memory_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    return info


def verify_dataset_sidecar(dataset_root: Path, memory_name: str) -> Dict:
    sidecar_root = dataset_root / 'memory' / memory_name
    info_path = sidecar_root / 'memory_info.json'
    if not info_path.exists():
        raise FileNotFoundError(f'Missing memory_info.json: {info_path}')
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    checked = 0
    for episode in info['episodes']:
        path = sidecar_root / episode['file']
        if not path.exists():
            raise FileNotFoundError(f'Missing sidecar episode file: {path}')
        data = np.load(path)
        tokens = data['memory_tokens']
        masks = data['memory_masks']
        if tokens.shape[0] != episode['num_frames']:
            raise ValueError(f'{path}: token frame count mismatch.')
        if masks.shape[:2] != tokens.shape[:2]:
            raise ValueError(f'{path}: mask shape mismatch.')
        checked += 1
    return {'episodes_checked': checked, 'info': info}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate causal memory sidecars for copied datasets.')
    parser.add_argument(
        '--dataset-root',
        action='append',
        type=Path,
        required=True,
        help='Copied dataset root. Repeat this flag for multiple datasets.')
    parser.add_argument(
        '--encoder',
        type=str,
        default='raw-state',
        choices=('raw-state', 'pi0-siglip-moviechat'),
        help='Memory encoder to run.')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/pi0/pi0_paligemma_libero_all_full_finetune.py'),
        help='PI0 config used to build the SigLIP vision backbone.')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Torch device for pi0-siglip-moviechat.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='bf16',
        choices=('bf16', 'fp16', 'fp32'),
        help='Model dtype for pi0-siglip-moviechat.')
    parser.add_argument(
        '--video-keys',
        nargs='+',
        default=[
            'observation.images.image',
            'observation.images.wrist_image',
        ],
        help='Video keys to decode for visual memory.')
    parser.add_argument(
        '--frame-memory-tokens',
        type=int,
        default=32,
        help='Pooled visual tokens per frame before MovieChat memory update.')
    parser.add_argument(
        '--short-memory-size',
        type=int,
        default=18,
        help='MovieChat short-memory length.')
    parser.add_argument(
        '--high-relevance-keep-frames',
        type=int,
        default=3,
        help='Frames kept when the segment is relevant.')
    parser.add_argument(
        '--low-relevance-keep-frames',
        type=int,
        default=1,
        help='Frames kept when the segment is weakly relevant.')
    parser.add_argument(
        '--relevance-threshold',
        type=float,
        default=0.25,
        help='MovieChat relevance threshold.')
    parser.add_argument(
        '--clip-relevance',
        action='store_true',
        help='Use MovieChat-style CLIP text-image relevance for compression.')
    parser.add_argument(
        '--clip-model',
        type=str,
        default='ViT-B/32',
        help='CLIP model used when --clip-relevance is enabled.')
    parser.add_argument(
        '--memory-name',
        type=str,
        default='raw_state_causal_v1',
        help='Subdirectory name under <dataset_root>/memory/.')
    parser.add_argument(
        '--max-memory-tokens',
        type=int,
        default=32,
        help='Fixed number of memory tokens saved per frame.')
    parser.add_argument(
        '--memory-dim',
        type=int,
        default=32,
        help='Fixed feature dimension for each memory token.')
    parser.add_argument(
        '--stride',
        type=int,
        default=4,
        help='Temporal stride for causal history downsampling.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing episode sidecar files.')
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify an existing sidecar.')
    parser.add_argument(
        '--log-every',
        type=int,
        default=10,
        help='Print preprocessing progress every N episodes.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_memory_tokens <= 0:
        raise ValueError('--max-memory-tokens must be positive.')
    if args.memory_dim <= 0:
        raise ValueError('--memory-dim must be positive.')
    if args.stride <= 0:
        raise ValueError('--stride must be positive.')

    encoder = None
    if not args.verify_only and args.encoder == 'pi0-siglip-moviechat':
        encoder = PI0SigLIPMovieChatEncoder(
            config_path=args.config,
            device=args.device,
            dtype=args.dtype,
            max_memory_tokens=args.max_memory_tokens,
            frame_memory_tokens=args.frame_memory_tokens,
            short_memory_size=args.short_memory_size,
            high_relevance_keep_frames=args.high_relevance_keep_frames,
            low_relevance_keep_frames=args.low_relevance_keep_frames,
            relevance_threshold=args.relevance_threshold,
            video_keys=args.video_keys,
            clip_relevance=args.clip_relevance,
            clip_model=args.clip_model,
        )

    for dataset_root in args.dataset_root:
        dataset_root = dataset_root.resolve()
        if args.verify_only:
            result = verify_dataset_sidecar(dataset_root, args.memory_name)
            print(f'[OK] {dataset_root}: verified '
                  f'{result["episodes_checked"]} episodes')
            continue

        info = generate_dataset_sidecar(
            dataset_root=dataset_root,
            memory_name=args.memory_name,
            max_memory_tokens=args.max_memory_tokens,
            memory_dim=args.memory_dim,
            stride=args.stride,
            overwrite=args.overwrite,
            encoder_name=args.encoder,
            encoder=encoder,
            log_every=args.log_every,
        )
        print(f'[OK] {dataset_root}: wrote {info["num_episodes"]} episodes, '
              f'{info["num_frames"]} frames to memory/{args.memory_name}')


if __name__ == '__main__':
    main()
