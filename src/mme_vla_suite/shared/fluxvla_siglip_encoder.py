from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import SiglipVisionConfig, SiglipVisionModel


class FluxVLASigLIPEncoder:
    """FluxVLA-compatible 1152-d SigLIP encoder for MovieChat rollout."""

    expects_uint8_images = True

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
        dtype: str = "bf16",
        frame_memory_tokens: int = 32,
    ):
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"FluxVLA SigLIP checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = self._parse_dtype(dtype)
        self.frame_memory_tokens = frame_memory_tokens

        config = SiglipVisionConfig(
            attention_dropout=0.0,
            hidden_act="gelu_pytorch_tanh",
            hidden_size=1152,
            image_size=224,
            intermediate_size=4304,
            layer_norm_eps=1e-6,
            model_type="siglip_vision_model",
            num_attention_heads=16,
            num_channels=3,
            num_hidden_layers=27,
            patch_size=14,
            projection_dim=2048,
            vision_use_head=False,
        )
        self.model = SiglipVisionModel(config)
        self._load_fluxvla_vision_weights()
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        self.mean = torch.tensor(
            [123.515625, 116.04492188, 103.59375],
            dtype=torch.float32,
            device=self.device,
        )[None, :, None, None]
        self.std = torch.tensor(
            [58.27148438, 57.02636719, 57.27539062],
            dtype=torch.float32,
            device=self.device,
        )[None, :, None, None]

    @staticmethod
    def _parse_dtype(dtype: str) -> torch.dtype:
        normalized = str(dtype).lower()
        if normalized in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if normalized in {"fp16", "float16", "half"}:
            return torch.float16
        if normalized in {"fp32", "float32", "float"}:
            return torch.float32
        raise ValueError(f"Unsupported FluxVLA SigLIP dtype: {dtype}")

    def _load_fluxvla_vision_weights(self) -> None:
        prefix = "paligemma_with_expert.paligemma.model.vision_tower."
        state_dict = {}
        with safe_open(str(self.checkpoint_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state_dict[key[len(prefix):]] = handle.get_tensor(key)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected FluxVLA SigLIP checkpoint keys: {unexpected[:8]}")
        required_missing = [
            key for key in missing
            if not key.startswith("vision_model.head")
        ]
        if required_missing:
            raise RuntimeError(f"Missing FluxVLA SigLIP checkpoint keys: {required_missing[:8]}")

    def _preprocess(self, images: np.ndarray) -> torch.Tensor:
        images = np.asarray(images)
        if images.ndim != 5:
            raise ValueError(f"images must be (t, v, h, w, 3), got {images.shape}")
        t, v = images.shape[:2]
        flat = images.reshape(t * v, *images.shape[2:])
        tensor = torch.from_numpy(flat).to(device=self.device, dtype=torch.float32)
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2)
        elif tensor.shape[1] != 3:
            raise ValueError(f"images must be HWC or CHW RGB, got {images.shape}")
        tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        tensor = (tensor - self.mean) / self.std
        return tensor.to(dtype=self.dtype)

    @torch.no_grad()
    def __call__(self, images: np.ndarray) -> np.ndarray:
        t, v = images.shape[:2]
        pixel_values = self._preprocess(images)
        tokens = self.model(pixel_values).last_hidden_state
        if tokens.size(1) > self.frame_memory_tokens:
            tokens = F.adaptive_avg_pool1d(
                tokens.transpose(1, 2),
                self.frame_memory_tokens,
            ).transpose(1, 2)
        tokens = tokens.reshape(t, v, tokens.shape[1], tokens.shape[2])
        return tokens.float().cpu().numpy()
