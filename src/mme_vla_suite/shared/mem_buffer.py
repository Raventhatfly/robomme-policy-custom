import jax
import jax.numpy as jnp
import einops
import math
import heapq
from copy import deepcopy
import numpy as np
from typing import Callable
import cv2
from collections import defaultdict


from openpi.shared import image_tools
from mme_vla_suite.shared.data_utils import *


def create_dict(indices):
    result = defaultdict(lambda: defaultdict(list)) # {step_idx: {view_idx: [patch_idx]}}
    for step_idx, view_idx, patch_idx in indices:
        result[step_idx][view_idx].append(patch_idx)
    return result


class MemoryBuffer:
    # We use 8x8 tokens each image for token dropping and 4x4 tokens for frame sampling.
    def __init__(
        self,
        num_views: int = 1,  # number of camera views to be used as memory, better use fixed-views
        img_emb_dim: int = 2048,  # the initila image token embedding dimension, siglip is 2048
        pos_emb_dim: int = 768, 
        state_emb_dim: int = 8,
        max_steps: int = 4096,
        compute_token_drop_score: bool = False, # set true when using token dropping
        token_drop_keptsize: int = 2048,   # max number of tokens to be kept in the heap for similarity score calculation
        token_drop_stride: int = 8,        # timestep stride for similarity score calculation, if too close, the difference will be too small
        prepare_buffer: bool = False,      # set true during online evaluation, set false during offline training
        pool_type: str = "mean",           # pooling type for the image tokens, mean or max
        vision_enc_fn: Callable | None = None,   # vision encoding function
    ):
        self.num_views = num_views
        self.img_emb_dim = img_emb_dim        
        self.pos_emb_dim = pos_emb_dim
        self.state_emb_dim = state_emb_dim

        self.pool_type = pool_type
        self.token_drop_keptsize = token_drop_keptsize
        self.token_drop_stride = token_drop_stride
        self.token_drop_last_frame = -1   
        self.scored_token_heap = []
        self.compute_token_drop_score = compute_token_drop_score
        
        self._history_feats = {}
        
        
        if prepare_buffer:
            # load models in GPUs to provide embeedings
            # used for dataset buliding and evaluation
            from mme_vla_suite.shared.posemb_3d import PosEmb3D
            if vision_enc_fn is None:
                from mme_vla_suite.shared.siglip_tokenizer import SigLipTokenizer
                siglip_tokenizer = SigLipTokenizer()
                self.vision_enc = jax.jit(siglip_tokenizer.__call__)
            else:
                self.vision_enc = vision_enc_fn
            pos_embedder = PosEmb3D(dim=pos_emb_dim)
            ranges = jnp.arange(max_steps)
            self.pos_emb_dict = {
                "8x8": np.array(pos_embedder(ranges, 8)),
                "4x4": np.array(pos_embedder(ranges, 4)),
                "2x2": np.array(pos_embedder(ranges, 2)),
            }
        else:
            # used in dataset loading mode
            self.vision_enc = None
            self.pos_emb_dict = None
        
    def add_buffer(
        self,
        images, #: (t v h w 3), np.int8
        states, #: (t d), np.float32
        step_idx_list: list[int],
    ):
        assert self.vision_enc is not None, "encode is not initialized"
        
        t, v, _, _, _ = images.shape
        assert v == self.num_views
        
        image_jnp = jnp.array(
            images.astype(np.float32) / 255.0 * 2.0 - 1.0
        )
        image_jnp = einops.rearrange(image_jnp, "t v h w c -> (t v) h w c")
        image_jnp = image_tools.resize_with_pad(image_jnp, 224, 224)
        image_jnp = einops.rearrange(image_jnp, "(t v) h w c -> t v h w c", t=t, v=v)
        output_emb = self.vision_enc(image_jnp)  # (t, v, 64, 2048)
        
        pooled_emb_8x8 = pool_tokens_to_size(output_emb, 64)  # (t, v, 64, 2048)
        pooled_emb_4x4 = pool_tokens_to_size(output_emb, 16)  # (t, v, 16, 2048)
        pooled_emb_2x2 = pool_tokens_to_size(output_emb, 4)  # (t, v, 4, 2048)
        
        for i, step_idx in enumerate(step_idx_list):
            image_emb_8x8 = jax.device_get(pooled_emb_8x8)[i]  # (v, 64, 2048)
            image_emb_4x4 = jax.device_get(pooled_emb_4x4)[i]  # (v, 16, 2048)
            image_emb_2x2 = jax.device_get(pooled_emb_2x2)[i]  # (v, 4, 2048)
            
            pos_emb_8x8 = self.pos_emb_dict["8x8"][
                step_idx*self.num_views : (step_idx+1)*self.num_views]  # (v, 64, 768)
            pos_emb_4x4 = self.pos_emb_dict["4x4"][
                step_idx*self.num_views : (step_idx+1)*self.num_views]  # (v, 16, 768)
            pos_emb_2x2 = self.pos_emb_dict["2x2"][
                step_idx*self.num_views : (step_idx+1)*self.num_views]  # (v, 4, 768)
            
            token_emb_to_save = {
                "image_pixels": images[i].copy(), # fp32, this is for visualization
                "image_emb_8x8": image_emb_8x8, # bf16
                "image_emb_4x4": image_emb_4x4, # bf16
                "image_emb_2x2": image_emb_2x2, # bf16
                "pos_emb_8x8": pos_emb_8x8, # fp32
                "pos_emb_4x4": pos_emb_4x4, # fp32
                "pos_emb_2x2": pos_emb_2x2, # fp32
                "state_emb": states[i], # fp32
            }
            assert step_idx not in self._history_feats, f"step_idx {step_idx} already in buffer"
            self._history_feats[step_idx] = token_emb_to_save
            
            if self.compute_token_drop_score:
                self._process_token_drop_score(step_idx)    
            
    
    def get_history_feats(self, step_idx: int, remove_image_pixels: bool = True):
        if remove_image_pixels:
            return {k: v for k, v in self._history_feats[step_idx].items() if k != "image_pixels"}
        return self._history_feats[step_idx]

    
    def _process_token_drop_score(self, step_idx):
        if step_idx == 0: # all tokens in the first frame should be kept, only later frames are considered for token dropping
            for patch_idx in range(64): # 64 patches in 8x8 grid, which is fixed here for now
                for view_idx in range(self.num_views):
                    heapq.heappush(
                        self.scored_token_heap, (1000.0, step_idx, view_idx, patch_idx)
                    )
        
        if step_idx == self.token_drop_last_frame + self.token_drop_stride:
            prev_img = self._history_feats[max(0, self.token_drop_last_frame)]["image_pixels"]
            curr_img = self._history_feats[step_idx]["image_pixels"]
            
            prev_img = prev_img.astype(np.float32) / 255.0 * 2.0 - 1.0
            curr_img = curr_img.astype(np.float32) / 255.0 * 2.0 - 1.0

            prev_img_tokens = einops.rearrange(prev_img, "v (ph h) (pw w) c -> v (ph pw) (h w c)", ph=8, pw=8)
            curr_img_tokens = einops.rearrange(curr_img, "v (ph h) (pw w) c -> v (ph pw) (h w c)", ph=8, pw=8)
            
            for view_idx in range(self.num_views):
                # pixel works better than siglip embedding
                difference = np.abs(prev_img_tokens[view_idx] - curr_img_tokens[view_idx]).mean(axis=-1) 
                for patch_idx in range(64):
                    if difference[patch_idx] < 1e-4:
                        # too similar, skip this patch
                        continue
                    heapq.heappush(
                        self.scored_token_heap,
                        (
                            difference[patch_idx],
                            step_idx,     # timestep index
                            view_idx,     # view index
                            patch_idx,    # patch index
                        ),
                    )
                    if len(self.scored_token_heap) > self.token_drop_keptsize:
                        heapq.heappop(self.scored_token_heap)
            
            self.token_drop_last_frame += self.token_drop_stride
                
    
    def clear(self):
        self.scored_token_heap.clear()
        self.token_drop_last_frame = -1
        self._history_feats.clear()
        
        
    def get_token_dropping_indices(self):
        selected_tokens_heap = deepcopy(self.scored_token_heap)
        kept_indices = []
        while selected_tokens_heap:
            _, buffer_idx, view_idx, patch_idx = heapq.heappop(selected_tokens_heap)
            kept_indices.append((buffer_idx, view_idx, patch_idx))
                
        return kept_indices
    
    @staticmethod
    def filter_token_dropping_indices(kept_indices, step_idx, token_budget, is_sorted=True):
        kept_indices_filtered = [item for item in kept_indices if item[0] <= step_idx]
        kept_indices = kept_indices_filtered[-token_budget:]
        if is_sorted:
            kept_indices = sorted(kept_indices)
        return kept_indices
    
    
    def _prepare_token_dropping(self, history_feats, sorted_kept_indices, token_budget):
        img_emb = np.zeros((token_budget, self.img_emb_dim), dtype=np.float32)
        pos_emb = np.zeros((token_budget, self.pos_emb_dim), dtype=np.float32)
        state_emb = np.zeros((token_budget, self.state_emb_dim), dtype=np.float32)
        mask = np.zeros((token_budget), dtype=np.bool_)

        for idx, (buffer_idx, view_idx, patch_idx) in enumerate(sorted_kept_indices[:token_budget]):
            img_emb[idx] = history_feats[buffer_idx][f"image_emb_8x8"][view_idx][patch_idx]
            pos_emb[idx] = history_feats[buffer_idx][f"pos_emb_8x8"][view_idx][patch_idx]
            state_emb[idx] = history_feats[buffer_idx]["state_emb"]
            mask[idx] = True
        
        # print("effective token length: ", np.sum(mask))
                
        return img_emb, pos_emb, state_emb, mask
    

    def prepare_token_dropping(self, step_idx, token_budget, history_feats_gather_fn, kept_indices=None, *args, **kwargs):
        if kept_indices is None:
            indices_to_load = self.get_token_dropping_indices()
        else:
            indices_to_load = kept_indices
        
        sorted_kept_indices = self.filter_token_dropping_indices(
            indices_to_load, step_idx, token_budget, is_sorted=True)
        
        indices_to_load = sorted(set([item[0] for item in sorted_kept_indices]))
        # print("step_idx: ", step_idx)
        # print(f"indices_to_load (length: {len(indices_to_load)}/{len(sorted_kept_indices)}): {indices_to_load}")
        # self._visualize_token_dropping(sorted_kept_indices, step_idx)
        # import pdb; pdb.set_trace()
        
        history_feats = history_feats_gather_fn(indices_to_load, *args, **kwargs)

        return self._prepare_token_dropping(history_feats, sorted_kept_indices, token_budget)
    
    
    def _visualize_token_dropping(self, kept_indices, step_idx):
        dic = create_dict(sorted(kept_indices))
        images = [self._history_feats[idx]["image_pixels"][0] for idx in range(len(self._history_feats))]
        
        images_anno = []
        for step_idx in dic:
            img_anno = images[step_idx].copy()

            img_anno = cv2.putText(
                img_anno,
                f"{step_idx}",
                (img_anno.shape[1] // 2, img_anno.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
            )
            mask = np.zeros(img_anno.shape[:2], dtype=np.uint8)

            for patch_idx in range(64):
                if patch_idx in dic[step_idx][0]:
                    continue
                h_center = patch_idx // 8 * 32 + 16
                w_center = patch_idx % 8 * 32 + 16
                mask = cv2.rectangle(
                    mask,
                    (w_center - 16, h_center - 16),
                    (w_center + 16, h_center + 16),
                    255,
                    -1,
                )
            img_anno[mask == 255] = (
                img_anno[mask == 255] * 0.3 + np.array([255, 255, 255]) * 0.7
            )
            
            images_anno.append(img_anno)
        
        import imageio, os
        imageio.mimsave(
            os.path.join(
                "debug",
                f"token_drop_anno_step_{step_idx}.mp4",
            ),
            images_anno,
            fps=10,
        )
        
    
    def get_frame_sampling_indices(self, step_idx, token_budget, token_per_image):
        max_size = token_budget // (token_per_image * self.num_views)
        return even_sampling_indices(step_idx, max_size)
    
    
    def _prepare_frame_sampling(self, history_feats, indices_to_load, token_budget, token_per_image):
        spatial_size = str(int(math.sqrt(token_per_image)))
        spatial_key = f"{spatial_size}x{spatial_size}"
        max_size = token_budget // (token_per_image * self.num_views)
        

        sampled_img_emb = self._load_emb(history_feats, indices_to_load, f"image_emb_{spatial_key}")
        sampled_pos_emb = self._load_emb(history_feats, indices_to_load, f"pos_emb_{spatial_key}")
        sampled_state_emb = self._load_emb(history_feats, indices_to_load, "state_emb")
        mask = np.ones((sampled_img_emb.shape[0]), dtype=np.bool_)
                
        # we use right padding to the perceptual memory
        sampled_img_emb, sampled_pos_emb, sampled_state_emb, mask = right_padding_token_emb(
            sampled_img_emb, sampled_pos_emb, sampled_state_emb, mask, max_size
        )
                
        img_emb =  np.reshape(sampled_img_emb, (-1, self.img_emb_dim))
        pos_emb = np.reshape(sampled_pos_emb, (-1, self.pos_emb_dim))
        mask = np.repeat(mask, self.num_views * token_per_image)
        state_emb = np.repeat(sampled_state_emb, self.num_views * token_per_image, axis=0)
        
        # print("effective token length: ", np.sum(mask))
                
        return img_emb, pos_emb, state_emb, mask
            
    
    def prepare_frame_sampling(self, step_idx, token_budget, token_per_image, history_feats_gather_fn,  *args, **kwargs):
        indices_to_load = self.get_frame_sampling_indices(step_idx, token_budget, token_per_image)
        # print("step_idx: ", step_idx, "indices_to_load: ", indices_to_load, "length: ", len(indices_to_load))
        # self._visualize_frame_sampling(indices_to_load, step_idx)
        history_feats = history_feats_gather_fn(indices_to_load, *args, **kwargs)
        return self._prepare_frame_sampling(history_feats, indices_to_load, token_budget, token_per_image)


    def _visualize_frame_sampling(self, indices_to_load, step_idx):
        images = [self._history_feats[idx]["image_pixels"][0] for idx in indices_to_load]
        
        import imageio, os
        imageio.mimsave(
            os.path.join(
                "debug",
                f"frame_sampling_step_{step_idx}.mp4",
            ),
            images,
            fps=2,
        )
                
    @staticmethod
    def _load_emb(history_feats: dict, indices_to_load: list[int], key: str):
        return np.stack(
            [history_feats[idx][key] for idx in indices_to_load],
            axis=0,
        )
        
    def default_history_feats_gather_fn(self, indices_to_load, *args, **kwargs):
        return {idx: self._history_feats[idx] for idx in indices_to_load}


class MemoryBufferRecurrent(MemoryBuffer):
    # By default, we use 8x8 tokens each image for recurrent memory.
    def __init__(
        self,
        input_obs_horizon: int = 8, 
        max_recur_steps: int = 64, # max number of recurrent steps for all history 
        max_video_steps: int = 40, # max number of recurrent steps to load for the video-based observations for certain video-conditioned tasks
        *args, **kwargs
    ):
        
        super().__init__(*args, **kwargs)
                
        self.input_obs_horizon = input_obs_horizon
        self.max_recur_steps = max_recur_steps
        self.max_video_steps = max_video_steps

    
    def add_buffer(
        self,
        images, #: (t v h w 3), np.int8
        state, #: (t d), np.float32
        step_idx_list: list[int],
    ):
        assert self.vision_enc is not None, "encode is not initialized"
        
        t, v, _, _, _ = images.shape
        assert v == self.num_views
        
        image_jnp = jnp.array(
            images.astype(np.float32) / 255.0 * 2.0 - 1.0
        )
        image_jnp = einops.rearrange(image_jnp, "t v h w c -> (t v) h w c")
        image_jnp = image_tools.resize_with_pad(image_jnp, 224, 224)
        image_jnp = einops.rearrange(image_jnp, "(t v) h w c -> t v h w c", t=t, v=v)
        output_emb = self.vision_enc(image_jnp)  # (t, v, 64, 2048)
        
        # use 8x8 tokens, since recurrent memory can take more tokens than perceptual memory
        pooled_emb_8x8 = pool_tokens_to_size(output_emb, 64)  # (t, v, 64, 2048)
        
        for i, step_idx in enumerate(step_idx_list):
            image_emb_8x8 = jax.device_get(pooled_emb_8x8)[i]  # (v, 64, 2048)
            
            pos_emb_8x8 = self.pos_emb_dict["8x8"][
                step_idx*self.num_views : (step_idx+1)*self.num_views]  # (v, 64, 768)
            
            token_emb_to_save = { # recurrent memory only uses 8x8
                # "image_pixels": images[i].copy(), # fp32
                "image_emb_8x8": image_emb_8x8,
                "pos_emb_8x8": pos_emb_8x8,
                "state_emb": state[i],
            }
            assert step_idx not in self._history_feats, f"step_idx {step_idx} already in buffer"
            self._history_feats[step_idx] = token_emb_to_save  
    
    
    def get_token_recurrent_indices(self, step_idx, exec_start_idx):
        input_obs_horizon = self.input_obs_horizon
        max_recur_steps = self.max_recur_steps
        max_video_steps = self.max_video_steps
                
        assert step_idx >=0 and step_idx >= exec_start_idx
        
        if exec_start_idx == 0:
            # no videos
            if step_idx < input_obs_horizon:
                indices_to_load = [step_idx]
            else:
                start_idx = step_idx % input_obs_horizon
                indices_to_load = list(range(start_idx, step_idx+1, input_obs_horizon))
                indices_to_load = indices_to_load[-max_recur_steps:]
        else:
            if exec_start_idx <= input_obs_horizon * 2:
                # if too short, use more granular sampling
                video_indices = list(range(0, exec_start_idx, input_obs_horizon//2))
            elif exec_start_idx <= max_video_steps * input_obs_horizon:
                # if not too short, use regular sampling
                video_indices = list(range(0, exec_start_idx, input_obs_horizon))
            else:
                # if too long, use fixed-size sampling
                video_indices = np.linspace(0, exec_start_idx - 1, max_video_steps, dtype=int).tolist()
            
            if step_idx-exec_start_idx < input_obs_horizon:
                rest_indices = [step_idx]
            else:
                start_idx = (step_idx-exec_start_idx) % input_obs_horizon + exec_start_idx
                rest_indices = list(range(start_idx, step_idx+1, input_obs_horizon))
            
            combined_indices = video_indices + rest_indices
            indices_to_load = combined_indices[-max_recur_steps:]
            
        assert len(indices_to_load) > 0 and len(indices_to_load) <= max_recur_steps
            
        return indices_to_load
    
    
    def _prepare_token_recurrent(self, history_feats, indices_to_load, padding=True):        
        recur_image_embs = []
        recur_pos_embs = []
        recur_state_embs = []
        recur_masks = []
        
        for i in indices_to_load:
            recur_image_embs.append(history_feats[i]["image_emb_8x8"])  # (v, p, d)
            recur_pos_embs.append(history_feats[i]["pos_emb_8x8"])  # (v, p, d)
            recur_state_embs.append(history_feats[i]["state_emb"])  # (d)
            recur_masks.append(True)  # (1)
        
        recur_image_emb = np.stack(recur_image_embs, axis=0)
        recur_pos_emb = np.stack(recur_pos_embs, axis=0)
        recur_state_emb = np.stack(recur_state_embs, axis=0)
        recur_mask = np.stack(recur_masks, axis=0)
        
        if padding:
            recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask = left_padding_token_emb(
                recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask, self.max_recur_steps
            )                
        return recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask


    def prepare_token_recurrent(self, step_idx, exec_start_idx,  history_feats_gather_fn,  *args, **kwargs):
        indices_to_load = self.get_token_recurrent_indices(step_idx, exec_start_idx)
        # print("step_idx: ", step_idx, "exec_start_idx: ", exec_start_idx)
        # print(f"indices_to_load ({len(indices_to_load)}): {indices_to_load}")
        assert len(indices_to_load) > 0 and len(indices_to_load) <= self.max_recur_steps
        history_feats = history_feats_gather_fn(indices_to_load, *args, **kwargs)
        # self._visualize_token_recurrent(indices_to_load, step_idx)
        return self._prepare_token_recurrent(history_feats, indices_to_load)

    def _visualize_token_recurrent(self, indices_to_load, step_idx):
        images = [self._history_feats[idx]["image_pixels"][0] for idx in indices_to_load]
        import imageio, os
        imageio.mimsave(
            os.path.join(
                "debug",
                f"token_recurrent_step_{step_idx}.mp4",
            ),
            images,
            fps=2,
        )


class MemoryBufferMovieChat(MemoryBuffer):
    """MovieChat-style causal short/long memory for online rollout.

    This mirrors FluxVLA's MovieChatMemory update/read behavior, but keeps the
    implementation in numpy/JAX-land so the OpenPI policy server does not need
    to import FluxVLA. The model-facing output is the same as the offline
    sidecar: a fixed `(budget, img_emb_dim)` static memory bank.
    """

    def __init__(
        self,
        short_memory_size: int = 18,
        short_memory_merge: int = 2,
        long_memory_size: int = 30,
        high_relevance_keep_frames: int = 3,
        low_relevance_keep_frames: int = 1,
        relevance_threshold: float = 0.25,
        frame_memory_tokens: int = 32,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.short_memory_size = short_memory_size
        self.short_memory_merge = short_memory_merge
        self.long_memory_size = None if long_memory_size in (None, 0) else long_memory_size
        self.high_relevance_keep_frames = high_relevance_keep_frames
        self.low_relevance_keep_frames = low_relevance_keep_frames
        self.relevance_threshold = relevance_threshold
        self.frame_memory_tokens = frame_memory_tokens
        self._reset_moviechat_state()

    def _reset_moviechat_state(self):
        self.short_memory = None
        self.short_mask = None
        self.short_weights = None
        self.recent_short_memory = None
        self.recent_short_mask = None
        self.long_memory = None
        self.long_mask = None
        self.long_weights = None

    def clear(self):
        super().clear()
        self._reset_moviechat_state()

    @staticmethod
    def _append(old, new, axis=1):
        if old is None:
            return new
        if new is None or new.shape[axis] == 0:
            return old
        return np.concatenate([old, new], axis=axis)

    @staticmethod
    def _frame_weights(mask):
        return mask.sum(axis=-1, keepdims=True).astype(np.float32)[..., None]

    def _most_similar_adjacent_index(self, tokens, mask):
        if tokens.shape[1] <= 1:
            return 0
        scores = []
        for i in range(tokens.shape[1] - 1):
            left = tokens[:, i]
            right = tokens[:, i + 1]
            pair_mask = mask[:, i, :, None] & mask[:, i + 1, None, :]
            denom = np.maximum(pair_mask.sum(axis=(-1, -2)), 1.0)
            sim = np.matmul(left, np.swapaxes(right, -1, -2))
            per_batch = (sim * pair_mask.astype(np.float32)).sum(axis=(-1, -2)) / denom
            scores.append(float(per_batch.mean()))
        return int(np.argmax(np.asarray(scores)))

    @staticmethod
    def _merge_adjacent(tokens, mask, weights, merge_idx):
        left_weight = weights[:, merge_idx]
        right_weight = weights[:, merge_idx + 1]
        denom = np.maximum(left_weight + right_weight, 1.0)
        merged = (tokens[:, merge_idx] * left_weight + tokens[:, merge_idx + 1] * right_weight) / denom
        merged_mask = mask[:, merge_idx] | mask[:, merge_idx + 1]
        merged_weight = left_weight + right_weight

        tokens = np.concatenate(
            [tokens[:, :merge_idx], merged[:, None], tokens[:, merge_idx + 2:]], axis=1)
        mask = np.concatenate(
            [mask[:, :merge_idx], merged_mask[:, None], mask[:, merge_idx + 2:]], axis=1)
        weights = np.concatenate(
            [weights[:, :merge_idx], merged_weight[:, None], weights[:, merge_idx + 2:]], axis=1)
        return tokens, mask, weights

    def _consolidate(self, tokens, mask, weights):
        # Training sidecar generation used clip_relevance=False, so FluxVLA falls
        # back to the threshold and keeps the high-relevance target.
        target_frames = (
            self.high_relevance_keep_frames
            if self.relevance_threshold >= self.relevance_threshold
            else self.low_relevance_keep_frames
        )
        target_frames = min(target_frames, tokens.shape[1])
        while tokens.shape[1] > target_frames:
            merge_idx = self._most_similar_adjacent_index(tokens, mask)
            tokens, mask, weights = self._merge_adjacent(tokens, mask, weights, merge_idx)
        return tokens, mask, weights

    def _update_moviechat(self, frame_tokens):
        frame_tokens = np.asarray(frame_tokens, dtype=np.float32)
        if frame_tokens.ndim != 3:
            raise ValueError(f"frame_tokens must be (batch, num_tokens, dim), got {frame_tokens.shape}")
        if frame_tokens.shape[-1] != self.img_emb_dim:
            raise ValueError(
                f"MovieChat online encoder dim {frame_tokens.shape[-1]} does not match "
                f"training sidecar dim {self.img_emb_dim}."
            )
        frame_tokens = frame_tokens[:, None]
        frame_mask = np.ones(frame_tokens.shape[:-1], dtype=np.bool_)
        frame_weights = self._frame_weights(frame_mask)

        self.short_memory = self._append(self.short_memory, frame_tokens, axis=1)
        self.short_mask = self._append(self.short_mask, frame_mask, axis=1)
        self.short_weights = self._append(self.short_weights, frame_weights, axis=1)

        while self.short_memory is not None and self.short_memory.shape[1] >= self.short_memory_size:
            segment = self.short_memory[:, :self.short_memory_size]
            segment_mask = self.short_mask[:, :self.short_memory_size]
            segment_weights = self.short_weights[:, :self.short_memory_size]
            remaining = self.short_memory[:, self.short_memory_size:]
            remaining_mask = self.short_mask[:, self.short_memory_size:]
            remaining_weights = self.short_weights[:, self.short_memory_size:]

            self.recent_short_memory = segment
            self.recent_short_mask = segment_mask
            compressed, compressed_mask, compressed_weights = self._consolidate(
                segment, segment_mask, segment_weights)

            self.long_memory = self._append(self.long_memory, compressed, axis=1)
            self.long_mask = self._append(self.long_mask, compressed_mask, axis=1)
            self.long_weights = self._append(self.long_weights, compressed_weights, axis=1)
            if self.long_memory_size is not None and self.long_memory.shape[1] > self.long_memory_size:
                keep_from = self.long_memory.shape[1] - self.long_memory_size
                self.long_memory = self.long_memory[:, keep_from:]
                self.long_mask = self.long_mask[:, keep_from:]
                self.long_weights = self.long_weights[:, keep_from:]

            self.short_memory = remaining if remaining.shape[1] else None
            self.short_mask = remaining_mask if remaining_mask.shape[1] else None
            self.short_weights = remaining_weights if remaining_weights.shape[1] else None

    def add_buffer(
        self,
        images,
        states,
        step_idx_list: list[int],
    ):
        assert self.vision_enc is not None, "encode is not initialized"

        t, v, _, _, _ = images.shape
        if getattr(self.vision_enc, "expects_uint8_images", False):
            pooled_emb = self.vision_enc(images)
        else:
            image_jnp = jnp.array(images.astype(np.float32) / 255.0 * 2.0 - 1.0)
            image_jnp = einops.rearrange(image_jnp, "t v h w c -> (t v) h w c")
            image_jnp = image_tools.resize_with_pad(image_jnp, 224, 224)
            image_jnp = einops.rearrange(image_jnp, "(t v) h w c -> t v h w c", t=t, v=v)
            output_emb = self.vision_enc(image_jnp)
            pooled_emb = jax.device_get(pool_tokens_to_size(output_emb, self.frame_memory_tokens))

        for i, step_idx in enumerate(step_idx_list):
            frame_tokens = pooled_emb[i]
            self._update_moviechat(frame_tokens)
            self._history_feats[step_idx] = {"state_emb": states[i]}

    def _read_fixed_memory(self, token_budget):
        short_tokens = (
            self.recent_short_memory
            if self.recent_short_memory is not None else self.short_memory)
        short_mask = (
            self.recent_short_mask
            if self.recent_short_mask is not None else self.short_mask)
        tokens = self._append(self.long_memory, short_tokens, axis=1)
        mask = self._append(self.long_mask, short_mask, axis=1)

        img_emb = np.zeros((token_budget, self.img_emb_dim), dtype=np.float32)
        pos_emb = np.zeros((token_budget, self.pos_emb_dim), dtype=np.float32)
        state_emb = np.zeros((token_budget, self.state_emb_dim), dtype=np.float32)
        out_mask = np.zeros((token_budget,), dtype=np.bool_)
        if tokens is None:
            return img_emb, pos_emb, state_emb, out_mask

        tokens = tokens.reshape(-1, tokens.shape[-1]).astype(np.float32)
        mask = mask.reshape(-1).astype(np.bool_)
        if tokens.shape[0] > token_budget:
            tokens = self._adaptive_avg_pool_tokens(tokens, token_budget)
            mask = np.ones((token_budget,), dtype=np.bool_)

        keep = min(tokens.shape[0], token_budget)
        if keep > 0:
            img_emb[:keep] = tokens[-keep:]
            out_mask[:keep] = mask[-keep:]
        return img_emb, pos_emb, state_emb, out_mask

    @staticmethod
    def _adaptive_avg_pool_tokens(tokens, target_size):
        source_size = tokens.shape[0]
        pooled = np.zeros((target_size, tokens.shape[1]), dtype=np.float32)
        for i in range(target_size):
            start = int(np.floor(i * source_size / target_size))
            end = int(np.ceil((i + 1) * source_size / target_size))
            end = max(end, start + 1)
            pooled[i] = tokens[start:end].mean(axis=0)
        return pooled

    def prepare_moviechat_memory(
        self,
        step_idx,
        token_budget,
        token_per_image,
        history_feats_gather_fn,
        *args,
        **kwargs,
    ):
        del step_idx, token_per_image, history_feats_gather_fn, args, kwargs
        return self._read_fixed_memory(token_budget)
