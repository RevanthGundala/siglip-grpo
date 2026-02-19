from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from vla_coach.temporal_embedding import TemporalEmbedding


class SigLIPTReward:
    """Computes per-trajectory scalar rewards from SigLIP embeddings + temporal encoding.

    Matches SRPO pipeline: trajectory embedding → cosine similarity to success cluster → sigmoid reward.

    Args:
        embed_dim: SigLIP embedding dimension (1152 for ViT-SO400M).
        use_temporal: Whether to add temporal embeddings.
        freq_channels: Sinusoidal encoding channels for temporal MLP.
        sigmoid_steepness: Steepness of sigmoid reward mapping (SRPO default: 10.0).
        device: Torch device.
    """

    def __init__(
        self,
        embed_dim: int = 1152,
        use_temporal: bool = True,
        freq_channels: int = 256,
        sigmoid_steepness: float = 10.0,
        last_k_frames: int = 0,
        device: torch.device = None,
    ):
        self.embed_dim = embed_dim
        self.use_temporal = use_temporal
        self.sigmoid_steepness = sigmoid_steepness
        self.last_k_frames = last_k_frames
        self.device = device or torch.device("cpu")

        if use_temporal:
            self.temporal = TemporalEmbedding(
                embed_dim=embed_dim, freq_channels=freq_channels
            ).to(self.device)
        else:
            self.temporal = None

    def parameters(self):
        if self.temporal is not None:
            return self.temporal.parameters()
        return iter([])

    def state_dict(self):
        if self.temporal is not None:
            return self.temporal.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if self.temporal is not None:
            self.temporal.load_state_dict(state_dict)

    def embed_trajectory(self, frame_embeds: torch.Tensor) -> torch.Tensor:
        """(T, D) → (D,). Optionally adds temporal embeddings, pools final K frames."""
        frame_embeds = frame_embeds.to(self.device)
        if self.temporal is not None:
            T = frame_embeds.shape[0]
            temporal_emb = self.temporal.embed_trajectory(T, device=frame_embeds.device)
            frame_embeds = frame_embeds + temporal_emb
        if self.last_k_frames > 0:
            frame_embeds = frame_embeds[-self.last_k_frames:]
        return frame_embeds.mean(dim=0)

    def compute_batch_rewards(
        self,
        all_embeds: List[torch.Tensor],
        success_mask: List[bool],
        task_ids: List[int],
    ) -> Tuple[List[float], List[int]]:
        """Compute per-trajectory scalar rewards (SRPO-style).

        Returns (rewards, failed_indices). Successes get 1.0; failures get
        sigmoid(cosine similarity to success cluster) ∈ [0, 0.6], grouped by task.
        """
        n = len(all_embeds)
        traj_embeds = []
        for e in all_embeds:
            traj_embeds.append(self.embed_trajectory(e))
        traj_embeds_np = torch.stack(traj_embeds).detach().cpu().numpy()

        rewards = np.zeros(n, dtype=np.float64)
        failed_indices = []

        task_groups: Dict[int, List[int]] = defaultdict(list)
        for i, tid in enumerate(task_ids):
            task_groups[tid].append(i)

        for tid, indices in task_groups.items():
            indices = np.array(indices)
            succ_idx = indices[[success_mask[i] for i in indices]]
            fail_idx = indices[[not success_mask[i] for i in indices]]

            rewards[succ_idx] = 1.0

            if len(succ_idx) == 0 or len(fail_idx) == 0:
                failed_indices.extend(fail_idx.tolist())
                continue

            # Simple mean cluster center (skip DBSCAN for robustness)
            succ_embeds = traj_embeds_np[succ_idx]
            cluster_center = succ_embeds.mean(axis=0, keepdims=True)

            fail_embeds = traj_embeds_np[fail_idx]
            from scipy.spatial.distance import cdist
            cosine_sims = 1.0 - cdist(fail_embeds, cluster_center, "cosine").flatten()

            # Normalize cosine similarities and apply sigmoid → [0, 0.6]
            max_sim = cosine_sims.max()
            min_sim = cosine_sims.min()
            sim_range = max_sim - min_sim
            if sim_range < 1e-6:
                normalized = np.full_like(cosine_sims, 0.5)
            else:
                normalized = (cosine_sims - min_sim) / sim_range

            from scipy.special import expit
            sigmoid_inputs = self.sigmoid_steepness * (normalized - 0.5)
            reward_values = 0.6 * expit(sigmoid_inputs)

            rewards[fail_idx] = reward_values
            failed_indices.extend(fail_idx.tolist())

        return rewards.tolist(), failed_indices


def extract_siglip_embeddings(
    model,
    images: torch.Tensor,
    encoder_name: str = "siglip",
) -> torch.Tensor:
    """Extract (B, D) mean-pooled embeddings from the VLA's vision encoder.

    OpenVLA uses a fused DINOv2 + SigLIP backbone; the SigLIP featurizer
    is at model.vision_backbone.fused_featurizer (D=1152 for ViT-SO400M).
    """
    with torch.no_grad():
        vision_backbone = getattr(model, "vision_backbone", None)
        if vision_backbone is None:
            vision_backbone = getattr(model, "model", model)
            vision_backbone = getattr(vision_backbone, "vision_backbone", vision_backbone)

        # PrismaticVisionBackbone:
        #   .featurizer     = DINOv2 (embed_dim=1024)
        #   .fused_featurizer = SigLIP (embed_dim=1152)
        if encoder_name == "siglip" and hasattr(vision_backbone, "fused_featurizer"):
            featurizer = vision_backbone.fused_featurizer
        elif encoder_name == "dinov2" and hasattr(vision_backbone, "featurizer"):
            featurizer = vision_backbone.featurizer
        else:
            # Fallback: use the full backbone
            featurizer = vision_backbone

        # SigLIP expects 224x224; LIBERO frames are 256x256
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

        # Match featurizer dtype (model is bf16, interpolate outputs fp32)
        images = images.to(dtype=next(featurizer.parameters()).dtype)

        features = featurizer(images)

        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.dim() == 3:
            features = features.mean(dim=1)

        return features


def get_siglip_transform(model):
    """Get the SigLIP image transform from a VLA model (or None if unavailable)."""
    vision_backbone = getattr(model, "vision_backbone", None)
    if vision_backbone is None:
        vision_backbone = getattr(model, "model", model)
        vision_backbone = getattr(vision_backbone, "vision_backbone", vision_backbone)

    transform = getattr(vision_backbone, "image_transform", None)

    if transform is not None and hasattr(transform, "siglip_image_transform"):
        return transform.siglip_image_transform

    # PrismaticVisionBackbone doesn't expose image_transform; return None to
    # signal callers should use a standard SigLIP preprocessing pipeline.
    return transform
