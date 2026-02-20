from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from siglip_grpo.utils import setup_logging

logger = setup_logging()


def _unwrap_model(model):
    """Unwrap PEFT/LoRA layers to get the underlying VLA model.

    PeftModel.base_model → LoraModel, LoraModel.model → PrismaticForConditionalGeneration.
    LoRA weights are injected in-place, so the unwrapped model still uses them.
    """
    unwrapped = model
    if hasattr(unwrapped, "base_model"):
        unwrapped = unwrapped.base_model  # LoraModel
        if hasattr(unwrapped, "model"):
            unwrapped = unwrapped.model  # actual VLA
    return unwrapped


def compute_grpo_advantages(
    rewards: List[float],
    task_ids: List[int],
    epsilon: float = 1e-6,
) -> List[float]:
    """GRPO group-relative advantage: (reward - group_mean) / group_std per task."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, (r, tid) in enumerate(zip(rewards, task_ids)):
        groups[tid].append((i, r))

    advantages = [0.0] * len(rewards)
    for tid, entries in groups.items():
        group_rewards = [r for _, r in entries]
        mean = sum(group_rewards) / len(group_rewards)
        std = (sum((r - mean) ** 2 for r in group_rewards) / max(len(group_rewards) - 1, 1)) ** 0.5
        std = max(std, epsilon)
        for idx, r in entries:
            advantages[idx] = (r - mean) / std

    return advantages


def compute_token_log_probs(
    model,
    query_inputs: dict,
    response_ids: Optional[torch.Tensor] = None,
    unnorm_key: str = "libero_spatial_no_noops",
    enable_grad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute sum of token-level log-probs for action tokens.

    Returns (log_prob_sum, token_ids). If response_ids is None, derives
    token IDs via argmax (matching predict_action's discrete path).
    """
    device = next(model.parameters()).device
    model_base = _unwrap_model(model)

    input_ids = query_inputs["input_ids"].to(device)
    pixel_values = query_inputs["pixel_values"].to(device, dtype=torch.bfloat16)
    attention_mask = query_inputs["attention_mask"].to(device)
    if response_ids is not None:
        response_ids = response_ids.to(device)

    with torch.set_grad_enabled(enable_grad):
        try:
            from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, IGNORE_INDEX
        except ImportError:
            ACTION_DIM = 7
            NUM_ACTIONS_CHUNK = 8
            IGNORE_INDEX = -100

        n_action_tokens = ACTION_DIM * NUM_ACTIONS_CHUNK

        # 29871 is an eos token
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], device=device)), dim=1
            )
            attention_mask = torch.cat(
                (attention_mask, torch.ones(1, 1, device=device, dtype=attention_mask.dtype)), dim=1
            )

        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1

        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        input_ids_prep, attention_mask_prep = model_base._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        labels = model_base._prepare_labels_for_action_prediction(labels, input_ids_prep)

        input_embeddings = model_base.get_input_embeddings()(input_ids_prep)
        all_actions_mask = model_base._process_action_masks(labels)

        all_actions_mask_3d = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask_3d

        projected = model_base._process_vision_features(pixel_values)

        NUM_PATCHES = model_base.vision_backbone.get_num_patches() * model_base.vision_backbone.get_num_images_in_input()

        multimodal_embeddings, multimodal_attention_mask = model_base._build_multimodal_attention(
            input_embeddings, projected, attention_mask_prep
        )

        lm_output = model_base.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            inputs_embeds=multimodal_embeddings,
            output_hidden_states=False,
            return_dict=True,
        )

        logits = lm_output.logits
        action_start = NUM_PATCHES + NUM_PROMPT_TOKENS
        action_end = action_start + n_action_tokens
        action_logits = logits[:, action_start:action_end, :]

        if response_ids is None:
            response_ids = action_logits.argmax(dim=2)

        log_probs = F.log_softmax(action_logits, dim=-1)
        if response_ids.dim() == 1:
            response_ids = response_ids.unsqueeze(0)
        n_tokens = min(log_probs.shape[1], response_ids.shape[-1])
        gathered = log_probs[:, :n_tokens, :].gather(
            2, response_ids[:, :n_tokens].unsqueeze(-1)
        ).squeeze(-1)

        return gathered.sum(), response_ids.detach()


def grpo_loss(
    log_probs: List[torch.Tensor],
    old_log_probs: List[torch.Tensor],
    advantages: List[float],
    clip_epsilon: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """PPO clipped surrogate loss at the trajectory level."""
    all_losses = []
    all_ratios = []

    for lp, old_lp, adv in zip(log_probs, old_log_probs, advantages):
        adv_t = torch.tensor(adv, device=lp.device, dtype=lp.dtype).detach()
        ratio = torch.exp(lp - old_lp.detach())
        all_ratios.append(ratio.item())

        clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surrogate = torch.min(ratio * adv_t, clipped_ratio * adv_t)
        all_losses.append(-surrogate)

    loss = torch.stack(all_losses).mean()

    metrics = {
        "grpo_loss": loss.item(),
        "avg_ratio": sum(all_ratios) / max(len(all_ratios), 1),
    }
    return loss, metrics


class GRPOTrainer:
    """Manages GRPO training state: optimizer, clipping, gradient norm."""

    def __init__(
        self,
        model,
        optimizer,
        reward_model=None,
        clip_epsilon: float = 0.2,
        kl_coeff: float = 0.01,
        gamma: float = 0.99,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.reward_model = reward_model
        self.clip_epsilon = clip_epsilon
        self.max_grad_norm = max_grad_norm
        self.step_count = 0

    def update(
        self,
        log_probs: List[torch.Tensor],
        old_log_probs: List[torch.Tensor],
        advantages: List[float],
    ) -> Dict[str, float]:
        """Perform one GRPO gradient update. Returns training metrics."""
        loss, metrics = grpo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            clip_epsilon=self.clip_epsilon,
        )

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            metrics["grad_norm"] = grad_norm.item()
        self.optimizer.step()

        self.step_count += 1
        metrics["step"] = self.step_count
        return metrics
