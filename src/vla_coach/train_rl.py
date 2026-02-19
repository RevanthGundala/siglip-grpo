"""SigLIP-T self-improvement loop: rollout → reward → GRPO → repeat."""

import argparse
import json
import math
import os
import time
from collections import deque

import numpy as np
import torch
from PIL import Image

from vla_coach.grpo import GRPOTrainer, compute_grpo_advantages, compute_token_log_probs
from vla_coach.reward import SigLIPTReward, extract_siglip_embeddings
from vla_coach.rollout import collect_rollouts, compute_batch_statistics
from vla_coach.utils import load_config, setup_logging

logger = setup_logging()


def train_rl(cfg: dict) -> dict:
    """Run the SigLIP-T self-improvement loop.

    Each iteration: rollout VLA on LIBERO → extract SigLIP embeddings (free
    from inference) → compute dense rewards via cosine similarity to successful
    trajectories → GRPO policy gradient update.

    Returns history dict with per-iteration metrics.
    """
    from libero.libero.benchmark import get_benchmark

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load VLA model
    logger.info("Loading VLA model...")
    model, processor, norm_stats = _load_vla(cfg, device)
    logger.info("VLA loaded.")

    if torch.cuda.is_available():
        logger.info(f"VRAM after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Setup reward model
    embed_dim = cfg.get("embed_dim", 1152)
    use_temporal = cfg.get("use_temporal", True)
    reward_model = SigLIPTReward(
        embed_dim=embed_dim,
        use_temporal=use_temporal,
        last_k_frames=cfg.get("last_k_frames", 0),
        device=device,
    )
    label = "SigLIP-T" if use_temporal else "SigLIP-only"
    n_reward_params = sum(p.numel() for p in reward_model.parameters())
    logger.info(f"Reward model: {label} ({n_reward_params:,} trainable params)")

    # Setup LIBERO environment
    benchmark_name = cfg.get("benchmark", "libero_spatial")
    BENCHMARK_MAP = {
        "libero_spatial": "LIBERO_SPATIAL",
        "libero_object": "LIBERO_OBJECT",
        "libero_goal": "LIBERO_GOAL",
        "libero_10": "LIBERO_10",
    }
    benchmark_cls = get_benchmark(BENCHMARK_MAP[benchmark_name])
    benchmark = benchmark_cls(cfg.get("task_order_index", 0))
    n_tasks = min(cfg.get("n_tasks", benchmark.n_tasks), benchmark.n_tasks)
    logger.info(f"Benchmark: {benchmark_name}, {n_tasks} tasks")

    # Setup optimizer (VLA LoRA + temporal MLP)
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    trainable_params += list(reward_model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.get("lr", 1e-5),
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    grpo_cfg = cfg.get("grpo", {})
    trainer = GRPOTrainer(
        model=model,
        optimizer=optimizer,
        reward_model=reward_model,
        clip_epsilon=grpo_cfg.get("clip_epsilon", 0.2),
        kl_coeff=grpo_cfg.get("kl_coeff", 0.01),
        gamma=grpo_cfg.get("gamma", 0.99),
        max_grad_norm=grpo_cfg.get("max_grad_norm", 1.0),
    )

    def extract_fn(images):
        return extract_siglip_embeddings(model, images)

    # Training loop
    n_iterations = cfg.get("n_iterations", 3)
    rollouts_per_task = cfg.get("rollouts_per_task", 8)
    max_steps = cfg.get("max_steps", 300)
    checkpoint_dir = cfg.get("checkpoint_dir", "results/siglip_t/checkpoints")
    video_dir = cfg.get("video_dir", "results/siglip_t/videos") if cfg.get("save_videos") else None
    os.makedirs(checkpoint_dir, exist_ok=True)

    history = {
        "iterations": [],
        "cfg": cfg,
        "label": label,
    }

    logger.info(f"Starting {label} self-improvement: {n_iterations} iterations, "
                f"{rollouts_per_task} rollouts/task, {n_tasks} tasks")
    logger.info("=" * 60)

    for iteration in range(1, n_iterations + 1):
        t0 = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration}/{n_iterations}")
        logger.info(f"{'='*60}")

        iter_video_dir = None
        if video_dir:
            iter_video_dir = os.path.join(video_dir, f"iter{iteration}")

        # Step 1: Collect rollouts
        logger.info("Collecting rollouts...")
        all_trajectories = []
        for task_idx in range(n_tasks):
            env = _create_env(benchmark, task_idx, cfg)
            task_trajs = collect_rollouts(
                policy=_wrap_as_policy(model, processor, benchmark, task_idx,
                                       norm_stats=norm_stats,
                                       use_center_crop=cfg.get("use_center_crop", False)),
                env=env,
                task_id=task_idx,
                num_episodes=rollouts_per_task,
                max_steps=max_steps,
                extract_embeds_fn=extract_fn,
                save_video_dir=iter_video_dir,
                device=device,
            )
            all_trajectories.extend(task_trajs)
            env.close()

        batch_stats = compute_batch_statistics(all_trajectories)
        logger.info(f"Rollout stats: {batch_stats['n_success']}/{batch_stats['n_total']} success "
                     f"({batch_stats['success_rate']:.1%})")

        # Step 2: Compute SigLIP-T rewards
        logger.info("Computing rewards...")
        all_embeds = []
        success_mask = []
        task_id_list = []
        for traj in all_trajectories:
            if traj.siglip_embeds:
                embed_tensor = torch.stack(traj.siglip_embeds)
            else:
                embed_tensor = torch.zeros(max(len(traj.actions), 1), embed_dim)
            all_embeds.append(embed_tensor)
            success_mask.append(traj.success)
            task_id_list.append(traj.task_id)

        rewards, failed_indices = reward_model.compute_batch_rewards(
            all_embeds, success_mask, task_id_list
        )
        n_failed = len(failed_indices)
        avg_reward = sum(rewards[i] for i in failed_indices) / max(n_failed, 1)
        logger.info(f"Computed rewards for {n_failed} failed trajectories, avg={avg_reward:.4f}")

        # Step 3: GRPO policy update (token-level log-probs)
        if n_failed > 0 and any(len(all_trajectories[i].query_inputs) > 0 for i in failed_indices):
            logger.info("GRPO policy update...")

            advantages = compute_grpo_advantages(rewards, task_id_list)

            # Compute old log-probs (detached). Token IDs derived via argmax
            # since rollout doesn't store them. Per-query (not per-trajectory)
            # so importance sampling ratios remain valid.
            old_query_log_probs = {}
            stored_token_ids = {}
            for idx in failed_indices:
                traj = all_trajectories[idx]
                query_lps = []
                traj_token_ids = []
                for q_inputs in traj.query_inputs:
                    lp, token_ids = compute_token_log_probs(
                        model, q_inputs, response_ids=None,
                        unnorm_key=cfg.get("unnorm_key", "libero_spatial_no_noops"),
                        enable_grad=False,
                    )
                    query_lps.append(lp.detach())
                    traj_token_ids.append(token_ids)
                old_query_log_probs[idx] = query_lps
                stored_token_ids[idx] = traj_token_ids

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Compute new log-probs with per-query gradient accumulation (avoids OOM).
            failed_advantages = [advantages[i] for i in failed_indices]
            try:
                optimizer.zero_grad()
                total_loss = 0.0
                total_queries = 0
                all_ratios = []

                for traj_i, idx in enumerate(failed_indices):
                    traj = all_trajectories[idx]
                    adv = failed_advantages[traj_i]
                    old_lps = old_query_log_probs[idx]
                    for q_i, (q_inputs, token_ids) in enumerate(zip(traj.query_inputs, stored_token_ids[idx])):
                        lp, _ = compute_token_log_probs(
                            model, q_inputs, response_ids=token_ids,
                            unnorm_key=cfg.get("unnorm_key", "libero_spatial_no_noops"),
                            enable_grad=True,
                        )
                        # PPO clipped surrogate — per-query ratio (not per-trajectory!)
                        adv_t = torch.tensor(adv, device=device, dtype=lp.dtype).detach()
                        ratio = torch.exp(lp - old_lps[q_i])
                        if total_queries == 0:
                            logger.debug(f"First query: lp={lp.item():.2f} old_lp={old_lps[q_i].item():.2f} "
                                         f"ratio={ratio.item():.4f} adv={adv:.4f}")
                        all_ratios.append(ratio.item())
                        clipped = torch.clamp(ratio, 0.8, 1.2)
                        query_loss = -torch.min(ratio * adv_t, clipped * adv_t)
                        query_loss.backward()
                        total_loss += query_loss.item()
                        total_queries += 1
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # Average gradients and step
                if total_queries > 0:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad /= total_queries

                if grpo_cfg.get("max_grad_norm", 1.0) > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        grpo_cfg.get("max_grad_norm", 1.0),
                    ).item()
                else:
                    grad_norm = 0.0
                optimizer.step()
                trainer.step_count += 1

                metrics = {
                    "grpo_loss": total_loss / max(total_queries, 1),
                    "grad_norm": grad_norm,
                    "avg_ratio": sum(all_ratios) / max(len(all_ratios), 1),
                    "step": trainer.step_count,
                    "n_queries": total_queries,
                }
            except RuntimeError as e:
                logger.warning(f"GRPO backward failed: {e}")
                metrics = {"grpo_loss": 0.0, "step": trainer.step_count}
            logger.info(f"GRPO step {metrics.get('step',0)}: loss={metrics.get('grpo_loss',0):.4f} "
                        f"grad_norm={metrics.get('grad_norm',0):.4f} n_queries={metrics.get('n_queries',0)}")
        else:
            logger.info("Skipped GRPO (no failed trajectories with stored queries)")
            metrics = {"grpo_loss": 0.0, "step": trainer.step_count}

        elapsed = time.time() - t0
        iter_record = {
            "iteration": iteration,
            "batch_stats": batch_stats,
            "avg_reward": avg_reward,
            "grpo_metrics": metrics,
            "elapsed_seconds": elapsed,
            "vram_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        }
        history["iterations"].append(iter_record)

        logger.info(f"Iteration {iteration} complete in {elapsed:.1f}s")
        logger.info(f"  Success rate: {batch_stats['success_rate']:.1%}")
        logger.info(f"  GRPO loss: {metrics['grpo_loss']:.4f}")
        if torch.cuda.is_available():
            logger.info(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # Save checkpoint — only LoRA weights to avoid 15GB full model
        ckpt = {
            "iteration": iteration,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()
                                 if "lora" in k.lower()},
            "reward_state_dict": reward_model.state_dict(),
            "history": history,
        }
        ckpt_path = os.path.join(checkpoint_dir, f"siglip_t_iter{iteration}.pt")
        torch.save(ckpt, ckpt_path)
        logger.info(f"  Checkpoint saved: {ckpt_path}")

        # Save running history as JSON
        history_path = os.path.join(checkpoint_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(_make_serializable(history), f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("SigLIP-T training complete!")
    logger.info(f"Final success rate: {history['iterations'][-1]['batch_stats']['success_rate']:.1%}")
    logger.info("=" * 60)
    return history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_vla(cfg: dict, device: torch.device):
    """Load VLA model with optional 4-bit quantization and LoRA."""
    model_name = cfg.get("model_name", "Sylvest/OpenVLA-AC-PD-1traj-libero-spatial")

    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        logger.info(f"Loading {model_name}...")

        load_in_4bit = cfg.get("load_in_4bit", True)

        kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            logger.info("Using 4-bit quantization (QLoRA)")

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(model_name, **kwargs)

        # Get vision backbone reference
        vb = getattr(model, "vision_backbone", None)
        if vb is None and hasattr(model, "model"):
            vb = getattr(model.model, "vision_backbone", None)

        # Download fine-tuned normalization statistics
        from huggingface_hub import hf_hub_download
        try:
            ds_path = hf_hub_download(repo_id=model_name, filename="dataset_statistics.json")
            with open(ds_path, "r") as f:
                norm_stats = json.load(f)
            model.norm_stats = norm_stats
            logger.info(f"Loaded fine-tuned norm_stats keys: {list(norm_stats.keys())}")
        except Exception as e:
            logger.warning(f"Could not load dataset_statistics.json: {e}")
            norm_stats = getattr(model, "norm_stats", None)

        # Enable LoRA for training
        try:
            from peft import LoraConfig, get_peft_model
            if load_in_4bit:
                from peft import prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model)
            lora_config = LoraConfig(
                r=cfg.get("lora_r", 16),
                lora_alpha=cfg.get("lora_alpha", 32),
                target_modules=cfg.get("lora_targets", ["q_proj", "v_proj"]),
                lora_dropout=cfg.get("lora_dropout", 0.05),
            )
            model = get_peft_model(model, lora_config)
            model.gradient_checkpointing_enable()
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            logger.info(f"LoRA enabled: {trainable:,} / {total:,} params trainable "
                        f"({100 * trainable / total:.2f}%)")
        except ImportError:
            logger.warning("peft not installed — training full model (not recommended)")

        if vb and hasattr(vb, "fused_featurizer"):
            logger.info(f"SigLIP embed_dim: {vb.fused_featurizer.embed_dim}")

        return model, processor, norm_stats

    except Exception as e:
        logger.warning(f"Could not load {model_name}: {e}")
        logger.info("Falling back to MLPPolicy for development/testing")
        from vla_coach.policy import MLPPolicy
        model = MLPPolicy().to(device)
        return model, None, None


def _create_env(benchmark, task_idx: int, cfg: dict):
    """Create a LIBERO environment for a given task."""
    from vla_coach.utils import create_env
    bddl_file_path = benchmark.get_task_bddl_file_path(task_idx)
    env = create_env(
        bddl_file=bddl_file_path,
        img_h=cfg.get("img_h", 256),
        img_w=cfg.get("img_w", 256),
    )
    env.seed(0)  # Match official eval — affects object positions
    return env


def _center_crop_pil(image, crop_scale=0.9):
    """Center crop a PIL image to crop_scale area, then resize to 224x224."""
    w, h = image.size
    new_h = int(h * math.sqrt(crop_scale))
    new_w = int(w * math.sqrt(crop_scale))
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    image = image.crop((left, top, left + new_w, top + new_h))
    image = image.resize((224, 224), Image.LANCZOS)
    return image


def _get_libero_image(obs):
    """Extract agentview image (rotated 180° to match training)."""
    img = obs["agentview_image"]
    return img[::-1, ::-1].copy()


def _get_libero_wrist_image(obs):
    """Extract wrist camera image (rotated 180° to match training)."""
    img = obs["robot0_eye_in_hand_image"]
    return img[::-1, ::-1].copy()


def _process_action(action):
    """Normalize gripper [0,1]→[-1,+1], binarize, invert sign."""
    action = action.copy()
    action[..., -1] = 2 * action[..., -1] - 1
    action[..., -1] = np.sign(action[..., -1])
    action[..., -1] *= -1.0
    return action


class _VLAPolicyWrapper:
    """Wraps a VLA model for rollout collection.

    Handles image preprocessing, 8-step action chunking, gripper post-processing,
    and captures query data for token-level GRPO.
    """

    def __init__(self, model, processor, task_instruction,
                 norm_stats=None, unnorm_key="libero_spatial_no_noops",
                 use_center_crop=False, use_wrist=False):
        self.model = model
        self.processor = processor
        self.task_instruction = task_instruction
        self.norm_stats = norm_stats
        self.unnorm_key = unnorm_key
        self.use_center_crop = use_center_crop
        self.use_wrist = use_wrist
        self.action_queue = deque()
        self._last_query_data = None  # cached query data from most recent model call

    def reset(self):
        self.action_queue.clear()
        self._last_query_data = None

    def get_action(self, obs: dict) -> np.ndarray:
        action, _ = self.get_action_with_tokens(obs, step=0)
        return action

    def get_action_with_tokens(self, obs: dict, step: int = 0):
        """Run VLA inference, returning (action, query_data_or_None)."""
        if self.processor is None:
            return self.model.get_action(obs), None

        if len(self.action_queue) > 0:
            return self.action_queue.popleft(), None

        from PIL import Image

        agentview = _get_libero_image(obs)
        if self.use_center_crop:
            agentview_pil = _center_crop_pil(Image.fromarray(agentview))
        else:
            agentview_pil = Image.fromarray(agentview).resize((224, 224), Image.LANCZOS)

        prompt = f"In: What action should the robot take to {self.task_instruction.lower()}?\nOut:"

        inputs = self.processor(prompt, agentview_pil).to(self.model.device, dtype=torch.bfloat16)

        if self.use_wrist and "robot0_eye_in_hand_image" in obs:
            wrist = _get_libero_wrist_image(obs)
            if self.use_center_crop:
                wrist_pil = _center_crop_pil(Image.fromarray(wrist))
            else:
                wrist_pil = Image.fromarray(wrist).resize((224, 224), Image.LANCZOS)
            wrist_inputs = self.processor(prompt, wrist_pil).to(self.model.device, dtype=torch.bfloat16)
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
            )

        # Run inference
        with torch.no_grad():
            action_result, _hidden_states = self.model.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False,
            )

        # Handle action chunks (8, 7) or single actions (7,)
        # Handle action chunks (8, 7) or single actions (7,)
        action_chunk = np.array(action_result, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk.reshape(1, -1)
        chunk_len = len(action_chunk)

        for i in range(chunk_len):
            processed = _process_action(action_chunk[i])
            self.action_queue.append(processed)

        # Capture query data for GRPO (CPU to save GPU memory)
        query_data = {
            "inputs": {
                "input_ids": inputs["input_ids"].cpu(),
                "pixel_values": inputs["pixel_values"].cpu(),
                "attention_mask": inputs["attention_mask"].cpu(),
            },
            "step_range": (step, step + chunk_len),
        }

        return self.action_queue.popleft(), query_data


def _wrap_as_policy(model, processor, benchmark, task_idx,
                    norm_stats=None, use_center_crop=False, use_wrist=False):
    """Create a policy wrapper for rollout collection."""
    task = benchmark.get_task(task_idx)
    return _VLAPolicyWrapper(
        model=model,
        processor=processor,
        task_instruction=task.language,
        norm_stats=norm_stats,
        use_center_crop=use_center_crop,
        use_wrist=use_wrist,
    )


def _make_serializable(obj):
    """Convert numpy/torch types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(description="SigLIP-T self-improvement training")
    parser.add_argument("--config", type=str, default="configs/train_rl.yaml")
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--use-temporal", action="store_true", default=None)
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument("--save-videos", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.benchmark is not None:
        cfg["benchmark"] = args.benchmark
    if args.n_iterations is not None:
        cfg["n_iterations"] = args.n_iterations
    if args.use_temporal:
        cfg["use_temporal"] = True
    if args.no_temporal:
        cfg["use_temporal"] = False
    if args.save_videos:
        cfg["save_videos"] = True
    if args.seed is not None:
        cfg["seed"] = args.seed

    history = train_rl(cfg)
    final_sr = history["iterations"][-1]["batch_stats"]["success_rate"]
    print(f"\nFinal success rate: {final_sr:.1%}")


if __name__ == "__main__":
    main()
