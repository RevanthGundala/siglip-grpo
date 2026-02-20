"""LIBERO rollout collection for RL training."""

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from siglip_grpo.utils import setup_logging

logger = setup_logging()


class Trajectory:
    """A single rollout trajectory."""

    __slots__ = ["frames", "actions", "siglip_embeds", "success", "total_reward",
                 "task_id", "query_inputs", "query_response_ids", "query_steps"]

    def __init__(self):
        self.frames: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.siglip_embeds: List[torch.Tensor] = []
        self.success: bool = False
        self.total_reward: float = 0.0
        self.task_id: int = -1
        # For token-level GRPO: stored on CPU
        self.query_inputs: list = []      # list of (input_ids, pixel_values, attention_mask) per model query
        self.query_response_ids: list = []  # list of response token id tensors per model query
        self.query_steps: list = []       # list of (start_step, end_step) per model query


def collect_rollouts(
    policy,
    env,
    task_id: int,
    num_episodes: int,
    max_steps: int = 300,
    extract_embeds_fn=None,
    save_video_dir: Optional[str] = None,
    device: torch.device = None,
) -> List[Trajectory]:
    """Collect rollout trajectories, optionally extracting SigLIP embeddings."""
    trajectories = []
    device = device or torch.device("cpu")

    for ep in range(num_episodes):
        traj = Trajectory()
        traj.task_id = task_id

        obs = env.reset()
        policy.reset()
        done = False
        video_frames = []

        # Let objects stabilize (matches official eval)
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

        for step in range(max_steps):
            if "agentview_image" in obs:
                frame = obs["agentview_image"][::-1, ::-1].copy()
                traj.frames.append(frame)
                if save_video_dir is not None:
                    video_frames.append(frame.copy())

            action, query_data = policy.get_action_with_tokens(obs, step)
            traj.actions.append(action)

            if query_data is not None:
                traj.query_inputs.append(query_data["inputs"])
                traj.query_response_ids.append(None)
                traj.query_steps.append(query_data["step_range"])

            obs, reward, done, info = env.step(action.tolist() if hasattr(action, 'tolist') else action)
            traj.total_reward += reward

            if done:
                break

        traj.success = bool(env.is_success() if hasattr(env, "is_success") else done)

        if extract_embeds_fn is not None and traj.frames:
            traj.siglip_embeds = _extract_trajectory_embeds(
                traj.frames, extract_embeds_fn, device
            )

        if save_video_dir is not None and video_frames:
            _save_video(video_frames, save_video_dir, task_id, ep, traj.success)

        trajectories.append(traj)
        status = "✓" if traj.success else "✗"
        logger.debug(
            f"  Task {task_id} ep {ep}: {status} ({len(traj.actions)} steps, reward={traj.total_reward:.2f})"
        )

    n_success = sum(t.success for t in trajectories)
    logger.info(
        f"Task {task_id}: {n_success}/{num_episodes} successes "
        f"({100 * n_success / max(num_episodes, 1):.0f}%)"
    )
    return trajectories


def _extract_trajectory_embeds(
    frames: List[np.ndarray],
    extract_fn,
    device: torch.device,
    batch_size: int = 32,
) -> List[torch.Tensor]:
    """Extract embeddings for all frames in a trajectory, batched for efficiency."""
    all_embeds = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i : i + batch_size]
        batch_tensor = torch.stack(
            [torch.from_numpy(f).float().permute(2, 0, 1) / 255.0 for f in batch_frames]
        ).to(device)
        embeds = extract_fn(batch_tensor)  # (B, D)
        all_embeds.append(embeds.cpu())
    return [e for e in torch.cat(all_embeds, dim=0)]


def _save_video(frames: List[np.ndarray], save_dir: str, task_id: int, ep: int, success: bool):
    """Save rollout frames as an MP4 video."""
    os.makedirs(save_dir, exist_ok=True)
    label = "success" if success else "fail"
    path = os.path.join(save_dir, f"task{task_id}_ep{ep}_{label}.mp4")

    try:
        import cv2
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 20, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        logger.debug(f"  Saved video: {path}")
    except ImportError:
        np_path = path.replace(".mp4", ".npy")
        np.save(np_path, np.array(frames))
        logger.debug(f"  Saved frames: {np_path} (cv2 not available)")


def compute_batch_statistics(trajectories: List[Trajectory]) -> Dict:
    """Compute summary statistics for a batch of trajectories."""
    n_total = len(trajectories)
    n_success = sum(t.success for t in trajectories)
    avg_len = np.mean([len(t.actions) for t in trajectories]) if trajectories else 0
    avg_reward = np.mean([t.total_reward for t in trajectories]) if trajectories else 0

    return {
        "n_total": n_total,
        "n_success": n_success,
        "success_rate": n_success / max(n_total, 1),
        "avg_episode_length": float(avg_len),
        "avg_total_reward": float(avg_reward),
    }
