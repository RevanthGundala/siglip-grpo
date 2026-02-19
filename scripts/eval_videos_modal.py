"""Capture demo rollout videos on Modal for blog/paper.

Runs the baseline VLA (no RL checkpoint) on selected tasks and saves
MP4 videos. Captures a mix of successes and failures across tasks with
varying difficulty.

Usage:
    modal run scripts/eval_videos_modal.py
    # Then download:
    modal volume get vla-coach-rl-results siglip_t/videos/eval/ results/videos/
"""

import modal

# Reuse the same image + volume from training
from src.vla_coach.train_rl_modal import image, vol

app = modal.App("vla-coach-eval", image=image)


@app.function(
    gpu="A10G",
    timeout=3600,  # 1 hour max
    volumes={"/root/results": vol},
    memory=32768,
)
def eval_videos_remote():
    """Capture rollout videos + compute rewards for blog figures."""
    import os
    import sys
    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/opt/openvla-oft")

    import json
    import logging
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger(__name__)

    import numpy as np
    import torch
    import yaml

    log.info(f"Device: {torch.cuda.get_device_name()}")

    # Load config
    with open("/root/configs/train_rl.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda")

    # Override for eval
    video_dir = "/root/results/siglip_t/videos/eval"
    os.makedirs(video_dir, exist_ok=True)

    # Load model (same as training)
    from vla_coach.train_rl import _load_vla, _create_env, _wrap_as_policy
    model, processor, norm_stats = _load_vla(cfg, device=device)
    log.info("VLA loaded.")

    # Determine benchmark
    from libero.libero.benchmark import get_benchmark
    BENCHMARK_MAP = {
        "libero_spatial": "LIBERO_SPATIAL",
        "libero_object": "LIBERO_OBJECT",
        "libero_goal": "LIBERO_GOAL",
        "libero_10": "LIBERO_10",
    }
    benchmark_name = cfg.get("benchmark", "libero_spatial")
    benchmark = get_benchmark(BENCHMARK_MAP[benchmark_name])(cfg.get("task_order_index", 0))

    # Run rollouts: 5 tasks × 4 episodes = 20 videos
    from vla_coach.rollout import collect_rollouts
    n_tasks = 5
    rollouts_per_task = 4
    max_steps = cfg.get("max_steps", 220)

    all_trajs = []
    for task_idx in range(n_tasks):
        env = _create_env(benchmark, task_idx, cfg)
        policy = _wrap_as_policy(model, processor, benchmark, task_idx,
                                  norm_stats=norm_stats,
                                  use_center_crop=cfg.get("use_center_crop", False))
        task_trajs = collect_rollouts(
            policy=policy,
            env=env,
            task_id=task_idx,
            num_episodes=rollouts_per_task,
            max_steps=max_steps,
            save_video_dir=video_dir,
        )
        n_succ = sum(1 for t in task_trajs if t.success)
        log.info(f"Task {task_idx}: {n_succ}/{rollouts_per_task} successes ({n_succ/rollouts_per_task:.0%})")
        all_trajs.extend(task_trajs)
        env.close()

    n_success = sum(1 for t in all_trajs if t.success)
    n_total = len(all_trajs)
    log.info(f"Rollouts: {n_success}/{n_total} successes ({n_success/n_total:.0%})")

    # --- Compute rewards (all-frame AND last-20) for "reward is blind" figure ---
    log.info("Computing SigLIP embeddings for reward comparison...")
    from vla_coach.reward import SigLIPTReward, extract_siglip_embeddings
    import torchvision.transforms as T
    from PIL import Image

    siglip_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    all_frame_embeds = []
    for i, traj in enumerate(all_trajs):
        frames_pil = [Image.fromarray(f) for f in traj.frames]
        batch_size = 16
        frame_embs = []
        for b_start in range(0, len(frames_pil), batch_size):
            batch = frames_pil[b_start:b_start + batch_size]
            imgs = torch.stack([siglip_transform(f) for f in batch]).to(device)
            emb = extract_siglip_embeddings(model, imgs)
            frame_embs.append(emb.float().cpu())
        all_frame_embeds.append(torch.cat(frame_embs, dim=0))
        if (i + 1) % 5 == 0:
            log.info(f"  Embedded {i+1}/{n_total} trajectories")

    success_mask = [t.success for t in all_trajs]
    traj_task_ids = [t.task_id for t in all_trajs]

    reward_allframe = SigLIPTReward(embed_dim=all_frame_embeds[0].shape[1],
                                     use_temporal=False, last_k_frames=0, device=torch.device("cpu"))
    reward_lastk = SigLIPTReward(embed_dim=all_frame_embeds[0].shape[1],
                                  use_temporal=False, last_k_frames=20, device=torch.device("cpu"))

    rewards_af, _ = reward_allframe.compute_batch_rewards(all_frame_embeds, success_mask, traj_task_ids)
    rewards_lk, _ = reward_lastk.compute_batch_rewards(all_frame_embeds, success_mask, traj_task_ids)

    eval_data = []
    for i, traj in enumerate(all_trajs):
        eval_data.append({
            "task_id": traj.task_id,
            "success": traj.success,
            "n_steps": len(traj.frames),
            "reward_allframe": float(rewards_af[i]),
            "reward_lastk20": float(rewards_lk[i]),
        })
        status = "✓" if traj.success else "✗"
        log.info(f"  Task {traj.task_id} ep {i % rollouts_per_task}: {status}  "
                 f"steps={len(traj.frames)}  "
                 f"rwd_all={rewards_af[i]:.4f}  rwd_lk20={rewards_lk[i]:.4f}")

    eval_meta_path = os.path.join(video_dir, "eval_metadata.json")
    with open(eval_meta_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    log.info(f"Saved eval metadata: {eval_meta_path}")

    log.info("Saved files:")
    for fname in sorted(os.listdir(video_dir)):
        fpath = os.path.join(video_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        log.info(f"  {fname} ({size_kb:.0f} KB)")

    vol.commit()
    return {"n_success": n_success, "n_total": n_total, "eval_data": eval_data}


@app.local_entrypoint()
def main():
    print("Launching eval video capture on Modal A10G...")
    result = eval_videos_remote.remote()
    print(f"\nEval complete: {result['n_success']}/{result['n_total']} successes")
    print("\nDownload videos with:")
    print("  modal volume get vla-coach-rl-results siglip_t/videos/eval/ results/videos/")
