#!/usr/bin/env python3

import json
import sys

import matplotlib.pyplot as plt
import numpy as np


def plot_training(history_path: str, out_path: str = None):
    with open(history_path) as f:
        history = json.load(f)

    iters = history["iterations"]
    n = len(iters)
    xs = [it["iteration"] for it in iters]

    success_rates = [it["batch_stats"]["success_rate"] for it in iters]
    grpo_losses = [it["grpo_metrics"].get("grpo_loss", 0) for it in iters]
    grad_norms = [it["grpo_metrics"].get("grad_norm", 0) for it in iters]
    avg_rewards = [it.get("avg_reward", 0) for it in iters]
    vram = [it.get("vram_gb", 0) for it in iters]
    elapsed = [it.get("elapsed_seconds", 0) for it in iters]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("SigLIP-T Self-Improvement Training", fontsize=14, fontweight="bold")

    # 1. Success rate
    ax = axes[0, 0]
    ax.plot(xs, success_rates, "b-o", markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=success_rates[0], color="gray", linestyle="--", alpha=0.5, label=f"Baseline: {success_rates[0]:.1%}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. GRPO loss
    ax = axes[0, 1]
    valid_losses = [(x, l) for x, l in zip(xs, grpo_losses) if l != 0 and not np.isnan(l)]
    if valid_losses:
        ax.plot(*zip(*valid_losses), "r-o", markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("GRPO Loss")
    ax.set_title("GRPO Loss")
    ax.grid(True, alpha=0.3)

    # 3. Gradient norm
    ax = axes[0, 2]
    valid_grads = [(x, g) for x, g in zip(xs, grad_norms) if g > 0]
    if valid_grads:
        ax.plot(*zip(*valid_grads), "g-o", markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Grad Norm")
    ax.set_title("LoRA Gradient Norm")
    ax.grid(True, alpha=0.3)

    # 4. Average failed trajectory reward
    ax = axes[1, 0]
    ax.plot(xs, avg_rewards, "m-o", markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg Reward (failed)")
    ax.set_title("Avg Failed Trajectory Reward")
    ax.set_ylim(-0.05, 0.65)
    ax.grid(True, alpha=0.3)

    # 5. Time per iteration
    ax = axes[1, 1]
    ax.bar(xs, [e / 60 for e in elapsed], color="steelblue", alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Minutes")
    ax.set_title("Time per Iteration")
    ax.grid(True, alpha=0.3)

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis("off")
    total_time = sum(elapsed) / 3600
    summary = (
        f"Iterations: {n}\n"
        f"Initial success: {success_rates[0]:.1%}\n"
        f"Final success: {success_rates[-1]:.1%}\n"
        f"Δ success: {success_rates[-1] - success_rates[0]:+.1%}\n"
        f"Total time: {total_time:.1f} hrs\n"
        f"Est. cost: ${total_time * 1.10:.2f}\n"
        f"Avg GRPO loss: {np.nanmean([l for l in grpo_losses if l != 0]):.4f}\n"
        f"Avg grad norm: {np.mean([g for g in grad_norms if g > 0]):.2f}"
    )
    ax.text(0.1, 0.5, summary, transform=ax.transAxes, fontsize=12,
            verticalalignment="center", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title("Summary")

    plt.tight_layout()
    if out_path is None:
        out_path = history_path.replace(".json", "_plots.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_training.py <history.json> [output.png]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else None
    plot_training(sys.argv[1], out)
