#!/usr/bin/env python3
"""Publication-quality plots for SigLIP-T results.

Generates clean, minimal plots inspired by scaling-law paper style:
- Muted color palette, whitegrid background
- Proper axis labels and tight layouts
- Confidence-band style where applicable
- Separate per-task breakdown

Usage:
    python scripts/plot_publication.py results/history.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

COLORS = {
    "primary": "#2171b5",
    "secondary": "#cb181d",
    "tertiary": "#238b45",
    "muted": "#807dba",
    "gray": "#636363",
    "light": "#bdbdbd",
}

# Per-task data extracted from training logs (Task × Iteration matrix)
TASK_SUCCESS = np.array([
    # Task 0  Task 1  Task 2  Task 3  Task 4
    [2/8,     0/8,    8/8,    7/8,    5/8],   # iter 1
    [3/8,     0/8,    8/8,    8/8,    3/8],   # iter 2
    [2/8,     0/8,    8/8,    7/8,    4/8],   # iter 3
    [2/8,     0/8,    7/8,    5/8,    4/8],   # iter 4
    [2/8,     0/8,    8/8,    6/8,    4/8],   # iter 5
    [2/8,     0/8,    6/8,    7/8,    4/8],   # iter 6
    [4/8,     0/8,    8/8,    8/8,    3/8],   # iter 7
    [3/8,     0/8,    8/8,    5/8,    4/8],   # iter 8
    [2/8,     0/8,    7/8,    6/8,    5/8],   # iter 9
    [2/8,     0/8,    8/8,    7/8,    5/8],   # iter 10
])

TASK_LABELS = [
    "Task 0\n(pick & place)",
    "Task 1\n(hard)",
    "Task 2\n(easy)",
    "Task 3\n(medium-easy)",
    "Task 4\n(medium)",
]


def load_history(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def fig1_success_rate(iters, out_dir: Path):
    """Main result: success rate over iterations with baseline reference."""
    xs = [it["iteration"] for it in iters]
    ys = [it["batch_stats"]["success_rate"] * 100 for it in iters]
    baseline = ys[0]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, ys, "-o", color=COLORS["primary"], markersize=5, linewidth=1.8,
            label="SigLIP-T + GRPO", zorder=3)
    ax.axhline(baseline, color=COLORS["light"], linestyle="--", linewidth=1.2,
               label=f"BC baseline ({baseline:.0f}%)", zorder=1)
    # SRPO reference
    ax.axhline(99.2, color=COLORS["secondary"], linestyle=":", linewidth=1.2,
               label="SRPO (published, 99.2%)", zorder=1)

    ax.set_xlabel("GRPO Iteration")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(0.5, len(xs) + 0.5)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("SigLIP-T Self-Improvement: No Trend")

    plt.tight_layout()
    out = out_dir / "fig1_success_rate.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


def fig2_per_task(out_dir: Path):
    """Per-task success rates: grouped bar chart across iterations."""
    n_iters, n_tasks = TASK_SUCCESS.shape
    means = TASK_SUCCESS.mean(axis=0) * 100
    stds = TASK_SUCCESS.std(axis=0) * 100

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(n_tasks)
    bars = ax.bar(x, means, yerr=stds, width=0.6, color=COLORS["primary"],
                  edgecolor="white", linewidth=0.5, capsize=3, error_kw={"linewidth": 1.2})

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{m:.0f}%", ha="center", va="bottom", fontsize=9, color=COLORS["gray"])

    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS, fontsize=8)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Per-Task Average Success Rate (±1 std)")

    plt.tight_layout()
    out = out_dir / "fig2_per_task.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


def fig3_per_task_over_time(out_dir: Path):
    """Per-task success rate curves over iterations."""
    n_iters, n_tasks = TASK_SUCCESS.shape
    xs = np.arange(1, n_iters + 1)
    task_colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for t in range(n_tasks):
        ax.plot(xs, TASK_SUCCESS[:, t] * 100, "-o", color=task_colors[t],
                markersize=4, linewidth=1.4, label=TASK_LABELS[t].replace("\n", " "),
                alpha=0.85)

    ax.set_xlabel("GRPO Iteration")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 110)
    ax.set_xlim(0.5, n_iters + 0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8,
              framealpha=0.9)
    ax.set_title("Per-Task Success Rate Over Training")

    plt.tight_layout()
    out = out_dir / "fig3_per_task_time.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


def fig4_reward_signal(iters, out_dir: Path):
    """Average failed trajectory reward over iterations — shows weak signal."""
    xs = [it["iteration"] for it in iters]
    rewards = [it.get("avg_reward", 0) for it in iters]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, rewards, "-s", color=COLORS["muted"], markersize=5, linewidth=1.8,
            label="SigLIP-T reward (failed trajs)")
    ax.fill_between(xs, 0, rewards, alpha=0.12, color=COLORS["muted"])
    ax.axhline(0.6, color=COLORS["light"], linestyle=":", label="Max possible (0.6)")
    ax.axhline(np.mean(rewards), color=COLORS["gray"], linestyle="--", linewidth=0.8,
               label=f"Mean: {np.mean(rewards):.3f}")

    ax.set_xlabel("GRPO Iteration")
    ax.set_ylabel("Avg SigLIP-T Reward")
    ax.set_ylim(0, 0.7)
    ax.set_xlim(0.5, len(xs) + 0.5)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_title("Reward Signal Quality: Too Weak to Drive Learning")

    plt.tight_layout()
    out = out_dir / "fig4_reward_signal.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


def fig5_grpo_diagnostics(iters, out_dir: Path):
    """Two-panel: GRPO loss and gradient norm — shows pipeline is working."""
    xs = [it["iteration"] for it in iters]
    losses = [it["grpo_metrics"]["grpo_loss"] for it in iters]
    grads = [it["grpo_metrics"]["grad_norm"] for it in iters]
    ratios = [it["grpo_metrics"]["avg_ratio"] for it in iters]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # Loss
    ax1.plot(xs, losses, "-o", color=COLORS["secondary"], markersize=4, linewidth=1.5)
    ax1.set_xlabel("GRPO Iteration")
    ax1.set_ylabel("GRPO Loss")
    ax1.set_title("Policy Loss (finite, non-trivial)")
    ax1.set_xlim(0.5, len(xs) + 0.5)

    # Grad norm
    ax2.plot(xs, grads, "-^", color=COLORS["tertiary"], markersize=4, linewidth=1.5,
             label="Grad norm")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(xs, ratios, "-s", color=COLORS["muted"], markersize=3, linewidth=1.2,
                  alpha=0.7, label="PPO ratio")
    ax2.set_xlabel("GRPO Iteration")
    ax2.set_ylabel("Gradient Norm", color=COLORS["tertiary"])
    ax2_twin.set_ylabel("Avg PPO Ratio", color=COLORS["muted"])
    ax2_twin.spines["right"].set_visible(True)
    ax2_twin.spines["top"].set_visible(False)
    ax2.set_title("LoRA Gradients & PPO Ratios")
    ax2.set_xlim(0.5, len(xs) + 0.5)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    plt.tight_layout()
    out = out_dir / "fig5_grpo_diagnostics.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


def fig6_resource_comparison(out_dir: Path):
    """Bar chart comparing resources: SRPO vs SigLIP-T."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    methods = ["SRPO\n(V-JEPA)", "SigLIP-T\n(ours)"]
    srpo_color = COLORS["gray"]
    ours_color = COLORS["primary"]

    # VRAM
    ax = axes[0]
    vram = [80, 15.2]  # SRPO uses 4×A100 (~80GB total), we use 15GB on 1×A10G
    bars = ax.bar(methods, vram, color=[srpo_color, ours_color], width=0.5, edgecolor="white")
    ax.set_ylabel("VRAM (GB)")
    ax.set_title("GPU Memory", pad=14)
    for bar, v in zip(bars, vram):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.0f} GB",
                ha="center", fontsize=9)

    # Extra params
    ax = axes[1]
    params = [1100, 1.6]  # 1.1B vs 1.6M
    bars = ax.bar(methods, params, color=[srpo_color, ours_color], width=0.5, edgecolor="white")
    ax.set_ylabel("Extra Parameters (M)")
    ax.set_title("Reward Encoder Size", pad=14)
    ax.set_yscale("log")
    for bar, v in zip(bars, params):
        label = f"{v:.0f}M" if v >= 1 else f"{v:.1f}M"
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.3, label,
                ha="center", fontsize=9)

    # Hardware
    ax = axes[2]
    gpus = ["4× A100\n80GB", "1× A10G\n24GB"]
    cost_per_hr = [16.0, 1.10]  # approximate $/hr
    bars = ax.bar(gpus, cost_per_hr, color=[srpo_color, ours_color], width=0.5, edgecolor="white")
    ax.set_ylabel("Cost ($/hr)")
    ax.set_title("Hardware Cost", pad=14)
    for bar, v in zip(bars, cost_per_hr):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"${v:.2f}/hr",
                ha="center", fontsize=9)

    plt.tight_layout()
    out = out_dir / "fig6_resources.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


def main(history_path: str):
    history = load_history(history_path)
    iters = history["iterations"]
    out_dir = Path(history_path).parent / "figures"
    out_dir.mkdir(exist_ok=True)

    print(f"Generating publication plots → {out_dir}/")
    fig1_success_rate(iters, out_dir)
    fig2_per_task(out_dir)
    fig3_per_task_over_time(out_dir)
    fig4_reward_signal(iters, out_dir)
    fig5_grpo_diagnostics(iters, out_dir)
    fig6_resource_comparison(out_dir)
    print("Done! 6 figures generated.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_publication.py results/history.json")
        sys.exit(1)
    main(sys.argv[1])
