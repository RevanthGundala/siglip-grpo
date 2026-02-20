# VLA-Coach: Self-Referential RL for Vision-Language-Action Models

> *Can a VLA be its own reward model?* We replace SRPO's 1.1B-parameter V-JEPA world model with the VLA's own SigLIP encoder augmented by a lightweight temporal MLP (~200K params), achieving self-improvement at zero extra VRAM cost.

## Abstract

Vision-Language-Action (VLA) models can be improved through reinforcement learning, but existing methods like [SRPO](https://arxiv.org/abs/2502.09466) require a separate V-JEPA world model (1.1B params, ~5GB VRAM) to compute dense rewards. **VLA-Coach** shows that the VLA's own SigLIP vision encoder — already computed during inference — can serve as a reward signal when paired with a GR00T-style temporal embedding (sinusoidal + learned MLP). This **SigLIP-T** reward model replaces V-JEPA at zero marginal VRAM cost, using only ~200K trainable parameters.

The self-improvement loop:
1. **Rollout** the VLA on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) tasks
2. **Extract SigLIP embeddings** per frame (free — already computed during inference)
3. **Add temporal embeddings** (GR00T-style sinusoidal → learned MLP)
4. **Compute dense rewards** via distance to successful trajectory cluster
5. **GRPO policy gradient update** → repeat

## Method

```
Frame t ──→ SigLIP ViT-SO400M ──→ (1152-dim) ──+──→ Mean Pool ──→ Trajectory Embedding
                                                  │
              Frame index t ──→ Sinusoidal(256) ──→ MLP(256→1152) ──┘
                                   (parameter-free)    (~200K params)
```

**Reward computation** (matches SRPO exactly):
- Success trajectories → reward = 1.0
- Failed trajectories → reward = sigmoid(distance to success cluster) ∈ [0, 0.6]
- GRPO advantage = (reward − group_mean) / group_std (per task)

**Training**: QLoRA (4-bit, r=16) on the VLA's language model, with PPO-clipped surrogate loss and per-query gradient accumulation to fit in A10G VRAM.

## Reproduce

### Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10.

```bash
git clone --recurse-submodules https://github.com/<your-org>/siglip-grpo.git
cd siglip-grpo
pip install -e ".[rl]"
```

### Train locally

```bash
PYTHONPATH=src python -m siglip_grpo.train_rl --config configs/train_rl.yaml
```

### Train on Modal (A10G GPU)

```bash
pip install modal
modal run src/siglip_grpo/train_rl_modal.py --config configs/train_rl.yaml

# SigLIP-only ablation (no temporal embedding)
modal run src/siglip_grpo/train_rl_modal.py --config configs/train_rl.yaml --no-temporal
```

### Configuration

See [`configs/train_rl.yaml`](configs/train_rl.yaml) for all hyperparameters (benchmark, LoRA rank, GRPO clipping, reward pooling, etc.).

## Project Structure

```
siglip-grpo/
├── src/siglip_grpo/
│   ├── reward.py              # SigLIP-T reward: embedding → distance → sigmoid
│   ├── temporal_embedding.py  # GR00T-style sinusoidal + MLP (~200K params)
│   ├── grpo.py                # GRPO trainer, PPO loss, token-level log-probs
│   ├── rollout.py             # LIBERO rollout collection + trajectory storage
│   ├── train_rl.py            # Main self-improvement loop
│   ├── train_rl_modal.py      # Modal wrapper (zero logic, container setup only)
│   └── utils.py               # Config, logging, environment helpers
├── configs/train_rl.yaml      # Training hyperparameters
├── scripts/                   # Plotting and video capture utilities
├── tests/test_pipeline.py     # Unit tests (CPU-only, no GPU needed)
└── vendor/                    # LIBERO submodules
```

## Citation

```bibtex
@misc{vlacoach2025,
  title   = {VLA-Coach: Self-Referential Reinforcement Learning for Vision-Language-Action Models},
  author  = {<authors>},
  year    = {2025},
  url     = {https://github.com/<your-org>/siglip-grpo}
}
```

## License

MIT
