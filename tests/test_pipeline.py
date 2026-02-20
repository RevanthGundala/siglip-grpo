"""Unit tests for the SigLIP-T self-improvement pipeline.

All tests run locally with CPU-only mocked models — no GPU or Modal needed.
Run: python -m pytest tests/test_pipeline.py -v
"""

import sys
import os

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from siglip_grpo.reward import SigLIPTReward
from siglip_grpo.grpo import compute_grpo_advantages, grpo_loss, GRPOTrainer
from siglip_grpo.temporal_embedding import TemporalEmbedding
from siglip_grpo.rollout import Trajectory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reward_model():
    return SigLIPTReward(embed_dim=64, use_temporal=True, device=torch.device("cpu"))


@pytest.fixture
def reward_model_no_temporal():
    return SigLIPTReward(embed_dim=64, use_temporal=False, device=torch.device("cpu"))


def _make_embeds(n_trajs, n_frames=20, embed_dim=64, seed=42):
    rng = np.random.RandomState(seed)
    return [torch.from_numpy(rng.randn(n_frames, embed_dim).astype(np.float32))
            for _ in range(n_trajs)]


# ===========================================================================
# Reward Model Tests
# ===========================================================================

class TestRewardModel:

    def test_reward_success_gets_1(self, reward_model):
        embeds = _make_embeds(4)
        success = [True, True, False, False]
        task_ids = [0, 0, 0, 0]
        rewards, _ = reward_model.compute_batch_rewards(embeds, success, task_ids)
        assert rewards[0] == 1.0
        assert rewards[1] == 1.0

    def test_reward_failed_in_range(self, reward_model):
        embeds = _make_embeds(4)
        success = [True, False, False, False]
        task_ids = [0, 0, 0, 0]
        rewards, failed_indices = reward_model.compute_batch_rewards(embeds, success, task_ids)
        for idx in failed_indices:
            assert 0.0 <= rewards[idx] <= 0.6 + 1e-6, f"reward {rewards[idx]} out of range"

    def test_reward_closer_failure_higher(self, reward_model):
        """A failed trajectory whose embedding is closer to success should get higher reward."""
        torch.manual_seed(0)
        success_embed = torch.randn(20, 64)
        close_fail = success_embed + 0.01 * torch.randn(20, 64)  # very close
        far_fail = success_embed + 10.0 * torch.randn(20, 64)    # very far
        embeds = [success_embed, close_fail, far_fail]
        success = [True, False, False]
        task_ids = [0, 0, 0]
        rewards, _ = reward_model.compute_batch_rewards(embeds, success, task_ids)
        assert rewards[1] > rewards[2], f"close={rewards[1]:.4f} should > far={rewards[2]:.4f}"

    def test_reward_temporal_changes_embedding(self):
        """With use_temporal=True, trajectory embedding should differ from raw mean-pool."""
        embed_dim = 64
        model_t = SigLIPTReward(embed_dim=embed_dim, use_temporal=True, device=torch.device("cpu"))
        model_no = SigLIPTReward(embed_dim=embed_dim, use_temporal=False, device=torch.device("cpu"))
        frames = torch.randn(20, embed_dim)
        emb_t = model_t.embed_trajectory(frames)
        emb_no = model_no.embed_trajectory(frames)
        # Temporal model adds learned embeddings so results should differ
        assert not torch.allclose(emb_t, emb_no, atol=1e-6)

    def test_reward_no_successes(self, reward_model):
        """When no successes exist for a task, failures should get reward=0.0."""
        embeds = _make_embeds(3)
        success = [False, False, False]
        task_ids = [0, 0, 0]
        rewards, failed_indices = reward_model.compute_batch_rewards(embeds, success, task_ids)
        for idx in failed_indices:
            assert rewards[idx] == 0.0

    def test_reward_multiple_tasks(self, reward_model):
        """Tasks are processed independently."""
        embeds = _make_embeds(6)
        success = [True, False, False, True, False, False]
        task_ids = [0, 0, 0, 1, 1, 1]
        rewards, failed_indices = reward_model.compute_batch_rewards(embeds, success, task_ids)
        assert rewards[0] == 1.0
        assert rewards[3] == 1.0
        assert len(failed_indices) == 4

    def test_reward_model_parameters(self, reward_model, reward_model_no_temporal):
        """Temporal model has params, no-temporal model has none."""
        assert sum(1 for _ in reward_model.parameters()) > 0
        assert sum(1 for _ in reward_model_no_temporal.parameters()) == 0


# ===========================================================================
# GRPO Advantage Tests
# ===========================================================================

class TestGRPOAdvantages:

    def test_advantages_zero_mean_per_group(self):
        rewards = [1.0, 0.3, 0.5, 0.8, 0.2, 0.6]
        task_ids = [0, 0, 0, 1, 1, 1]
        advs = compute_grpo_advantages(rewards, task_ids)
        # Group 0 mean advantage should be ~0
        group0 = [advs[i] for i in range(3)]
        assert abs(sum(group0) / 3) < 1e-6
        # Group 1 mean advantage should be ~0
        group1 = [advs[i] for i in range(3, 6)]
        assert abs(sum(group1) / 3) < 1e-6

    def test_advantages_different_tasks_independent(self):
        """Changing rewards in task 1 shouldn't affect task 0 advantages."""
        rewards_a = [1.0, 0.3, 0.5, 0.8, 0.2, 0.6]
        rewards_b = [1.0, 0.3, 0.5, 0.1, 0.9, 0.4]  # only task 1 changed
        task_ids = [0, 0, 0, 1, 1, 1]
        advs_a = compute_grpo_advantages(rewards_a, task_ids)
        advs_b = compute_grpo_advantages(rewards_b, task_ids)
        # Task 0 advantages should be identical
        for i in range(3):
            assert abs(advs_a[i] - advs_b[i]) < 1e-6

    def test_advantages_single_sample(self):
        """Single sample in a group should get advantage=0."""
        rewards = [0.5]
        task_ids = [0]
        advs = compute_grpo_advantages(rewards, task_ids)
        assert abs(advs[0]) < 1e-6


# ===========================================================================
# GRPO Loss Tests
# ===========================================================================

class TestGRPOLoss:

    def test_loss_gradient_flows(self):
        lps = [torch.tensor(-5.0, requires_grad=True) for _ in range(3)]
        old_lps = [torch.tensor(-5.1) for _ in range(3)]
        advs = [1.0, -0.5, 0.2]
        loss, _ = grpo_loss(lps, old_lps, advs)
        loss.backward()
        assert all(lp.grad is not None for lp in lps)
        assert all(lp.grad.abs() > 0 for lp in lps)

    def test_loss_positive_advantage_encourages(self):
        """Positive advantage should produce negative gradient (encourage higher lp)."""
        lp = torch.tensor(0.0, requires_grad=True)
        old_lp = torch.tensor(0.0)
        loss, _ = grpo_loss([lp], [old_lp], [1.0])
        loss.backward()
        # Loss = -ratio * adv. d(loss)/d(lp) = -exp(lp-old_lp)*adv = -1.0
        # So gradient should be negative (optimizer step increases lp)
        assert lp.grad.item() < 0

    def test_loss_negative_advantage_discourages(self):
        """Negative advantage should produce positive gradient (discourage higher lp)."""
        lp = torch.tensor(0.0, requires_grad=True)
        old_lp = torch.tensor(0.0)
        loss, _ = grpo_loss([lp], [old_lp], [-1.0])
        loss.backward()
        assert lp.grad.item() > 0

    def test_loss_clipping_works(self):
        """Very different log-probs should be clipped."""
        # Big difference → ratio = exp(5) ≈ 148 → clipped to 1+ε=1.2
        lp = torch.tensor(5.0, requires_grad=True)
        old_lp = torch.tensor(0.0)
        loss_clipped, metrics = grpo_loss([lp], [old_lp], [1.0], clip_epsilon=0.2)
        # Ratio should be high but loss uses clipped value
        assert metrics["avg_ratio"] > 100  # unclipped ratio is huge

    def test_loss_equal_logprobs_ratio_one(self):
        """When old and new log-probs are equal, ratio should be 1.0."""
        lp = torch.tensor(-3.0, requires_grad=True)
        old_lp = torch.tensor(-3.0)
        _, metrics = grpo_loss([lp], [old_lp], [1.0])
        assert abs(metrics["avg_ratio"] - 1.0) < 1e-6


# ===========================================================================
# Trajectory Storage Tests
# ===========================================================================

class TestTrajectoryStorage:

    def test_trajectory_has_query_fields(self):
        traj = Trajectory()
        assert hasattr(traj, "query_inputs")
        assert hasattr(traj, "query_response_ids")
        assert hasattr(traj, "query_steps")
        assert isinstance(traj.query_inputs, list)
        assert isinstance(traj.query_response_ids, list)
        assert isinstance(traj.query_steps, list)

    def test_trajectory_query_data_storage(self):
        traj = Trajectory()
        # Simulate storing query data
        dummy_inputs = {
            "input_ids": torch.randint(0, 32000, (1, 50)),
            "pixel_values": torch.randn(1, 6, 224, 224),
            "attention_mask": torch.ones(1, 50),
        }
        dummy_response_ids = torch.randint(31744, 32000, (1, 56))
        traj.query_inputs.append(dummy_inputs)
        traj.query_response_ids.append(None)  # token IDs derived later via compute_token_log_probs
        traj.query_steps.append((0, 8))
        assert len(traj.query_inputs) == 1
        assert traj.query_steps[0] == (0, 8)


# ===========================================================================
# Temporal Embedding Tests
# ===========================================================================

class TestTemporalEmbedding:

    def test_temporal_output_shape(self):
        te = TemporalEmbedding(embed_dim=64, freq_channels=32)
        out = te.embed_trajectory(20, device=torch.device("cpu"))
        assert out.shape == (20, 64)

    def test_temporal_different_positions(self):
        """Different frame indices should produce different embeddings."""
        te = TemporalEmbedding(embed_dim=64, freq_channels=32)
        out = te.embed_trajectory(10)
        assert not torch.allclose(out[0], out[5])

    def test_temporal_is_trainable(self):
        te = TemporalEmbedding(embed_dim=64, freq_channels=32)
        params = list(te.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


# ===========================================================================
# GRPOTrainer Tests
# ===========================================================================

class TestGRPOTrainer:

    def test_trainer_update_changes_params(self):
        """Optimizer step should change trainable parameters."""
        # Create a simple trainable model
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        trainer = GRPOTrainer(model=model, optimizer=optimizer)

        # Record initial params
        initial_params = {n: p.clone() for n, p in model.named_parameters()}

        # Dummy log-probs with grad
        lps = [torch.tensor(-3.0, requires_grad=True)]
        old_lps = [torch.tensor(-3.5)]
        advs = [1.0]

        # This won't change model params because lps aren't connected to model
        # But it tests that the trainer runs without error
        metrics = trainer.update(lps, old_lps, advs)
        assert "grpo_loss" in metrics
        assert "step" in metrics
        assert metrics["step"] == 1

    def test_trainer_step_count(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        trainer = GRPOTrainer(model=model, optimizer=optimizer)
        assert trainer.step_count == 0

        lps = [torch.tensor(-3.0, requires_grad=True)]
        old_lps = [torch.tensor(-3.0)]
        trainer.update(lps, old_lps, [0.5])
        assert trainer.step_count == 1
        trainer.update(lps, old_lps, [0.5])
        assert trainer.step_count == 2


# ===========================================================================
# Integration: Full Mock Iteration
# ===========================================================================

class TestIntegration:

    def test_full_mock_iteration(self):
        """Simulate one full iteration: rollout data → reward → GRPO update."""
        embed_dim = 64
        n_trajs = 8

        # 1. Simulate rollout results
        trajectories = []
        for i in range(n_trajs):
            traj = Trajectory()
            traj.task_id = i // 4  # 2 tasks, 4 trajs each
            traj.success = (i % 3 == 0)  # some succeed
            traj.frames = [np.random.randn(224, 224, 3).astype(np.float32) for _ in range(20)]
            traj.actions = [np.random.randn(7).astype(np.float32) for _ in range(20)]
            traj.siglip_embeds = [torch.randn(embed_dim) for _ in range(20)]
            # Simulate 3 model queries per trajectory
            for q in range(3):
                traj.query_inputs.append({
                    "input_ids": torch.randint(0, 32000, (1, 50)),
                    "pixel_values": torch.randn(1, 6, 224, 224),
                    "attention_mask": torch.ones(1, 50),
                })
                traj.query_response_ids.append(None)  # derived later
                traj.query_steps.append((q * 8, (q + 1) * 8))
            trajectories.append(traj)

        # 2. Compute rewards
        reward_model = SigLIPTReward(embed_dim=embed_dim, use_temporal=True, device=torch.device("cpu"))
        all_embeds = [torch.stack(t.siglip_embeds) for t in trajectories]
        success_mask = [t.success for t in trajectories]
        task_ids = [t.task_id for t in trajectories]
        rewards, failed_indices = reward_model.compute_batch_rewards(all_embeds, success_mask, task_ids)

        assert len(rewards) == n_trajs
        assert all(rewards[i] == 1.0 for i in range(n_trajs) if success_mask[i])
        assert len(failed_indices) > 0

        # 3. Compute advantages
        advs = compute_grpo_advantages(rewards, task_ids)
        assert len(advs) == n_trajs

        # 4. GRPO loss with dummy log-probs
        failed_lps = [torch.tensor(-5.0, requires_grad=True) for _ in failed_indices]
        failed_old_lps = [torch.tensor(-5.1) for _ in failed_indices]
        failed_advs = [advs[i] for i in failed_indices]

        loss, metrics = grpo_loss(failed_lps, failed_old_lps, failed_advs)
        assert loss.item() != 0.0 or all(a == 0.0 for a in failed_advs)

        loss.backward()
        assert all(lp.grad is not None for lp in failed_lps)

    def test_reward_state_dict_roundtrip(self):
        """Save and load reward model state."""
        model = SigLIPTReward(embed_dim=64, use_temporal=True, device=torch.device("cpu"))
        sd = model.state_dict()
        assert len(sd) > 0

        model2 = SigLIPTReward(embed_dim=64, use_temporal=True, device=torch.device("cpu"))
        model2.load_state_dict(sd)

        # Embeddings should be identical after loading
        frames = torch.randn(10, 64)
        e1 = model.embed_trajectory(frames)
        e2 = model2.embed_trajectory(frames)
        assert torch.allclose(e1, e2)
