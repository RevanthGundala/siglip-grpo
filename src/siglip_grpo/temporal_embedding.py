import math

import torch
import torch.nn as nn


class SinusoidalEncoding(nn.Module):
    """Sinusoidal positional encoding for scalar timesteps (DDPM / Transformer-style)."""

    def __init__(self, num_channels: int = 256, max_period: float = 10000.0):
        super().__init__()
        self.num_channels = num_channels
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """(B,) → (B, num_channels) sinusoidal encoding."""
        if timesteps.dim() == 2:
            timesteps = timesteps.squeeze(-1)

        half = self.num_channels // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        args = timesteps[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TemporalEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 1152, freq_channels: int = 256):
        super().__init__()
        self.sinusoidal = SinusoidalEncoding(num_channels=freq_channels)
        self.mlp = nn.Sequential(
            nn.Linear(freq_channels, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """(B,) frame indices → (B, embed_dim) temporal embedding."""
        sinusoidal_features = self.sinusoidal(frame_indices)
        return self.mlp(sinusoidal_features)

    def embed_trajectory(self, num_frames: int, device: torch.device = None) -> torch.Tensor:
        """Compute temporal embeddings for all frames 0..num_frames-1."""
        indices = torch.arange(num_frames, dtype=torch.float32, device=device)
        return self.forward(indices)
