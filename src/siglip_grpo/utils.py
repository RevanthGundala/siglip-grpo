"""Shared utilities for config, logging, environment setup, and results I/O."""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str) -> dict:
    """Load a YAML config file and return as dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return a logger for siglip_grpo."""
    logger = logging.getLogger("siglip_grpo")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    return logger


def create_env(bddl_file: str, img_h: int = 128, img_w: int = 128):
    """Create a single LIBERO OffScreenRenderEnv."""
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=img_h,
        camera_widths=img_w,
    )


def create_vec_env(bddl_file: str, num_envs: int, img_h: int = 128, img_w: int = 128):
    """Create a vectorized LIBERO environment."""
    from libero.libero.envs import SubprocVectorEnv, DummyVectorEnv, OffScreenRenderEnv

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": img_h,
        "camera_widths": img_w,
    }

    if num_envs == 1:
        return DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args)])
    return SubprocVectorEnv(
        [lambda: OffScreenRenderEnv(**env_args) for _ in range(num_envs)]
    )


def save_results(results: dict, output_dir: str, prefix: str = "eval"):
    """Save evaluation results as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, cls=_NpEncoder)
    return filepath


class _NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
