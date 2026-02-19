"""Modal wrapper — runs train_rl() on a remote A10G GPU. Contains zero training logic."""

import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "cmake", "build-essential", "clang",
        "libgl1-mesa-glx", "libegl1-mesa", "libglib2.0-0",
        "libosmesa6-dev", "libglew-dev", "patchelf",
        "libgl1-mesa-dev", "libxrandr2", "libxinerama1", "libxcursor1",
        "libinput-dev", "libudev-dev",
    )
    .run_commands(
    # 1. CUDA-enabled PyTorch
        "pip install torch==2.2.0 torchvision==0.17.0 torchaudio==0.2.0 --index-url https://download.pytorch.org/whl/cu121",
        # 2. OpenVLA-OFT
        "git clone https://github.com/moojink/openvla-oft.git /opt/openvla-oft",
        "cd /opt/openvla-oft && pip install --no-deps -e .",
        # 3. Dependencies
        "pip install 'accelerate>=0.25.0' 'draccus==0.8.0' einops huggingface_hub json-numpy jsonlines matplotlib 'peft==0.11.1' protobuf rich 'sentencepiece==0.1.99' 'timm==0.9.10' 'tokenizers==0.19.1' wandb 'diffusers==0.30.3' imageio uvicorn fastapi 'numpy<2'",
        # 4. Custom transformers fork
        "pip install --no-deps git+https://github.com/moojink/transformers-openvla-oft.git",
        "pip install --no-deps bitsandbytes==0.43.0",
        # 5. Patch prismatic imports
        "for f in /opt/openvla-oft/prismatic/__init__.py "
        "/opt/openvla-oft/prismatic/vla/__init__.py "
        "/opt/openvla-oft/prismatic/models/__init__.py "
        "/opt/openvla-oft/prismatic/models/vlms/__init__.py "
        "/opt/openvla-oft/prismatic/models/vlas/__init__.py "
        "/opt/openvla-oft/prismatic/training/__init__.py; do "
        "echo '# Patched for minimal import' > $f; done",
    )
    .pip_install(
        "pillow>=10.0",
        "pyyaml>=6.0",
        "easydict>=1.9",
        "opencv-python>=4.8.0",
        "scipy>=1.11.0",
    )
    .pip_install("robosuite>=1.4.0", "robomimic>=0.2.0")
    .run_commands("pip install libero")
    .run_commands(
        # Pre-create LIBERO config to avoid interactive prompt
        "mkdir -p ~/.libero",
        "python -c \""
        "import yaml, os; "
        "libero_root = os.path.join(os.path.dirname(__import__('libero').__file__), 'libero'); "
        "d = {"
        "'benchmark_root': libero_root, "
        "'bddl_files': os.path.join(libero_root, 'bddl_files'), "
        "'init_states': os.path.join(libero_root, 'init_files'), "
        "'datasets': '/tmp/libero_datasets', "
        "'assets': os.path.join(libero_root, 'assets')"
        "}; "
        "yaml.dump(d, open(os.path.expanduser('~/.libero/config.yaml'), 'w'))"
        "\"",
    )
    .env({"MUJOCO_GL": "egl"})
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir(
        "vendor/LIBERO/libero/libero/bddl_files",
        remote_path="/usr/local/lib/python3.10/site-packages/libero/libero/bddl_files",
    )
)

app = modal.App("vla-coach-rl", image=image)

vol = modal.Volume.from_name("vla-coach-rl-results", create_if_missing=True)


@app.function(
    gpu="A10G",
    timeout=43200,  # 12 hours max
    volumes={"/root/results": vol},
    memory=32768,  # 32GB RAM
)
def train_rl_remote(cfg: dict):
    """Run SigLIP-T self-improvement on a remote A10G GPU."""
    import sys
    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/opt/openvla-oft")

    # Parse YAML config if passed as raw text
    if "_config_text" in cfg:
        import yaml
        parsed = yaml.safe_load(cfg.pop("_config_text"))
        parsed.update(cfg)
        cfg = parsed

    cfg["checkpoint_dir"] = "/root/results/siglip_t/checkpoints"
    if cfg.get("save_videos"):
        cfg["video_dir"] = "/root/results/siglip_t/videos"

    from vla_coach.train_rl import train_rl
    history = train_rl(cfg)

    vol.commit()
    return history


@app.local_entrypoint()
def main(
    config: str = "configs/train_rl.yaml",
    benchmark: str = None,
    n_iterations: int = None,
    n_tasks: int = None,
    no_temporal: bool = False,
    save_videos: bool = False,
    seed: int = None,
):
    """Local entrypoint — dispatches to remote GPU."""
    with open(config) as f:
        config_text = f.read()

    cfg = {"_config_text": config_text}

    if benchmark is not None:
        cfg["benchmark"] = benchmark
    if n_iterations is not None:
        cfg["n_iterations"] = n_iterations
    if n_tasks is not None:
        cfg["n_tasks"] = n_tasks
    if no_temporal:
        cfg["use_temporal"] = False
    if save_videos:
        cfg["save_videos"] = True
    if seed is not None:
        cfg["seed"] = seed

    print(f"Launching SigLIP-{'T' if cfg.get('use_temporal', True) else 'only'} on Modal A10G...")
    history = train_rl_remote.remote(cfg)

    final = history["iterations"][-1]
    print("\nTraining complete!")
    print(f"  Final success rate: {final['batch_stats']['success_rate']:.1%}")
    print(f"  Total GRPO steps: {final['grpo_metrics'].get('step', 0)}")
    print(f"  Total time: {sum(i['elapsed_seconds'] for i in history['iterations']):.0f}s")
