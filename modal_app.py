import modal
from modal import gpu

app = modal.App("gstar-grpo")

# Persist outputs and HF cache across runs for speed and reproducibility
# Use from_name(...) for broader client compatibility
out_vol = modal.Volume.from_name("gstar-output", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Build the image per README guidance: vLLM, transformers, torch, triton, datasets, flash-attn
# Use CUDA 12.4 wheels for Torch to match flash-attn compatibility
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "python3-dev", "python3-pip", "pkg-config")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:${PATH}",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}",
    })
    .run_commands([
    "python3 -m pip install --upgrade pip setuptools wheel",
    "python3 -m pip install torch==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124",
    "python3 -m pip install flash-attn==2.7.4.post1 --no-build-isolation",
    ])
    .pip_install(
        "vllm==0.7.2",
        "triton==3.1.0",
        "transformers==4.51.3",
        "datasets",
        "tensorboard",
        "gpustat",
        "python-dotenv",
    )
    .env({"VLLM_USE_V1": "0"})
    .add_local_file("starter.py", "/root/app/starter.py")
    .add_local_file("zero_shot_eval.py", "/root/app/zero_shot_eval.py")
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60 * 12,
    volumes={
        "/root/output": out_vol,
        "/root/.cache/huggingface": hf_vol,
    },
    secrets=[modal.Secret.from_name("hf-token")],
)
def train():
    import os, sys
    os.chdir("/root/app")
    if "/root/app" not in sys.path:
        sys.path.insert(0, "/root/app")
    import starter
    starter.main()


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60 * 4,
    volumes={
        "/root/output": out_vol,
        "/root/.cache/huggingface": hf_vol,
    },
    secrets=[modal.Secret.from_name("hf-token")],
)
def eval(model_id: str = "Qwen/Qwen3-1.7B", max_tokens: int = 256):
    import os, sys
    os.chdir("/root/app")
    if "/root/app" not in sys.path:
        sys.path.insert(0, "/root/app")
    sys.argv = ["zero_shot_eval.py", "--model", model_id, "--max_tokens", str(max_tokens)]
    import zero_shot_eval
    zero_shot_eval.main()


@app.local_entrypoint()
def main(run: str = "train", model_id: str = "Qwen/Qwen3-1.7B", max_tokens: int = 256):
    if run == "train":
        train.remote()
    else:
        eval.remote(model_id, max_tokens)


