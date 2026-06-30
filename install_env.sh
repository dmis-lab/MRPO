# Usage: bash install_env.sh   (CUDA 13.0 / cu130)
set -euo pipefail
ENV_NAME=MRPO
PIP="python -m pip install --no-cache-dir --prefer-binary"

# conda + compiler + fresh env
source "$(conda info --base)/etc/profile.d/conda.sh"
source /usr/share/Modules/init/bash 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load compilers/gcc/10.2.0 2>/dev/null || true
command -v gcc >/dev/null && export CC=$(command -v gcc) CXX=$(command -v g++)
conda create -n "$ENV_NAME" python=3.11 -y
conda activate "$ENV_NAME"
python -m pip install --no-cache-dir --upgrade pip

# pyarrow from conda-forge (glibc-safe), then cu130 torch
conda install -n "$ENV_NAME" -c conda-forge "pyarrow>=21" -y
$PIP "numpy==2.4.6"
$PIP --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
  "torch==2.12.1+cu130" "torchvision==0.27.1+cu130"

# rest of the stack
$PIP \
  "accelerate==1.12.0" \
  "bert-score" \
  "datasets==4.5.0" \
  "deepspeed==0.15.4" \
  "einops" \
  "huggingface_hub" \
  "math-verify" \
  "matplotlib" \
  "nltk" \
  "ninja" \
  "num2words" \
  "openai" \
  "packaging" \
  "pandas" \
  "peft==0.18.1" \
  "pillow" \
  "python-dotenv" \
  "qwen-vl-utils" \
  "rouge-score" \
  "sentencepiece" \
  "timm" \
  "transformers==4.57.3" \
  "trl==0.17.0" \
  "wandb"

# flash-attn: source build against the system CUDA toolkit (CUDA_HOME from nvcc)
command -v nvcc >/dev/null && export CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
MAX_JOBS=${MAX_JOBS:-4} python -m pip install --no-cache-dir flash-attn==2.8.3 --no-build-isolation
