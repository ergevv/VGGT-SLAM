#!/usr/bin/env bash
# 在项目内配置 Python 3.11、基础 SLAM 依赖及官方模型权重。
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export UV_CACHE_DIR="$PROJECT_DIR/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_DIR/.cache/python"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
mkdir -p .cache/bootstrap logs third_party .cache/torch/hub/checkpoints

# 优先使用本机较新的 Python，只用它引导安装 uv，不修改系统环境。
if [[ ! -x .cache/bootstrap/bin/uv ]]; then
    BOOTSTRAP_PYTHON="$(command -v python3.10 || command -v python3)"
    "$BOOTSTRAP_PYTHON" -m pip install --target .cache/bootstrap uv
fi
UV_BIN="$PROJECT_DIR/.cache/bootstrap/bin/uv"
"$UV_BIN" python install 3.11
if [[ ! -x .venv/bin/python ]]; then
    "$UV_BIN" venv --python 3.11 .venv
fi
"$UV_BIN" pip install --python .venv/bin/python -r requirements.txt -c constraints-local.txt

# 复用已下载的源码，重复配置不会覆盖本地依赖中的修改。
if [[ ! -d third_party/vggt ]]; then
    git clone --depth 1 https://github.com/MIT-SPARK/VGGT_SPARK.git third_party/vggt
fi
if [[ ! -d third_party/salad ]]; then
    git clone --depth 1 https://github.com/Dominic101/salad.git third_party/salad
fi
"$UV_BIN" pip install --python .venv/bin/python -e ./third_party/vggt -e ./third_party/salad -e .

# 先保存为临时文件，下载完整后再替换，避免下次读取不完整的权重。
if [[ ! -f .cache/torch/hub/checkpoints/model.pt ]]; then
    curl -fLsS --retry 3 --connect-timeout 20 \
        -o .cache/torch/hub/checkpoints/model.pt.part \
        https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt
    mv .cache/torch/hub/checkpoints/model.pt.part .cache/torch/hub/checkpoints/model.pt
fi
if [[ ! -f .cache/torch/hub/checkpoints/dino_salad.ckpt ]]; then
    curl -fLsS --retry 3 --connect-timeout 20 \
        -o .cache/torch/hub/checkpoints/dino_salad.ckpt.part \
        https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt
    mv .cache/torch/hub/checkpoints/dino_salad.ckpt.part .cache/torch/hub/checkpoints/dino_salad.ckpt
fi
# SALAD 会通过 torch.hub 加载 DINOv2，预先缓存以便启动时直接使用。
if [[ ! -d .cache/torch/hub/facebookresearch_dinov2_main ]]; then
    git clone --depth 1 https://github.com/facebookresearch/dinov2.git \
        .cache/torch/hub/facebookresearch_dinov2_main
fi
if [[ ! -f .cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth ]]; then
    curl -fLsS --retry 3 --connect-timeout 20 \
        -o .cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth.part \
        https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
    mv .cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth.part \
        .cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth
fi
if [[ ! -d office_loop ]]; then
    unzip -q office_loop.zip
fi
"$UV_BIN" pip check --python .venv/bin/python
echo "本地环境配置完成，运行 ./run_local.sh 启动示例。"
