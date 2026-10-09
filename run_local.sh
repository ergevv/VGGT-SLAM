#!/usr/bin/env bash
# 使用独立环境运行 office_loop 示例，默认参数适配 6GB 显存。
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
    echo "未找到项目环境，请先运行 ./setup_local.sh。" >&2
    exit 1
fi

# 将运行缓存集中到项目内，方便重启时复用已下载的权重。
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
mkdir -p logs

# 额外命令行参数可覆盖默认参数，例如 --image_folder 指定自己的图像序列。
exec .venv/bin/python -u main.py \
    --image_folder office_loop \
    --submap_size 2 \
    --max_loops 1 \
    --vis_map \
    --vis_voxel_size 0.01 \
    --keep_alive \
    --log_results \
    --log_path logs/office_loop_poses.txt \
    "$@"
