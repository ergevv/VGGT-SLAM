"""根据本机硬件加载 VGGT，兼容不支持 BF16 的显卡。"""

import os

import torch
from vggt.models.vggt import VGGT


def get_inference_dtype(device):
    """为 CUDA 选择支持的半精度，CPU 使用单精度。"""
    if torch.device(device).type == "cuda":
        # GTX 10 系列即使驱动报告支持 BF16，也没有原生 BF16 运算单元。
        capability = torch.cuda.get_device_capability(torch.device(device))
        return torch.bfloat16 if capability[0] >= 8 else torch.float16
    return torch.float32


class CompatibleVGGT(VGGT):
    """复用上游网络，按硬件选择精度并省去 SLAM 未使用的预测头。"""

    def __init__(self):
        """只启用 SLAM 实际使用的相机与深度预测头，减少显存占用。"""
        super().__init__(enable_point=False, enable_track=False)

    def forward(self, images, query_points=None, compute_similarity=False):
        """保持上游返回格式，特征提取使用半精度，几何预测使用单精度。"""
        if query_points is not None:
            raise ValueError("SLAM 模型未启用点跟踪预测头。")
        if images.ndim == 4:
            images = images.unsqueeze(0)

        dtype = next(self.aggregator.parameters()).dtype
        tokens, patch_start_idx, target_tokens, match_ratio = self.aggregator(
            images.to(dtype), compute_similarity
        )
        # 相机和深度预测保留单精度，避免 FP16 的取值范围影响几何结果。
        geometry_tokens = [token.float() for token in tokens]
        pose_enc_list = self.camera_head(geometry_tokens)
        depth, depth_conf = self.depth_head(
            geometry_tokens, images=images.float(), patch_start_idx=patch_start_idx
        )
        return {
            "pose_enc": pose_enc_list[-1],
            "pose_enc_list": pose_enc_list,
            "depth": depth,
            "depth_conf": depth_conf,
            "images": images.float(),
            "target_tokens": target_tokens,
            "image_match_ratio": match_ratio,
        }


def load_vggt_model(device):
    """读取官方权重并配置推理精度，模型缓存沿用 PyTorch 的目录约定。"""
    checkpoint_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.isfile(checkpoint_path):
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.hub.download_url_to_file(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            checkpoint_path,
        )

    # 内存映射避免一次性将数 GB 权重全部复制到内存中。
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    state_dict = {
        key: value for key, value in state_dict.items()
        if not key.startswith(("point_head.", "track_head."))
    }
    # 直接使用映射的权重替换初始化参数，减少加载过程中的复制。
    model = CompatibleVGGT()
    model.load_state_dict(state_dict, assign=True)
    dtype = get_inference_dtype(device)
    model.aggregator.to(dtype=dtype)
    model = model.eval().to(device)
    print(f"VGGT 特征提取精度: {dtype}，几何预测精度: torch.float32")
    return model
