# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any
from matplotlib import pyplot as plt

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

viz_attention = False

def plot_first_token_cosine_matrix(token1, token2, eps=1e-8):
    """绘制两组 token 的余弦相似度矩阵，主要用于离线调试特征。

    参数
    ----
    token1, token2:
        NumPy 数组，预期形状为 ``(N, P, C)``。当前调试代码会跳过前
        5 个特殊 token，再对每张图的 patch token 求平均，因此实际使用
        的是 ``token[:, 5:, :]``，而不是字面意义上的第 0 个 token。
    eps:
        L2 归一化时添加的数值稳定项，防止零向量除零。

    返回
    ----
    numpy.ndarray
        形状为 ``(N, N)`` 的余弦相似度矩阵。函数同时弹出 Matplotlib
        图像，适合研究 token 是否具有跨图像相似性，不参与正常推理。
    """

    # Normalize

    # Extract first tokens (shape: n x 1024)
    t1 = np.mean(token1[:, 5:, :], axis=1)  # (n x 1024)
    t2 = np.mean(token2[:, 5:, :], axis=1)  # (n x 1024)

    t1 = t1 / (np.linalg.norm(t1, axis=-1, keepdims=True) + eps)
    t2 = t2 / (np.linalg.norm(t2, axis=-1, keepdims=True) + eps)

    # Compute cosine similarity: (t1 · t2^T)
    sim = t1 @ t2.T  # (n x n) matrix

    # Plot the matrix
    plt.figure(figsize=(5, 5))
    plt.imshow(sim, interpolation="nearest")
    plt.colorbar(label="cosine similarity")
    plt.title("Cosine Similarity Matrix (First Token)")
    plt.xlabel("Image index")
    plt.ylabel("Image index")
    plt.tight_layout()
    plt.show()

    return sim

# from image_retrieval import run_asmk

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

import torch
import matplotlib.pyplot as plt

def mean_top_quarter(arr):
    """计算数组中最高四分之一数值的平均值。

    该函数先把任意形状的数组展平，使用第 75 百分位作为门槛，再对
    大于等于门槛的元素求均值。`get_similarity()` 用它压缩注意力比值，
    让少量高响应 token 对最终分数贡献更大。
    """

    # 将输入展平，统一处理标量向量、矩阵以及更高维数组。
    flat = arr.ravel()
    # 取第 75 百分位作为“最高四分之一”的近似分界。
    thresh = np.percentile(flat, 75)
    # 保留门槛以上的值；重复值可能使保留数量多于四分之一。
    top_vals = flat[flat >= thresh]
    # 返回高响应部分的均值，作为稳健的汇总分数。
    return top_vals.mean()


# k_selected = k[:, :, token_idx+frame_idx*tokens_per_img:token_idx+frame_idx*tokens_per_img+1, :] 
# attn = q @ k_selected.transpose(-2, -1)
# attn = attn.transpose(-2, -1) 
# attn = attn.softmax(dim=-1)    
# attn_mean = attn.mean(dim=1)
# attn_mean = attn_mean.view(B, N)
# shown = attn_mean[:, self.token_offset:tokens_per_img]

def get_similarity(k, q, token_offset=5, image_height=21, image_width=37):
    """根据双帧 global attention 计算图像匹配分数。

    参数 ``k`` 和 ``q`` 来自 Transformer attention，通常形状为
    ``(B, heads, 2P, head_dim)``。当前实现假定序列恰好有两张图，每帧
    P 个 token。它只保留第一帧中跳过特殊 token 后的 key，再让两帧的
    所有 query 与这些 key 计算响应。结果沿 query 维 softmax、对 head
    求平均，然后用第一帧的最大响应归一化第二帧响应，最后取所有第二帧
    query 的最佳响应中最高四分之一的均值。

    返回值是 Python/NumPy 标量，数值越大表示当前实现定义的注意力匹配
    越强。它不是像素内点率、RANSAC 比例或几何重投影误差；参数
    ``image_height`` 和 ``image_width`` 仅保留给可视化调试路径。
    """

    # 当前回环验证固定把“查询帧 + 历史帧”组成两帧输入。
    num_imgs = 2
    tokens_per_img = q.shape[2] // num_imgs
    # print(k.shape, q.shape)
    # 只保留第一帧的 patch key，跳过 camera/register 等特殊 token。
    k = k[:,:,token_offset:tokens_per_img , :]
    # print(k.shape)

    # q: [B,H,2P,D]，k: [B,H,P-offset,D]；乘积得到所有 query 对
    # 第一帧 patch key 的响应，再转置为 [B,H,P-offset,2P]。
    attn = q @ k.transpose(-2, -1)
    attn = attn.transpose(-2, -1)
    # print(attn.shape)
    # 对两帧的全部 query token 做归一化，再对多头结果求平均。
    attn = attn.softmax(dim=-1)
    attn = attn.mean(dim=1)

    # 最后一维前 P 个 query 属于第一帧，后 P 个属于第二帧。
    all_token_to_first_frame = attn[..., :tokens_per_img]
    all_token_to_second_frame = attn[..., tokens_per_img:]

    # 每个第一帧 patch key 在第一帧 query 中的最大响应作为归一化基准。
    max_per_token_first_img = all_token_to_first_frame.max(dim=-1)[0]
    # max_per_token_second_img = all_token_to_second_frame.max(dim=-1)[0]

    # 比较同一组第一帧 key 对第二帧 query 的响应相对基准有多强。
    attn_second_frame_normalized = all_token_to_second_frame / (max_per_token_first_img.unsqueeze(-1) + 1e-8)
    # 对每个第二帧 query，取所有第一帧 patch key 中的最佳匹配。
    ratio = attn_second_frame_normalized.max(dim=1)[0]

    # ratio = max_per_token_second_img / (max_per_token_first_img + 1e-8)

    # 脱离计算图并移到 CPU，最后只保留高响应 query 的汇总标量。
    ratio = ratio.float().detach().cpu().numpy()
    ratio = mean_top_quarter(ratio)


    print("Average of top quarter attention values (all frames):", ratio)
    # print("First Frame, Second Frame:", avg_top_quarter_first_img, avg_top_quarter_second_img)
    # plt.figure(figsize=(6,6))
    # plt.imshow(max_attn[1].reshape(image_height, image_width))
    # plt.colorbar()
    # plt.show()

    return ratio

class CosineAttentionVisualizer:
    """交互式查看 token 余弦相似度和 attention 热力图的调试工具。

    该类不参与 `Aggregator` 的正常前向计算。`Aggregator` 在全局注意力
    调试开关打开时，把每层 token、key 和 query 保存到这里；用户可以
    点击某一帧的 patch，查看它在所有帧中的响应，并用键盘切换层和模式。
    """

    def __init__(self, image_height=21, token_offset=5):
        """初始化可视化状态和 token 网格参数。

        ``token_offset`` 表示 camera/register 等特殊 token 的数量；默认
        5 与 Aggregator 的 1 个 camera token 加 4 个 register token 对应。
        """

        self.image_width = 37
        self.image_height= image_height
        self.token_offset = token_offset  # number of special tokens to skip (e.g., CLS, etc.)
        self.layers = []  # each: {"name": str, "x_norm": tensor[B,N,C], "attn": tensor[B,N,N] or None}
        self.current_layer = 0
        self.mode = "cosine"  # "cosine" | "attention"
        self.last_clicked = None  # (frame_idx, row, col, token_idx)
        self.fig = None
        self.axes = []
        self.ims = []
        self.colorbar = None

    def add_layer(self, name, x, k, q):
        """保存一层的归一化 token、key 和 query，供后续交互绘图。

        `_process_global_attention()` 传入的 ``x`` 已恢复为逐帧布局，形状
        是 ``(B*S, P, C)``；``k`` 和 ``q`` 仍是 global attention 布局，
        形状为 ``(B, heads, S*P, head_dim)``。这里不复制数据，因而调用方
        需要确保调试期间这些 Tensor 仍然有效。
        """

        eps = 1e-8
        x_norm = x / (x.norm(dim=-1, keepdim=True) + eps)

        self.layers.append({"name": name, "x_norm": x_norm, "k": k, "q": q})

    # ---------- plotting & interaction ----------

    def show(self):
        """创建窗口并显示当前层的 token 热力图。

        若用户尚未点击 token，则默认选择第 0 帧中间附近的 patch 作为
        参考；之后 `_redraw()` 根据当前模式绘制所有帧的响应。
        """

        if not self.layers:
            print("No layers added.")
            return

        # Default reference: center-ish pixel in frame 0
        if self.last_clicked is None:
            r = min(self.image_height // 2, self.image_width - 1)
            c = min(self.image_width // 2, self.image_width - 1)
            token_idx = self.token_offset + r * self.image_width + c
            self.last_clicked = (0, r, c, token_idx)

        self._init_figure()
        self._redraw()
        plt.show()

    def _init_figure(self):
        """按 batch 中的图像数量创建 Matplotlib 子图并注册交互事件。"""

        layer = self.layers[self.current_layer]
        B = layer["x_norm"].shape[0]

        frames_per_row = 5
        ncols = min(B, frames_per_row)
        nrows = (B + ncols - 1) // ncols

        self.fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5*ncols, 4.5*nrows))
        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]

        self.axes = [ax for row in axes for ax in row]
        self.ims = []

        # connect events once
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _compute_maps(self):
        """计算当前参考 token 的热力图和颜色范围。

        返回 ``(maps, vmin, vmax)``，其中 ``maps`` 形状为
        ``(B, image_height * image_width)``。`cosine` 模式比较 token
        向量，`attention` 模式使用保存的 q/k 计算响应；两种模式都会
        去掉特殊 token，只保留图像 patch 网格。
        """

        layer = self.layers[self.current_layer]
        x_norm = layer["x_norm"]      # (B, N, C)
        # x_norm = torch.tensor(np.load('/home/dominic/vggt/tokens1.npy'))
        k = layer["k"]
        q = layer["q"]
        print(self.current_layer, k.shape, q.shape, x_norm.shape)
        B, N, C = x_norm.shape

        frame_idx, row, col, token_idx = self.last_clicked

        tokens_per_img = self.token_offset + self.image_height*self.image_width

        # guard if click outside valid token area
        if token_idx < self.token_offset or token_idx >= N:
            raise ValueError("Clicked token index out of range.")

        if self.mode == "cosine":
            # Use the *vector from the clicked frame* as the query, compare to ALL tokens in EVERY frame
            ref_vec = x_norm[frame_idx, token_idx, :]                     # (C,)
            # (C) x (B,N,C) -> (B,N)
            maps_full = torch.einsum("c,bnc->bn", ref_vec, x_norm)
            # display only the image tokens (skip special tokens)
            shown = maps_full[:, self.token_offset:tokens_per_img]
            # run_asmk(x_norm)
            print(self.current_layer, shown.shape)
            # np.save("/home/dominic/Documents/vggt/test_tokens/target_tokens12.npy", x_norm.detach().cpu().numpy())
            # plot_first_token_cosine_matrix(x_norm.detach().cpu().numpy(), x_norm.detach().cpu().numpy(), eps=1e-8)

            vmin, vmax = 0.0, 1.0
            return shown, vmin, vmax

        else:  # attention
            k_selected = k[:, :, token_idx+frame_idx*tokens_per_img:token_idx+frame_idx*tokens_per_img+1, :] 
            attn = q @ k_selected.transpose(-2, -1)
            print("here",k_selected.shape, q.shape, attn.shape)

            attn = attn.transpose(-2, -1) 
            attn = attn.softmax(dim=-1)   
            print(attn.shape) 
            attn_mean = attn.mean(dim=1)
            attn_mean = attn_mean.view(B, N)
            shown = attn_mean[:, self.token_offset:tokens_per_img]
            # dynamic range across all shown frames for this selection
            vmin = float(shown.min().detach().cpu())
            vmax = float(shown.max().detach().cpu())
            # protect against degenerate ranges
            if vmin == vmax:
                vmax = vmin + 1e-8
            
            get_similarity(k, q)

            return shown, vmin, vmax

    def _redraw(self):
        """根据当前层、参考 token 和模式创建或更新热力图。"""

        maps, vmin, vmax = self._compute_maps()  # (B, W*W)
        B = maps.shape[0]
        W = self.image_width
        H = self.image_height

        # draw/update images
        if not self.ims:
            for i in range(len(self.axes)):
                ax = self.axes[i]
                if i < B:
                    frame = maps[i].float().detach().cpu().reshape(H, W).numpy()
                    im = ax.imshow(frame, cmap="viridis", vmin=vmin, vmax=vmax)
                    ax.set_title(f"Frame {i}")
                    # mark clicked pixel (same row/col) so you can see it
                    if i == self.last_clicked[0]:
                        ax.plot(self.last_clicked[2], self.last_clicked[1], marker='x', markersize=8, mew=2, color='black')
                    # else:
                    #     ax.plot(self.last_clicked[2], self.last_clicked[1], marker='x', markersize=6, alpha=0.4)
                    self.ims.append(im)
                ax.axis("off")

            # one shared colorbar
            self.colorbar = self.fig.colorbar(self.ims[0], ax=self.axes[:B],
                                              orientation="horizontal", fraction=0.03, pad=0.08)
        else:
            # update existing
            for i in range(B):
                frame = maps[i].float().detach().cpu().reshape(H, W).numpy()
                self.ims[i].set_data(frame)
                self.ims[i].set_clim(vmin, vmax)
                ax = self.axes[i]
                # clear old markers and replot marker
                # (remove all lines in this axes; cheap if there are only markers)
                for ln in list(ax.lines):
                    ln.remove()
                if i == self.last_clicked[0]:
                    ax.plot(self.last_clicked[2], self.last_clicked[1], marker='x', markersize=8, mew=2, color='black')
                # else:
                #     ax.plot(self.last_clicked[2], self.last_clicked[1], marker='x', markersize=6, alpha=0.4)

            if self.colorbar is not None:
                self.colorbar.update_normal(self.ims[0])

        self.fig.suptitle(
            f"Mode: {self.mode} • Layer: {self.layers[self.current_layer]['name']} • "
            f"Ref: frame={self.last_clicked[0]}, token_idx={self.last_clicked[3]} (row={self.last_clicked[1]}, col={self.last_clicked[2]})",
            fontsize=12
        )
        self.fig.canvas.draw_idle()

    # ---------- events ----------

    def _on_click(self, event):
        """把鼠标点击位置转换为 patch token，并触发热力图刷新。"""

        if event.inaxes not in self.axes:
            return
        ax_idx = self.axes.index(event.inaxes)

        # guard against clicks outside image
        if event.xdata is None or event.ydata is None:
            return

        col = int(event.xdata)
        row = int(event.ydata)

        # map (row, col) -> token index
        token_idx = self.token_offset + row * self.image_width + col

        self.last_clicked = (ax_idx, row, col, token_idx)
        self._redraw()

    def _on_key(self, event):
        """响应左右方向键和 m/a 键，切换层或余弦/attention 模式。"""

        if event.key in ("right", "n"):
            self.current_layer = (self.current_layer + 1) % len(self.layers)
            self._redraw()
        elif event.key in ("left", "p"):
            self.current_layer = (self.current_layer - 1) % len(self.layers)
            self._redraw()
        elif event.key in ("m", "a"):  # toggle mode
            self.mode = "attention" if self.mode == "cosine" else "cosine"
            self._redraw()

class Aggregator(nn.Module):
    """
    VGGT 的多帧特征聚合器。

    这个模块把输入图像先编码成 patch token，再在两种注意力之间交替计算：

    - frame attention：每一帧内部独立建模，理解本帧的纹理和空间关系；
    - global attention：把同一 batch 的所有帧拼接起来，建立跨帧几何关系。

    两种注意力各自有一组 Transformer block。每次处理后，代码把 frame
    和 global 的中间特征沿最后一维拼接，形成后续 camera head、DPT depth
    head 使用的多尺度表示。默认 ``embed_dim=1024``，所以拼接后的通道数
    是 2048。

    输入形状为 ``[B, S, 3, H, W]``，其中 ``B`` 是 batch size，``S`` 是
    同一序列中的图像数。经过 patch embedding 后，每帧会在 patch token
    前面添加 1 个 camera token 和若干 register token。token 数取决于实际
    输入尺寸：518×518 输入共有 ``1 + 4 + 37×37 = 1374`` 个 token；本项目
    常见的 294×518 宽图输入共有 ``1 + 4 + 21×37 = 782`` 个 token。

    当 ``compute_similarity=True`` 时，模块会在 ``target_layer`` 指定的
    global block 取出 q/k，并计算双帧 `image_match_ratio`。这个分支供回环
    验证使用，不改变主干的几何输出。当前实现的 `target_tokens` 检查发生在
    block 索引递增之后，所以它保存的是前一个 global block 的输出；代码中
    已在对应位置明确标出这两个索引语义。

    训练模式下使用 ``torch.utils.checkpoint`` 重算 attention block，以较少
    激活显存换取额外计算；推理模式直接执行 block。若希望启用 checkpoint，
    调用方需要先执行 ``model.train()``。

    参数
    ----
    img_size:
        patch 编码器预训练/初始化使用的图像边长；运行时仍可接受满足 patch
        整除条件的矩形图像。
    patch_size:
        一个 patch 的边长，默认 14。
    embed_dim:
        每个 token 的通道维度，默认 1024。
    depth:
        frame blocks 和 global blocks 各自的层数，默认各 24 层。
    num_heads:
        每个 attention block 的注意力头数。
    mlp_ratio:
        Transformer MLP 隐藏层宽度相对 ``embed_dim`` 的倍率。
    num_register_tokens:
        每帧添加的 register token 数量。
    block_fn:
        构造 frame/global block 使用的模块类型，默认 `Block`。
    qkv_bias, proj_bias, ffn_bias:
        分别控制 qkv、attention 输出投影和前馈网络是否使用 bias。
    patch_embed:
        patch 编码器名称，可使用简单卷积或 DINOv2 ViT 变体。
    aa_order:
        交替注意力顺序，例如 ``["frame", "global"]``。
    aa_block_size:
        切换注意力类型前连续执行的同类 block 数量。
    qk_norm:
        是否对每个注意力头的 query/key 应用归一化层。
    rope_freq:
        二维旋转位置编码的基础频率；小于等于 0 时关闭 RoPE。
    init_values:
        Transformer LayerScale 的初始缩放值。
    target_layer:
        提取回环 q/k 的 global block 零基索引。上层 `VGGT` 当前默认传入
        21，会覆盖本类直接实例化时的默认值 20。
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        target_layer=20
    ):
        """创建 patch 编码器、位置编码、两组注意力 block 和特殊 token。

        ``aa_order`` 决定每个交替周期先执行哪种注意力，``aa_block_size``
        决定连续执行多少个同类 block。默认的 ``["frame", "global"]`` 与
        1 表示每次 frame attention 后立即进行一次 global attention。

        ``camera_token`` 和 ``register_token`` 的第二维大小为 2：第 0 组
        专门给序列第一帧使用，第 1 组复用于其余帧。这与 VGGT 对第一帧
        和后续帧的特殊初始化约定一致。
        """

        super().__init__()
        # 可视化对象只在 viz_attention 打开时使用，不参与普通推理的输出。
        self.viz = CosineAttentionVisualizer()
        # 创建 DINOv2/卷积 patch embedding，将图像变成 [B*S, P_patch, C]。
        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # frequency <= 0 时关闭二维旋转位置编码，便于调试或使用无位置编码的变体。
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.target_layer = target_layer

        # frame_blocks 只在每帧内部做 attention；global_blocks 负责跨帧 attention。
        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # 每个交替周期包含 aa_block_size 个同类 block，因此 depth 必须整除它。
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # 两组 camera/register token：第一帧使用第 0 组，其余帧使用第 1 组。
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # patch token 在特殊 token 之后开始；默认偏移量为 1 + 4 = 5。
        # 这是由网络结构决定的整数，表示跳过多少个特殊 token，不是图像预测值。
        self.patch_start_idx = 1 + num_register_tokens

        # 用很小的随机值初始化特殊 token，避免它们在训练初期压制图像特征。
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # 注册 ImageNet/ResNet 风格标准化常量；persistent=False 表示不写入 checkpoint。
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        # checkpoint 使用非 reentrant 实现，兼容当前 PyTorch 的推荐接口。
        self.use_reentrant = False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """根据配置创建图像到 patch token 的编码器。

        当 ``patch_embed`` 名称包含 ``"conv"`` 时使用项目中的简单卷积
        `PatchEmbed`；否则从 DINOv2 风格的 ViT 工厂中选择 small/base/large/
        giant2 之一。ViT 编码器可以插值位置编码，以支持与预训练尺寸不同
        的输入。若编码器带有 mask token，这里关闭它的梯度，因为当前聚合器
        只做普通 RGB 推理，不执行掩码图像训练。
        """

        # 简单卷积路径：输出形状为 [batch, patch 数量, embed_dim]。
        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            # 通过名称选择对应的 DINOv2 ViT 主干。
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # 当前任务不使用 mask token，因此不让它参与梯度更新。
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor, compute_similarity=False) -> Tuple[List[torch.Tensor], int]:
        """对一批多帧图像执行交替 frame/global attention。

        参数
        ----
        images:
            范围为 ``[0, 1]`` 的 RGB Tensor，形状为 ``[B, S, 3, H, W]``。
            ``S`` 是一次共同推理的帧数；global attention 会在这 ``S`` 帧
            的全部 token 上计算跨帧关系。
        compute_similarity:
            仅在双帧回环验证时打开。到达 ``target_layer`` 的 global block
            时保存 q/k，并调用 `get_similarity()` 返回 attention 匹配分数。
            q/k 只在 eval 分支返回，因此该选项应与 ``model.eval()`` 配合。

        返回
        ----
        实际返回四元组，依次是以下对象（本函数历史类型注解只列出了前两项）。

        output_list : list[torch.Tensor]
            从空列表开始，在每轮 frame/global 计算完成后加入拼接结果。
            默认 depth=24、aa_block_size=1，因此列表包含 24 项；列表索引
            表示网络层，图像索引则保存在每项 Tensor 的 S 维。
            记第 l 轮帧内输出为 F_l，随后跨帧输出为 G_l，则
            output_list[l] = cat([F_l, G_l], dim=-1)，形状为 [B,S,P,2C]。
            其中 P=(H//patch_size)*(W//patch_size)+1+num_register_tokens。
            前 C 个通道来自 F_l，后 C 个通道来自 G_l。每个向量是训练得到
            的上下文特征，单个通道没有预先规定的深度、角度或坐标单位。
            默认 294x518 三帧输入时，每项为 [1,3,782,2048]。
            上层把这些特征转 FP32：相机头取最后一层的 token 0；深度头取
            第 [4,11,17,23] 层，去掉特殊 token 后恢复二维网格预测深度。

        patch_start_idx : int
            在 __init__ 中由 1+num_register_tokens 计算，默认是 5。
            每帧 token 的索引 0 为 camera，1~4 为 register，5 起为图像
            patch。因此 output_list[l][:,:,patch_start_idx:,:] 提取全部
            patch 特征，供 DPT 恢复成 H//patch_size x W//patch_size 网格。

        target_tokens : torch.Tensor or None
            初始为 None；当 global_idx 在 block 执行后等于 target_layer
            时，复制当前 global 输出，并沿最后一维归一化每个 token：
            target_tokens = G / (norm(G, dim=-1, keepdim=True)+1e-8)。
            默认 aa_block_size=1 时，这份 G 来自 global_blocks[target_layer-1]。
            上层 VGGT 默认传 target_layer=21，因此保存 G_20（索引从 0 起）。
            它只包含 global 输出，未拼接 frame 输出，形状为 [B*S,P,C]；
            上述三帧例子为 [3,782,1024]，camera/register/patch 全部保留。
            compute_similarity=False 时，只要命中该层也会生成这份快照。
            clone 会复制数据，但不会自动 detach；SLAM 推理由外层 no_grad
            禁用梯度。CompatibleVGGT 将它透传到 predictions，当前项目的
            相机/深度头和 SALAD 检索不消费它；它可供特征分析使用。

        image_match_ratio : numpy scalar or None
            初始为 None；只有 compute_similarity=True 且执行前的 global_idx
            命中 target_layer，才取该 global block 的 q/k 并调用 get_similarity。
            默认实际 VGGT 的 q/k 来自 global_blocks[21]，与上面的 G_20 不同。
            get_similarity 假定两帧输入：取第一帧 patch key，对两帧所有 query
            计算响应；按第一帧 query 的最大响应归一化第二帧响应，再对每个
            第二帧 query 取最佳 key，最后平均最高四分位的值。
            这是经 float().detach().cpu().numpy() 汇总得到的标量，可大于 1，
            不表示正确匹配像素的比例。solver.py 用小于 0.95 的条件拒绝回环。
            普通窗口不请求相似度时返回 None；双帧回环验证时才用于决策。
        """

        # 约定输入保留 batch 维和序列维，后续会在两种布局之间来回变形。
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")
        
        # global viz_attention
        # if compute_similarity:
        #     viz_attention = True

        # 按注册的均值/标准差标准化图像；输入本身仍保持 RGB 语义。
        images = (images - self._resnet_mean) / self._resnet_std

        # patch embedding 逐帧工作，因此先把 [B,S] 合并成 [B*S]。
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # 按“第一帧/其余帧”规则展开两类特殊 token，并匹配 B、S。
        camera_token = slice_expand_and_flatten(self.camera_token, B, S) #这是加载模型的时候加载的参数
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # 每帧 token 顺序：camera token、register tokens、patch tokens。
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            # 只为 patch 网格生成二维位置；特殊 token 的位置稍后置零。
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # camera/register token 不使用二维空间位置，因此对应位置设为 0。
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # 加入特殊 token 后，P 表示每帧完整 token 数，不再只是 patch 数。
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        # 每项保存一层的多帧特征；随着网络层推进增加，列表长度不由帧数 S 决定。
        output_list = []

        # 目标层快照与匹配分数各自有独立的赋值条件；普通推理也会保存快照。
        target_tokens = None

        # eps = 1e-8
        # patch_tokens_save = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + eps)
        # np.save("/home/dominic/vggt/tokens2_decoder.npy", patch_tokens_save.detach().cpu().numpy())

        image_match_ratio = None
        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                # skip_to = 17
                # if global_idx == 10:
                #     global_idx = skip_to
                    # frame_idx = skip_to
                # if frame_idx == 24 or global_idx == 24:
                #     break
                if attn_type == "frame":
                    # 帧内特征路径：tokens [B*S,P,C] 与 pos [B*S,P,2] 一起
                    # 进入 frame_blocks[frame_idx]，每帧的 P 个 token 独立做自注意力。
                    # 返回 tokens 供下一步 global attention 使用；返回的新 frame_idx
                    # 指向下一个 frame block；frame_intermediates 按 [B,S,P,C]
                    # 保存本次各 block 的输出，稍后与 global_intermediates 拼接。
                    # 默认只执行一个 block，所以 frame_idx 每次加 1，列表长度为 1。
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    if compute_similarity and global_idx == self.target_layer:
                        # global_idx 是执行前的零基 block 索引；命中目标层时保留 q/k。
                        tokens, global_idx, global_intermediates, k, q = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos, compute_similarity=True
                        )
                        image_match_ratio = get_similarity(k, q)
                    else:
                        # 普通 global attention 只返回更新后的 token 和中间特征。
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos
                        )
                    if global_idx == self.target_layer:
                        # 此处 global_idx 已经加 1，因此保存的是索引 target_layer-1
                        # 的 block 输出；它与上面的 q/k 目标 block 相差一层。
                        print(self.target_layer)
                        # target_tokens = tokens[:,5:, :].clone()  # exclude camera and register tokens
                        target_tokens = tokens.clone()  # 保留 camera/register 和全部 patch token。
                        eps = 1e-8
                        # 沿 C 维计算每个 token 的 L2 长度，得到近似单位向量；
                        # 仍是 [B*S,P,C]，不合并帧、不平均 patch、不拼接帧内输出。
                        target_tokens = target_tokens / (target_tokens.norm(dim=-1, keepdim=True) + eps)


                        ###### For testing loading a token #####
                        # loaded_tokens = np.load("/home/dominic/vggt/tokens1.npy")
                        # print(loaded_tokens.shape, loaded_tokens.dtype)
                        # loaded_tokens = torch.from_numpy(loaded_tokens).to(target_tokens.device)
                        # tokens[0,:] = loaded_tokens[10:11,:,:]
                        # tokens = torch.cat([tokens, loaded_tokens[10:11,:,:]], dim=0)
                        # S += 1
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # 对齐同一交替周期的 frame/global 特征，拼成 [B,S,P,2C]。
                # frame_intermediates[i] 是本轮帧内 block 的输出；tokens 随后
                # 又经过 global block。二者保留不同阶段的信息，在通道维拼接。
                # 具体索引 [b,s,p,c] 表示：某层、第 b 个序列、第 s 帧、第 p 个
                # token 的第 c 个特征通道；c<C 取帧内信息，c>=C 取跨帧信息。
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        if viz_attention:
            self.viz.show()

        # 删除循环中间引用，帮助较大窗口尽快释放 Python 层的临时对象。
        del concat_inter
        del frame_intermediates
        del global_intermediates
        # 普通三帧、294x518、默认网络配置的返回示例：
        # ([24 个 [1,3,782,2048] Tensor], 5, [3,782,1024] Tensor, None)。
        # 双帧回环验证时 S=2，target_tokens 为 [2,782,1024]，最后一项为匹配分数。
        return output_list, self.patch_start_idx, target_tokens, image_match_ratio

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """执行一组帧内 Transformer block，更新特征并收集本次的中间输出。

        参数与布局
        ----------
        tokens:
            包含 camera、register、patch token 的特征。内部统一为
            [B*S,P,C]；必要时从 [B,S*P,C] 恢复逐帧布局。
        B, S, P, C:
            B 为独立序列数，S 为每个序列的帧数，P 为每帧完整 token 数，
            C 为每个 token 的特征维度。合并 B、S 后，每帧成为独立的
            attention 批次项；所有帧共享当前 block 的参数。
        frame_idx:
            本次从哪个 frame_blocks 索引开始执行，与 global_idx 独立计数。
        pos:
            每个 token 的二维坐标，内部布局为 [B*S,P,2]，最后一维是
            (行,列)。它在 attention 内用于旋转 q/k，坐标数值不会被更新。

        默认配置的数据流（B=1、S=3、P=782、C=1024）
        ------------------------------------------------
        tokens [3,782,1024] + pos [3,782,2]
          -> Block.forward：LayerNorm -> Attention -> LayerScale -> 残差相加
          -> LayerNorm -> MLP(1024->4096->1024) -> LayerScale -> 残差相加
          -> tokens [3,782,1024]；通道数和 token 数不变，特征值已被更新。

        Attention 内部先把 q/k/v 拆为 [3,16,782,64]，其中 16 为注意力头数，
        64 为每头维度；q/k 经归一化和 RoPE 后，沿同一帧的 782 个 token
        计算注意力，再合并各头并投影回 1024 维。本步不新增跨帧交互；
        后续轮次的输入可能已包含上一轮 global attention 融合的跨帧信息。

        返回
        ----
        tokens:
            本次最后一个 block 的输出 [B*S,P,C]，交给后续 global attention。
        frame_idx:
            下一个待执行 frame block 的索引，等于输入索引加 aa_block_size。
        intermediates:
            本次执行的每个 block 输出的列表，长度为 aa_block_size；每项
            为 [B,S,P,C] 的视图，用于与本轮 global 中间特征拼接。它每次
            调用都会重新创建，并非跨 24 轮积累的 output_list。
        """

        # 1. 将每帧作为独立批次项；必要时先恢复 [B,S,P,C]，再合并 B、S。
        # view 只调整张量的布局，不重新计算特征，也不在不同帧之间交换数据。
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        # 位置张量必须同步采用逐帧布局，确保 tokens[j,i] 与 pos[j,i] 对应。
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        # 2. 只收集这一次调用中的 block 输出；默认会存入一项。
        intermediates = []

        # 3. 连续执行 aa_block_size 个 frame block；默认值为 1。
        for _ in range(self.aa_block_size):
            if self.training:
                # 训练时通过 checkpoint 保存必要状态，并在反向时按需重算激活；
                # 使用的仍是当前 frame block，输入输出布局与直接调用相同。
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                # 推理时进入 Block.forward(tokens, pos=pos, viz_attention=False)。
                # Attention 在 P 维聚合本帧各 token 信息，MLP 逐 token 变换通道；
                # 两个子层经残差连接后，输出仍为 [B*S,P,C]，此处只返回特征。
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos, viz_attention=False)

            # 4. 当前层执行完成，计数器指向下一层。例如从 0 变为 1 后，
            # 下一轮使用 frame_blocks[1]；这里是网络层索引，不是图像帧号。
            frame_idx += 1

            # 恢复显式的 batch/帧维，记录当前 block 的完整输出（含特殊 token）。
            # 此 view 与当前 tokens 共享存储，未做 clone；后面的 global block
            # 使用新的计算结果推进主分支，本轮帧内输出仍用于最终的特征拼接。
            intermediates.append(tokens.view(B, S, P, C))

        # 5. 主分支特征继续向下游流动；层索引回传给外层循环；中间列表用于拼接。
        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, compute_similarity=False):
        """执行一组 global attention block，建立窗口内的跨帧关系。

        global attention 将 token 变形为 ``(B, S*P, C)``，同一 batch 内的
        所有帧共享 attention 范围。若 ``compute_similarity`` 或全局调试开关
        打开，底层 `Block` 还会返回 q/k，供 `get_similarity()` 或交互式
        `CosineAttentionVisualizer` 使用。

        返回值在普通路径中是 ``(tokens, next_global_idx, intermediates)``；
        相似度路径额外返回 ``(k, q)``。
        """

        # global attention 需要把帧维和 token 维拼成一个序列维。
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # 默认每轮只执行一个 global block，与 frame block 交替推进。
        for _ in range(self.aa_block_size):
            if self.training:
                # 训练时同样通过 checkpoint 换取较低激活显存。
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                if viz_attention or compute_similarity:
                    # 只有调试/相似度分支才请求 q/k，避免正常推理保存额外张量。
                    tokens, k, q = self.global_blocks[global_idx](tokens, pos=pos, viz_attention=True)
                else:
                    # 普通推理只保留 block 的 token 输出。
                    tokens = self.global_blocks[global_idx](tokens, pos=pos, viz_attention=False)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        # 交给下一个 frame attention 前，恢复逐帧布局 [B*S,P,C]。
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if viz_attention:
            # prior_tokens = torch.tensor(np.load("/home/dominic/Documents/vggt/token_logs/" + str(global_idx) + ".npy"))
            # prior_tokens_k = torch.tensor(np.load("/home/dominic/Documents/vggt/token_logs/k" + str(global_idx) + ".npy"))
            # prior_tokens_q = torch.tensor(np.load("/home/dominic/Documents/vggt/token_logs/q" + str(global_idx) + ".npy"))
            # temp_concat = torch.cat([prior_tokens, tokens.detach().cpu()], dim=0)
            # temp_concat_k = torch.cat([prior_tokens_k, k.detach().cpu()], dim=2)
            # temp_concat_q = torch.cat([prior_tokens_q, q.detach().cpu()], dim=2)
            # np.save("/home/dominic/Documents/vggt/token_logs/" + str(global_idx) + ".npy", tokens.detach().cpu().numpy())
            # np.save("/home/dominic/Documents/vggt/token_logs/k" + str(global_idx) + ".npy", k.detach().cpu().numpy())
            # np.save("/home/dominic/Documents/vggt/token_logs/q" + str(global_idx) + ".npy", q.detach().cpu().numpy())
            # self.viz.add_layer("Layer " +str(global_idx), temp_concat, temp_concat_k, temp_concat_q)
            self.viz.add_layer("Layer " +str(global_idx), tokens, k, q)
        
        if compute_similarity:
            return tokens, global_idx, intermediates, k, q
        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """展开第一帧/后续帧专用 token，并整理成 attention 输入布局。

    参数 ``token_tensor`` 的形状为 ``(1, 2, X, C)``：第 2 维的第 0
    个切片给序列第一帧，第 1 个切片给其余 ``S-1`` 帧。函数先沿 batch
    和序列维扩展，得到 ``(B, S, X, C)``，再展平为 ``(B*S, X, C)``。
    这样既保留了第一帧的特殊参数，又能让 frame attention 以逐帧 batch
    的形式高效处理。

    返回
    ----
    torch.Tensor
        形状为 ``(B*S, X, C)`` 的特殊 token。
    """

    # 取出第一帧专用 token，扩展到 batch 维，形状为 (B, 1, X, C)。
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # 取出后续帧共用 token，沿序列维复制 S-1 次。
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # 将第一帧和后续帧拼成完整序列，形状为 (B, S, X, C)。
    combined = torch.cat([query, others], dim=1)

    # frame attention 使用 B*S 作为 batch 维，因此最后展平前两维。
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
