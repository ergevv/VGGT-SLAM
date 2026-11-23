import math
import torch
import torch.nn as nn

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(S, dustbin_score = 1.0, num_iters=3, reg=1.0):
    """通过 Sinkhorn 得到聚类槽与 patch 之间的对数分配权重。

    S 为原始打分 [B, M, N]，M 是聚类槽数量，N 是 patch 数量。
    额外加入一行 dustbin，使特征可以分配给“未被普通聚类槽吸收”的部分。
    log_otp_solver 在对数域交替调整行、列质量，避免直接指数计算溢出。

    返回 [B, M+1, N] 的对数权重；调用者需要 exp() 才得到正权重。
    按当前质量设置和返回缩放，收敛时每个普通槽的行质量为 1，dustbin
    的行质量为 N-M，每个 patch 的列质量为 1。有限次迭代只近似满足
    两组约束；N-M 是质量额度，不表示硬性删除 N-M 个 patch。
    当前公式使用 log(N-M)，因此要求 patch 数 N 大于聚类槽数 M。
    """
    batch_size, m, n = S.size()
    # 加入 dustbin 槽：最后一行的所有 patch 共用可训练的 dustbin_score。
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # 在对数域设置行/列目标质量；dustbin 比普通槽有更大的质量额度。
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n-m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(
        log_a,
        log_b,
        S_aug,
        num_iters=num_iters,
        reg=reg
    )
    return log_P - norm


class SALAD(nn.Module):
    """
    This class represents the Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.

    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
        token_dim (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
    """
    def __init__(self,
            num_channels=1536,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
            dropout=0.3,
        ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters= num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        
        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.))


    def forward(self, x):
        """将 DINOv2 的局部特征和整图 token 聚合成固定长度的图像描述子。

        本函数对 batch 中的每张图像独立计算描述子。核心思路是：把每个
        patch 特征投影到较低维，用 Sinkhorn 计算它分配到各个聚类槽的
        权重，再按槽加权求和，并拼接整图 CLS token 的投影特征。

        参数
        ----
        x:
            元组 (局部特征图, CLS token)，由 DINOv2.forward() 返回：
            - 局部特征图：[B, D, h, w]，h、w 是 patch 网格大小；
            - CLS token：[B, D]，表示每张图像的整体特征。
            这里 B 为图像数量，D 为主干特征维度，N=h*w 为 patch 数量。
            M=self.num_clusters 为槽数，L=self.cluster_dim 为每槽维度，
            G=self.token_dim 为整图分支的输出维度。

        返回
        ----
        torch.Tensor
            L2 归一化的图像描述子 [B, G+L*M]，可用于图像检索。
            本项目 eval.py 配置 D=768、M=64、L=128、G=256；224x224
            图像对应 16x16 的 patch 网格，所以 N=256，输出为 [B, 8448]。

        中间张量
        --------
        f：从 [B, L, N] 的 patch 特征变为 [B, G+L*M] 的最终描述子；
        p：从 [B, M, N] 的原始分数变为 Sinkhorn 的软分配权重；
        t：从 [B, D] 的 CLS token 变为 [B, G] 的整图投影特征。
        聚类槽是学习得到的聚合位置，不预设为某种物体类别；分配是连续
        权重，同一个 patch 可以对多个槽有贡献。
        """
        # 解包主干输出：这里左边的 x 已从元组变为局部特征图，t 是 CLS token。
        x, t = x # Extract features and token

        # 1. 局部特征分支：两个 1x1 卷积在每个 patch 上变换通道 D->512->L。
        # flatten(2) 只合并空间维 h,w，得到 [B,L,N]；patch 与分数的顺序一致。
        # 本项目的形状变化为 [B,768,16,16] -> [B,128,256]。
        f = self.cluster_features(x).flatten(2)

        # 2. 分配打分分支：每个 patch 为 M 个槽产生原始打分，得到 [B,M,N]。
        # p[b,k,i] 表示第 b 张图的第 i 个 patch 对第 k 个槽的打分，尚未归一化。
        p = self.score(x).flatten(2)

        # 3. 整图分支：MLP 将 CLS token 从 D 维投影到 G 维，得到 [B,G]。
        t = self.token_features(t)

        # 4. Sinkhorn 软分配：加入 dustbin 后从 [B,M,N] 变为 [B,M+1,N]。
        # 它交替调整行/列质量，让普通槽的分配较均衡；本次执行 3 轮迭代。
        # 函数返回的是对数权重 log(P)，不是可直接与特征相乘的正权重。
        p = get_matching_probs(p, self.dust_bin, 3)

        # exp 将对数权重恢复为正的分配权重，形状仍为 [B,M+1,N]。
        p = torch.exp(p)

        # 删除最后的 dustbin 行，只用 M 个普通槽参与描述子聚合： [B,M,N]。
        # 此处是切片，不是重新归一化，也不是从特征图中删除具体 patch。
        # 留给 dustbin 的质量不会再分给普通槽，因此某些 patch 的贡献会降低。
        p = p[:, :-1, :]

        # 5. 对齐局部特征与权重的维度，二者最终都为 [B,L,M,N]。
        # p 在 L 维复制：同一个槽/patch 的分配权重对全部 L 个特征通道共用。
        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)

        # f 在 M 维复制：同一个 patch 的 L 维特征可按不同权重贡献给各个槽。
        # repeat 会生成重复数据；这里维持原实现，便于观察逐元素乘法的维度。
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        # 6. 每槽加权汇总：A[b,l,k] = sum_i(f[b,l,k,i] * p[b,l,k,i])。
        # (f*p).sum(-1) 沿 N 个 patch 求和，结果为 [B,L,M]；仍保留每槽 L 维。
        # dim=1 的 L2 归一化逐槽执行，再 flatten(1) 得到 [B,L*M]。
        # 展平顺序是特征通道在前、槽索引在后，与训练时保持一致。
        # t 先在 G 维归一化为 [B,G]，再与局部汇总特征拼为 [B,G+L*M]。
        f = torch.cat([
            nn.functional.normalize(t, p=2, dim=-1),
            nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
        ], dim=-1)

        # 7. 对拼接后的整条描述子做最终 L2 归一化，非零向量的范数接近 1。
        # 本项目得到 [B,256+128*64]=[B,8448]，下游按描述子的 L2 距离检索。
        return nn.functional.normalize(f, p=2, dim=-1)
