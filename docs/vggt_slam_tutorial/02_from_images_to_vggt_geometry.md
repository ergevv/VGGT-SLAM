# VGGT-SLAM 教程（二）：从 RGB 图像到相机坐标几何

办公室中的墙在照片里只有颜色，模型却需要给出“这个像素距离相机多远”和“拍摄这些照片的相机如何移动”。本篇沿着图像到三维点的因果链，解释 VGGT 内部计算，而不把它留在“调用大模型”这一句话里。

依赖基准为 `third_party/vggt` 的 `6e6e16107b88e8e76c751826af10d4295d87ecd2`。这是本项目安装的 VGGT_SPARK 分支，其中加入了注意力相似度接口；本地 `CompatibleVGGT` 又选择了硬件精度、关闭了未使用预测头。本文区分这两层改动。

## 1. 图像中未知的是什么

设相机看到办公室墙上的一个像素，像素位置为 $(u,v)$，单位为像素。光线方向可以由相机内参算出，但沿这条光线的距离尚未知。多个三维点可能投影到同一个像素，单个 RGB 像素无法独立决定深度。

VGGT，即 Visual Geometry Grounded Transformer，是一个已训练的视觉几何预测网络。它结合图像纹理、训练中学到的几何先验及当前窗口跨帧关系，输出相机和深度。本仓库只做推理，不从本次视频重新训练网络，也没有 IMU 或 RealSense 实测深度参与这个主流程。[代码/依赖确认]

先约定三个空间：

- **像素空间**：$(u,v)$，横向向右、纵向向下；
- **相机系 $C_i$**：第 $i$ 帧相机的 x 向右、y 向下、z 向前；
- **局部参考系 $L$**：这一批 VGGT 预测共同使用的参考。它与最终图世界系 $W$ 要分别对待。

图像序号 $i$ 和坐标系 $C_i$ 是一一对应的，但局部预测并不直接说明该相机在最终全局图中的位置。下一步先让图像变成网络可以共同处理的表示。

## 2. 预处理改变了哪些信息

### 2.1 文件颜色、尺寸和 Tensor

光流通过 OpenCV 使用 BGR，网络通过 PIL 重新读取路径并转 RGB；二者不是共享一个被就地改色的数组。默认 `load_and_preprocess_images(mode='crop')` 的过程是：

```text
PIL 读取 -> RGBA 与白背景融合（如果有透明通道）
-> RGB -> 宽度 518
-> 高度按比例缩放并四舍五入为 14 的倍数
-> 高度超过 518 时中心裁剪
-> 范围 [0,1] 的 C×H×W Tensor -> stack 成 S×3×H×W
```

若各帧形状不同，补白至批内最大高宽。默认 crop 与可选 pad 的语义不同：前者可能丢掉上下部分，后者保留图像并补到方形。Solver 没有给这次调用传 mode，所以实际使用 crop。[依赖确认]

office_loop 的原图为 1080×1920。高度计算为：

$$
H'=14\operatorname{round}\left(\frac{1080\cdot518}{1920\cdot14}\right)=294,
\qquad W'=518.
$$

因此一个三帧窗口变成 `3×3×294×518`。每张网络输入有 152292 个像素。当前 FOV 转内参使用的是这个输入尺寸，不能把原图 1920×1080 的像素坐标直接代入后面的反投影。

### 2.2 网络内部的标准化

aggregator 接收 `[0,1]` 图像，按通道做：

$$
I'_c=\frac{I_c-\mu_c}{\sigma_c},
\quad \boldsymbol\mu=(0.485,0.456,0.406),
\quad \boldsymbol\sigma=(0.229,0.224,0.225).
$$

这个操作是改变数值表示，不是改成灰度，也不代表光照校准。原始 `[0,1]` images 仍用于颜色和输出。标准化后的图像被送入 patch embedding，开始建立区域级特征。

## 3. 像素怎样变成 token

### 3.1 为什么不在每个像素上直接做全局注意力

三帧共有约 45 万像素。若直接比较每个像素与其他所有像素，配对数量会极大。模型先把图像分成 14×14 的 patch，再用视觉骨干编码每个小区域。

**token** 是一个区域或特定角色的特征向量，保存网络学到的视觉描述。它不是三维点，也不是一个物体 ID；一个 patch 覆盖了多个像素，特征维度不能当作空间坐标维度。

对于 294×518 输入，patch 网格为 21×37，共 777 个 patch token。aggregator 又在每帧前面添加：

- 1 个 camera token，给相机预测头读取整帧信息；
- 4 个 register token，作为可训练的额外信息槽位；
- 777 个 patch token，给密集几何恢复使用。

于是每帧 token 数 $P=782$，通道维数 $C=1024$。第一帧和后续帧使用两组不同的 camera/register 参数，帧数扩大时后续组被展开复用。[依赖确认]

### 3.2 token 的形状和所有权

| 表示 | office_loop 三帧实例 | 来源/更新 | 生命周期/消费者 |
|---|---|---|---|
| 图像 | `B=1,S=3,3,294,518` | PIL + 转 batch | Submap.frames 长期保存输入 |
| patch 特征 | `B*S,777,1024` | 内部 DINOv2 风格骨干 | forward 中间值 |
| 加特殊 token | `B*S,782,1024` | camera + register + patch 拼接 | frame/global blocks |
| frame/global 拼接特征 | `B,S,782,2048` | 对应两类注意力结果拼接 | camera/DPT heads |
| `target_tokens` | 目标层的归一化特征 | clone 与按通道范数归一化 | 推理输出；不是 SALAD 检索向量 |

整个 aggregator 共 24 组交替层，输出列表保留各层拼接特征。DPT 需要多个深度的特征，因而即使 `torch.no_grad()` 不保存反向传播图，也仍需保留这些前向结果直到预测头用完。

## 4. 交替注意力怎样让不同照片互相解释

### 4.1 注意力不是直接的像素匹配

对 token 矩阵 $X\in\mathbb R^{N\times C}$，三个可训练投影产生查询 $Q$、键 $K$ 和值 $V$。某个查询 token 在问“其他 token 中哪些信息对我有用”；键决定相关程度，值提供被汇总的信息。

单头简化形式为：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
$$

$$
A_{ab}=\frac{\exp(q_a^\mathsf T k_b/\sqrt{d})}{\sum_j\exp(q_a^\mathsf T k_j/\sqrt{d})},
\qquad Y_a=\sum_b A_{ab}v_b.
$$

$d$ 是每个头的通道数，$A$ 是注意力权重，不是三维配准的内点掩码。依赖实际还使用 q/k normalization、二维旋转位置编码和输出投影；上式是解释共同机制的 [教学简化]，不是省略后可以替代该网络的实现。

例如两个 key 的缩放得分为 $(0,\log3)$，softmax 得到 $(0.25,0.75)$。若对应 value 为 $(2,6)$，输出是 $0.25\cdot2+0.75\cdot6=5$。token 被其他区域的信息改变，但这并不能说明第二个区域就是同一个物理点。

### 4.2 frame attention 先解释一张图

`_process_frame_attention()` 将张量整理为 `(B*S,P,C)`。每个帧是独立批项，注意力在自己的一帧内工作：墙纹理、地面和视野上下文都能影响本帧相机 token，但尚不能直接向其他帧取信息。

这一步解决局部图像结构问题。仅靠它，三张图仍然是三份独立解释，因此需要 global attention。

### 4.3 global attention 再连接多帧

`_process_global_attention()` 将张量整理为 `(B,S*P,C)`，一个批项里包含整个窗口的 token。此时图 0 的墙 patch 可以读图 1 和图 2 的信息，camera token 也能综合跨帧上下文。

两者交替 24 次，最后每层把帧内和全局的特征拼接为 2048 通道。模型不是先完整做所有帧内层、再完整做所有全局层，而是交替更新。[依赖确认]

概念上的注意力比较规模为 $S P^2$ 与 $(SP)^2$。这说明增大窗口尤其会增加跨帧处理规模；实际 PyTorch SDPA 的峰值显存还取决于选中的算子，不应直接把理论注意力矩阵大小当作实测 CUDA 分配。

### 4.4 `Aggregator` 的函数分工

| 函数 | 输入布局 | 核心职责 | 输出/下游 |
|---|---|---|---|
| `__init__()` | 模型配置 | 创建 patch encoder、RoPE、24 个 frame blocks、24 个 global blocks 和特殊 token | 持久模型参数 |
| `__build_patch_embed__()` | 编码器名称与图像/patch 尺寸 | 选择卷积编码器或 DINOv2 ViT，并关闭未使用 mask token 的梯度 | `self.patch_embed` |
| `forward()` | `B×S×3×H×W` | 标准化、生成 token、交替执行两种 attention、拼接中间特征 | camera/DPT heads、目标层 token、回环 ratio |
| `_process_frame_attention()` | `B*S×P×C` | 让每一帧的 P 个 token 独立交互 | `B×S×P×C` 中间特征 |
| `_process_global_attention()` | `B×S*P×C` | 让一个窗口的所有帧共享 attention；按需返回 q/k | 跨帧中间特征和回环 q/k |
| `slice_expand_and_flatten()` | `1×2×X×C` 特殊 token | 第一组只给首帧，第二组复用到后续帧，再展平 B、S | `B*S×X×C` |
| `get_similarity()` | 双帧 global q/k | 用第一帧 patch key 衡量第二帧 query 的相对高响应 | `image_match_ratio` 标量 |

当前源码有一个需要按执行顺序理解的索引细节：相似度分支在执行 global block **之前**检查 `global_idx == target_layer`，所以 q/k 来自零基索引 `target_layer` 的 block；保存 `target_tokens` 的检查发生在 block 将 `global_idx` 加 1 **之后**，所以保存的是索引 `target_layer-1` 的输出。上层 `VGGT` 当前默认传入 `target_layer=21`。[依赖确认]

## 5. 相机头：从整帧特征到 9 个参数

### 5.1 camera token 的消费者

相机头读取最后一层拼接结果中的 token 0，获得 `(B,S,2048)`。它采用四轮网络预测修正：第一轮使用 learned empty pose，后续把上一轮预测嵌入到特征中，生成新的 9 维增量。

$$
p^{(0)}=\Delta p^{(0)},
\qquad p^{(k+1)}=p^{(k)}+\Delta p^{(k+1)}.
$$

$p$ 是相机编码向量，不是 4×4 李群状态。这里的修正是神经网络迭代，不是 GTSAM 的 LM 优化，也没有使用后端因子。推理取 `pose_enc_list[-1]`。[依赖/代码确认]

9 维的实际顺序如下：

| 索引 | 语义 | 单位/约定 |
|---|---|---|
| `0:3` | world-to-camera 的平移 $t$ | 模型局部长度单位；不自动等于相机中心 |
| `3:7` | 四元数 $(q_x,q_y,q_z,q_w)$ | 实部最后；转换时按范数处理 |
| `7` | 垂直视场角 FOV_h | 弧度 |
| `8` | 水平视场角 FOV_w | 弧度 |

依赖的 FOV 激活用 ReLU；这只保证非负，不是证明焦距永远合理。四元数转换要求非零范数。

### 5.2 外参的方向必须写出来

令 $L$ 为本批局部参考，$C_i$ 为相机系。模型外参满足：

$$
p_{C_i}=R_{C_iL}p_L+t_{C_iL}.
$$

相机中心在 $L$ 中的位置来自 $p_{C_i}=0$，解得：

$$
c_L=-R_{C_iL}^{\mathsf T}t_{C_iL}.
$$

因此外参平移 $t$ 不等于相机在地图中的坐标。例如 $R=I$、相机中心 $(1,0,0)$，world-to-camera 的 $t$ 应为 $(-1,0,0)$。本项目用 `closed_form_inverse_se3()` 得到 camera-to-local 的 4×4 矩阵，又保存其逆作为 `Submap.poses`。

### 5.3 从 FOV 恢复焦距

在中心对称针孔模型中，半张图宽对应半个视场角：

$$
\tan(\theta_w/2)=\frac{W/2}{f_x},
\qquad f_x=\frac{W/2}{\tan(\theta_w/2)}.
$$

同理 $f_y=(H/2)/\tan(\theta_h/2)$，主点设为 $(W/2,H/2)$。若 `W=518`、水平 FOV 为 90°，则 $f_x=259$ 像素。若把角度 90 当成弧度传入公式，结果会完全错误。

内参 $K$ 的单位是像素，它将相机空间坐标的比值变成像素位置；它不承担相机位置或世界尺度估计。

## 6. DPT：从区域特征恢复每个像素的深度

### 6.1 为什么只用最后一个 token 还不够

camera token 汇总整帧适合输出少量相机参数，但每个像素需要一个深度。因此深度头要重新利用 patch 网格的空间排列。

DPT（Dense Prediction Transformer，密集预测头）读取 aggregator 输出的层索引 `[4,11,17,23]`，去掉 camera/register token，将 patch token 转为二维特征图。各层通过 1×1 卷积投影、不同倍率缩放，再由逐层融合块恢复高分辨率输出。[依赖确认]

```text
四个深度层的 patch tokens
 -> B*S×2048×21×37 特征网格
 -> 各层通道投影和缩放
 -> 多尺度融合
 -> 插值到 H×W
 -> 2 通道 raw 输出
 -> 深度 + 置信度
```

多层融合保留浅层空间细节和深层跨帧信息。这是对网络设计的机制解释，不意味着每个层索引只负责某一个可直接命名的物体。

### 6.2 深度和置信度不是同一个数

本地 VGGT 深度头配置是 `output_dim=2, activation='exp', conf_activation='expp1'`。若 raw 的两个通道为 $a,b$，则：

$$
z=\exp(a),\qquad c=1+\exp(b).
$$

例如 $a=\log2$、$b=\log3$，输出深度 2，置信度 4。置信度可以大于 1，不能按百分之四百成功率解释。`conf_threshold=25` 也不是要求概率超过 25%，而是从子地图完整置信度数组计算第 25 百分位。

DPT 默认按最多 8 帧一块处理密集头，减轻头部激活压力；aggregator 仍处理整个输入窗口，因此这不是把整个模型切成了 8 帧独立推理。

## 7. 从深度到相机坐标点：完整推导

### 7.1 前向投影先说明深度的含义

相机点 $p_C=(x,y,z)$ 的针孔投影为：

$$
u=f_x\frac{x}{z}+c_x,\qquad v=f_y\frac{y}{z}+c_y.
$$

这里 $z$ 是光轴坐标，不是射线距离 $\sqrt{x^2+y^2+z^2}$。已知像素 $(u,v)$ 和深度 $z$，把公式分别对 $x,y$ 求解：

$$
x=\frac{(u-c_x)z}{f_x},\qquad y=\frac{(v-c_y)z}{f_y}.
$$

于是：

$$
p_C=z K^{-1}[u,v,1]^{\mathsf T},
\qquad K=\begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}.
$$

这是一个 3 维向量，没有齐次第四分量；之后再附加 1，才能乘图节点的 4×4 homography。

### 7.2 手算墙上的一个点

用便于计算的教学相机：$f_x=f_y=500$ 像素，主点 $(320,240)$，墙像素 $(420,290)$，深度 2。反投影为：

$$
p_C=(0.4,0.2,2),
\qquad \|p_C\|=\sqrt{0.16+0.04+4}=\sqrt{4.2}\approx2.049.
$$

深度 2 与射线长 2.049 不同。若相机到世界变换是无旋转、平移 $(1,0,0)$，世界点变成 $(1.4,0.2,2)$。第 3 篇将说明这个变换在图里如何得到和更新。

### 7.3 当前依赖中函数名与返回值不一致

`depth_to_world_coords_points()` 返回三项：世界/局部参考坐标、相机坐标、有效深度掩码。但当前 fork 的 `unproject_depth_map_to_point_map()` 追加到结果列表的是第二项 `cam_coords_points`，并丢掉掩码。

`Solver.add_points()` 把这个返回值命名为 `world_points`，并保存到 `Submap.pointclouds`。**它的实际坐标系仍是逐帧相机系。** 不能依据变量名说代码已经做了外参逆变换。原掩码虽然检查 `depth>eps`，却没在这里成为筛选条件；实际点云读取主要使用置信度掩码。[固定依赖确认]

若深度出现 NaN、Inf 或极端值，单纯“正深度激活”不能代替数值有效性过滤。独立 `view_pcd.py` 在文件读取阶段会删除 NaN/Inf，但这不是建图阶段已经做了相同处理。

## 8. 置信度门槛如何改变可读取点

设一个极小子地图的置信度排序为 `(2,4,6,8)`，`np.percentile(...,25)` 在线性插值下为 3.5，实际阈值为 `3.5+1e-6`；点的读取条件是 $c>\tau$，因此保留 4、6、8 三个点。

$$
\tau=\operatorname{percentile}(\{c_{iuv}\},p)+10^{-6},
\qquad m_{iuv}=(c_{iuv}>\tau).
$$

这里阈值是在**整个子地图所有帧**上计算，不是每帧分别保证保留 75%。若置信度全部相同，`+1e-6` 与严格大于会使所有点被拒绝；所以“25 百分位一定保留 75%”只对没有大量并列值的通常分布近似成立。

完整局部点和颜色仍保留在内存；读取时才生成高置信度视图。因此调高门槛可能降低输出点数，却不按同样比例降低持久数组内存。

## 9. 精度路径与生命周期

`CompatibleVGGT` 关闭 point/track 头，aggregator 选择 FP16 或 BF16，geometry tokens 转为 FP32 再进入 camera/DPT heads。模型输出接口不变，但精度选择来自本地适配层。

```text
半精度模型权重/特征提取
 -> 多层 tokens
 -> FP32 geometry_tokens
 -> camera/depth 头
 -> CPU NumPy 几何数组
```

`torch.no_grad()` 禁用梯度记录，`model.eval()` 改变训练/推理行为；两者都不会把模型参数自动移回 CPU。`Submap.frames` 和 SALAD vectors 仍可能长期驻留 CUDA。因此 GPU 内存要拆成权重、瞬时激活、持久图像/描述子、缓存四部分，而不是只看窗口帧数。

## 10. 本篇证据与下一步

| 结论 | 源码入口 |
|---|---|
| crop/pad、518 与 14 | `vggt/utils/load_fn.py::load_and_preprocess_images` |
| 特殊 token、24组、交替注意力 | `vggt/models/aggregator.py::Aggregator` |
| SDPA、q/k 处理 | `vggt/layers/attention.py::Attention.forward` |
| 四次相机修正 | `vggt/heads/camera_head.py::trunk_fn` |
| 多尺度深度、exp/expp1 | `vggt/heads/dpt_head.py`、`head_act.py` |
| 编码順序与外参 | `vggt/utils/pose_enc.py`、`rotation.py` |
| 实际相机坐标返回 | `vggt/utils/geometry.py::unproject_depth_map_to_point_map` |
| 本地精度适配 | `vggt_slam/model_utils.py` |

现在我们有了每帧相机系点，仍然缺少不同窗口共享的世界参考。下一篇将把局部外参和重叠帧转成 SL(4) 节点、尺度测量和因子，并追踪后端如何真正修改地图。
