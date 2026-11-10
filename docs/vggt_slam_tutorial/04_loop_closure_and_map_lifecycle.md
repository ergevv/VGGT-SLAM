# VGGT-SLAM 教程（四）：从 SALAD 检索到回环约束

相机沿走廊前进时，邻近 Submap 的 `BetweenFactor` 可以维持局部连续性；但走完整个办公室后，许多小误差已经累积，回到起点时，当前局部坐标可能和起点相距很远。单靠“最近时间帧”无法发现这次重访，因此系统增加一条外观旁路：先检索历史相似画面，再让 VGGT 验证两帧的跨图像关系，最后把验证结果转成 PoseGraph 的额外 factor。

本篇特别区分三个相似但不同的量：

```text
SALAD L2 距离       -> 外观检索候选，越小越像
image_match_ratio   -> VGGT global attention 派生统计，越大越像
GTSAM factor error   -> 图约束与当前节点状态的不一致代价
```

只有前两个用于决定候选是否进入图；第三个由后端优化阶段计算，不能拿来替代图像检索。

## 1. 回环问题的三层门控

### 1.1 第一层：历史搜索

当前窗口有若干查询帧。系统要在历史普通 Submap 中找“可能看过的同一地点”，但不能把最新普通块和当前块拿来当回环：它们本来就相邻，会把正常里程计关系误报成回环。

### 1.2 第二层：跨图像关系验证

外观相似并不等于同一位置。白墙、相似桌面和重复走廊会产生相近描述子，因此候选要经过双帧 VGGT 的 `compute_similarity=True`，用 aggregator global attention 统计跨帧 token 关系。

### 1.3 第三层：图约束应用

只有验证通过的候选才创建一个回环 Submap 并新增 between factors。回环不是在 `ImageRetrieval` 内直接移动历史节点；它只产生观测和约束，真正修改节点值仍由 `PoseGraph.optimize()` 完成。

## 2. SALAD 描述子是怎样产生的

### 2.1 输入预处理

`ImageRetrieval` 初始化时固定 `input_size=224`。每张 Submap frame 经过：

```text
Tensor -> PIL -> resize(224,224) -> ToTensor -> ImageNet mean/std normalize
```

这与 VGGT 的 518 宽 crop 不是同一输入。SALAD 不使用 VGGT 的 `target_tokens`，而是独立加载 DINOv2 backbone。

### 2.2 DINOv2 输出

固定 SALAD 配置为 `dinov2_vitb14`，通道数 768。224×224 除以 patch size 14 后是 16×16 的局部网格；DINOv2 同时给出：

- `f`：`B×768×16×16` 的局部 feature map；
- `t`：`B×768` 的全局 token。

模型 forward 中，前面的 DINOv2 blocks 在 `no_grad` 中运行，最后 4 个 blocks（SALAD `num_trainable_blocks=4`）保留可训练路径；推理时都在 eval/no_grad 语境中使用。[依赖确认]

### 2.3 Sinkhorn 局部聚合

SALAD 名称来自 *Sinkhorn Algorithm for Locally Aggregated Descriptors*。它建立两个局部量：

- `cluster_features` 把每个空间位置映射到 128 维 cluster 特征；
- `score` 为 64 个 cluster 产生每位置的分数。

再加一个 dustbin 行，使用 3 次 Sinkhorn 迭代得到近似质量分配 $P$。局部聚合的教学形式可以写成：

$$
 g_k=\sum_{u}P_{ku}f_u,
\qquad
\hat g_k=\frac{g_k}{\lVert g_k\rVert_2},
$$

其中 $u$ 是 16×16 局部位置，$k$ 是 64 个 cluster，$f_u\in\mathbb R^{128}$。同时将全局 token 经 MLP 变成 256 维并归一化。最终描述子拼接为：

$$
 d=[t_{256},\hat g_1,\ldots,\hat g_{64}]\in\mathbb R^{256+64\times128}=\mathbb R^{8448},
$$

最后整体 L2 归一化。代码默认模型输出的确是 8448 维；实际向量保存在 GPU 上，随 `Submap.retrieval_vectors` 长期保存。[依赖/代码确认]

一个理解 Sinkhorn 的小例子：两个空间位置和两个 cluster 的分配矩阵若为

$$
P=\begin{bmatrix}0.8&0.2\\0.2&0.8\end{bmatrix},
$$

第一个 cluster 的特征是 $0.8f_1+0.2f_2$，而不是只复制最强位置。这使描述子对局部纹理的一点遮挡不那么敏感，但不保证在重复结构中唯一。

## 3. 候选检索的精确代码路径

### 3.1 每帧描述子进入历史集合

`run_predictions()` 创建 `new_submap` 后立即调用：

```text
ImageRetrieval.get_all_submap_embeddings(new_submap)
  -> get_batch_descriptors(frames)
  -> SALAD model
  -> S×8448 CUDA tensor
  -> new_submap.retrieval_vectors
```

候选搜索发生在 `GraphMap` 已有的 `submaps` 中；当前 `new_submap` 尚未通过 `add_points()` 加入 map，所以不会与自身比较。

### 3.2 距离和筛选

对查询描述子 $q$ 与历史向量 $d_j$，代码使用：

$$
\delta_j=\lVert d_j-q\rVert_2,
\qquad
j^*=\arg\min_j\delta_j.
$$

由于向量被 L2 归一化，$
0\le \delta_j\le2$；代码的 `lc_thres=0.95` 是距离上限而不是相似度下限。接受条件为：

```text
submap_id != current_submap_id
submap 不是 latest non-loop 普通块
submap 不是回环辅助块
best_score < lc_thres
```

每个查询帧只保留自己的最佳历史帧。`LoopMatchQueue(max_size=max_loops)` 再用容量堆限制整个新子地图最多保留 `max_loops` 个候选。`max_loops=0` 的实际结果是空队列，因此可以关闭回环。[代码确认]

### 3.3 一个手算距离例子

假设两个已归一化的二维描述子：

$$
q=(0.8,0.6),\qquad d_1=(0.8,0.6),\qquad d_2=(0.6,0.8).
$$

则：

$$
\delta_1=0,\qquad
\delta_2=\sqrt{(0.6-0.8)^2+(0.8-0.6)^2}=0.283.
$$

若阈值 0.2，只接受 $d_1$；若阈值 0.95，两者都有可能成为候选，但每个查询只取最小的 $d_1$。这个门控只说明“值得几何验证”，不能说明相机位姿已经相等。

## 4. `image_match_ratio` 不是匹配点内点率

### 4.1 什么时候计算

如果 SALAD 返回一个或多个候选，`run_predictions()` 取当前查询帧和历史匹配帧，构造两帧 batch，并调用：

```python
predictions_lc = model(lc_frames, compute_similarity=True)
```

`CompatibleVGGT.forward()` 将 `compute_similarity` 传给 aggregator。aggregator 在 `target_layer` 的 global attention 中请求 q/k，调用 `get_similarity(k,q)`。

### 4.2 实际统计步骤

按当前 `third_party/vggt` 实现，过程可以展开为：

1. q、k 经过 attention head 维度处理；
2. 计算 `q @ k.transpose(-2,-1)`；
3. 转置并在最后一维 softmax；
4. 对 head 求均值；
5. 按 `tokens_per_img` 切出第一、第二帧区域；
6. 用第一帧每个查询 token 的最大注意力做归一化；
7. 对第二帧方向取 token 维最大值；
8. 取这些值的 top quarter 平均。

可以用符号写成一个教学近似：

$$
A=\operatorname{mean}_{h}\left[\operatorname{softmax}\left((q_hK_h^T)^T\right)\right],
$$

$$
 r=\operatorname{mean}\left(\operatorname{TopQuarter}\left(\max_{u\in\text{second}}\frac{A_{u}}{\max_{v\in\text{first}}A_v+10^{-8}}\right)\right).
$$

这不是几何点对应数量、RANSAC 内点比例或深度误差。它是 attention 派生的标量。当前 `Solver.run_predictions()` 使用 `image_match_ratio < 0.95` 拒绝候选，等价于保留 ratio 至少 0.95 的候选。[代码/依赖确认]

### 4.3 为什么这个数字可能超出直觉

该统计做了归一化和 top-quarter 平均，没有显式 clamp；attention 的实现、token 切片和 q/k shape 都影响数值。不要把 `0.95` 解读成“95% 像素正确”。它只是当前项目选定的二次门槛。

如果候选被拒绝，`predictions_lc=None`，`predictions["detected_loops"]` 设为空，当前窗口仍可作为普通 Submap 继续进入图；回环失败不会阻止正常建图。[代码确认]

## 5. 回环 Submap 怎样进入 PoseGraph

### 5.1 普通窗口先完成自己的路径

`add_points()` 先处理当前普通 Submap：

1. 把 VGGT 外参反投影结果写入 current working Submap；
2. 将普通块加入 `GraphMap`；
3. 用时间/窗口重叠调用 `add_edge(..., is_loop_closure=False)`；
4. 为块首节点、内部节点和内部因子建图。

这样当前查询帧首先有普通图节点，回环测量才有一个已存在的 query endpoint。

### 5.2 通过验证后创建辅助块

每个 `LoopMatch` 会创建一个新的 `lc_submap`：

- `set_lc_status(True)`；
- 保存两帧 `frames_lc` 和对应路径；
- `last_non_loop_frame_index=1`；
- 写入两帧的外参、内参扩展矩阵、局部点和置信度。

它的两帧分别来自当前查询和历史检索帧；两帧必须用一次双帧 VGGT 产生共同的相对解释。

### 5.3 两次 `add_edge()` 产生连接

代码对回环辅助块调用：

```text
add_edge(lc_id, 0, query_submap_id, query_frame, is_loop_closure=False)
add_edge(detected_submap_id, detected_frame, lc_id, 1, is_loop_closure=True)
```

第一次把回环块首节点放在查询帧附近，并为辅助块两帧加入内部关系；第二次以历史帧为 current、辅助块第二帧为 prior，加入历史节点与辅助节点之间的 between factor。`is_loop_closure=True` 分支只建立跨图约束，不再为 current 节点添加新初值，也不继续添加 current 子图内部边。[代码确认]

概念上它形成：

```text
query graph node  -- first relation --  lc frame 0
lc frame 0        -- internal edge  --  lc frame 1
history node      -- loop relation --  lc frame 1
```

图优化会通过这条链把 query 和 history 耦合。回环子地图不是把历史节点直接重命名，也不是把两份点云先验平均。

## 6. 回环实体的生命周期和下游影响

| 阶段 | 对象 | 状态 | 后续消费者 |
|---|---|---|---|
| 外观检索 | 描述子距离 | 候选计算 | `LoopMatchQueue` |
| 队列门控 | `LoopMatch` | 临时候选 | 双帧 VGGT |
| ratio 拒绝 | 候选 | 丢弃 | 无；普通窗口继续 |
| ratio 通过 | `lc_submap` | 持久保存在 `GraphMap`，标记 LC | PoseGraph、点云、Viser |
| LM 优化 | 回环 between factor | 图中的长期约束 | 新 `Values` |
| 导出轨迹 | LC 子地图 | `get_all_cam_matricies` 跳过 | `poses.txt` |
| 导出点云 | 当前 `write_points_to_file` 遍历所有子地图 | LC 点仍进入 PCD | binary PCD |

这解释了一个容易冲突的结论：回环子地图是“辅助约束块”，但不是“运行后一定被删除的临时变量”。普通轨迹统计跳过它，点云写出当前代码却没有跳过它。[代码确认]

## 7. 回环失败的边界和可观测量

### 7.1 漏检

真实回到旧位置却没有候选，可能是：

- `lc_thres` 太小；
- SALAD 权重没有加载到同一缓存或输入颜色/尺度异常；
- 历史块被跳过（例如只有一个普通 Submap）；
- 关键帧选择没有保存包含重访证据的画面；
- 图像外观变化超出描述子鲁棒性。

此时 `detected_loops=[]` 是正常数据路径，不等于 graph 或 depth 失败。

### 7.2 误检

外观候选错误但 ratio 恰好很高，可能产生错误 factor。后端目前使用固定 diagonal noise，没有在 `PoseGraph` 这里实现 robust kernel 或 switchable constraint；错误回环可能牵动多个节点。因此调大 `lc_thres` 会同时增加召回和误检风险，必须查看 factor error、轨迹跳变和候选帧图像。[代码确认 + 工程风险]

### 7.3 诊断顺序

```text
SALAD checkpoint 是否存在
 -> 当前 query/historical submap 是否被跳过
 -> best_score 和 lc_thres
 -> retrieved frame 文件名
 -> image_match_ratio
 -> detected_loops 长度
 -> graph.num_loop_closures
 -> optimize 前后 total error
 -> Viser/PCD 是否读取更新后的 H
```

不要只看最后的 `Total number of loop closures`；它在 ratio 通过时递增，可能早于 `add_points()` 以及最终 LM 成功完成。

## 8. 本篇证据入口

| 主题 | 文件/符号 |
|---|---|
| 描述子模型 | `third_party/salad/salad/eval.py::load_model` |
| DINO 特征和 token | `third_party/salad/salad/models_salad/backbones/dinov2.py` |
| Sinkhorn 与 SALAD 聚合 | `third_party/salad/salad/models_salad/aggregators/salad.py` |
| L2 历史搜索 | `vggt_slam/map.py::retrieve_best_score_frame` |
| 候选队列 | `vggt_slam/loop_closure.py::LoopMatchQueue` |
| 双帧验证 | `vggt_slam/solver.py::run_predictions` |
| 注意力 ratio | `third_party/vggt/vggt/models/aggregator.py::get_similarity` |
| 辅助块和 factor | `vggt_slam/solver.py::add_points/add_edge` |

下一篇把优化后的 `Values` 接到输出：一个 graph homography 怎样作用于数千万个局部点，为什么 Viser 的 voxel 参数与 PCD 文件大小无关，以及如何用独立点云查看器绕开浏览器 WebGL 问题。
