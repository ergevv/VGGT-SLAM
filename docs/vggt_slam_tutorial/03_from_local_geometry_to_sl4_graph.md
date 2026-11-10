# VGGT-SLAM 教程（三）：从局部几何到 SL(4) 位姿图

上一篇得到的是每帧的 `extrinsic`、`intrinsic` 和相机坐标点。它们仍然属于各自的局部解释：第一个 Submap 可能以自己的相机为原点，第二个 Submap 又以另一个窗口的局部原点表达。若直接拼接点，就会产生重影。

本篇解决的核心问题是：**共享帧和局部相对几何怎样变成一个可优化的全局图？** 文中会先解释 `add_edge()` 的工程路径，再追踪本机 GTSAM SL(4) 依赖中真实的 `BetweenFactor` 残差、15 维局部坐标、雅可比和重牵引。不要把普通 SE(3) 位姿图公式直接替换过来：当前项目保存的是 `SL4`。

## 1. 从两个局部窗口开始

假设第一块包含相机帧 $C_0,C_1,C_2$，第二块包含重叠帧 $C_2,C_3,C_4$。第一块的局部点记为 $p^{(0)}$，第二块的局部点记为 $p^{(1)}$。即使两个 `C_2` 来自同一张图片，VGGT 的窗口坐标解释也可能带有尺度或投影差异。

目标不是逐点把两块平均，而是找到每个帧节点的 4×4 变换 $H_i$，使得：

```text
局部点 p_i 经过 H_i 后进入同一个世界坐标
相邻窗口的重叠帧关系尽量一致
每个 Submap 内部的帧间关系尽量一致
回环提供的远距离关系也能一起满足
```

$H_i$ 是 projective homography（齐次投影变换），不是被代码限制为纯旋转平移的 `Pose3`。它作为 `gtsam.SL4` 写入 `PoseGraph.values`。

## 2. 先把代码里的矩阵方向固定下来

### 2.1 `pose` 是 world-to-camera 外参

`pose_encoding_to_extri_intri()` 得到 $E_i=[R_{C_iL}\mid t_{C_iL}]$，满足：

$$
p_{C_i}=R_{C_iL}p_L+t_{C_iL}.
$$

`Solver.add_points()` 保存：

```python
cam_to_world = closed_form_inverse_se3(extrinsics_cam)
world_to_cam = np.linalg.inv(cam_to_world)
current_submap.add_all_poses(world_to_cam)
```

所以 `Submap.poses` 仍是 `world_to_cam` 外参，名字 `poses` 不表示 camera-to-world。`cam_to_world` 只作为中间量，也没有直接写入 Submap。[代码确认]

### 2.2 `proj_mats` 不是相机外参

`K_4x4` 由每帧内参 $K_i$ 放入 4×4 单位矩阵左上角得到：

$$
K_i^{(4)}=\begin{bmatrix}f_x&0&c_x&0\\0&f_y&c_y&0\\0&0&1&0\\0&0&0&1\end{bmatrix}.
$$

`Submap.add_all_points()` 把它保存为 `proj_mats`。该字段名容易让人以为它是 VGGT 的投影矩阵；当前代码里它实际是扩展的 intrinsics。`get_all_poses_world()` 后来用 `proj_mats[index] @ inv(homography_world)` 构造 4×4 projective camera matrix，再通过 RQ 分解提取旋转和平移。[代码确认]

## 3. 相邻子地图怎样估计尺度

### 3.1 为什么只看重叠帧

两个窗口若没有共同观测，无法把当前局部点和上一块的点直接比较。代码选择上一普通子地图最后一个非回环帧与当前子地图第一帧：

```text
prior_submap.last_non_loop_frame
current_submap.frame_0
```

先从两个置信度图生成 `good_mask`。正常路径要求 prior 和 current 都高于各自阈值；点数小于 100 时降级为 prior 高置信度；再不足则使用正掩码。这个选择不是几何算法的普遍定理，而是当前工程代码的可观察门控。

### 3.2 `estimate_scale_pairwise()` 的确切公式

代码接收同形状的两组点 $X,Y\in\mathbb R^{N\times3}$，按每个点到各自局部原点的欧氏范数计算比值：

$$
\rho_i^X=\lVert X_i\rVert_2,\qquad
\rho_i^Y=\lVert Y_i\rVert_2,
$$

$$
 s_i=\frac{\rho_i^Y}{\rho_i^X},\qquad
\hat{s}=\operatorname{median}_{i=1}^{N}s_i.
$$

代码没有显式除零保护；如果 $\rho_i^X=0$，NumPy 会产生无穷或 NaN，最终中位数可能污染后续图。正确阅读方式是：尺度函数假定传入点和 mask 已经足够良好，而不是它自己保证鲁棒。[代码确认]

例如 $X_1=(1,0,0)$、$Y_1=(2,0,0)$，比值是 2；另外三个点比值为 1.9、2.1、8。中位数为 2，单个错误点不会像均值那样把尺度拉到约 3。这个抗离群例子是 [通用统计解释]，不是对任意错误对应都安全的保证。

### 3.3 当前代码如何组成重叠变换

代码先计算：

```python
P_temp = inv(prior_submap.proj_mats[-1]) @ current_submap.proj_mats[0]
t1 = P_temp[0:3, 0:3] @ current_points[good_mask]
t2 = prior_points[good_mask]
scale_factor = estimate_scale_pairwise(t1, t2)[0]
H_scale = diag(scale_factor, scale_factor, scale_factor, 1)
H_overlap = inv(prior_K4) @ current_K4 @ H_scale
```

这里 `P_temp` 使用的是两个扩展内参矩阵；它不是从完整 SE(3) 外参直接求出的对齐矩阵。代码再把 `H_overlap` 与 prior 节点的 graph homography 相乘，得到当前 Submap 第一个节点的初值：

$$
H_{w,submap}=H_{w,prior\_overlap}\,H_{overlap,prior\_current}.
$$

矩阵的物理解释需结合当前 fork 的投影参数约定；不要把变量名自动改写成标准 `world_T_camera`。本段直接描述实现，不给它添加源码没有声明的纯刚体性质。

## 4. Submap 内部关系和节点生命周期

### 4.1 第一个节点

第一块没有 previous submap。代码要求它必须是 `submap_id=0, frame_id=0`，插入单位矩阵：

$$
H_0=I_4.
$$

然后加入 `PriorFactorSL4(X(0), SL4(I_4), anchor_noise)`。它固定全局参考，否则只有相对因子的图可以整体乘上同一个群元素而保持相对关系不变。[代码确认 + 通用 gauge freedom]

### 4.2 后续窗口的第一个节点

当存在 prior overlap 时，当前首节点先作为新的 `SL4` value 加入，然后增加上一重叠节点到当前首节点的 `BetweenFactorSL4`。这条边的噪声是 `intra_submap_noise`，其 15 个 diagonal sigma 当前均为 0.05。

### 4.3 当前 Submap 内部节点

对 `world_to_cam` 中的第 `index` 帧，代码构造：

$$
H_{inner,index}=E_{index-1}E_{index}^{-1}.
$$

并把当前节点初值设成：

$$
H_{current}=H_{previous}H_{inner,index}.
$$

同时增加前一节点到当前节点的 `BetweenFactorSL4`，噪声使用 `inner_submap_noise`。这不是让点云每个像素都成为节点，而是用每个相机帧一个节点把整块几何挂到图上。

节点的 key 用 `X(submap_id + index)`；`X()` 是 GTSAM 的符号键编码，实际保存的是 long key，不应与 Python 的 submap ID 直接当作同一类型。若 `add_homography()` 发现 key 已初始化，只打印后返回；若 `add_between_factor()` 两端没有先加入节点，则抛出 `ValueError`。[代码确认]

## 5. SL(4) 到底表示什么

### 5.1 15 维从哪里来

一个任意 4×4 矩阵有 16 个元素。SL(4) 要求行列式为 1，局部自由度减少 1，因而 GTSAM `SL4::dimension=15`。当前官方源码把李代数分成：

```text
6 个 4D 反对称旋转分量
6 个对称非对角剪切分量
3 个迹为零的对角缩放分量
```

三者合计 15。它不是“3 平移 + 3 旋转 + 1 尺度”的七维参数；SL(4) 允许更一般的投影形变。当前 `PoseGraph` 仍称 homography，且不调用本地 `normalize_to_sl4()`，因为构造 `SL4(global_h)` 时 GTSAM constructor 会重新处理矩阵。[代码/依赖确认]

### 5.2 构造时发生了什么

官方 GTSAM 快照中的 `SL4(Matrix44)`：

1. 对输入矩阵做 SVD，$M=U\Sigma V^T$；
2. 如果 $UV^T$ 是反射，翻转 $U$ 的最后一列；
3. 用修正后的 $U\Sigma V^T$ 重建矩阵；
4. 检查奇异值乘积非零且有限；
5. 除以 $\det(M)^{1/4}$，使内部矩阵行列式为 1。

因此传入的 `global_h` 可能不是 determinant 1 的矩阵，但它不会原样存入 `Values`。本机验证 `diag(2,2,2,1)` 经过构造后变为 determinant 约 1 的矩阵；这解释了为什么“打印出的 homography”和 Python 初始矩阵可能有统一尺度差异。[依赖确认：本机 wheel + 2026-10-02 官方源码快照]

## 6. BetweenFactor 的真实残差：不是直接矩阵减法

### 6.1 物理关系

边 $i\to j$ 携带一个测量 $Z_{ij}\in SL(4)$。若当前节点是 $H_i,H_j$，由节点预测的相对变换为：

$$
\hat Z_{ij}=H_i^{-1}H_j.
$$

GTSAM `BetweenFactor` 的源码先调用 `traits<T>::Between(p1,p2)` 得到上式，再把测量和预测送入 `Local(measured, hx)`。对 SL4，当前 `ChartAtOrigin::Local` 使用 `Vee(T-I)`，不是默认把矩阵送入矩阵对数。因而精确的当前依赖表达为：

$$
 r_{ij}=\operatorname{Local}\left(Z_{ij},H_i^{-1}H_j\right),
$$

$$
 r_{ij}=\operatorname{Vee}\left(Z_{ij}^{-1}H_i^{-1}H_j-I_4\right)
\quad\text{（当前 SL4 chart 的展开）}.
$$

`Vee` 按固定的 15 维正交基抽取反对称、对称非对角和无迹对角分量。这里 residual 的维度为 15，不是 16，因为 determinant 约束去掉一个方向。

### 6.2 零残差意味着什么

当

$$
Z_{ij}^{-1}H_i^{-1}H_j=I_4
$$

且位于当前 chart 的有效范围时，残差为零，表示节点预测的相对关系与测量一致。它不表示每个节点的绝对世界坐标正确；整体 gauge 仍需 prior，且局部错误可能互相抵消。

对 $Z_{ij}=I$、$H_i=I$、$H_j=I$，残差显然为 0。对 $H_j=\operatorname{diag}(1.3,1/1.3,1,1)$，残差非零；本机 GTSAM 探针得到 15 维向量只有无迹对角分量非零，数值约为 `(0.3753, 0.0283, 0.0200)`，它不是“沿 x 方向 0.3 米”的直接编码。[依赖确认]

### 6.3 为什么教学中常见的 Log 公式仍会出现

一般李群教程常写：

$$
\log\left(Z_{ij}^{-1}H_i^{-1}H_j\right).
$$

这表示在群上取一个局部误差的通用思路，但当前 GTSAM SL4 `Local` 的具体 chart 是 `Vee(T-I)`；官方 `SL4::Logmap` 是另一个静态接口。两者在误差足够小时都表达“离单位元有多远”，数值却不一定相同。

因此本文严格区分：

- **[依赖精确形式]** `BetweenFactor`：`Local(measured, between)`，SL4 chart 展开为 `Vee(relative-I)`；
- **[教学简化]** `Log(Z^{-1}H_i^{-1}H_j)`，用于解释李群误差概念，不用于声称逐行等价。

## 7. 从残差到节点更新

### 7.1 右侧重牵引和 15 维增量

GTSAM 的 LieGroup 默认用：

$$
H_i' = H_i\operatorname{Retract}(\delta\xi_i),
\qquad \delta\xi_i\in\mathbb R^{15}.
$$

当前 SL4 chart 的 retraction 首先尝试：

$$
\operatorname{Retract}(\delta\xi)=\operatorname{SL4}\left(I_4+\operatorname{Hat}(\delta\xi)\right),
$$

若候选行列式非正或非有限，才退回矩阵指数 `Expmap`。`Hat` 把 15 个数放进 4×4 无迹李代数矩阵。每次 LM 更新不是直接把 15 个数加到矩阵 16 个元素上。

### 7.2 Jacobian 的两个非零块

在当前图值附近，残差只依赖 $H_i$ 与 $H_j$：

$$
 r_{ij}(\xi+\delta\xi)
 \approx r_{ij}+J_i\delta\xi_i+J_j\delta\xi_j.
$$

`BetweenFactor` 源码把 `H1/H2` 传给 `traits<T>::Between` 和 `Local`。对于当前 SL4 chart，本机对编译后的 factor 做 `linearize()` 探针，得到 $15\times15$ 的两块：

$$
J_i=-\operatorname{Ad}_{h(x)^{-1}},
\qquad J_j=I_{15},
\qquad h(x)=H_i^{-1}H_j
$$

这里的 `h(x)` 是 factor 在**当前图值**处的预测相对变换；只有在残差接近零、$h(x)\approx Z_{ij}$ 时，才可以把它近似写成 $-\operatorname{Ad}_{Z_{ij}^{-1}}$。本机对简单状态 `H_i=I`、`H_j=D` 做 `linearize()` 探针，确实得到第一块为 $-\operatorname{Ad}_{D^{-1}}$、第二块为单位阵。边不连接的其他节点 Jacobian 块为零。于是一个回环边可以通过共享节点把远处轨迹耦合进同一个正规方程。

### 7.3 加权最小二乘和 LM 阻尼

所有因子堆叠为：

$$
\min_{\delta\xi}\sum_k\|r_k+J_k\delta\xi\|_{\Omega_k}^2,
\qquad
\|r\|_{\Omega}^2=r^T\Omega r.
$$

若把所有变量按节点拼接，线性化后得到：

$$
(J^T\Omega J+\lambda D)\delta\xi=-J^T\Omega r.
$$

这里 $D$ 是 LM 使用的阻尼对角结构，$\lambda$ 随尝试步长调整。GTSAM `LevenbergMarquardtOptimizer` 先 linearize 图，再建立带阻尼系统，尝试增量；若真实代价下降不符合预测，就调大 lambda 重试。`PoseGraph.optimize()` 不设置自定义 lambda，也不读取最终迭代统计，只把 `optimizer.optimize()` 返回的 `Values` 存回 `self.values`。[依赖/代码确认]

对于代码中的噪声：`inner_noise` 和 `intra_noise` 都是 15 个 `0.05` 的 diagonal sigma，anchor 是 15 个 `1e-6`。若按高斯噪声解释，信息权重约为 $1/0.05^2=400$；anchor 的权重约为 $10^{12}$，所以首节点几乎固定。[工程解释；数值来自代码]

### 7.4 间接更新点云

优化变量不是每个点，而是节点 $H_i$。真正的传播路径为：

```text
Between/Prior residual
  -> 两个节点的非零 Jacobian 块
  -> 共享节点的 Hessian/正规方程耦合
  -> 15维 delta xi
  -> SL4 retract 写回 Values
  -> get_homography() 读出新矩阵
  -> Submap 点齐次变换改变
  -> Viser/PCD 看到校正后的位置
```

因此浏览器刷新发生在优化之后，但点的局部三维值没有被网络重新预测。这个区分对解释回环后的旧地图移动非常重要。

## 8. 两节点数值例子

构造一个只含无迹对角缩放的教学状态：

```text
H_i = I
H_j = diag(1.3, 1/1.3, 1, 1)
Z_ij = I
```

预测相对关系就是 $H_j$，所以残差为：

$$
 r=\operatorname{Vee}(H_j-I).
$$

它不等于把 `1.3-1` 直接当作一个平移误差，因为 SL4 的基包含对角缩放方向，且矩阵需要保持 determinant 1。优化器会寻找一个 15 维增量，让相对关系靠近 $I$；若首节点有强 prior，主要修正第二节点；若两节点都有软约束，误差可以按权重分摊。

物理画面是：两张局部地图的相对尺度/投影关系不一致，后端移动的是节点解释，而不是把墙上的每个点独立拉到平均位置。若一条边的端点未先 `add_homography`，代码会直接报错，这比产生一个无约束假结果更容易排查。

## 9. 本篇源码入口和边界

| 问题 | 入口 |
|---|---|
| 读取局部预测并挂图 | `vggt_slam/solver.py::add_points` |
| 尺度和重叠矩阵 | `vggt_slam/solver.py::add_edge`、`scale_solver.py` |
| 节点、prior、between | `vggt_slam/graph.py::PoseGraph` |
| 局部点变世界点 | `vggt_slam/submap.py::get_points_in_world_frame` |
| SL4 维度、constructor、retract、Local | GTSAM `SL4.h/SL4.cpp`，参考快照记录在附录 A |
| between 残差 | GTSAM `BetweenFactor.h::evaluateError` |
| LM 阻尼 | GTSAM `LevenbergMarquardtOptimizer.cpp` |

下一篇处理尚未解决的长程问题：即使相邻边都成立，回到旧地点时仍可能因累计漂移产生第二份办公室。需要把外观检索候选、注意力相似度和回环子地图接入这张图。
