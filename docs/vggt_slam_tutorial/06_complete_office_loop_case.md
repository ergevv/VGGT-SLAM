# VGGT-SLAM 教程（六）：`office_loop` 完整案例

前五篇按表示层拆开了系统。本篇把同一条 `office_loop` 运行重新走一遍：从 473 个磁盘文件开始，经过光流关键帧、窗口、VGGT、回环和图优化，最后落到 311 行轨迹与约 5,366 万点的 PCD。每一节都标出“运行产物确认”或“源码推导”，避免把观察到的结果和没有日志直接记录的内部状态混为一谈。

## 1. 案例输入和验收对象

本机项目目录中的案例数据和输出如下：

| 对象 | 位置 | 已确认内容 |
|---|---|---|
| 图像序列 | `office_loop/` | 473 个文件（运行前端会排除名称含 `depth`、`txt`、`db` 的文件） |
| 轨迹文件 | `logs/office_loop_poses.txt` | 311 行，每行是 `frame_id x y z qx qy qz qw` |
| 点云文件 | `logs/office_loop_poses_points.pcd` | binary PCD，53,657,975 点，约 858 MB |
| 独立查看器 | `view_pcd.py` | 默认读取上述 PCD，在 8081 端口提供 Viser |
| 浏览器启动脚本 | `open_viewer.sh` | 用独立 Chrome 配置和 SwiftShader 规避本机 WebGL 初始化失败 |

这里的 311 行不是“原始输入 473 帧全部保留”的证明。前端只把通过位移门槛的帧加入窗口，窗口又包含重叠帧；回环辅助帧还会被轨迹写出跳过。正确做法是沿每一层的计数器确认结果。

## 2. 启动参数怎样决定这次运行

主入口 `main.py` 的关键默认值是：

```text
submap_size = 16
overlapping_window_size = 1
max_loops = 1
min_disparity = 50
conf_threshold = 25.0
lc_thres = 0.95
```

它们分属不同层：

| 参数 | 影响的层 | 具体作用 |
|---|---|---|
| `min_disparity` | 前端 | KLT 追踪的当前帧相对关键帧位移门槛 |
| `submap_size` | 窗口调度 | 每块新增帧数，不包含重叠帧 |
| `overlapping_window_size` | 图连接 | 下一块保留多少上一块尾部帧；当前代码只支持默认 1 的路径 |
| `conf_threshold` | 点输出 | 每个 Submap 过滤其置信度最低百分位 |
| `lc_thres` | 外观检索 | SALAD L2 距离上限，越大越容易产生候选 |
| `max_loops` | 回环候选 | 每个新 Submap 最多保留的 LoopMatch 数，0 可关闭 |
| `vis_voxel_size` | 浏览器 | 只降低发送给 Viser 的点数 |
| `--log_results` | 文件输出 | 写轨迹文本，并默认写稠密 PCD |
| `--keep_alive` | 服务生命周期 | 建图结束后继续运行 8080 服务 |

一次可复现的离线命令可以写成：

```bash
.venv/bin/python main.py \
  --image_folder office_loop \
  --max_loops 1 \
  --vis_voxel_size 0.05 \
  --log_results \
  --log_path logs/office_loop_poses.txt \
  --keep_alive
```

若目标是先确认算法输出，再查看保存文件，可以暂时不加 `--keep_alive`，建图完成后运行独立查看器。若要观察逐块更新，再增加 `--vis_map`；若启用 `--vis_imgs`，浏览器还要接收每个相机视锥的图像纹理，交互压力会上升。[代码确认]

## 3. 第一层：473 个文件怎样变成关键帧窗口

### 3.1 文件筛选和排序

`main.py` 用 `glob` 读取目录，排除文件名中包含 `depth`、`txt` 或 `db` 的条目，再用 `sort_images_by_number()` 按文件名中的数字排序。`office_loop` 的 473 个文件进入循环后，并不是每一张都自动送入 VGGT。

### 3.2 KLT 追踪的实际判定

`FrameTracker.compute_disparity()` 的状态只有上一关键帧、灰度图和角点。第一帧或有效角点少于 10 个时立即初始化并返回 True。其他帧执行 `calcOpticalFlowPyrLK`，保留 `status==1` 的点。

代码写法是：

```python
displacement = np.linalg.norm(good_next - good_kf, axis=1)
mean_disparity = np.mean(displacement)
```

由于 OpenCV 点数组仍是 `N×1×2`，沿 `axis=1` 后得到 `N×2`，所以当前实现实际平均的是两个坐标方向的绝对差分：

$$
d_{code}=\frac{1}{2N}\sum_{i=1}^{N}\left(|u_i'-u_i|+|v_i'-v_i|\right),
$$

而不是先对每个点算二维欧氏长度再平均。只有 `d_code > min_disparity` 才将该帧追加到 `image_names_subset`，并更新参考关键帧。[代码确认]

这处细节会影响“为什么某一张图没有进入模型”的解释：不能直接把参数名 `disparity` 当成一个经过实现保证的 Euclidean disparity。

### 3.3 窗口触发和重叠

循环每通过一帧就可能追加到 `image_names_subset`。当：

```text
len(image_names_subset) == submap_size + overlapping_window_size
或当前遍历到最后一个文件
```

就调用 `solver.run_predictions()`。默认设置下，完整窗口通常由 16 个新增帧加 1 个重叠帧组成；处理结束后列表重置为最后 1 帧。这个重叠帧会在后续窗口再次出现，用来估计相邻块之间的尺度和 BetweenFactor。[代码确认]

## 4. 第二层：一个窗口内部发生了什么

### 4.1 图像张量和 Submap 初始化

`run_predictions()` 调用 `load_and_preprocess_images()`，将窗口图像送到 CUDA（若可用）。随后创建一个新的 `Submap`：

```text
frames          -> S×3×H×W GPU Tensor
frame_ids       -> 从文件名解析出的数值
retrieval_vectors -> SALAD S×8448 CUDA Tensor
last_non_loop_frame_index -> S-1
```

新块此时仍在 `current_working_submap` 中，尚未通过 `add_points()` 放进 `GraphMap`。这解释了为什么当前查询窗口不会在自己的历史库里检索自己。

### 4.2 VGGT 一次 forward 的输出

`CompatibleVGGT` 关闭 `point_head` 和 `track_head`，只计算相机与深度相关路径。`run_predictions()` 之后把 `pose_enc` 变成 `extrinsic` 和 `intrinsic`，并保留：

```text
depth          -> S×H×W×1
depth_conf     -> S×H×W
world_points   -> 上游命名，后续由 depth/intrinsic 再次生成相机坐标点
target_tokens  -> 特殊保留项，未统一转 CPU
```

普通预测中的 Tensor 会转成 float、CPU NumPy，再交给 `add_points()`。这样做释放了模型推理激活，但 `Submap.frames` 和 `retrieval_vectors` 的持久 GPU 状态仍可能保留；显存排查不能只看 `predictions` 是否已经转 CPU。[代码确认]

## 5. 第三层：普通 Submap 如何加入图

### 5.1 第一个块

第一块的 `submap_id=0`。`add_points()`：

1. 用深度和内参生成相机坐标点；
2. 将图像转成颜色数组；
3. 计算 `world_to_cam` 外参并保存；
4. 用置信度百分位设置块阈值；
5. 加入 `GraphMap`；
6. `add_edge(0,0,None,None,is_loop_closure=False)` 创建第一个节点和 `PriorFactorSL4`。

第一个节点提供 gauge anchor。没有它，所有节点同时乘一个全局变换时，优化问题没有固定参考。

### 5.2 后续块

后续普通块先和上一普通块的最后非回环帧比较重叠点。代码按共同像素索引取点，计算从当前局部尺度到先前局部尺度的中位数比例，构造：

```text
H_scale = diag(scale, scale, scale, 1)
H_overlap = inv(K_prior) @ K_current @ H_scale
H_first_world = H_prior_last_world @ H_overlap
```

然后把当前块的第一节点和相邻节点的 BetweenFactor 写进 GTSAM。块内部的相邻帧也根据 `world_to_cam[index-1] @ inv(pose)` 建立内部边。此时世界点还没有被永久重采样；图值是局部点进入世界系的唯一桥梁。

### 5.3 每块后的优化

`main.py` 在 `solver.add_points(predictions)` 后调用 `solver.graph.optimize()`。GTSAM LM 使用当前 `Values` 和 factor graph 计算增量，更新 `PoseGraph.values`。下一次 `get_points_in_world_frame()` 才会读到这些新 homography。

若使用 `--vis_map`：

```text
无回环 -> update_latest_submap_vis()
有回环 -> update_all_submap_vis()
```

后者是必要的，因为回环可能改变旧块和当前块之间的全局相对位置。[代码确认]

## 6. 第四层：office_loop 的回环路径

### 6.1 SALAD 只产生候选

新块先计算每帧 8448 维描述子，再在历史普通块中找 L2 距离最小的帧。当前代码跳过新块、最新普通块和已经标记为回环的块。默认 `max_loops=1`，每个新块最多把一个最佳候选交给双帧 VGGT。

一个候选记录的是：

```text
query_submap_id / query_submap_frame
detected_submap_id / detected_submap_frame
best_score
```

`best_score < 0.95` 才能进入下一层；它是归一化描述子的欧氏距离上限。[代码确认]

### 6.2 双帧 VGGT 验证

通过检索后，代码将当前查询帧和历史帧堆成两帧 batch，再调用：

```python
model(lc_frames, compute_similarity=True)
```

aggregator 的 global attention 派生 `image_match_ratio`。当前实现要求 ratio 至少 0.95；否则清空 `predictions["detected_loops"]`，当前窗口仍按普通块处理。只有通过时，才把双帧的外参、内参、深度和置信度保存到 `predictions`，并递增图中的 loop closure 计数。[代码确认]

因此“检测到候选”和“图里真的增加回环因子”是两个不同事件；排障时应分别记录 `best_score`、`image_match_ratio`、`detected_loops` 和 `graph.get_num_loops()`。

### 6.3 辅助回环块

通过验证后，`add_points()` 创建 `is_lc_submap=True` 的两帧辅助块。它不直接修改历史普通节点，而是加入一条连接链：

```text
query 普通节点 -- 辅助块第 0 帧
辅助块第 0 帧 -- 辅助块第 1 帧
历史普通节点 -- 辅助块第 1 帧（回环 BetweenFactor）
```

`write_poses_to_file()` 跳过这类辅助块；`write_points_to_file()` 当前会遍历它们。这就是轨迹文件和 PCD 文件对回环观测处理不同的原因。[代码确认]

## 7. 第五层：输出怎样对应到本次运行

### 7.1 轨迹文件

本地 `logs/office_loop_poses.txt` 有 311 行。每行 8 个浮点数：第一列是从文件名解析的 frame id，接着是平移和四元数。开头几行包含：

```text
1.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
10.0 -0.0293 -0.0231 0.0839 ...
14.0 -0.0297 -0.0525 0.1616 ...
```

第一行接近单位姿态，符合第一个图节点被 PriorFactor 固定在单位矩阵的初始化路径。轨迹末尾的 frame id 到 472，说明输入文件编号和轨迹行数是两个不同维度。[运行产物确认 + 源码推导]

### 7.2 稠密 PCD

本地 PCD 头部为：

```text
FIELDS x y z rgb
TYPE F F F F
WIDTH 53657975
POINTS 53657975
DATA binary
```

它是 `GraphMap.write_points_to_file()` 将每个子地图的高置信度世界点和颜色拼接后写出的结果。文件大小约 858 MB，所以直接用浏览器显示全部点会同时承受 Open3D 读取、WebSocket 传输和 WebGL 缓冲压力。

### 7.3 独立查看验证

先运行：

```bash
.venv/bin/python view_pcd.py --voxel_size 0.05 --port 8081
```

终端会打印原始点数和显示点数。另开终端：

```bash
./open_viewer.sh 8081
```

本地记录的独立查看器日志显示浏览器成功连接并创建 SwiftShader WebGL；这条路径不加载 VGGT、不执行回环，也不依赖主程序是否仍在运行。它验证的是 PCD 内容和 Viser 浏览器链路。[运行产物确认]

## 8. “网页空白”的案例诊断

这次问题的浏览器 Console 报告：

```text
THREE.WebGLRenderer: A WebGL context could not be created
GL_VENDOR = Disabled
GL_RENDERER = Disabled
Sandboxed = yes
```

同时服务日志出现：

```text
(viser) Connection opened
(viser) Connection closed
```

两者并不矛盾：WebSocket 可以打开并接收场景消息，Three.js 仍可能在创建渲染器时失败。主服务没有因为这条日志证明“数据不存在”，而是浏览器在绘制阶段失败。

推荐的分层排查顺序：

```text
1. curl/浏览器确认 8080 HTTP 页面能打开
2. 看 Console 是否是 WebGL context 错误
3. 用 open_viewer.sh 启动 SwiftShader
4. 用 view_pcd.py 直接读取 PCD
5. 只有独立查看也失败时，检查 PCD、Open3D 和服务端数据
```

如果 `view_pcd.py` 能显示，说明点云文件至少能被 Open3D 解析；如果主 Viser 仍空白，问题集中在主服务的场景更新、端口或浏览器渲染配置，不必重新运行 VGGT。

## 9. 用案例反查源码的最短路线

| 观察到的现象 | 先读的符号 | 要确认的量 |
|---|---|---|
| 某些图像没有进模型 | `FrameTracker.compute_disparity` | 有效角点、`d_code`、`min_disparity` |
| 窗口大小不符合预期 | `main.main` | `submap_size + overlap` 和最后一组触发 |
| 新块与旧块比例异常 | `Solver.add_edge` | good mask、scale median、`H_overlap` |
| 回环只检索不入图 | `Solver.run_predictions` | SALAD score、ratio、`detected_loops` |
| 轨迹有 311 行但点更多 | `GraphMap.get_all_cam_matricies/write_points_to_file` | LC 跳过规则与稠密点拼接 |
| 浏览器白屏但连接打开 | Chrome Console、`open_viewer.sh` | WebGL context，不是 WebSocket 是否打开 |
| 文件很大但显示还可以 | `Solver.set_point_cloud` | Viser voxel 与 PCD 独立 |

这条路线把一个“结果不对”的模糊问题拆成可观察的状态：输入数量、关键帧数量、子地图节点、候选数量、优化后的值、世界点数量和浏览器渲染器。后续改参数时，只需定位是哪一层的状态先偏离预期。

## 10. 本篇证据入口

| 主题 | 文件/符号 |
|---|---|
| 离线调度 | `main.py::main` |
| KLT 选择 | `vggt_slam/frame_overlap.py::FrameTracker` |
| 窗口推理 | `vggt_slam/solver.py::run_predictions` |
| 点和图写入 | `vggt_slam/solver.py::add_points/add_edge` |
| 输出文件 | `vggt_slam/map.py::write_poses_to_file/write_points_to_file` |
| 真实运行产物 | `logs/office_loop_poses.txt`、`logs/office_loop_poses_points.pcd` |
| 独立查看路径 | `view_pcd.py`、`open_viewer.sh` |

附录会把这些入口按依赖版本、参数和源码符号重新索引，方便读者在修改实现后复核本教程哪些结论仍然成立。
