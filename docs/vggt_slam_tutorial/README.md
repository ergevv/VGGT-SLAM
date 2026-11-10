# VGGT-SLAM 详细教程

这套教程按照 Hydra 教程的生成流程重整 VGGT-SLAM 的阅读路径。它不把源码目录直接翻译成章节，而是沿着一条可复述的数据生命周期展开：

```text
磁盘图像 / RealSense 帧
  -> 预处理与关键帧候选
  -> 关键帧窗口与 Submap
  -> VGGT 共享视觉特征
  -> 相机、深度、置信度
  -> 相机坐标点云
  -> SL(4) 节点和 Between/Prior 因子
  -> SALAD 回环候选与双帧验证
  -> 优化后的世界系点云和相机轨迹
  -> Viser、poses.txt、binary PCD
```

每一层都回答六个问题：上游传入什么、内存中怎样表示、哪些候选被保留、数据如何更新、什么时候失效或结束、下游谁消费它。源码只作为行为证据，论文方法、固定依赖、通用数学和工程推断分别标记。

## 推荐阅读顺序

```text
01 系统问题与架构
  -> 01a 运行时管线与状态所有权
  -> 02 图像怎样经过 VGGT 变成几何
  -> 03 局部几何怎样进入 SL(4) 图
  -> 04 回环怎样生成附加约束
  -> 05 世界地图、Viser 和文件输出
  -> 06 office_loop 完整案例
  -> 附录 A 实现边界和版本
  -> 附录 B 参数、显存与故障排查
  -> 附录 C 源码阅读地图
```

| 篇目 | 主问题 | 阅读结果 |
|---|---|---|
| [01：为什么需要 VGGT-SLAM 与 Submap](01_why_vggt_slam_needs_submaps.md) | 单张图像、整段视频和全局地图之间缺少什么表示 | 能说清前端、模型、后端和可视化的职责边界 |
| [01a：运行时管线、队列和状态所有权](01a_runtime_pipeline_and_state_ownership.md) | 一帧图像在哪些阶段改变表示，谁持有可变状态 | 能定位显存、并发、窗口和生命周期问题 |
| [02：从图像到相机坐标几何](02_from_images_to_vggt_geometry.md) | VGGT 的 token、注意力、相机头、DPT 深度头如何产生三维点 | 能从像素、深度和内参手算一个点 |
| [03：从局部点到 SL(4) 位姿图](03_from_local_geometry_to_sl4_graph.md) | 子地图如何连接，尺度如何估计，GTSAM 因子如何更新节点 | 能追踪残差、雅可比、增量和点云写回 |
| [04：从外观检索到回环约束](04_loop_closure_and_map_lifecycle.md) | SALAD 候选、VGGT 注意力相似度和回环子地图的真实作用 | 不会把检索分数误认为几何配准 |
| [05：从图状态到世界地图和浏览器](05_from_graph_to_world_map.md) | 优化后的节点如何改变点云、轨迹、Viser 和 PCD | 能区分显示降采样和磁盘稠密输出 |
| [06：完整 office_loop 案例](06_complete_office_loop_case.md) | 用一条实际运行把所有表示串起来 | 能从日志反查某个结果来自哪一层 |
| [附录 A：实现边界与依赖](appendix_a_implementation_boundaries_and_dependencies.md) | 当前代码、VGGT/SALAD、GTSAM 和论文各自负责什么 | 不把外部依赖行为写成项目自有实现 |
| [附录 B：参数、显存和排障](appendix_b_parameters_memory_and_debugging.md) | 参数怎样沿数据链改变结果，如何排查空白页面和 OOM | 有可执行的检查顺序和中间量 |
| [附录 C：源码阅读地图](appendix_c_source_reading_map.md) | 读哪些文件、按什么顺序、每个符号验证什么 | 可以独立继续研究和改造项目 |
| [研究审计底稿](RESEARCH_LEDGER.md) | 记录表示来源、证据差异和复核结果 | 用于审查教程结论是否仍与当前代码一致 |

## 贯穿案例和读者契约

贯穿案例是一台相机沿办公室走廊移动，最后回到曾经经过的位置。它会反复出现：

- 相邻帧变化太小时，光流选择器拒绝新关键帧；
- 视差足够大时，若干帧组成一个 `Submap`；
- VGGT 为窗口共享 token，并输出每帧的 `pose_enc`、深度和置信度；
- 局部点先保留在相机/子地图坐标，图优化后才变成世界坐标；
- SALAD 只提出外观候选，双帧 VGGT 相似度再决定是否创建回环约束；
- 回环优化后，旧子地图通过新的 homography 重新投影，浏览器和 PCD 读取同一份更新后的图状态。

读者只需要会读 Python、理解矩阵乘法和基本三维坐标。不会默认读者已经了解 SLAM、VGGT、SALAD、SL(4)、GTSAM 或 WebGL；每个术语第一次出现时会说明它解决的具体问题。

## 统一证据标签

- **[代码确认]**：当前仓库中可以直接定位到的行为。
- **[依赖库确认]**：当前 checkout 实际调用的 `third_party/vggt`、`third_party/salad`、GTSAM、Open3D 或 Viser 行为。
- **[通用理论]**：相机模型、齐次坐标、李群和最小二乘的标准知识。
- **[论文描述]**：README 引用的 VGGT-SLAM 研究方法；不能自动替代运行时代码。
- **[工程推断]**：由输入规模、内存和调用关系推断出的原因或建议。

固定源码版本记录在附录 A。GTSAM 的 Python wheel 能确认 API 和版本，但不携带完整源码 commit；涉及 SL(4) 的精确实现时，附录 A 会同时给出构建时间附近的官方 GTSAM 源码快照，并把它标为依赖证据。

## 先建立总表示链

| 层 | 输入 | 存储 | 选择/过滤 | 更新或生命周期 | 消费者 |
|---|---|---|---|---|---|
| 原始图像 | 路径或相机 BGR 帧 | 文件、临时 `numpy` 数组 | 排除深度、txt、db；按文件名数字排序 | 离线直到读取；实时帧可丢弃 | 光流和预处理 |
| 关键帧 | 当前帧与上一关键帧角点 | `FrameTracker` 状态 | KLT 成功点至少 10 个；平均位移大于阈值 | 通过则替换参考关键帧 | `image_names_subset` |
| 子地图 | 窗口内关键帧 | `Submap` 的 frames、文件名、检索向量 | 窗口大小等于新帧数加重叠帧 | 加入 `GraphMap` 后保留；回环块有独立标记 | VGGT、图、检索、可视化 |
| VGGT 表示 | `S×3×H×W` 图像 | aggregator tokens 与 DPT 特征 | 使用相机/深度头；关闭 point/track 头 | 一次 forward 的临时激活 | 外参、内参、深度、置信度 |
| 局部几何 | 深度、内参、外参 | `Submap.pointclouds`、colors、conf | 读取时用子地图置信度百分位阈值 | 原始数组保留；高置信度视图按需生成 | 尺度、homography、PCD |
| 图后端 | 节点初值与相对 homography | GTSAM `Values` 和 factor graph | 节点必须先初始化；边连接已存在节点 | 每个子地图后 LM 优化；值被替换 | 世界点和相机轨迹 |
| 回环 | SALAD 描述子与双帧注意力 | `LoopMatch`、回环 `Submap`、附加边 | 最近邻 L2 距离门槛、最大候选数、match ratio | 候选拒绝即丢弃；通过则块保留并参与图/导出 | `PoseGraph.optimize()` |
| 输出 | 世界点、姿态 | Viser 对象、txt、binary PCD | Viser 可 voxel 降采样；PCD 不使用该参数 | 服务会话或磁盘文件 | 浏览器、Open3D、评测 |

## 阅读后的验收目标

读完主线后，应该可以回答：

1. 一个 `image_names_subset` 是怎样从原始目录产生的？
2. `world_points` 为什么在 `Solver.add_points()` 中实际还是相机坐标点？
3. `Submap.proj_mats`、`PoseGraph` homography 和 `Submap.get_all_poses_world()` 分别表示什么？
4. SALAD 距离、`image_match_ratio` 和 GTSAM factor error 哪一个是外观候选、哪一个是注意力相似度、哪一个是后端代价？
5. LM 求解的 15 维增量如何通过 SL(4) retraction 改变 `Values`，再改变一个点的世界位置？
6. 为什么 `--vis_voxel_size` 能降低浏览器压力，却不能降低一次 VGGT forward 的显存？
7. 如果网页空白但 WebSocket 曾经连接，应该先查数据还是先查 WebGL？

如果其中任一问题答不出来，应回到对应章节的“来源—表示—筛选—更新—消费者”表，而不是只看术语表。
