# 附录 C：源码阅读地图

如果要继续改造项目，建议按数据流而不是按目录字母顺序阅读。下面的路线把每个关键符号和需要验证的问题配在一起。

## C.1 从入口开始

| 顺序 | 文件/符号 | 读完应回答 |
|---:|---|---|
| 1 | `main.py::main` | 输入文件怎样筛选、窗口何时触发、优化和可视化何时调用 |
| 2 | `main_realtime.py::main` | RealSense 帧怎样进入队列，solver 如何串行化 |
| 3 | `vggt_slam/frame_overlap.py::FrameTracker` | 一帧为什么成为关键帧，位移量实际如何计算 |
| 4 | `vggt_slam/solver.py::Solver.__init__` | Viewer、GraphMap、PoseGraph、ImageRetrieval 怎样组合 |
| 5 | `vggt_slam/solver.py::run_predictions` | 一个窗口的 GPU/CPU 边界、回环顺序和输出字典 |
| 6 | `vggt_slam/solver.py::add_points` | 预测如何变成 Submap 字段和图边 |

## C.2 模型和几何链

```text
load_and_preprocess_images
  -> CompatibleVGGT.forward
  -> aggregator
  -> camera_head / depth_head
  -> pose_encoding_to_extri_intri
  -> unproject_depth_map_to_point_map
  -> Submap.add_all_points
```

| 文件/符号 | 重点 |
|---|---|
| `vggt_slam/model_utils.py::CompatibleVGGT` | 关闭的 head、dtype 分支、权重加载和 `target_tokens` |
| `third_party/vggt/vggt/models/aggregator.py` | patch token、register token、global attention、`get_similarity` |
| `third_party/vggt/vggt/heads/camera_head.py` | pose encoding 的四次迭代更新 |
| `third_party/vggt/vggt/heads/dpt_head.py` | inverse depth、confidence、上采样输出 |
| `third_party/vggt/vggt/utils/geometry.py` | 深度反投影、正深度掩码和返回值语义 |
| `vggt_slam/slam_utils.py` | 图像预处理、外参反演、相机分解 |

读到 `world_points` 时，立刻回到 `geometry.py` 看函数的第二个返回值，确认它是相机坐标还是世界坐标；不要只看预测字典键名。

## C.3 子地图和图后端链

```text
Submap.add_all_poses/add_all_points
  -> Solver.add_edge
  -> PoseGraph.add_homography
  -> PoseGraph.add_between_factor/add_prior_factor
  -> PoseGraph.optimize
  -> Submap.get_points_in_world_frame/get_all_poses_world
```

| 文件/符号 | 重点 |
|---|---|
| `vggt_slam/submap.py::Submap` | 所有字段的所有权、置信度阈值、局部点持久化 |
| `vggt_slam/solver.py::add_edge` | overlap mask、尺度、节点初值、内部边和回环分支 |
| `vggt_slam/graph.py::PoseGraph` | `SL4`、噪声向量、factor 加入和 LM |
| `vggt_slam/map.py::GraphMap` | 子地图 id、普通/回环分类、输出遍历顺序 |
| GTSAM `SL4` traits | 15 维 chart、Local/ Retract、BetweenFactor 残差 |

最容易读反的两个方向是：

```text
Submap.poses       保存 world_to_cam
graph homography   作用在局部点上得到 world 点
```

先在纸上写齐次点的左右乘方向，再读 `get_points_in_world_frame()`，比凭变量名猜测更可靠。

## C.4 回环链

```text
ImageRetrieval.get_all_submap_embeddings
  -> GraphMap.retrieve_best_score_frame
  -> LoopMatchQueue
  -> Solver.run_predictions(compute_similarity=True)
  -> image_match_ratio
  -> Solver.add_points
  -> lc_submap + loop BetweenFactor
```

| 文件/符号 | 重点 |
|---|---|
| `vggt_slam/loop_closure.py::ImageRetrieval` | SALAD checkpoint、输入尺寸、描述子保存位置 |
| `vggt_slam/loop_closure.py::LoopMatchQueue` | 最大候选数和堆排序 |
| `vggt_slam/map.py::retrieve_best_score_frame` | L2 距离方向、跳过最新块和回环块 |
| `third_party/salad/salad/models_salad/aggregators/salad.py` | Sinkhorn、cluster、dustbin、描述子维度 |
| `third_party/vggt/vggt/models/aggregator.py::get_similarity` | q/k attention 的 token 切分和 top-quarter 统计 |
| `solver.py::add_points` 回环分支 | 两帧辅助块、两个 `add_edge()` 的方向 |

先分清三个数字：SALAD 距离、`image_match_ratio`、GTSAM factor error。它们来自三层不同的状态，不能交叉解释。

## C.5 输出和可视化链

```text
Submap.get_points_in_world_frame
  -> Solver.set_point_cloud
  -> Viser scene.add_point_cloud

GraphMap.write_poses_to_file
  -> Submap.get_all_poses_world
  -> poses.txt

GraphMap.write_points_to_file
  -> Submap.get_points_in_world_frame
  -> Open3D write_point_cloud
  -> binary PCD
```

| 文件/符号 | 重点 |
|---|---|
| `vggt_slam/solver.py::set_point_cloud` | Viser voxel 发生在哪一步 |
| `vggt_slam/viewer.py::visualize_frames` | camera-to-world 视锥和图像纹理 |
| `vggt_slam/map.py::get_all_cam_matricies` | 为什么姿态跳过回环块 |
| `vggt_slam/map.py::write_points_to_file` | 为什么 PCD 仍遍历全部子地图 |
| `view_pcd.py` | 独立读取、过滤无效值、自动视角 |
| `open_viewer.sh` | SwiftShader 浏览器参数 |

## C.6 修改前后的证据表

修改一个函数前，先写一行“输入—表示—输出—消费者”：

| 要改的地方 | 输入表示 | 输出表示 | 必须复核 |
|---|---|---|---|
| 关键帧门槛 | BGR 图像和 KLT 点 | bool + 新关键帧状态 | 窗口数量、最后一组触发 |
| 深度过滤 | `depth_conf` 和点图 | mask 后点/颜色 | 尺度估计点数、PCD 长度 |
| 回环阈值 | 归一化描述子 | LoopMatch 或空 | 召回、误检、ratio 二次门控 |
| 图噪声 | SL4 节点和测量 | factor error/新 Values | residual 方向、轨迹稳定性 |
| 显示降采样 | 世界点和颜色 | Viser 数组 | PCD 是否保持原规模 |
| 实时队列 | 相机帧 | 待处理窗口 | 丢帧上限、solver 锁、内存 |

## C.7 生成文档时的质量门

本教程的每篇核心章节都应满足：

1. 至少有一条从源码函数到结论的证据入口；
2. 关键状态说明创建、保留、更新和结束条件；
3. 公式写清变量形状和矩阵方向；
4. 把代码事实、依赖事实、通用理论和工程推断分开；
5. 能由一个案例或命令验证，而不是只给术语解释；
6. 相对链接和代码路径在当前仓库中存在。

若新增模块，先把它放进总 README 的表示链，再决定是否新增章节。这样可以避免文档变成按文件罗列的 API 索引，保持 Hydra 风格的因果阅读顺序。
