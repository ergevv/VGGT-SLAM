# VGGT-SLAM 项目讲解

本次聊天按“启动问题 → 模型内部 → 几何与图优化 → 回环 → 主循环”的顺序整理在 [docs/代码讲解/VGGT-SLAM-代码讲解.md](代码讲解/VGGT-SLAM-代码讲解.md)；它固定使用 1920×1080 → 294×518、patch=14、P=782 等具体数字追踪张量和坐标变换。

这份入口文档已经按 Hydra 教程的生成流程拆分：先建立表示链，再沿着数据生命周期阅读每一层，最后用 `office_loop` 的真实输出复盘。详细内容位于 [`docs/vggt_slam_tutorial/`](vggt_slam_tutorial/README.md)。

## 推荐阅读顺序

```text
系统问题与架构
  -> 运行时管线和状态所有权
  -> 图像怎样经过 VGGT 变成几何
  -> 局部几何怎样进入 SL(4) 图
  -> SALAD 怎样提出回环约束
  -> 图状态怎样变成世界地图和 PCD
  -> office_loop 完整案例
  -> 依赖边界、参数排障、源码阅读地图
```

| 章节 | 主要问题 |
|---|---|
| [教程索引](vggt_slam_tutorial/README.md) | 全部章节、表示链和读者验收问题 |
| [01：为什么需要 Submap](vggt_slam_tutorial/01_why_vggt_slam_needs_submaps.md) | 前端、局部几何、全局图和输出层怎样分工 |
| [01a：运行时管线与状态所有权](vggt_slam_tutorial/01a_runtime_pipeline_and_state_ownership.md) | 离线/实时路径、队列、GPU/CPU 边界和生命周期 |
| [02：从图像到相机坐标几何](vggt_slam_tutorial/02_from_images_to_vggt_geometry.md) | token、相机头、DPT 深度、置信度和反投影 |
| [03：从局部几何到 SL(4) 图](vggt_slam_tutorial/03_from_local_geometry_to_sl4_graph.md) | 尺度、节点、因子、残差、雅可比和 LM 更新 |
| [04：从 SALAD 到回环约束](vggt_slam_tutorial/04_loop_closure_and_map_lifecycle.md) | 描述子候选、双帧验证和回环辅助 Submap |
| [05：从图状态到世界地图](vggt_slam_tutorial/05_from_graph_to_world_map.md) | homography 如何改变点云、Viser 与 PCD 的分叉 |
| [06：office_loop 完整案例](vggt_slam_tutorial/06_complete_office_loop_case.md) | 用本地 473 个文件、311 行轨迹和稠密 PCD 串起全流程 |
| [附录 A：实现边界与依赖](vggt_slam_tutorial/appendix_a_implementation_boundaries_and_dependencies.md) | 项目代码、VGGT/SALAD、GTSAM 和论文的证据边界 |
| [附录 B：参数、显存与排障](vggt_slam_tutorial/appendix_b_parameters_memory_and_debugging.md) | OOM、回环、PCD 和 WebGL 空白页的检查顺序 |
| [附录 C：源码阅读地图](vggt_slam_tutorial/appendix_c_source_reading_map.md) | 按数据流定位函数和复核修改影响 |

## 统一阅读方法

每一层都按下面六个问题阅读：

1. 上游传入什么形状和坐标系的数据？
2. 内存中由哪个对象持有？
3. 哪些候选会被门槛或 mask 筛掉？
4. 优化、回环或显示更新时谁会改变它？
5. 它什么时候结束、被跳过或继续保留？
6. 哪个下游函数消费它？

文档中的证据标签含义如下：

- **[代码确认]**：可以直接定位到当前仓库的实现。
- **[依赖库确认]**：来自当前 checkout 实际调用的 VGGT、SALAD、GTSAM、Open3D 或 Viser。
- **[运行产物确认]**：来自本机日志、轨迹或 PCD 文件的测量。
- **[通用理论]**：相机模型、齐次坐标、李群和最小二乘的标准知识。
- **[工程推断]**：由调用关系、输入规模或资源表现推断出的解释，会单独标记。

## 当前案例的快速入口

```bash
# 直接读取已保存的稠密点云，不加载 VGGT
.venv/bin/python view_pcd.py --voxel_size 0.05 --port 8081
./open_viewer.sh 8081
```

本机 `logs/office_loop_poses_points.pcd` 的头部记录了 53,657,975 个点，文件约 858 MB；Viser 使用的 `--vis_voxel_size` 只影响浏览器发送数据，不会缩小这个 PCD，也不会降低 VGGT 一次 forward 的显存峰值。网页空白时，先看浏览器 Console 是否出现 `Could not create a WebGL context`，再决定查浏览器渲染还是查点云数据。

详细教程的研究审计底稿保存在 [`RESEARCH_LEDGER.md`](vggt_slam_tutorial/RESEARCH_LEDGER.md)，用于记录源码证据、依赖边界和易混淆结论。
