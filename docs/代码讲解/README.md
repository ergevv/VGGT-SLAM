# VGGT-SLAM 代码讲解

这里保存从项目启动到 VGGT-SLAM 完成建图的代码阅读材料。

主文档现已包含聊天主线和仓库中 vggt_slam_tutorial 的完整章节原文，约 3290 行；不需要跳转到其他文件才能看到 Aggregator、Camera/DPT head、SALAD、SL(4)、回环、PCD 和排障的详细解释。

如果要查看 10 月 10 日周六 14:32 开始的原始聊天，请直接打开 [2026-10-10-14-32-VGGT-SLAM-聊天原文.md](2026-10-10-14-32-VGGT-SLAM-聊天原文.md)。该文件按原顺序保存用户和助手消息，不是改写后的教程。

推荐先读 [VGGT-SLAM-代码讲解.md](VGGT-SLAM-代码讲解.md)，再按需要展开：

1. [运行时管线、状态所有权与线程边界](../vggt_slam_tutorial/01a_runtime_pipeline_and_state_ownership.md)
2. [从图像到 VGGT 几何](../vggt_slam_tutorial/02_from_images_to_vggt_geometry.md)
3. [从局部几何到 SL(4) 图](../vggt_slam_tutorial/03_from_local_geometry_to_sl4_graph.md)
4. [从 SALAD 到回环约束](../vggt_slam_tutorial/04_loop_closure_and_map_lifecycle.md)
5. [从图状态到世界地图、Viser 和 PCD](../vggt_slam_tutorial/05_from_graph_to_world_map.md)
6. [office_loop 完整案例](../vggt_slam_tutorial/06_complete_office_loop_case.md)
7. [参数、显存和故障排查](../vggt_slam_tutorial/appendix_b_parameters_memory_and_debugging.md)

总文档中使用的数字分为两类：

- 当前代码或本地运行数字：例如网络宽度 518、office_loop 的网络高度 294、patch 大小 14、PCD 的 53,657,975 个点。
- 固定教学实例数字：例如 B=1、S=5 或 S=17，用于逐层追踪张量，不代表每次运行都一定得到同样大小。
