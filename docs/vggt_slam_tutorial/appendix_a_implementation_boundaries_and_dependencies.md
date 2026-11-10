# 附录 A：实现边界、依赖版本与证据分层

详细教程把“项目代码做了什么”“依赖库做了什么”“论文或通用理论说了什么”分开。本附录给出复核边界，避免升级依赖后仍把旧行为当成当前行为。

## A.1 当前项目负责的部分

| 功能 | 项目入口 | 项目实际决定的行为 |
|---|---|---|
| 离线序列调度 | `main.py` | 文件筛选、排序、关键帧循环、窗口触发、日志和服务生命周期 |
| 实时采集 | `main_realtime.py` | RealSense 读取、捕获线程、待处理队列、solver 锁和断线重连 |
| 关键帧选择 | `vggt_slam/frame_overlap.py` | KLT 参数、有效点数量、位移门槛和参考帧替换 |
| 模型桥接 | `vggt_slam/model_utils.py` | 权重加载、dtype、关闭 point/track head、设备选择 |
| 推理编排 | `vggt_slam/solver.py` | Submap 创建、VGGT/SALAD 顺序、输出转换和回环验证 |
| 局部数据实体 | `vggt_slam/submap.py` | 点、颜色、置信度、投影矩阵和世界坐标读取 |
| 地图容器 | `vggt_slam/map.py` | 子地图登记、候选检索、轨迹与 PCD 导出 |
| 图结构 | `vggt_slam/graph.py` | SL(4) 节点、Prior/Between 因子和 LM 调用 |
| 浏览器服务 | `vggt_slam/viewer.py` | Viser 场景对象、相机视锥和可选语义查询 |

项目代码并不重新实现 VGGT 的 aggregator、DPT、SALAD 的 Sinkhorn、GTSAM 的 LM 求解器或 Viser 的 WebGL 前端；它通过固定接口把这些依赖组合起来。

## A.2 本机实际环境

以下版本从本机 `.venv`/`logs/environment.txt` 记录读取，适合复现本教程中的形状和输出路径：

| 组件 | 版本/记录 |
|---|---|
| Python | 3.11（本机 `.venv`） |
| PyTorch | 2.3.1，CUDA 12.1 wheel |
| torchvision | 0.18.1 |
| Open3D | 0.19.0 |
| OpenCV | 4.11.0.86 |
| viser | 0.2.23 |
| gtsam-develop | `4.3.2.dev202610020315` |
| NumPy | 1.26.4 |
| VGGT checkout | `third_party/vggt`，固定 commit `6e6e16107b88e8e76c751826af10d4295d87ecd2` |
| SALAD checkout | `third_party/salad`，固定 commit `33ca9c0ca1e10cbb21efc0d6a5fcb6d45688e42d` |
| VGGT checkpoint | `.cache/torch/hub/checkpoints/model.pt` |

`requirements.txt` 是依赖意图，`logs/environment.txt` 是一次实际安装快照；两者都不保证未来重新安装时底层 CUDA 驱动完全相同。

## A.3 GTSAM SL(4) 的证据边界

本机 Python wheel 能直接确认：

```python
gtsam.SL4
gtsam.Values
gtsam.PriorFactorSL4
gtsam.BetweenFactorSL4
```

它不能从 wheel 元数据直接给出完整 C++ 源码 commit。为解释教程里的 15 维局部坐标和 BetweenFactor 残差，曾用构建时间附近的官方 GTSAM 源码快照做对照：

```text
参考目录：/tmp/vggt_slam_gtsam_reference
参考 commit：8444f9d3fb8325eb4c6928d3001daa2d6ffdba45
```

这份快照是依赖实现的研究证据，不是项目源码的一部分。升级 GTSAM 后应重新验证：

1. `SL4::SL4(Matrix44)` 的合法化和行列式处理；
2. `ChartAtOrigin::Local/ Retract` 的具体定义；
3. `BetweenFactor` 对 `Local(measured, between)` 的残差方向；
4. LM 阻尼和 `Values` 替换行为。

教程中的精确残差写成当前对照实现下的形式：

$$
r_{ij}=\operatorname{Local}\left(Z_{ij},H_i^{-1}H_j\right),
$$

而不是把它概括成 SE(3) 的 6 维平移旋转误差。

## A.4 上游和本项目之间的命名差异

| 名称 | 代码中的位置 | 阅读时的实际含义 |
|---|---|---|
| `world_points` | `solver.py::add_points` 入参 | 上游命名；当前返回值实际是相机坐标点 |
| `proj_mats` | `Submap` | 扩展 4×4 内参，不是完整 world-to-camera 投影矩阵 |
| `poses` | `Submap` | 保存 `world_to_cam` 外参 |
| `get_all_poses_world()` | `Submap` | 用 `K·H⁻¹` 重新构造并分解的相机表达 |
| `image_match_ratio` | VGGT aggregator | attention 派生统计，不是像素内点率 |
| `lc_thres` | `Solver` | SALAD L2 距离上限，不是 ratio 阈值 |
| `conf_threshold` | `Submap` | 置信度百分位参数，不是置信度数值阈值 |

命名是接口的一部分，但不能代替数据流证据。修改任一字段名时，最好同步更新本教程的表示链表格。

## A.5 论文方法与运行时代码

论文可以解释为什么要用窗口、回环和 projective 图，但不能自动证明当前 checkout 的每个行为。例如：

- 论文中的几何点名词不决定当前函数返回的是相机系还是世界系；
- 论文中的相似度概念不决定当前阈值使用距离还是相似度；
- 论文中的连续优化描述不决定项目是否在回环后重新渲染所有 Viser 对象；
- 论文中的置信度含义不决定工程代码采用百分位过滤还是绝对阈值。

涉及这些问题时，以当前源码、依赖源码和运行产物的组合证据为准，并在文中保留“[代码确认]”“[依赖库确认]”“[运行产物确认]”标签。

## A.6 依赖升级后的复核清单

| 升级对象 | 需要重新测量 |
|---|---|
| PyTorch/CUDA | GPU 显存峰值、BF16/FP16 分支、checkpoint 加载 |
| VGGT | token 数、pose/depth 输出形状、`get_similarity` 统计 |
| SALAD/DINOv2 | 描述子维度、归一化方式、GPU/CPU 所有权 |
| GTSAM | SL4 chart、factor residual、LM 优化后 `Values` |
| Open3D | 体素降采样结果、PCD 写出类型和速度 |
| Viser/Chrome | WebSocket 协议、点云句柄替换、WebGL 初始化 |

这些检查比只运行一次 `python main.py --help` 更能发现不兼容，因为帮助文本只能证明参数仍然存在。
