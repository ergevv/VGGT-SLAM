# 附录 B：参数、显存、文件规模与故障排查

本附录把前文的状态链转换成可执行排查表。建议每次只改变一个层面的参数，并记录输入数量、窗口形状、候选数量、点数和浏览器端错误；否则多个门槛同时变化时很难判断是哪一层改变了结果。

## B.1 参数沿数据链的影响

| 参数 | 默认值 | 增大后的直接影响 | 可能代价 |
|---|---:|---|---|
| `min_disparity` | 50 | 需要更大运动才选关键帧，关键帧更稀疏 | 细节遗漏、窗口触发变慢、回环候选减少 |
| `submap_size` | 16 | 每次 VGGT 看更多新增帧 | 单次显存和推理时间增加；局部图约束更长 |
| `overlapping_window_size` | 1 | 块间共享帧更多 | 当前 `main.py` 和 solver 路径只明确支持默认 1，盲目增大可能破坏节点假设 |
| `conf_threshold` | 25.0 | 按百分位过滤更多低置信度点 | 点云更稀疏；小块中可用于尺度的点可能不足 |
| `lc_thres` | 0.95 | SALAD 距离门槛放宽 | 候选和误检都增加，双帧 VGGT 次数上升 |
| `max_loops` | 1 | 每块允许更多候选 | 回环辅助块、深度推理和图规模增加 |
| `vis_voxel_size` | `None` | 显示体素变大 | 浏览器更轻，但只影响 Viser，不影响模型和 PCD |
| `--vis_imgs` | 关闭 | 显示相机图像纹理 | 浏览器传输和渲染开销增加 |
| `--skip_dense_log` | 关闭 | 不写稠密 PCD | 节省磁盘和写出时间，不改变图和轨迹 |

一个有用的实验顺序是：先固定 `max_loops=0` 验证普通子地图；再打开回环；最后开启 `--vis_map`。这样能把图优化问题和浏览器压力分开。

## B.2 显存为什么可能不够

### B.2.1 一次窗口的峰值

`run_predictions()` 至少会同时涉及：

```text
S×3×H×W 输入图像
VGGT aggregator token 和 attention 激活
相机头、DPT 深度头的中间张量
当前 predictions
Submap.frames 的 GPU Tensor
SALAD retrieval_vectors
```

其中 $S$ 由关键帧窗口决定，默认是 `submap_size + overlap`，但最后一组可能更短。attention 的 token 相关内存随窗口长度增加明显，不能只用最终点数估算显存。

### B.2.2 为什么测试环境和用户运行不同

常见差异包括：

1. 测试用的窗口最后一组更短，峰值低于完整窗口；
2. 测试前没有保留旧 Python 进程或浏览器 GPU 缓冲；
3. 用户开启了 `--vis_imgs`、`--run_os` 或更大的窗口；
4. 不同 PyTorch/CUDA 版本的 allocator 保留块不同；
5. 当前项目会把 `Submap.frames` 和 SALAD 向量长期保留在 GPU，已处理块越多，持久占用可能越大；
6. 浏览器 Viser 走独立 GPU/共享显存，和 Python 的 CUDA OOM 不是同一个分配器，但都可能让系统整体内存紧张。

### B.2.3 可复现的显存记录

在每个 `run_predictions()` 前后记录：

```python
print(torch.cuda.memory_allocated() / 1024**3)
print(torch.cuda.memory_reserved() / 1024**3)
print(torch.cuda.max_memory_allocated() / 1024**3)
```

记录时要先 `torch.cuda.reset_peak_memory_stats()`，并把 `submap_size`、窗口实际帧数、`vis_imgs`、`run_os` 和历史子地图数写在同一行。否则单看一次“当前 allocated”无法区分临时峰值和缓存保留。

### B.2.4 降低峰值的优先顺序

```text
1. 先减小 submap_size，例如 16 -> 8
2. 关闭 --vis_imgs 和 --run_os
3. 设置 max_loops=0 验证普通路径
4. 确认不是多个 main.py 进程同时占用 GPU
5. 让模型 forward 使用 no_grad（当前路径已使用）
6. 再考虑清理不需要长期保留的 GPU Submap 字段
```

`--vis_voxel_size` 不在这个列表前面，因为它发生在世界点已生成之后，不能降低 VGGT forward 的峰值。[代码确认]

## B.3 普通建图的检查顺序

| 步骤 | 检查命令/位置 | 结果含义 |
|---|---|---|
| 1 | `python main.py --help` | 参数入口是否正常 |
| 2 | 打印 `Found N images` | 输入目录筛选是否正确 |
| 3 | 打印每个 `image_names_subset` | KLT 和窗口触发是否合理 |
| 4 | `Preprocessed images shape` | VGGT 输入 `S×3×H×W` 是否符合预期 |
| 5 | `Created new submap` | Submap id 和帧数是否连续 |
| 6 | `detected_loops` / `image_match_ratio` | 候选是否只在验证后入图 |
| 7 | `scale factor` | 相邻块尺度是否出现 NaN/Inf/极端值 |
| 8 | `Total number of submaps/loop closures` | 后端实体数量是否符合预期 |
| 9 | 轨迹行数和 PCD header | 导出路径是否真正执行 |

如果第 3 步就不对，不要先改 GTSAM；如果第 9 步正常但浏览器空白，不要先重新训练或重跑模型。

## B.4 PCD 的检查顺序

```bash
# 查看文件头和大小
ls -lh logs/office_loop_poses_points.pcd
head -n 11 logs/office_loop_poses_points.pcd

# 用独立脚本读取，并把显示点数控制在浏览器可承受范围
.venv/bin/python view_pcd.py logs/office_loop_poses_points.pcd \
  --voxel_size 0.05 --port 8081
```

若 Open3D 提示文件为空或无法读取：

1. 确认 PCD 的 `DATA binary` 与头部字段完整；
2. 确认写出进程没有在文件中途被终止；
3. 确认文件路径没有把轨迹 `.txt` 当成 PCD；
4. 检查磁盘剩余空间和进程退出码。

若能读取但视角看不到地图，`view_pcd.py` 会按 0.5% 和 99.5% 分位数设置初始相机，仍看不到时先打印 `points.min/max`，排除少量离群点把尺度拉大。

## B.5 Viser/WebGL 空白页

### B.5.1 区分三种故障

| 现象 | 更可能的层 |
|---|---|
| 页面打不开、无法建立 WebSocket | 端口、进程或代理 |
| WebSocket opened 后立即断开，Console 是 WebGL context 错误 | 浏览器渲染器初始化 |
| 页面有 UI、没有点云，Console 没有 WebGL 错误 | 场景更新、点数组为空或数据传输 |
| 独立 PCD 查看器正常，主服务空白 | 主服务场景/端口/浏览器配置 |

### B.5.2 本机软件 WebGL 路径

```bash
./open_viewer.sh 8080
```

脚本使用独立 Chrome 用户目录、`--use-angle=swiftshader` 和 `--enable-unsafe-swiftshader`。独立配置避免已有 Chrome 进程复用旧的 GPU 参数。若使用其他端口，脚本参数和 Python 查看器的 `--port` 要一致。

### B.5.3 浏览器 Console 证据

以下错误说明 Three.js 创建 WebGL renderer 失败：

```text
THREE.WebGLRenderer: A WebGL context could not be created
Error creating WebGL context
GL_VENDOR = Disabled
GL_RENDERER = Disabled
```

此时先换 SwiftShader 或确认浏览器 GPU/沙箱策略；只重启 Python 服务不会修复浏览器渲染器。

## B.6 回环排查

| 量 | 期望含义 | 异常时先查 |
|---|---|---|
| `best_score` | SALAD 归一化 L2 距离，越小越像 | 输入颜色、checkpoint、历史块跳过规则 |
| `lc_thres` | 距离上限 | 是否把相似度阈值误当距离阈值 |
| `image_match_ratio` | 双帧 attention 派生统计 | q/k token 切片、模型版本 |
| `detected_loops` | 通过 ratio 后的当前窗口候选 | ratio 是否被置空 |
| `graph.get_num_loops()` | 已加入图的回环计数 | 候选是否在 `add_points()` 处理 |
| `is_lc_submap` | 辅助块标志 | 轨迹和 PCD 的不同遍历规则 |

不要只用最终轨迹是否闭合判断回环成功；错误回环也可能让轨迹“看起来闭合”，漏检也可能在短序列中不明显。

## B.7 修改代码后的最小验证

新增代码应有中文注释和函数说明。对于本项目，建议每次修改后执行：

```bash
python -m py_compile main.py main_realtime.py \
  vggt_slam/*.py view_pcd.py
python view_pcd.py --help
git diff --check
```

`detect_seg_track` 子项目在本机缺少 RK3588 依赖，不把它的本地编译失败当作 VGGT-SLAM 主链失败。若修改只涉及文档，可以跳过模型推理，至少检查 Markdown 链接、公式围栏和代码路径是否仍存在。
