# VGGT-SLAM 教程（五）：从图状态到世界地图、Viser 和文件输出

前几篇把局部深度变成了图节点和因子，但图优化本身并不会自动改写 `Submap.pointclouds`。当前项目采用“局部数据长期保留、读取时按最新 homography 变换”的方式：同一份相机坐标点，在 `Values` 更新后重新投影到世界系，Viser 和 PCD 都从这条读取路径获得结果。

这一点解释了三个常见现象：

```text
优化后不用重新运行 VGGT，点云仍然会改变位置；
Viser 的 voxel 降采样只影响浏览器发送的数据；
写出的 binary PCD 很大，因为它不会沿用 Viser 的显示体素参数。
```

## 1. 三个坐标层次

### 1.1 `Submap.pointclouds`：局部相机坐标

`Solver.add_points()` 调用 `unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)`。上游函数返回的是相机坐标点，虽然返回值变量和预测字段沿用了 `world_points` 这个名字。数组形状为 `S×H×W×3`，每一帧都有一张局部深度几何图。

同一阶段还保存：

| 字段 | 形状/类型 | 作用 |
|---|---|---|
| `colors` | `S×H×W×3`、`uint8` | 从输入图像转成 RGB 颜色 |
| `conf` | `S×H×W` | VGGT 深度置信度原值 |
| `conf_masks` | `S×H×W` | 当前代码再次保存的置信度数组 |
| `conf_threshold` | 标量 | `np.percentile(conf, 25)+1e-6`（默认过滤最低 25%） |
| `proj_mats` | `S×4×4` | 左上角嵌入 `intrinsic` 的扩展内参矩阵 |
| `poses` | `S×4×4` | `world_to_cam` 外参，而不是最终世界姿态 |

点云和颜色会保留在 `Submap` 中。它们不是每轮优化都复制一份的新世界点云。

### 1.2 图中的 homography：局部点到世界系

对第 `i` 帧，`PoseGraph` 节点键是 `submap_id + i`。设图中当前值为 $H_i$，局部齐次点为 $	ilde p_i=[x,y,z,1]^T$，读取时执行：

$$
\tilde p_w=H_i\tilde p_i,
\qquad
p_w=\frac{\tilde p_w[0:3]}{\tilde p_w[3]}.
$$

`Submap.get_points_in_world_frame()` 对每一帧都这样计算，再用 `conf_masks[index] > conf_threshold` 过滤并拼接。因而它反映的是**读取时的图值**，而不是 `add_points()` 当时的初始值。[代码确认]

这里的除以第四坐标很重要：SL(4) 允许一般的齐次投影变换，不能假定最后一维永远等于 1。把矩阵简单截取前三行会丢掉 projective 分量。

### 1.3 相机轨迹：由 `K·H⁻¹` 重新分解

`Submap.get_all_poses_world()` 并不直接返回图节点作为相机位姿。它先构造：

$$
P_i=K_i^{(4)}H_i^{-1},
$$

再将 $P_i$ 的前三行交给 `decompose_camera()`，得到相机内参、旋转、平移和尺度；默认返回一个 4×4 camera-to-world 风格矩阵，`give_camera_mat=True` 时返回归一化后的投影矩阵。[代码确认]

因此应分别理解：

```text
H_i             局部点 -> 世界点的图节点
K_i^(4)H_i^-1   当前图值下的相机投影表达
decompose(...)  用于轨迹输出或 Viser 相机姿态
```

## 2. 置信度过滤怎样改变输出

### 2.1 百分位阈值不是固定置信度单位

默认 `--conf_threshold 25.0` 传给 `Submap.add_all_points()`。代码将其作为百分位数：

$$
\tau=\operatorname{percentile}(C,25)+10^{-6},
$$

然后保留 $C>\tau$ 的点。它过滤的是每个子地图自己的最低置信度分布，而不是“置信度大于 0.25”。不同窗口的数值尺度不同，保留比例也只近似等于 75%，因为严格大于和重复值会改变精确数量。

### 2.2 深度头的正深度掩码没有沿用到这里

上游 `unproject_depth_map_to_point_map()` 会计算一个正深度掩码，但该函数最终只返回点图。`Submap` 之后使用 `conf_masks` 和百分位阈值重新筛选。因此阅读当前工程时，不能把“上游曾经算过正深度掩码”写成“PCD 一定只包含正深度点”。真正的输出条件是：

```text
点被投影到世界坐标后仍然是有限值
且对应 conf_masks[index] > conf_threshold
```

这是当前实现的边界，不代表所有 VGGT-SLAM 版本都采用相同过滤顺序。[代码确认]

### 2.3 颜色与点的对应关系

`get_points_in_world_frame()` 依据 `conf_masks` 筛点；`get_points_colors()` 通过 `filter_data_by_confidence()` 依据 `self.conf` 筛颜色。正常路径中两者都来自 `depth_conf`，所以布尔掩码一致。若后续改代码让 `conf_masks` 和 `conf` 分离，就必须同时修改颜色读取，否则 PCD 的点和颜色长度会不一致。

## 3. Viser 显示路径

### 3.1 一次更新做了什么

`Solver.set_submap_point_cloud(submap)` 的路径是：

```text
submap.get_points_in_world_frame(graph)
  -> submap.get_points_colors()
  -> Solver.set_point_cloud(...)
  -> viewer.server.scene.add_point_cloud(...)
```

如果 `main.py` 使用默认的 `--vis_map` 关闭状态，建图结束后调用 `update_all_submap_vis()`，一次性遍历所有子地图。若启用 `--vis_map`，每个普通窗口优化后先更新最新块；发生回环时才更新全部块，因为回环可能改变旧节点的世界位置。[代码确认]

### 3.2 `--vis_voxel_size` 只作用在发送前

`Solver.set_point_cloud()` 在已有世界点和颜色都生成之后，才用 Open3D：

```python
pcd = pcd.voxel_down_sample(self.vis_voxel_size)
```

降采样后的数组才传给 Viser。这个参数不会改变：

- VGGT 输入窗口和一次 forward 的显存；
- `Submap.pointclouds`、`conf` 或图节点；
- `logs/*_points.pcd` 的写出逻辑；
- 下一轮回环检索使用的 SALAD 描述子。

它降低的是浏览器 WebGL、WebSocket 和 Viser 场景的点数。对于 `office_loop` 这类稠密输出，建议先用 `--vis_voxel_size 0.02` 或 `0.05` 验证交互，再决定是否导出全量 PCD。[代码确认 + 工程推断]

### 3.3 当前 Viser 不会自动删除旧点云句柄

点云名称形如 `pcd_<submap_id>`。回环后 `update_all_submap_vis()` 会再次添加同名场景对象，具体替换行为依赖 Viser 的场景句柄语义；如果修改可视化代码，应该显式保存句柄或在更新前清理对象，避免调试时误以为画面中的重叠来自算法本身。当前项目的可靠数据源仍是 `get_points_in_world_frame()` 和 PCD 文件。[代码确认 + 工程推断]

## 4. PCD 写出路径和文件规模

### 4.1 `--log_results` 触发两个文件

当 `main.py` 使用：

```bash
python main.py --log_results --log_path logs/office_loop_poses.txt ...
```

代码先调用：

```text
GraphMap.write_poses_to_file(...)
  -> logs/office_loop_poses.txt
GraphMap.write_points_to_file(...)
  -> logs/office_loop_poses_points.pcd
```

如果加上 `--skip_dense_log`，只跳过第二个稠密 PCD，不影响轨迹文本。

### 4.2 姿态文本跳过回环辅助块

`get_all_cam_matricies()` 遍历子地图时跳过 `is_lc_submap=True`，再由 `write_poses_to_file()` 将相机投影矩阵分解成：

```text
frame_id x y z qx qy qz qw
```

默认不是 KITTI 的 12 个矩阵元素格式。普通窗口的重叠帧可能在相邻子地图重复出现，因此行数不一定等于原始图像数。

### 4.3 PCD 当前会遍历所有子地图

`write_points_to_file()` 没有像轨迹输出那样跳过回环块，而是对 `ordered_submaps_by_key()` 中的每个块调用 `get_points_in_world_frame()`，然后拼接点和颜色。结果可能包含：

- 重叠窗口的重复观测；
- 回环辅助块的两帧观测；
- 每帧高置信度像素的稠密点。

这能保留更多原始观测，但也会让文件远大于显示用点云。当前 `office_loop` 实测：

| 文件 | 实测值 |
|---|---:|
| `logs/office_loop_poses.txt` | 311 行，约 28 KB |
| `logs/office_loop_poses_points.pcd` | 858,527,788 字节 |
| PCD `WIDTH/POINTS` | 53,657,975 |
| PCD 数据 | binary，字段 `x y z rgb` |

因此用户看到的 `logs/office_loop_poses_points.pcd` 确实是当前建图流程写出的稠密点云；文件很大是因为它保存了约 5366 万个点，不代表 Viser 页面必须一次显示同样数量。[运行产物确认]

## 5. 为什么浏览器空白但服务端已连接

Viser 的 WebSocket 连接只说明浏览器成功收到了场景消息；Three.js 还需要在浏览器侧创建 WebGL context。若控制台出现：

```text
THREE.WebGLRenderer: A WebGL context could not be created
GL_VENDOR = Disabled
GL_RENDERER = Disabled
Sandboxed = yes
```

数据连接可以已经打开，渲染器仍会在初始化阶段抛错，于是页面看起来空白。这是浏览器渲染能力问题，不能用“Connection opened”推断点云为空。

当前项目提供两个互补路径：

```bash
# 继续查看正在运行的 Viser 服务
./open_viewer.sh 8080

# 不加载模型，直接读取保存的 PCD
.venv/bin/python view_pcd.py --voxel_size 0.05 --port 8081
./open_viewer.sh 8081
```

`open_viewer.sh` 为独立 Chrome 配置启用 SwiftShader 软件 WebGL；`view_pcd.py` 默认读取 `logs/office_loop_poses_points.pcd`，先用 Open3D 过滤 NaN/Inf，再按显示体素降采样，连接浏览器时自动设置一个按点云分位数估计的视角。[代码确认]

如果独立点云服务能显示而主服务空白，优先检查主服务的 Viser 场景更新和浏览器端 WebGL；如果两者都空白，再检查 PCD 是否存在、Open3D 是否能读取、服务端端口和浏览器控制台。

## 6. 从一个点追踪到最终画面

假设某帧像素 $(u,v)$ 的深度为 $d$，内参为 $(f_x,f_y,c_x,c_y)$，局部点为：

$$
p_i=\left(\frac{u-c_x}{f_x}d,\frac{v-c_y}{f_y}d,d\right).
$$

从当前图状态取出 $H_i$ 后：

$$
\tilde p_w=H_i[p_i^T,1]^T,
\qquad
p_w=\tilde p_w[0:3]/\tilde p_w[3].
$$

这个 $p_w$ 有两条下游路径：

```text
Viser：可选 voxel -> scene.add_point_cloud -> 浏览器 WebGL
PCD：和其他 Submap 拼接 -> Open3D write_point_cloud -> binary 文件
```

两条路径共享图状态和置信度过滤，但在显示降采样、数据传输和文件格式上分开。这个分叉是排查“文件有点、网页没有点”的最短路径。

## 7. 本篇证据入口

| 主题 | 文件/符号 |
|---|---|
| 局部点和置信度写入 | `vggt_slam/solver.py::add_points` |
| 世界点读取 | `vggt_slam/submap.py::get_points_in_world_frame` |
| 相机轨迹读取 | `vggt_slam/submap.py::get_all_poses_world` |
| Viser 点云 | `vggt_slam/solver.py::set_point_cloud` |
| 姿态和 PCD 导出 | `vggt_slam/map.py::write_poses_to_file/write_points_to_file` |
| 浏览器服务 | `vggt_slam/viewer.py::Viewer` |
| 独立 PCD 查看 | `view_pcd.py`、`open_viewer.sh` |
| 本地运行产物 | `logs/office_loop_poses.txt`、`logs/office_loop_poses_points.pcd` |

下一篇把这些入口串成一次完整的 `office_loop` 运行，并区分“日志中实际发生的事实”和“从源码推断出的中间状态”。
