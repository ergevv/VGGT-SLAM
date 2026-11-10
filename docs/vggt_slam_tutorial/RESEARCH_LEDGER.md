# 教程研究底稿：范围、表示来源和验收

这是教程的生成与复核记录，不是正文阅读前提。参考本机 Hydra 的 `doc/hydra_tutorial/README.md` 和 `MIGRATION_PLAN.md`：先建立表示链，再完成各层机制，再进行生命周期、数学和跨篇审查。

## 1. 读者契约

用户未声明已经掌握 SLAM、Transformer 或李群；正文只依赖基本 Python、坐标和矩阵乘法，梯度、注意力、齐次坐标、因子、雅可比、重牵引在主定义章节解释。教学例子和当前 office_loop 文件统计分开。重点是当前项目主执行链；不逐层推导 DINOv2 训练，也不假定已配置 SAM3。

## 2. 因果大纲与主定义

| 层 | 上一步留下的问题 | 本层引入 | 解决的问题/输出 | 下一层需求 |
|---|---|---|---|---|
| 01 | 图像不等于地图 | Submap、前端、后端 | 系统边界 | 运行时是谁持有状态 |
| 01a | 状态归属不清 | 关键帧窗口、锁、设备和 ID | 文件/帧到推理输入 | 图像如何成为几何 |
| 02 | RGB 缺少距离 | token、交替注意力、相机/DPT 头、反投影 | 相机系点和置信度 | 如何连接不同局部系 |
| 03 | 局部几何尺度和原点不同 | 相对 H、SL4、因子、Local、LM | 更新后的节点值 | 如何纠正长程漂移 |
| 04 | 链状图缺少重访证据 | SALAD、Sinkhorn、候选、注意力验证、回环块 | 新因子 | 新图值如何改地图 |
| 05 | 图状态不是点云 | 齐次除法、相机分解、Viser、PCD、语义旁路 | 世界点和输出 | 完整过程能否复述 |
| 06 | 分层知识还未串起来 | 同一窗口、同一节点、同一点 | 完整案例 | 参数、版本、排障和源码复核 |

## 3. 状态来源账本

| 实体 | 来源 | 表示/单位 | 初始化/选择 | 更新 | 生命周期 | 消费者 | 证据 |
|---|---|---|---|---|---|---|---|
| 文件帧 | glob/RealSense 彩色流 | 路径/BGR uint8 | 目录排除特定名称；实时保存关键帧 | 新观测 | 文件长期、帧临时 | 光流、PIL | main/cameras |
| 光流角点 | goodFeaturesToTrack | N×1×2 像素 | quality/minDistance；KLT status=1 | 接受关键帧时重置 | 参考关键帧间 | 当前形状上的位移门槛 | frame_overlap |
| 窗口 | 接受帧路径 | list[str] | 目标大小=submap_size+overlap | 消费后保留尾 overlap | 单窗口；实时可截断 | run_predictions | 两入口 |
| frames | 预处理图像 | S×3×H×W CUDA Tensor | crop/resize/RGB | 赋给 Submap | 随 map 保留、非 forward 临时 | SALAD、回环、Viser、语义 | solver/submap |
| tokens | aggregator | B×S×P×2048 | camera/register/patch、24 层 | frame/global attention | forward 临时；target_tokens 例外 | heads、相似度 | 固定 VGGT |
| pose_enc | CameraHead | B×S×9 | 四轮网络修正 | 最后一轮输出 | 预测临时，poses 被保存 | pose_enc.py | 固定 VGGT |
| depth/conf | DPT/head_act | S×H×W×1/S×H×W | exp(raw_depth)/1+exp(raw_conf) | 单次预测；百分位阈值 | 局部点/置信度长期 | 反投影、掩码 | 固定 VGGT/submap |
| pointclouds/colors | geometry/RGB | S×H×W×3 | 实际相机坐标；置信度读取过滤 | 原始数组不变 | 普通与回环子地图皆保留 | scale、世界点 | geometry/submap |
| K4/poses | 内外参转换 | S×4×4；K像素/E局部世界到相机 | 构造扩展矩阵 | 保存不重估 | 子地图长期 | edge、相机分解 | solver/submap |
| 图节点 | 内部/重叠变换 | SL4 4×4；15维局部更新 | 首节点 I；ID=base+index | LM 替换 Values | 不删除 | 变换点和相机 | graph/GTSAM |
| 因子 | 相对测量/先验 | Between/Prior、15维 sigma | 两端先存在 | 增加，未鲁棒化 | 图长期 | LM | graph/GTSAM |
| 描述子 | SALAD DINO+聚合 | S×8448 CUDA、单位范数 | 224×224；64×128+256 | 每块预测一次 | 随 Submap 长期 | 历史 L2 检索 | 固定 SALAD |
| LoopMatch | 历史最小L2 | 分数+四个索引 | 跳过最新块；容量门槛 | ratio拒绝或建立块 | 候选临时 | add_points | loop/map/solver |
| 回环子地图 | 查询+历史双帧 | frames+局部点+2节点 | ratio≥0.95 | 三条连接路径使图闭合 | 留在map；轨迹跳过，PCD保留 | 后端、点云导出 | solver/map |
| 世界点 | graph H+局部点 | N×3；地图尺度 | 齐次除法+置信度 | 每次读取重算 | 输出视图临时 | Viser/PCD | submap |
| 语义向量/OBB | PE/SAM3 | 归一化嵌入/点掩码/PCA框 | 可选run_os | 查询新建OBB | 图像向量长期，框会话 | Viser | main/utils/viewer |
| PCD | 全部子地图世界点 | binary x/y/z/rgb | 写出置信度点、不作viewer voxel | 每次导出覆盖文件 | 磁盘 | 独立查看 | map/view_pcd |

## 4. 证据与已经补证的差异

- 光流实际形状 N×1×2，`norm(axis=1)` 后为 N×2，再整体均值；数值探针确认为 `(abs(dx)+abs(dy))/2` 平均，而非平均欧氏长度。
- VGGT geometry 返回相机坐标，局部变量名 world_points 并不决定坐标语义；有效深度掩码被丢弃。
- `good_mask` 双方阈值均引用 prior threshold；这是实现事实，不擅自修代码。
- 当前 GTSAM 安装版 `4.3.2.dev202610020315`：本机实测 SL4 constructor determinant normalization、Local≠Logmap、retract≠Expmap；BetweenFactor.unwhitenedError 等于 Local；linearize 的两块为 `-Ad(h^-1)` 和 I。
- 官方构建时间附近快照 `8444f9d3fb8325eb4c6928d3001daa2d6ffdba45`（2026-10-02），wheel 没记录源码 commit，因此只标为参考快照，不冒充精确构建来源。
- match_ratio 是 k/q 注意力派生统计，不是内点率；普通 no_grad 不覆盖它的数值异常防护。
- Submap 保存 CUDA frames 和 CUDA SALAD vectors，累计地图会保留 GPU 状态，不能说子地图完成就释放全部显存。
- first batch 填满需要 submap_size+overlap 个新关键帧；后续才新增 submap_size 个。

## 5. 审核方式

逐篇检查中央概念的来源、维度、筛选、更新、生命周期、消费者、算例、边界、证据及后续需要；同时运行 skill 检查器、相对链接检查、不可见字符检查、git diff --check。数学探针只使用本机 CPU 的 NumPy/GTSAM 和小矩阵，不重复加载模型或运行完整建图。
