# CompatibleVGGT.forward()：从图像序列到相机与深度结果

本文参考 [transformer.md](./transformer.md) 中“沿着张量形状追踪数据”的写法，讲解 [vggt_slam/model_utils.py:25-50](/home/erge/work/progress/tmp/VGGT-SLAM/vggt_slam/model_utils.py:25) 的 CompatibleVGGT.forward()。

这段代码不是重新实现 VGGT，而是在上游 VGGT 外面增加一层适配逻辑：

~~~text
统一输入形状
    ↓
根据硬件选择 aggregator 的半精度
    ↓
提取跨帧视觉 token
    ↓
把几何 token 转成单精度
    ↓
相机头 → 位姿编码
深度头 → 深度图和置信度图
    ↓
保持上游接口，返回一个字典
~~~

它解决的是工程接口和数值精度问题：SLAM 需要相机、深度和深度置信度；回环验证还需要一个跨帧匹配分数；而点跟踪预测头在本项目中没有使用。

## 一、先区分几类数据

阅读这段代码时，最容易混淆的是下面几类数据：

| 数据 | 含义 | 是否是三维点 | 主要消费者 |
|---|---|---:|---|
| tokens | Aggregator 的中间视觉特征 | 否 | camera_head、depth_head |
| pose_enc | 每帧 9 个相机参数编码 | 否 | 后续转换为外参和内参 |
| depth | 每个像素的深度预测 | 否，仍是二维图上的数值 | 反投影为相机坐标点 |
| depth_conf | 每个像素的深度置信度 | 否 | 过滤点、估计子地图尺度 |
| target_tokens | 目标层的视觉特征快照 | 否 | 调试或相似度相关路径 |

tokens 不是点云，pose_enc 也不是 4×4 位姿矩阵。相机编码需要经过后续的 pose_encoding_to_extri_intri()，深度需要结合内参才能反投影。

## 二、模型从哪里被调用

正常建图时，Solver.run_predictions() 调用：

~~~python
predictions = model(images)
~~~

这里的 images 通常来自图像预处理，形状是：

~~~text
[S, 3, H, W]
~~~

其中 S 是当前 Submap 的图像数量。

回环验证时，代码会把当前查询帧和历史匹配帧组成两帧输入，并打开相似度计算：

~~~python
predictions_lc = model(lc_frames, compute_similarity=True)
~~~

两种调用都会进入同一个 forward()；区别只是 compute_similarity 是否要求 Aggregator 额外计算 image_match_ratio。

完整调用链可以画成：

~~~text
main.py
  ↓
Solver.run_predictions()
  ↓ model(images)
CompatibleVGGT.forward()
  ↓
Aggregator
  ├─ CameraHead
  └─ DPT/DepthHead
  ↓
predictions 字典
  ↓
Solver.add_points()
~~~

## 三、输入形状：为什么要自动增加 batch 维度

代码首先检查输入：

~~~python
if images.ndim == 4:
    images = images.unsqueeze(0)
~~~

如果输入是：

~~~text
[S, 3, H, W]
~~~

则 unsqueeze(0) 会在最前面增加一个 batch 维度：

~~~text
[S, 3, H, W]
    ↓
[1, S, 3, H, W]
~~~

如果调用方已经提供：

~~~text
[B, S, 3, H, W]
~~~

则不会再次添加维度。

这里的两个维度含义不同：

~~~text
B：独立样本数量
S：一个样本中连续处理的图像帧数量
~~~

VGGT 的全局注意力需要在同一个 S 维度内让不同帧交换信息，所以不能把每一帧误当成独立 batch 样本。

例如一个 17 帧窗口：

~~~text
输入给 Solver：       [17, 3, H, W]
进入 VGGT：            [1, 17, 3, H, W]
~~~

此时模型会把 17 帧视为一个跨帧推理窗口。

## 四、为什么拒绝 query_points

代码直接检查：

~~~python
if query_points is not None:
    raise ValueError("SLAM 模型未启用点跟踪预测头。")
~~~

CompatibleVGGT.__init__() 使用：

~~~python
super().__init__(enable_point=False, enable_track=False)
~~~

也就是说，本项目加载模型时关闭了 point head 和 track head，只保留相机与深度相关路径。

因此：

~~~text
query_points=None       → 可以正常推理
query_points 有值       → 立即报错
~~~

这不是输入数据检查失败，而是功能边界检查。当前 SLAM 流程没有请求点跟踪预测，因此提前拒绝可以避免调用一个未初始化的预测头。

## 五、Aggregator 的精度边界

下一行读取 Aggregator 参数的数据类型：

~~~python
dtype = next(self.aggregator.parameters()).dtype
~~~

这里没有直接读取 images.dtype，而是以 Aggregator 的参数为准。模型加载函数在 [model_utils.py:53-76](/home/erge/work/progress/tmp/VGGT-SLAM/vggt_slam/model_utils.py:53) 中已经执行：

~~~python
model.aggregator.to(dtype=dtype)
~~~

其中：

| 设备 | Aggregator 精度 |
|---|---|
| CUDA 且计算能力主版本 >= 8 | torch.bfloat16 |
| CUDA 且计算能力主版本 < 8 | torch.float16 |
| CPU | torch.float32 |

随后：

~~~python
tokens, patch_start_idx, target_tokens, match_ratio = self.aggregator(
    images.to(dtype), compute_similarity
)
~~~

数据流是：

~~~text
原始 images
    ├─ images.to(dtype) ──→ Aggregator 半精度特征提取
    └─ images.float()   ──→ DepthHead 的单精度图像输入
~~~

需要注意，to(dtype) 不会原地修改原始 Tensor。它产生一个按照 Aggregator 精度转换后的输入，原来的 images 变量仍可在后面转成 FP32。

采用这种分工的原因是：

~~~text
Aggregator：计算量大、token 数量多，半精度可以降低显存和计算成本
几何预测头：深度、置信度和相机参数对数值范围更敏感，使用单精度更稳妥
~~~

所以“特征提取使用半精度，几何预测使用单精度”不是一句注释，而是代码中两次显式 dtype 转换共同形成的策略。

## 六、Aggregator 返回的四项数据

Aggregator 返回：

~~~python
tokens, patch_start_idx, target_tokens, match_ratio
~~~

它们的职责不同。

### 6.1 tokens：多层视觉特征列表

tokens 是一个 Python list，每个元素对应 Aggregator 的一个中间层输出。当前配置下，可以把单层特征抽象为：

~~~text
[B, S, P, C]
~~~

其中：

~~~text
B：batch 数量
S：帧数量
P：每帧 token 数量
C：token 通道数
~~~

在当前 VGGT 配置中，一帧通常包含：

~~~text
1 个 camera token
4 个 register token
777 个 patch token
------------------------
782 个 token
~~~

对 294×518 的输入，14×14 patch 网格是：

~~~text
21 × 37 = 777 个 patch token
~~~

Aggregator 内部会交替执行：

~~~text
frame attention：同一帧内部的 token 交互
global attention：同一窗口不同帧之间的 token 交互
~~~

这和 [transformer.md](./transformer.md) 中“注意力让某一行特征吸收其他位置的信息”的概念一致；区别在于这里的“位置”是图像帧和 patch token，而不是手工构造的二维轨迹。

数据流可以表示为：

~~~text
[B,S,3,H,W]
    ↓ patch embedding 与特殊 token
[B,S,P,C]
    ↓ frame/global attention 交替更新
多层 tokens：
[
    [B,S,P,C_0],
    [B,S,P,C_1],
    ...
]
~~~

相机头主要读取最后一层，深度头需要从多个中间层恢复空间细节。

### 6.2 patch_start_idx：告诉深度头从哪里开始是图像 patch

特殊 token 位于每帧序列前部：

~~~text
索引 0：camera token
索引 1~4：register tokens
索引 5~781：patch tokens
~~~

因此当前配置通常是：

~~~python
patch_start_idx = 5
~~~

深度头会执行类似：

~~~python
patch_tokens = tokens[layer_idx][:, :, patch_start_idx:, :]
~~~

如果输入是：

~~~text
[1, 17, 782, 2048]
~~~

去掉前 5 个特殊 token 后得到：

~~~text
[1, 17, 777, 2048]
~~~

接着再把 777 个 patch 恢复为：

~~~text
[1, 17, 2048, 21, 37]
~~~

最后交给 DPT 式多尺度融合，生成每个像素的深度。

所以 patch_start_idx 不是帧编号，也不是图像坐标；它是 token 序列中“图像 patch 区域”的起始下标。

### 6.3 target_tokens：中间特征快照

target_tokens 是 Aggregator 在目标层保留的一份特征快照。它不是：

~~~text
不是深度图
不是点云
不是 SALAD 的 8448 维检索描述子
~~~

它更像是：

~~~text
在某个 Transformer 深度截取的视觉表示
~~~

当前 forward() 不对它调用 float()，因此它通常保留 Aggregator 路径的 dtype。后续 Solver.run_predictions() 把普通 Tensor 转为 CPU NumPy 时，特意跳过了 target_tokens；这意味着它可能继续占用 GPU 内存。

### 6.4 match_ratio：可选的回环匹配统计量

普通建图调用：

~~~python
model(images)
~~~

通常不要求额外相似度统计，match_ratio 可以是 None。

回环验证调用：

~~~python
model(lc_frames, compute_similarity=True)
~~~

Aggregator 会在目标 global attention 层提取 q/k，并计算 image_match_ratio。它是 attention 派生的匹配分数，不是像素正确率，也不是 RANSAC 内点率。

Solver.run_predictions() 后续使用它：

~~~python
if image_match_ratio < 0.95:
    predictions_lc = None
~~~

因此 match_ratio 只影响回环候选是否继续进入几何建图，不会改变普通窗口的相机和深度预测分支。

## 七、为什么要把 token 转成 FP32

代码是：

~~~python
geometry_tokens = [token.float() for token in tokens]
~~~

这里对 tokens 列表中的每个层逐个转换，而不是把列表整体转换。转换后的数据流：

~~~text
半精度 tokens
    ↓ 对每个中间层执行 float()
FP32 geometry_tokens
    ├─ camera_head
    └─ depth_head
~~~

这一步有三个直接效果：

1. 相机参数头输入的特征为 FP32；
2. 深度和置信度头输入的特征为 FP32；
3. 半精度只保留在 Aggregator 的大规模特征提取阶段。

它没有把 Aggregator 本身改回 FP32，也没有把模型所有参数重新转换，只是建立了一个 FP32 的几何预测边界。

## 八、CameraHead 的数据流

代码：

~~~python
pose_enc_list = self.camera_head(geometry_tokens)
~~~

相机头通常读取最后一层特征中的第 0 个 token，因为第 0 个 token 是 camera token：

~~~text
[B,S,P,C]
    ↓ 取 token 0
[B,S,C]
~~~

在当前拼接特征配置中，C 可以理解为 2048。相机头进一步经过归一化和多轮 refinement，得到一个列表：

~~~text
pose_enc_list = [第1轮预测, 第2轮预测, ...]
~~~

每轮最终形状通常是：

~~~text
[B, S, 9]
~~~

9 个数的语义是：

~~~text
[tx, ty, tz, qx, qy, qz, qw, fov_h, fov_w]
~~~

当前代码返回：

~~~python
"pose_enc": pose_enc_list[-1]
~~~

也就是最后一轮 refinement 的结果。

需要区分两件事：

~~~text
camera_head 的多轮 refinement：神经网络内部前向计算
PoseGraph.optimize()：SLAM 后端的图优化
~~~

它们发生在不同阶段，前者不会读取 GTSAM 的边和节点。

## 九、DepthHead 的数据流

代码：

~~~python
depth, depth_conf = self.depth_head(
    geometry_tokens,
    images=images.float(),
    patch_start_idx=patch_start_idx,
)
~~~

DepthHead 同时使用两类输入：

~~~text
geometry_tokens：来自 Aggregator 的多层视觉特征
images.float()：单精度原始图像，用于深度头的图像相关处理
~~~

它使用 patch_start_idx 去掉 camera/register token，只保留 patch token，并从多个中间层恢复二维空间结构：

~~~text
多层 [B,S,777,2048]
    ↓ 去除特殊 token
多层 patch 特征
    ↓ 恢复 21×37 网格
多尺度投影与融合
    ↓ 上采样到 H×W
深度头原始输出
    ↓ 激活函数
depth、depth_conf
~~~

在当前上游配置中，深度头内部通常按两类通道产生结果：

~~~text
一个通道 → 深度
一个通道 → 置信度
~~~

深度和置信度的具体布局由上游 DPT head 决定，常见形式是：

~~~text
[B, S, 1, H, W]
~~~

进入 Solver.run_predictions() 后，代码会统一转为 CPU NumPy 并去掉 batch 维；如果现场遇到维度问题，应直接打印 depth.shape 和 depth_conf.shape，不要只依据变量名猜测最后三个维度的顺序。

## 十、返回字典：每个键流向哪里

最后返回：

~~~python
return {
    "pose_enc": pose_enc_list[-1],
    "pose_enc_list": pose_enc_list,
    "depth": depth,
    "depth_conf": depth_conf,
    "images": images.float(),
    "target_tokens": target_tokens,
    "image_match_ratio": match_ratio,
}
~~~

| 键 | 数据来源 | 作用 | 后续使用 |
|---|---|---|---|
| pose_enc | pose_enc_list[-1] | 最终相机编码 | 转换为 extrinsic、intrinsic |
| pose_enc_list | camera_head 全部迭代结果 | 保留每轮预测 | 调试或分析 refinement |
| depth | depth_head | 每像素深度 | 反投影为相机坐标点 |
| depth_conf | depth_head | 深度可信程度 | 置信度过滤、尺度估计 |
| images | 输入图像转 FP32 | 保持模型输出接口 | 保存帧、生成颜色 |
| target_tokens | Aggregator 目标层 | 中间特征快照 | 特征分析或保留 |
| image_match_ratio | 可选 q/k 相似度 | 回环验证分数 | 接受或拒绝回环 |

正常窗口的后处理大致是：

~~~text
pose_enc
    ↓ pose_encoding_to_extri_intri()
extrinsic、intrinsic

depth + intrinsic + extrinsic
    ↓ unproject_depth_map_to_point_map()
相机坐标点云

depth_conf
    ↓ Submap 置信度阈值
高置信度点
~~~

回环窗口还会额外使用：

~~~text
image_match_ratio
    ↓ 与 0.95 比较
是否保留回环候选
~~~

## 十一、正常建图与回环验证的两条数据流

### 11.1 普通窗口

~~~text
images: [S,3,H,W]
    ↓ unsqueeze
[1,S,3,H,W]
    ↓ images.to(aggregator_dtype)
半精度 Aggregator
    ↓
tokens、patch_start_idx、target_tokens、None
    ↓ tokens.float()
geometry_tokens（FP32）
    ├─ camera_head → pose_enc_list → pose_enc
    └─ depth_head  → depth、depth_conf
    ↓
返回 predictions
~~~

Solver.add_points() 再把几何结果写入当前 Submap，并加入 PoseGraph。

### 11.2 回环双帧

~~~text
lc_frames: [2,3,H,W]
    ↓ unsqueeze
[1,2,3,H,W]
    ↓ 半精度 Aggregator + compute_similarity=True
tokens、patch_start_idx、target_tokens、match_ratio
    ↓ FP32 geometry_tokens
camera/depth heads
    ↓
双帧外参、内参、深度、置信度 + 匹配分数
~~~

只有当 match_ratio 通过阈值时，Solver 才把这次双帧预测用于创建回环辅助 Submap。

## 十二、一个具体的形状例子

以 17 帧、输入尺寸 294×518 为例：

~~~text
输入图像：                 [17, 3, 294, 518]
增加 batch：               [1, 17, 3, 294, 518]
patch 网格：               21×37 = 777
每帧特殊 token：           1 + 4 = 5
每帧 token 总数：          5 + 777 = 782
~~~

在 Aggregator 输出的拼接特征可理解为：

~~~text
[1, 17, 782, 2048]
~~~

相机头取每帧第 0 个 camera token：

~~~text
[1, 17, 782, 2048]
    ↓
[1, 17, 2048]
    ↓ camera refinement
[1, 17, 9]
~~~

深度头切掉前 5 个 token：

~~~text
[1, 17, 782, 2048]
    ↓ tokens[:, :, 5:, :]
[1, 17, 777, 2048]
    ↓ 恢复 patch 网格
[1, 17, 2048, 21, 37]
    ↓ DPT 融合和上采样
深度与置信度图
~~~

这里的 2048 是当前特征配置下 frame/global 两路特征拼接后的通道数；如果更换上游模型配置，通道数和中间层数量可能变化，不能把这个数字当作 forward() 的硬编码接口。

## 十三、这段代码刻意没有做什么

forward() 只负责模型推理和输出整理，不负责：

~~~text
不负责把 pose_enc 转成最终外参矩阵
不负责把深度直接转换成世界坐标点
不负责把当前窗口写入 GraphMap
不负责执行 GTSAM 位姿图优化
不负责决定是否接受尺度约束
~~~

这些工作分别在 run_predictions()、add_points()、Submap 和 PoseGraph 中完成。这样分层后，模型输出仍然是局部测量，SLAM 后端再负责把多个窗口的局部结果连接成全局地图。

## 十四、阅读和调试时建议观察的变量

在 forward() 返回前，可以重点观察：

~~~python
print(images.shape, images.dtype)
print(next(self.aggregator.parameters()).dtype)
print(len(tokens), tokens[-1].shape)
print(patch_start_idx)
print(None if target_tokens is None else target_tokens.shape)
print(None if match_ratio is None else match_ratio)
print(pose_enc.shape, depth.shape, depth_conf.shape)
~~~

这些信息可以回答几个常见问题：

| 问题 | 应观察的变量 |
|---|---|
| 是否正确把帧序列送入模型 | images.shape 是否为 [B,S,3,H,W] |
| 是否使用了预期半精度 | Aggregator 参数 dtype 和 images.to(dtype) |
| DPT 是否拿到了正确 patch 区域 | patch_start_idx 和 token 数 |
| 回环分支是否真的计算相似度 | compute_similarity 和 match_ratio |
| 几何输出是否异常 | pose_enc.shape、depth.shape、depth_conf.shape |
| GPU 显存为何没有完全释放 | target_tokens 是否仍在 GPU |

最重要的边界可以记成一句话：

> CompatibleVGGT.forward() 用半精度 Aggregator 建立跨帧视觉表示，用单精度几何头输出相机与深度测量，再把这些测量交给后续的几何转换和 SLAM 图优化。

## 十五、完整案例：两张办公室图像从输入走到几何结果

前面的章节说明了每个部件的职责，下面固定一个可以逐步跟踪的案例：

~~~text
图像文件：
    office_loop/frame_0001.jpg
    office_loop/frame_0002.jpg

原始尺寸：
    1920 × 1080

输入帧数：
    S = 2

模型输入模式：
    crop

设备假设：
    支持 BF16 的 CUDA GPU
~~~

最后一个假设只影响 Aggregator 的 dtype。如果运行在不支持 BF16 的 CUDA GPU 上，下面所有 BF16 都替换成 FP16；如果运行在 CPU 上，则 Aggregator 也使用 FP32，张量形状和数据流不变。

### 15.1 第 0 步：从文件读取 RGB 图像

图像加载函数首先使用 PIL 读取图像。如果文件是 RGBA，会先和白色背景合成；本案例是 RGB，因此不需要 alpha 合成。

此时两个图像可以抽象为：

~~~text
frame_0001：uint8，形状 [1080, 1920, 3]
frame_0002：uint8，形状 [1080, 1920, 3]
~~~

例如第一张图像原始中心附近的一个 RGB 像素可以读到：

~~~text
(R,G,B) = (88, 84, 81)
~~~

这个值仍然是 0 到 255 的整数。它还没有进入神经网络，也不能直接当作模型的输入值。

### 15.2 第 1 步：缩放到模型输入尺寸

当前使用 crop 模式，宽度固定为 518：

$$
W'=518
$$

高度按原始宽高比缩放，并调整为 14 的整数倍：

$$
\frac{1080\times518}{1920}=291.375
$$

调整到最接近的 14 的倍数：

$$
H'=14\times\operatorname{round}(291.375/14)=14\times21=294
$$

所以每张图经过双三次插值后变成：

~~~text
[1080,1920,3]
    ↓ RGB resize
[294,518,3]
~~~

因为 294 小于 518，crop 模式不会再次裁剪高度。若某个输入比例导致缩放后的高度大于 518，函数会做中心裁剪；本案例不会发生这一步。

双三次插值会综合邻域像素，所以不能把原始像素 960、540 简单地当成缩放后像素 259、147 的完全相同数值。为了演示通道归一化，假设缩放后中心像素近似为：

~~~text
RGB = (88, 84, 81)
~~~

### 15.3 第 2 步：转换为 Tensor 并归一化到 0 到 1

ToTensor 会完成两件事：

1. 把 H×W×C 改成 C×H×W；
2. 把 uint8 除以 255，转换为 FP32 的 0 到 1 数值。

单张图像变成：

~~~text
[294,518,3]
    ↓ transpose
[3,294,518]
~~~

示例中心像素变成：

$$
\left[
\frac{88}{255},
\frac{84}{255},
\frac{81}{255}
\right]
\approx
[0.3451,0.3294,0.3176]
$$

两张图像堆叠后，Solver 中的 images 是：

~~~text
[2,3,294,518]
~~~

这里还没有 batch 维。第一个维度 2 表示两帧属于同一个待推理序列。

### 15.4 第 3 步：进入 CompatibleVGGT.forward()

调用方式是：

~~~python
predictions = model(images)
~~~

进入 forward() 后：

~~~python
if images.ndim == 4:
    images = images.unsqueeze(0)
~~~

形状变化：

~~~text
images
    [2,3,294,518]
        ↓ unsqueeze(0)
    [1,2,3,294,518]
~~~

此时：

~~~text
B = 1：只有一个序列样本
S = 2：这个序列包含两张图
H = 294
W = 518
~~~

这一步非常关键。若把两张图组织成 [2,1,3,H,W]，模型会把它们当成两个独立样本，global attention 就不会把第 1 帧和第 2 帧作为同一个序列来建立跨帧关系。

### 15.5 第 4 步：准备两条精度路径

假设模型加载时选择了：

~~~text
dtype = torch.bfloat16
~~~

代码把图像复制转换为 BF16 后送入 Aggregator：

~~~python
images_for_aggregator = images.to(torch.bfloat16)
~~~

形状不变：

~~~text
[1,2,3,294,518]、torch.float32
    ↓ to(torch.bfloat16)
[1,2,3,294,518]、torch.bfloat16
~~~

原变量 images 没有被原地修改，后面仍可以执行 images.float()。

### 15.6 第 5 步：Aggregator 的图像标准化

Aggregator 内部保存三通道均值和标准差：

~~~text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
~~~

对每个像素逐通道计算：

$$
I'_c=\frac{I_c-\mu_c}{\sigma_c}
$$

使用示例像素 [0.3451,0.3294,0.3176]：

$$
\begin{aligned}
R'&=(0.3451-0.485)/0.229\approx-0.611,\\
G'&=(0.3294-0.456)/0.224\approx-0.565,\\
B'&=(0.3176-0.406)/0.225\approx-0.393.
\end{aligned}
$$

标准化只改变数值分布，不改变像素的帧编号、行列位置或 RGB 通道语义。

### 15.7 第 6 步：合并 B 和 S，进入 patch embedding

Aggregator 为了逐帧调用 patch encoder，把：

~~~text
[B,S,3,H,W] = [1,2,3,294,518]
~~~

改成：

~~~text
[B*S,3,H,W] = [2,3,294,518]
~~~

然后调用 patch embedding。patch 大小是 14×14：

~~~text
patch_rows = 294 / 14 = 21
patch_cols = 518 / 14 = 37
patch_count = 21 × 37 = 777
~~~

因此每张图的 patch 特征可以写成：

~~~text
[3,294,518]
    ↓ patch embedding
[777,1024]
~~~

两张图合并后：

~~~text
patch_tokens = [2,777,1024]
~~~

这里的 1024 是 Aggregator 的 embed_dim。一个 1024 维向量对应一个 14×14 图像区域，不是一个三维点，也不是 1024 个像素。

### 15.8 第 7 步：添加 camera 和 register token

当前配置有：

~~~text
1 个 camera token
4 个 register token
777 个 patch token
~~~

每帧 token 数：

$$
P=1+4+777=782
$$

在 B=1、S=2 时，特殊 token 展平后的形状为：

~~~text
camera_token   = [2,1,1024]
register_token = [2,4,1024]
patch_tokens   = [2,777,1024]
~~~

沿 token 维拼接：

~~~text
[2,1,1024]
    + [2,4,1024]
    + [2,777,1024]
    ↓
tokens = [2,782,1024]
~~~

此时每帧的 token 排列是：

~~~text
token 0：camera
token 1~4：register
token 5~781：图像 patch
~~~

### 15.9 第 8 步：生成二维 patch 位置

二维位置编码只对应 777 个 patch：

~~~text
patch positions = [2,777,2]
~~~

最后一维的两个数表示 patch 网格中的 row 和 column。特殊 token 的位置补成零：

~~~text
special positions = [2,5,2]
patch positions   = [2,777,2]
                       ↓ 拼接
all positions      = [2,782,2]
~~~

这些位置坐标会在 attention 中参与二维 RoPE，帮助模型区分左上角 patch 和右下角 patch。它们不是相机坐标，也不会直接变成深度。

### 15.10 第 9 步：第一轮 frame attention

frame attention 先把两帧当作两个独立 attention batch：

~~~text
tokens = [2,782,1024]
~~~

每一项只在自己的 782 个 token 内做自注意力：

~~~text
第 1 帧：camera + register + 自己的 patch
第 2 帧：camera + register + 自己的 patch
~~~

默认注意力头数为 16，每个头的维度为：

$$
d=1024/16=64
$$

Q、K、V 的形状可以理解为：

~~~text
[2,782,1024]
    ↓ 拆成 16 个头
[2,16,782,64]
~~~

单个帧内，注意力分数矩阵的形状是：

~~~text
[16,782,782]
~~~

它表示这一帧的每个 query token 对同一帧 782 个 key token 的关注关系。此时第 1 帧不会直接读取第 2 帧。

经过 LayerNorm、attention、残差、MLP 和再次残差后，token 数和通道数不变：

~~~text
[2,782,1024] → [2,782,1024]
~~~

改变的是每个 token 的 1024 个特征值。

### 15.11 第 10 步：第一轮 global attention

global attention 先把两帧拼在一个序列中：

~~~text
[B,S*P,C]
    = [1,2*782,1024]
    = [1,1564,1024]
~~~

此时一个 query 可以读取两帧的 key：

~~~text
第 1 帧的 patch → 可以读取第 1 帧和第 2 帧的 token
第 2 帧的 patch → 可以读取第 1 帧和第 2 帧的 token
~~~

global attention 完成后，再恢复为逐帧布局：

~~~text
[1,1564,1024]
    ↓ reshape
[2,782,1024]
~~~

这就是跨帧几何信息进入 token 的位置。比如第 2 帧的 camera token 现在可以包含第 1 帧中相同墙面、桌面或纹理区域的信息。

### 15.12 第 11 步：frame/global 特征拼接

一次 frame block 和一次 global block 都完成后，Aggregator 将两个阶段的输出沿最后一维拼接：

~~~text
frame_intermediates  = [1,2,782,1024]
global_intermediates = [1,2,782,1024]
                           ↓ cat(dim=-1)
output_list[0]       = [1,2,782,2048]
~~~

其中：

~~~text
前 1024 个通道：帧内 attention 的结果
后 1024 个通道：跨帧 attention 的结果
~~~

这两个 1024 维向量不是简单的原始图像复制，而是两个不同注意力路径在当前层产生的上下文表示。

### 15.13 第 12 步：重复 24 个交替周期

默认 depth=24、aa_block_size=1、aa_order=[frame, global]。因此流程会重复：

~~~text
frame block 0 → global block 0 → output_list[0]
frame block 1 → global block 1 → output_list[1]
...
frame block 23 → global block 23 → output_list[23]
~~~

最终：

~~~text
len(output_list) = 24
output_list[i].shape = [1,2,782,2048]
~~~

重要的是：列表长度 24 表示网络深度，不表示帧数；每个元素内部的第二维才是 2 帧。

### 15.14 第 13 步：保存 target_tokens 和计算回环分数

上层 VGGT 创建 Aggregator 时传入 target_layer=21。

对于 target_tokens，代码是在 global block 执行后检查 global_idx。因此当索引 20 的 global block 执行结束、global_idx 变成 21 时，保存：

~~~text
target_tokens = [2,782,1024]
~~~

然后对每个 token 的 1024 个通道做 L2 归一化：

$$
\hat t=\frac{t}{\|t\|_2+10^{-8}}
$$

普通建图调用的 compute_similarity=False，所以：

~~~text
image_match_ratio = None
~~~

如果使用同样的两帧输入执行 compute_similarity=True，则在 global block 索引 21 执行前保存 q/k，执行完成后计算 image_match_ratio。此时 q/k 的目标层和 target_tokens 的保存层相差一层，这是代码中的两个独立索引条件。

### 15.15 第 14 步：geometry_tokens 转为 FP32

现在 Aggregator 的输出是：

~~~text
tokens：24 个 [1,2,782,2048]，BF16
patch_start_idx：5
target_tokens：[2,782,1024]，通常仍为 BF16
image_match_ratio：None
~~~

CompatibleVGGT 执行：

~~~python
geometry_tokens = [token.float() for token in tokens]
~~~

得到：

~~~text
geometry_tokens：24 个 [1,2,782,2048]，FP32
~~~

从这里开始，相机头和深度头都不再使用 BF16 的 token。

### 15.16 第 15 步：CameraHead 取 camera token

相机头只取最后一层：

~~~text
geometry_tokens[-1] = [1,2,782,2048]
~~~

取每帧 token 0：

~~~text
[1,2,782,2048]
    ↓ tokens[:,:,0]
[1,2,2048]
~~~

再经过 LayerNorm 和 4 轮 trunk refinement：

~~~text
第 1 轮 → [1,2,9]
第 2 轮 → [1,2,9]
第 3 轮 → [1,2,9]
第 4 轮 → [1,2,9]
~~~

为了把后续的矩阵转换写完整，下面使用一组教学输出。它们用于展示计算格式，不是本项目权重对这两张图的真实预测：

~~~text
frame 0：
[0.00, 0.00, 0.00, 0, 0, 0, 1, π/2, π/2]

frame 1：
[0.10, 0.00, 0.00, 0, 0, 0, 1, π/2, π/2]
~~~

每行的含义是：

~~~text
[tx, ty, tz, qx, qy, qz, qw, fov_h, fov_w]
~~~

相机头真实输出由图像和权重决定，不能只通过输入尺寸手工推出；这里的示例数字只用于继续推导张量如何进入几何矩阵。

### 15.17 第 16 步：9 维编码变成外参和内参

对上面的教学输出，四元数 [0,0,0,1] 对应单位旋转：

~~~text
R0 = R1 = identity(3)
~~~

外参是：

~~~text
frame 0：
[1,0,0,0]
[0,1,0,0]
[0,0,1,0]

frame 1：
[1,0,0,0.10]
[0,1,0,0.00]
[0,0,1,0.00]
~~~

FOV 到焦距的转换由：

$$
f_x=\frac{W/2}{\tan(fov_w/2)},
\qquad
f_y=\frac{H/2}{\tan(fov_h/2)}
$$

本案例中：

~~~text
W = 518
H = 294
fov_w = fov_h = π/2
~~~

因此：

$$
f_x=259,\qquad f_y=147,\qquad c_x=259,\qquad c_y=147
$$

两帧内参相同：

~~~text
K =
[259, 0, 259]
[0, 147, 147]
[0,   0,   1]
~~~

此时 pose_enc 已经变成：

~~~text
extrinsic：[1,2,3,4]
intrinsic：[1,2,3,3]
~~~

这一步在 Solver.run_predictions() 中调用 pose_encoding_to_extri_intri() 完成，不在 CompatibleVGGT.forward() 内完成。

### 15.18 第 17 步：DepthHead 切出 patch token

DepthHead 收到：

~~~text
geometry_tokens：24 个 [1,2,782,2048]，FP32
images.float()： [1,2,3,294,518]，FP32
patch_start_idx：5
~~~

它从 4 个中间层取特征，当前索引是：

~~~text
[4, 11, 17, 23]
~~~

以第 23 层为例：

~~~text
[1,2,782,2048]
    ↓ [:,:,5:,:]
[1,2,777,2048]
    ↓ reshape
[2,777,2048]
    ↓ permute 与 reshape
[2,2048,21,37]
~~~

四个层各自经过通道投影和 resize，再进入 scratch_forward 多尺度融合。融合后上采样回图像大小：

~~~text
[2,128,21,37]
    ↓ 多尺度融合
[2,128,294,518]
~~~

最后 output_conv2 生成两个通道：

~~~text
[2,128,294,518]
    ↓ 3×3 conv、ReLU、1×1 conv
[2,2,294,518]
~~~

第一通道进入 exp 激活得到深度，第二通道进入 expp1 激活得到置信度。恢复 batch 和帧维度后，当前上游 DPTHead 的实际布局是：

~~~text
depth      = [1,2,1,294,518]
depth_conf = [1,2,1,294,518]
~~~

这里的 1 是通道维。它不是高度或宽度。

### 15.19 第 18 步：用一个像素继续算到三维点

为了展示深度如何离开网络，假设 frame 0 在像素：

~~~text
u = 300
v = 160
~~~

模型在这个像素输出的 raw 深度通道假设为 ln(2)，经过 exp 后：

~~~text
depth[frame0, 0, v, u] = 2.0
~~~

raw 置信度通道假设为 ln(3)，经过 expp1 后：

~~~text
depth_conf[frame0, 0, v, u] = 1 + 3 = 4.0
~~~

使用内参：

~~~text
fx = 259
fy = 147
cx = 259
cy = 147
z  = 2
~~~

针孔反投影公式：

$$
x=\frac{(u-c_x)z}{f_x},
\qquad
y=\frac{(v-c_y)z}{f_y},
\qquad
z=z
$$

代入：

$$
x=\frac{(300-259)\times2}{259}\approx0.3166
$$

$$
y=\frac{(160-147)\times2}{147}\approx0.1769
$$

因此这个像素对应的相机坐标点约为：

~~~text
p_camera = [0.3166, 0.1769, 2.0000]
~~~

这就是 unproject_depth_map_to_point_map() 返回的相机坐标点。当前函数虽然变量名里有 world，但源码实际把 cam_coords_points 放入返回数组；它还没有成为最终 PoseGraph 世界坐标。

### 15.20 第 19 步：forward() 的最终返回值

对本案例，CompatibleVGGT.forward() 返回字典可以概括为：

~~~text
pose_enc：
    [1,2,9]，FP32

pose_enc_list：
    4 个 [1,2,9]，FP32

depth：
    [1,2,1,294,518]，FP32

depth_conf：
    [1,2,1,294,518]，FP32

images：
    [1,2,3,294,518]，FP32

target_tokens：
    [2,782,1024]，通常保留 Aggregator dtype

image_match_ratio：
    普通建图为 None
~~~

到这里，模型前向已经结束。它输出的是：

~~~text
每帧的相机编码
每个像素的深度
每个像素的深度置信度
输入图像的 FP32 副本
可选的目标层视觉特征
可选的双帧匹配分数
~~~

它还没有创建 Submap，也没有把点放入全局地图。

### 15.21 第 20 步：Solver 如何接收这些结果

Solver.run_predictions() 接着执行：

~~~python
extrinsic, intrinsic = pose_encoding_to_extri_intri(
    predictions["pose_enc"],
    images.shape[-2:],
)
predictions["extrinsic"] = extrinsic
predictions["intrinsic"] = intrinsic
~~~

然后把普通 Tensor 转成 FP32、CPU NumPy，并去掉 batch 维：

~~~python
predictions[key] = (
    predictions[key].float().cpu().numpy().squeeze(0)
)
~~~

于是，除了 target_tokens 外，数据大致变成：

~~~text
pose_enc：    [2,9]       NumPy
depth：       [2,1,294,518] NumPy
depth_conf：  [2,1,294,518] NumPy
images：      [2,3,294,518] NumPy
extrinsic：   [2,3,4]      NumPy
intrinsic：   [2,3,3]      NumPy
~~~

这里需要特别留意当前源码的布局边界：DPTHead 返回的是 [B,S,1,H,W]，而 geometry.py 的文档接口更常写成 [S,H,W,1] 或 [S,H,W]。如果实际运行时 unproject_depth_map_to_point_map() 报维度错误，应在这个边界明确转换：

~~~python
depth = np.transpose(depth, (0, 2, 3, 1))
depth_conf = np.transpose(depth_conf, (0, 2, 3, 1))
~~~

转换后才是：

~~~text
depth：      [2,294,518,1]
depth_conf： [2,294,518,1]
~~~

具体是否需要这一步，应以现场打印出的实际 shape 为准；不能仅根据变量名判断。

### 15.22 第 21 步：写入 Submap 和 PoseGraph

Solver.add_points() 会把：

~~~text
depth + intrinsic + extrinsic
    ↓
unproject_depth_map_to_point_map()
    ↓
[2,294,518,3] 的相机坐标点
~~~

保存到当前 Submap：

~~~text
pointclouds： [2,294,518,3]
poses：       [2,4,4]，world_to_cam
proj_mats：   [2,4,4]，扩展内参
conf：        [2,294,518]
conf_masks：  [2,294,518]
~~~

第一次建图时，当前子地图的第 0 帧使用图节点 0 和单位初值。后续帧的节点由当前子地图内部的相对变换建立；下一窗口则通过重叠帧估计尺度，把新子地图接到旧图上。

因此完整链路最后变成：

~~~text
RGB 文件
  → [0,1] 图像 Tensor
  → [B,S,3,H,W]
  → patch tokens
  → frame/global tokens
  → FP32 geometry tokens
  → pose_enc、depth、depth_conf
  → extrinsic、intrinsic
  → 相机坐标点云
  → Submap
  → PoseGraph 节点和因子
  → optimize() 后的全局地图
~~~

## 十六、这个案例中哪些数字是真实的，哪些只是演示

为了避免把文档中的教学数字误解成模型固定输出，可以按下面区分：

| 数字 | 性质 |
|---|---|
| 1920×1080 | 当前 office_loop 图像文件的实际尺寸 |
| 518、294、14、21、37、777、782、1024、2048 | 当前代码和模型配置决定的形状数字 |
| 24 个 Aggregator 输出层 | 当前默认 depth 配置 |
| [0,0,0,0,0,0,1,π/2,π/2] | 为了演示矩阵转换而设定的假设输出 |
| depth=2、confidence=4 | 为了演示反投影而设定的假设像素输出 |
| 真实 pose_enc、depth、depth_conf 数值 | 由 checkpoint、设备精度和输入图像共同决定 |

如果需要查看真实数值，应在真实 checkpoint 已加载后打印：

~~~python
print(predictions["pose_enc"][0, 0])
print(predictions["depth"][0, 0].min())
print(predictions["depth"][0, 0].max())
print(predictions["depth_conf"][0, 0].min())
print(predictions["depth_conf"][0, 0].max())
~~~

这样得到的才是当前权重对 frame_0001 和 frame_0002 的实际预测；文档中的示例数值只用于把公式和数据流完整接起来。

## 十七、案例中每一步对应的源码

如果要在调试器中逐步验证上面的形状，可以按下面顺序设置断点：

| 数据流阶段 | 源码位置 | 重点观察 |
|---|---|---|
| 文件读取、缩放、转 Tensor | [third_party/vggt/vggt/utils/load_fn.py](/home/erge/work/progress/tmp/VGGT-SLAM/third_party/vggt/vggt/utils/load_fn.py) | images 的尺寸和数值范围 |
| 输入补 batch、选择 dtype | [vggt_slam/model_utils.py:25-37](/home/erge/work/progress/tmp/VGGT-SLAM/vggt_slam/model_utils.py:25) | images.shape、dtype、geometry_tokens |
| patch embedding、特殊 token | [third_party/vggt/vggt/models/aggregator.py:700-730](/home/erge/work/progress/tmp/VGGT-SLAM/third_party/vggt/vggt/models/aggregator.py:700) | patch_tokens、tokens、patch_start_idx |
| frame/global attention | [third_party/vggt/vggt/models/aggregator.py:744-804](/home/erge/work/progress/tmp/VGGT-SLAM/third_party/vggt/vggt/models/aggregator.py:744) | output_list、target_tokens、match_ratio |
| 相机编码 | [third_party/vggt/vggt/heads/camera_head.py:55-110](/home/erge/work/progress/tmp/VGGT-SLAM/third_party/vggt/vggt/heads/camera_head.py:55) | pose_tokens、每轮 [B,S,9] |
| 深度和置信度 | [third_party/vggt/vggt/heads/dpt_head.py:85-180](/home/erge/work/progress/tmp/VGGT-SLAM/third_party/vggt/vggt/heads/dpt_head.py:85) | patch 网格、DPT 输出布局 |
| 字典返回 | [vggt_slam/model_utils.py:42-50](/home/erge/work/progress/tmp/VGGT-SLAM/vggt_slam/model_utils.py:42) | 每个返回键的 shape 和 dtype |
| 编码转外参/内参 | [third_party/vggt/vggt/utils/pose_enc.py:72-130](/home/erge/work/progress/tmp/VGGT-SLAM/third_party/vggt/vggt/utils/pose_enc.py:72) | extrinsic、intrinsic |
| 深度反投影 | [third_party/vggt/vggt/utils/geometry.py:15-115](/home/erge/work/progress/tmp/VGGT-SLAM/third_party/vggt/vggt/utils/geometry.py:15) | 相机坐标点和深度布局 |
| 写入 Submap | [vggt_slam/solver.py:199-254](/home/erge/work/progress/tmp/VGGT-SLAM/vggt_slam/solver.py:199) | pointclouds、poses、conf |

案例使用的两个实际输入文件是 [frame_0001.jpg](/home/erge/work/progress/tmp/VGGT-SLAM/office_loop/frame_0001.jpg) 和 [frame_0002.jpg](/home/erge/work/progress/tmp/VGGT-SLAM/office_loop/frame_0002.jpg)。如果要得到这两张图对应的真实模型数值，需要本地 checkpoint 已经存在，并在模型完成加载后打印第十六节列出的张量。

