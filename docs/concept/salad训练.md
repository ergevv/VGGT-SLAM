# SALAD 的训练过程：从地点批次到 8448 维描述子，再到 Multi-Similarity Loss

## 先给结论

SALAD 的训练和 DINOv2 的原始训练是两个不同阶段。

**DINOv2 原始训练：**

```text
没有人工类别标签
        ↓
教师—学生、自监督、一致性学习
        ↓
得到通用视觉主干
```

**SALAD 训练：**

```text
有地点级别标签
        ↓
同一个地点的图像作为正样本
不同地点的图像作为负样本
        ↓
DINOv2 主干 + SALAD 聚合器
        ↓
输出整图描述子
        ↓
Multi-Similarity Loss
        ↓
让同地点描述子更近，不同地点描述子更远
```

SALAD 不是通过“网球、建筑、天空”等像素级类别标签训练的。它使用的是**图像属于哪个地点**这一层级的监督信号。

因此，SALAD 学习的目标不是：

```text
识别图像中有什么物体
```

而是：

```text
两张图像是否来自同一个地点？
哪些局部区域有助于区分地点？
哪些局部区域应该被弱化或丢弃？
```

论文和官方代码采用 GSV-Cities 进行训练，训练 4 个完整 epoch，每个 batch 组织为 60 个地点、每个地点 4 张图像，总计 240 张图像；优化使用 AdamW 和 Multi-Similarity Loss。

---

# 一、先区分：SALAD 训练时到底优化哪些参数？

一个完整的 DINOv2-SALAD 模型可以拆成：

```text
DINOv2 ViT-B/14 主干
        +
SALAD 局部特征投影分支
        +
SALAD 槽位分数分支
        +
SALAD CLS 投影分支
        +
最终描述子归一化
```

训练时通常：

- DINOv2 使用预训练权重初始化；
- DINOv2 前面的大部分 Transformer block 冻结；
- 只微调最后 4 个 Transformer block；
- SALAD 的投影层、分数层、CLS 层参与训练；
- 最终通过地点识别损失更新这些参数。

论文明确报告了只微调 DINOv2 最后 4 层的方案；消融实验中，训练 2 层和 4 层的结果优于完全冻结或训练全部层。MSLS Val 的 Recall@1 结果为：

| 训练方式 | Recall@1 |
|---|---:|
| DINOv2 全部冻结 | 88.5 |
| 训练最后 2 层 | 92.0 |
| 训练最后 4 层 | 92.2 |
| 训练最后 6 层 | 91.6 |
| 训练全部层 | 89.2 |

这说明 SALAD 并不是把 DINOv2 全部推倒重训，而是在保留通用视觉能力的基础上，对后几层进行地点识别适配。

官方训练配置中，主干使用 `dinov2_vitb14`，并设置：

```python
backbone_config = {
    "num_trainable_blocks": 4,
    "return_token": True,
    "norm_layer": True,
}
```

SALAD 聚合器使用：

```python
agg_config = {
    "num_channels": 768,
    "num_clusters": 64,
    "cluster_dim": 128,
    "token_dim": 256,
}
``` :chatgpt-content-reference{index="2"}


---

# 二、训练数据是什么样的？

## 2.1 一个 batch 不是随便抽 240 张图像

SALAD 的 batch 有明确的分组结构。

假设我们构造一个简化 batch：

```text
地点 A：
A1、A2、A3、A4

地点 B：
B1、B2、B3、B4

地点 C：
C1、C2、C3、C4
```

对应的地点标签为：

```text
labels = [0, 0, 0, 0,
          1, 1, 1, 1,
          2, 2, 2, 2]
```

真实训练中，官方配置是：

```text
60 个地点
每个地点 4 张图像
总图像数 = 60 × 4 = 240
```

如果用张量表示，数据加载器可以概念性地返回：

\[
\text{places}\in\mathbb{R}^{60\times4\times3\times224\times224}
\]

以及：

\[
\text{labels}\in\mathbb{R}^{60\times4}.
\]

之后将图像维度和每个地点的图像数量展平：

\[
[60,4,3,224,224]
\rightarrow
[240,3,224,224]
\]

标签也展平为：

\[
[60,4]\rightarrow[240].
\]

实际 `VPRModel` 的训练流程会先计算整批图像的描述子，再用标签和在线 miner 构造正负样本对，最后计算损失。

## 2.2 为什么同一个地点要放 4 张图像？

因为每个地点至少需要多个观测，才能构造正样本关系。

对于一个地点的 4 张图像：

```text
A1、A2、A3、A4
```

任意两张都可以组成正样本对：

\[
(A_1,A_2),(A_1,A_3),(A_1,A_4),
(A_2,A_3),(A_2,A_4),(A_3,A_4)
\]

因此每个地点有：

\[
\binom42=6
\]

个无序正样本对。

60 个地点一共有：

\[
60\times6=360
\]

个无序正样本对。

240 张图像一共可以形成：

\[
\binom{240}{2}=28680
\]

个无序图像对。

去掉同地点正样本后，不同地点负样本约为：

\[
28680-360=28320.
\]

所以一个 batch 同时包含：

```text
相对较少但非常重要的同地点正样本

大量不同地点负样本
```

但 SALAD 不会把所有这些 pair 等权使用，而是通过 Multi-Similarity Miner 选出更有训练价值的 pair。

---

# 三、图像进入 SALAD 之前，先经过 DINOv2

训练图像采用：

\[
224\times224
\]

作为输入尺寸。因为 DINOv2 的 patch 大小是 \(14\times14\)，所以每边有：

\[
224/14=16
\]

个 patch，总数为：

\[
N=16\times16=256.
\]

DINOv2-B/14 的 token 维度为：

\[
C=768.
\]

因此，对整个 batch 的 240 张图像：

```text
输入图像：
[240, 3, 224, 224]

DINOv2 局部特征：
[240, 768, 16, 16]

DINOv2 CLS 特征：
[240, 768]
```

将局部空间维度展开：

\[
[240,768,16,16]
\rightarrow
[240,768,256].
\]

其中：

- 240 是图像数量；
- 768 是每个 patch 的特征维度；
- 256 是 patch 数量。

每张图像都得到：

```text
256 个局部 patch 特征
1 个 CLS 整图特征
```

SALAD 的输入不是原始 RGB，而是 DINOv2 已经编码过的局部 token 和 CLS token。DINOv2 负责生成视觉表示，SALAD 负责根据地点识别目标重新组织这些表示。

---

# 四、SALAD 的三个可学习分支

以一张图像为例，DINOv2 输出：

\[
X\in\mathbb{R}^{768\times16\times16}
\]

以及：

\[
t\in\mathbb{R}^{768}.
\]

SALAD 对这两部分分别处理。

---

## 4.1 局部特征分支：决定“每个 patch 贡献什么内容”

局部特征分支使用两个 \(1\times1\) 卷积：

```text
768 → 512 → 128
```

具体为：

```python
Conv2d(768, 512, kernel_size=1)
Dropout(0.3)
ReLU
Conv2d(512, 128, kernel_size=1)
```

它的作用不是混合邻近像素，而是对每个 patch 的 768 维向量做通道投影。

输入：

\[
X\in\mathbb{R}^{B\times768\times16\times16}
\]

输出：

\[
F\in\mathbb{R}^{B\times128\times16\times16}.
\]

展平空间维度：

\[
F\in\mathbb{R}^{B\times128\times256}.
\]

对于第 \(i\) 个 patch：

\[
f_i\in\mathbb{R}^{128}.
\]

这里的 \(f_i\) 表示：

> 如果这个 patch 被聚合到某个槽位，它将贡献什么内容？

---

## 4.2 分数分支：决定“这个 patch 应该分给哪些槽位”

分数分支也使用：

```text
768 → 512 → 64
```

具体为：

```python
Conv2d(768, 512, kernel_size=1)
Dropout(0.3)
ReLU
Conv2d(512, 64, kernel_size=1)
```

输出：

\[
S\in\mathbb{R}^{B\times64\times16\times16}.
\]

展平后：

\[
S\in\mathbb{R}^{B\times64\times256}.
\]

其中：

\[
S_{k,i}
\]

表示第 \(i\) 个 patch 对第 \(k\) 个聚合槽位的匹配分数。

这 64 个槽位不是人工指定的：

```text
槽位 1 = 建筑
槽位 2 = 树
槽位 3 = 道路
```

它们是训练出来的统一描述子通道。训练过程会让这些槽位逐渐形成有利于地点区分的分工。

---

## 4.3 CLS 分支：保留整图上下文

CLS 特征经过：

```text
768 → 512 → 256
```

具体是：

```python
Linear(768, 512)
ReLU
Linear(512, 256)
```

得到：

\[
g\in\mathbb{R}^{B\times256}.
\]

这条分支保留整图级别的信息。

SALAD 最终同时使用：

```text
局部槽位聚合结果
+
CLS 全局结果
```

因为只有局部 patch，可能丢掉整体布局；只有 CLS，又可能丢掉具体地点细节。

官方 SALAD 实现正是使用局部特征分支、分数分支和 token 分支，并将最终结果拼接后再进行整体归一化。

---

# 五、从分数到 Sinkhorn 分配

## 5.1 加入 dustbin

普通槽位分数为：

\[
S\in\mathbb{R}^{B\times64\times256}.
\]

再增加一个 dustbin 行：

\[
\bar S\in\mathbb{R}^{B\times65\times256}.
\]

其中最后一行由一个共享的可学习标量 \(z\) 填充：

\[
\bar S_{65,i}=z.
\]

这个 \(z\) 不是每张图像单独产生的，也不是每个 patch 单独预测的；它是 SALAD 模型中的一个共享参数。

对于同一张图像的所有 patch，以及同一个 batch 中的所有图像，使用的都是同一个参数值。论文明确将 dustbin 与单个可学习参数 \(z\) 建模。

## 5.2 Sinkhorn 的目标约束

对每张图像，假设有：

\[
N=256
\]

个 patch，以及：

\[
K=64
\]

个普通槽位。

每个 patch 拥有 1 单位质量，因此总质量为：

\[
256.
\]

普通槽位每个接收 1 单位质量：

\[
64\times1=64.
\]

dustbin 接收剩余：

\[
256-64=192.
\]

因此，包含 dustbin 的分配矩阵为：

\[
\bar P\in\mathbb{R}^{65\times256}.
\]

约束可以写成：

\[
\sum_{k=1}^{65}\bar P_{k,i}=1,
\]

表示第 \(i\) 个 patch 的总质量为 1；

\[
\sum_{i=1}^{256}\bar P_{k,i}=1,
\qquad k=1,\ldots,64,
\]

表示每个普通槽位接收 1 单位质量；

\[
\sum_{i=1}^{256}\bar P_{65,i}=192,
\]

表示 dustbin 接收剩余质量。

这不是普通的“每个 patch 独立 softmax”。

普通 softmax 只保证：

\[
\sum_k P_{k,i}=1.
\]

它没有保证每个槽位收到多少总量。

Sinkhorn 额外引入了槽位侧的总量约束，因此是一个全局耦合分配：

```text
某个 patch 分给槽位 1 的权重增加
        ↓
槽位 1 的可用质量减少
        ↓
其他 patch 对槽位 1 的分配也会受到影响
```

这正是 SALAD 与简单 patch-wise softmax 的差别。

---

# 六、SALAD 的前向输出如何形成 8448 维描述子？

去掉 dustbin 行后：

\[
P\in\mathbb{R}^{B\times64\times256}.
\]

对每个槽位做加权汇总：

\[
v_{b,k}
=
\sum_{i=1}^{256}P_{b,k,i}f_{b,i}.
\]

其中：

- \(b\) 是图像编号；
- \(k\) 是槽位编号；
- \(i\) 是 patch 编号；
- \(f_{b,i}\in\mathbb{R}^{128}\) 是局部 patch 特征；
- \(v_{b,k}\in\mathbb{R}^{128}\) 是槽位汇总结果。

所以：

\[
V\in\mathbb{R}^{B\times64\times128}.
\]

每个槽位分别进行 L2 归一化，然后展开：

\[
64\times128=8192.
\]

CLS 分支输出：

\[
g\in\mathbb{R}^{B\times256}.
\]

拼接后：

\[
8192+256=8448.
\]

最终：

\[
Y\in\mathbb{R}^{B\times8448}.
\]

再对整条描述子做 L2 归一化：

\[
y_b=\frac{Y_b}{\|Y_b\|_2}.
\]

因此，实际训练 batch 的描述子张量为：

\[
\boxed{
y\in\mathbb{R}^{240\times8448}
}
\]

SALAD 官方代码的聚合器正是：

1. 产生局部特征；
2. 产生槽位分数；
3. 加 dustbin；
4. 进行三轮 Sinkhorn；
5. 去掉 dustbin；
6. 加权聚合；
7. 拼接 CLS；
8. 最终归一化。

---

# 七、训练没有 patch 标签，模型怎么知道哪些区域应该进入 dustbin？

这是理解 SALAD 训练最关键的地方。

训练数据并没有告诉模型：

```text
这个 patch 是天空，所以进入 dustbin。

这个 patch 是建筑，所以进入槽位 1。

这个 patch 是车辆，所以进入 dustbin。
```

训练数据只告诉它：

```text
图像 A1、A2、A3、A4 来自同一个地点。

图像 B1、B2、B3、B4 来自另一个地点。
```

然后通过地点级别损失反向传播。

假设某个 patch 是天空：

```text
它在很多地点都类似
对区分地点帮助不大
```

如果把天空权重降低、把建筑细节权重提高，能够让同地点描述子更接近、不同地点描述子更远，那么这样的分配方式会降低损失。

经过大量训练后，模型可能逐渐学会：

```text
建筑立面、门窗、植被纹理：
通常有助于区分地点

天空、道路、动态车辆：
通常区分度较低
```

论文的可视化结果显示，模型通常会弱化天空、道路和动态物体，并更多关注建筑、植被和具有地点辨别力的区域。但这是训练数据和地点识别目标产生的结果，不是手工编写的规则。

---

# 八、地点级标签怎样生成正样本和负样本？

设描述子为：

\[
y_1,y_2,\ldots,y_B.
\]

因为最终描述子已经 L2 归一化，所以两个描述子的点积等于余弦相似度：

\[
s_{ij}=y_i^Ty_j.
\]

如果：

\[
\ell_i=\ell_j,
\]

表示两张图像属于同一个地点，则：

\[
(i,j)\in\mathcal P_i,
\]

是 anchor \(i\) 的正样本。

如果：

\[
\ell_i\ne\ell_j,
\]

则：

\[
(i,j)\in\mathcal N_i,
\]

是负样本。

注意：

> 这里的 label 是地点标签，不是物体类别标签。

例如：

```text
label = 17
```

只表示“这张图来自地点 17”，不表示图中有第 17 类物体。

---

# 九、为什么还需要 Multi-Similarity Miner？

如果直接使用所有 pair，会有一个问题：

```text
同地点图像已经非常相似的 pair：
训练价值很小

不同地点图像本来就非常不相似的 pair：
训练价值也很小
```

真正有价值的是：

```text
同地点但相似度偏低的 pair

不同地点但相似度偏高的 pair
```

这些才是容易混淆的样本。

SALAD 使用 `MultiSimilarityMiner`，margin 参数为：

\[
\epsilon=0.1.
\]

官方代码在训练时先调用 miner，再把 miner 找到的 pair 传给 Multi-Similarity Loss。

对于 anchor \(i\)，设：

- 正样本相似度为 \(s_{ip}\)；
- 负样本相似度为 \(s_{in}\)。

Miner 的规则可以解释为：

### 负样本选择规则

负样本 \(n\) 被选中，如果：

\[
s_{in}>
\min_{p\in\mathcal P_i}s_{ip}-\epsilon.
\]

也就是说，如果某个不同地点图像的相似度已经接近最难的正样本，就值得训练。

### 正样本选择规则

正样本 \(p\) 被选中，如果：

\[
s_{ip}<
\max_{n\in\mathcal N_i}s_{in}+\epsilon.
\]

也就是说，如果某个同地点图像的相似度还没有明显超过难负样本，也值得训练。

---

# 十、用具体数字演示 Miner

为了便于手算，先构造一个小 batch。

假设 anchor 是地点 A 的图像：

\[
y_a=[1,0].
\]

它已经单位归一化。

同地点正样本：

\[
y_p=[0.82,\sqrt{1-0.82^2}]
\approx[0.82,0.5724].
\]

两者相似度：

\[
s_{ap}=y_a^Ty_p=0.82.
\]

两个不同地点负样本：

\[
y_{n_1}=[0.78,\sqrt{1-0.78^2}]
\approx[0.78,0.6258],
\]

\[
s_{an_1}=0.78.
\]

另一个负样本：

\[
y_{n_2}=[0.2,\sqrt{1-0.2^2}]
\approx[0.2,0.9799],
\]

\[
s_{an_2}=0.2.
\]

对于 anchor \(a\)：

```text
正样本相似度：[0.82]
负样本相似度：[0.78, 0.20]
```

最难正样本为：

\[
\min s_{ap}=0.82.
\]

使用：

\[
\epsilon=0.1.
\]

对于负样本 \(n_1\)：

\[
0.78>0.82-0.1=0.72.
\]

所以 \(n_1\) 被选中。

对于负样本 \(n_2\)：

\[
0.2<0.72.
\]

它太容易区分，不参与这一轮的 hard-pair 损失。

再看正样本选择：

\[
\max s_{an}=0.78.
\]

正样本被选中的条件是：

\[
0.82<0.78+0.1=0.88.
\]

成立，所以 \(p\) 也被选中。

最终这个 anchor 的有效 pair 是：

```text
正样本：0.82
负样本：0.78
```

而不是把 0.2 这个容易负样本也放进来。

---

# 十一、Multi-Similarity Loss 的精确形式

SALAD 训练使用 Multi-Similarity Loss。一般形式为：

\[
\mathcal L_{\mathrm{MS}}
=
\frac1B\sum_{i=1}^{B}\mathcal L_i,
\]

其中每个 anchor 的损失为：

\[
\mathcal L_i
=
\frac1\alpha
\log
\left[
1+
\sum_{p\in\mathcal P_i}
\exp\left(
-\alpha(s_{ip}-\lambda)
\right)
\right]
\]

\[
+
\frac1\beta
\log
\left[
1+
\sum_{n\in\mathcal N_i}
\exp\left(
\beta(s_{in}-\lambda)
\right)
\right].
\]

其中：

- \(s_{ip}\) 是 anchor 与正样本的相似度；
- \(s_{in}\) 是 anchor 与负样本的相似度；
- \(\mathcal P_i\) 是被 miner 选出的正样本集合；
- \(\mathcal N_i\) 是被 miner 选出的负样本集合；
- \(\alpha\) 控制正样本项；
- \(\beta\) 控制负样本项；
- \(\lambda\) 是相似度基准。

SALAD 的损失工厂配置使用：

\[
\alpha=1,\qquad
\beta=50,\qquad
\lambda=0,
\]

并使用点积相似度；因为 SALAD 最终描述子已经 L2 归一化，所以点积就是余弦相似度。

Multi-Similarity Loss 的原始公式和 pair weighting 机制可见其论文；官方 PyTorch Metric Learning 文档也给出了相同形式。

在 \(\lambda=0\) 时，公式变为：

\[
\mathcal L_i
=
\log
\left[
1+
\sum_{p\in\mathcal P_i}
e^{-s_{ip}}
\right]
+
\frac1{50}
\log
\left[
1+
\sum_{n\in\mathcal N_i}
e^{50s_{in}}
\right].
\]

这表达了两个方向：

```text
正样本相似度越低：
正样本损失越大

负样本相似度越高：
负样本损失越大
```

---

# 十二、用具体数字计算一次 SALAD 损失

继续使用刚才的 pair：

\[
s_{ap}=0.82,
\]

\[
s_{an}=0.78.
\]

取：

\[
\alpha=1,\qquad
\beta=50,\qquad
\lambda=0.
\]

## 12.1 正样本项

\[
\mathcal L_+
=
\log(1+e^{-0.82}).
\]

因为：

\[
e^{-0.82}\approx0.4404,
\]

所以：

\[
\mathcal L_+
=
\log(1+0.4404)
\approx0.3649.
\]

## 12.2 负样本项

\[
\mathcal L_-
=
\frac1{50}\log(1+e^{50\times0.78}).
\]

因为：

\[
50\times0.78=39,
\]

所以：

\[
\mathcal L_-
=
\frac1{50}\log(1+e^{39})
\approx\frac{39}{50}
=0.78.
\]

## 12.3 总损失

\[
\mathcal L_i
=
\mathcal L_++\mathcal L_-
\]

\[
\boxed{
\mathcal L_i\approx0.3649+0.78=1.1449
}
\]

这个 anchor 的损失比较大，因为：

```text
正样本相似度只有 0.82，还不够高

负样本相似度却高达 0.78，非常容易混淆
```

假设训练一段时间以后：

\[
s_{ap}=0.95,
\]

\[
s_{an}=0.20.
\]

则：

\[
\mathcal L_+
=
\log(1+e^{-0.95})
\approx0.3270,
\]

\[
\mathcal L_-
=
\frac1{50}\log(1+e^{10})
\approx0.2000.
\]

总损失：

\[
\mathcal L_i\approx0.5270.
\]

相比之前的：

\[
1.1449\rightarrow0.5270.
\]

这说明描述子空间变得更符合地点识别目标。

---

# 十三、损失的梯度怎样推动相似度变化？

对于只有一个正样本和一个负样本的简化情况：

\[
\mathcal L_+
=
\log(1+e^{-s_{ap}}),
\]

其对正样本相似度的梯度为：

\[
\frac{\partial\mathcal L_+}{\partial s_{ap}}
=
-\frac{e^{-s_{ap}}}{1+e^{-s_{ap}}}.
\]

当：

\[
s_{ap}=0.82
\]

时：

\[
\frac{\partial\mathcal L_+}{\partial s_{ap}}
\approx-0.3058.
\]

梯度下降的更新是：

\[
s_{ap}\leftarrow
s_{ap}
-\eta\frac{\partial\mathcal L_+}{\partial s_{ap}}.
\]

因为梯度是负的，所以更新会提高 \(s_{ap}\)。

对于负样本：

\[
\mathcal L_-
=
\frac1{50}\log(1+e^{50s_{an}}),
\]

其梯度为：

\[
\frac{\partial\mathcal L_-}{\partial s_{an}}
=
\frac{e^{50s_{an}}}{1+e^{50s_{an}}}.
\]

当：

\[
s_{an}=0.78
\]

时：

\[
\frac{\partial\mathcal L_-}{\partial s_{an}}
\approx1.
\]

梯度下降会降低 \(s_{an}\)。

因此，一次更新的方向是：

```text
同地点：
描述子更加接近

不同地点：
描述子更加远离
```

但实际训练中，网络不是直接修改 \(s_{ap}\) 和 \(s_{an}\)，而是通过反向传播修改：

```text
Multi-Similarity Loss
        ↓
最终 8448 维描述子
        ↓
CLS 投影层、局部聚合层
        ↓
Sinkhorn 分配矩阵
        ↓
槽位分数网络
        ↓
DINOv2 最后 4 个 Transformer block
```

---

# 十四、梯度怎样让某些 patch 进入 dustbin？

这是 SALAD 最有意思的地方。

对于第 \(k\) 个槽位：

\[
v_k=\sum_iP_{k,i}f_i.
\]

最终损失是：

\[
\mathcal L=\mathcal L(y).
\]

损失对局部特征 \(f_i\) 的梯度包含：

\[
\frac{\partial\mathcal L}{\partial f_i}
=
\sum_k
P_{k,i}
\frac{\partial\mathcal L}{\partial v_k}.
\]

这说明：

> 如果 patch \(i\) 被分给普通槽位的权重 \(P_{k,i}\) 很小，那么它对最终描述子的直接贡献也很小。

另一方面，损失也会通过分配矩阵反向传播：

\[
\frac{\partial\mathcal L}{\partial P_{k,i}}
=
\left(
\frac{\partial\mathcal L}{\partial v_k}
\right)^T
f_i.
\]

于是模型会逐渐学习：

```text
哪些 patch 的内容会让同地点图像更接近
哪些 patch 的内容会让不同地点图像混淆
哪些 patch 应该更多进入普通槽位
哪些 patch 应该更多进入 dustbin
```

这不是一个直接的二分类监督：

```text
有用 patch / 无用 patch
```

而是一个间接结果：

```text
哪个分配方式能让最终地点描述子更好？
```

如果分配到 dustbin 后，地点损失降低，那么这个方向会得到强化。

如果某个区域虽然视觉上很显眼，但在很多地点都重复出现，保留它可能反而降低检索性能，那么模型可能逐渐降低它进入普通槽位的权重。

---

# 十五、一个完整训练 step 的伪代码

可以把 SALAD 的训练过程写成下面这样：

```python
# places:
# [num_places, images_per_place, 3, 224, 224]
# 例如 [60, 4, 3, 224, 224]

# labels:
# [num_places, images_per_place]
# 例如：
# [[0, 0, 0, 0],
#  [1, 1, 1, 1],
#  ...
#  [59,59,59,59]]

places, labels = batch

num_places = places.shape[0]
images_per_place = places.shape[1]

# 展平图像维度
images = places.reshape(
    num_places * images_per_place,
    3,
    224,
    224,
)

labels = labels.reshape(-1)

# 前向
descriptors = model(images)
# [240, 8448]
# 每一行是一张图像的 SALAD 描述子
# 已经 L2 normalize

# 在线挖掘 hard positive / hard negative
mined_pairs = miner(descriptors, labels)

# Multi-Similarity Loss
loss = loss_fn(
    descriptors,
    labels,
    mined_pairs,
)

# 反向传播
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 学习率调度
scheduler.step()
```

官方 `VPRModel` 的训练逻辑就是先通过模型得到描述子，再通过 miner 选择 pair，将 miner 结果传递给 loss。[GitHub](https://github.com/serizba/salad/blob/main/vpr_model.py?utm_source=chatgpt.com)

---

# 十六、训练超参数及其作用

SALAD 论文给出的主要训练设置包括：

| 项目 | 设置 |
|---|---:|
| 训练数据 | GSV-Cities |
| 训练 epoch | 4 |
| 每 batch 地点数 | 60 |
| 每地点图像数 | 4 |
| 每 step 图像数 | 240 |
| 训练图像尺寸 | \(224\times224\) |
| 主干 | DINOv2-B/14 |
| 可训练 DINOv2 block | 最后 4 层 |
| 聚合槽位数 | 64 |
| 每槽位维度 | 128 |
| CLS 输出维度 | 256 |
| dropout | 0.3 |
| 优化器 | AdamW |
| 初始学习率 | \(6\times10^{-5}\) |
| 学习率变化 | 线性下降到初始值的 20% |
| 训练时间 | 约 30 分钟，RTX 3090 |

论文明确说明训练使用 224×224，而评估可以使用 322×322；由于模型只要求图像尺寸能够被 14 整除，改变输入分辨率会改变 patch 数量，但不改变最终描述子维度。[arxiv.org](https://arxiv.org/html/2311.15937v2?utm_source=chatgpt.com)

官方 `main.py` 中的 AdamW 配置还包括非常小的 weight decay：

```python
lr = 6e-5
optimizer = "adamw"
weight_decay = 9.5e-9
```

并设置：

```python
loss_name = "MultiSimilarityLoss"
miner_name = "MultiSimilarityMiner"
miner_margin = 0.1
``` :chatgpt-content-reference{index="12"}


---

# 十七、为什么不用普通分类交叉熵？

如果用普通分类交叉熵，需要把地点当成类别：

```text
地点 A = 类别 0
地点 B = 类别 1
地点 C = 类别 2
……
```

但视觉地点识别有一个问题：

> 推理时会遇到训练集中没有出现过的新地点。

如果训练成固定类别分类器，模型更容易学习：

```text
判断它属于训练时见过的地点类别
```

而 SALAD 的 metric learning 目标是：

```text
同地点描述子相似
不同地点描述子不相似
```

这样，推理阶段可以把新图像与新建立的数据库描述子比较，而不需要提前把新地点注册成一个神经网络类别。

所以 SALAD 的输出不是：

```text
地点 A 的概率
地点 B 的概率
地点 C 的概率
```

而是：

```text
8448 维描述子
```

这条描述子可以用于：

```text
数据库检索
回环候选检测
地点匹配
图像序列中的重定位
```

---

# 十八、为什么要用 DINOv2，而不是随机初始化的 CNN？

SALAD 的任务需要局部特征满足两个性质：

```text
同一地点在视角、光照和季节变化下仍然相对稳定

不同地点的局部结构仍然具有区分性
```

DINOv2 已经通过大规模自监督训练获得了较强的局部视觉表示，尤其适合输出高空间分辨率的 patch 特征。

SALAD 论文的消融实验表明：

- DINOv2 + SALAD 明显优于不使用 DINOv2 的方案；
- DINOv2 + NetVLAD 也有效；
- 但 SALAD 在相近描述子维度下表现更好；
- SALAD 对高空间分辨率特征特别适合。

论文指出，DINOv2 的局部特征适合密集视觉感知和局部聚合，这也是 SALAD 选择它作为 backbone 的原因。[arxiv.org](https://arxiv.org/html/2311.15937v2?utm_source=chatgpt.com)

---

# 十九、为什么只微调最后四层，而不是全部 DINOv2？

这可以从三个方面理解。

## 19.1 保留通用视觉能力

DINOv2 前层已经学习了比较通用的：

```text
边缘
纹理
局部形状
颜色组合
```

如果全部层都大幅更新，可能破坏原来的通用表示。

## 19.2 让后层适应地点识别

后面的 Transformer block 更接近最终语义和全局上下文，适合调整为：

```text
哪些信息对地点区分更重要
哪些外观变化应该被忽略
哪些建筑细节应该保留
```

## 19.3 降低训练代价和过拟合风险

DINOv2-B 有大量参数。只更新最后 4 层可以：

```text
减少反向传播开销
减少训练数据要求
降低过拟合风险
保留预训练知识
```

论文的消融结果也显示，训练全部 block 反而不如训练最后 4 个 block。[arxiv.org](https://arxiv.org/html/2311.15937v2?utm_source=chatgpt.com)

---

# 二十、训练过程可以用一句话概括

对每个 batch，SALAD 做的是：

```text
把 240 张图像变成 240 条 8448 维描述子
        ↓
根据地点标签建立正负关系
        ↓
找出最难的正样本和负样本
        ↓
让同地点描述子更接近
让不同地点描述子更远
        ↓
梯度反向影响：
DINOv2 最后 4 层
局部特征投影
槽位分数预测
dustbin/ Sinkhorn 分配
CLS 投影
```

完整数学链路可以写成：

\[
I_b
\overset{\text{DINOv2}}{\longrightarrow}
(F_b^{\mathrm{local}},t_b^{\mathrm{CLS}})
\]

\[
F_b^{\mathrm{local}}
\overset{\text{projection}}{\longrightarrow}
F_b
\]

\[
F_b^{\mathrm{local}}
\overset{\text{score head}}{\longrightarrow}
S_b
\]

\[
S_b+\text{dustbin}
\overset{\text{Sinkhorn}}{\longrightarrow}
P_b
\]

\[
V_b=F_bP_b^T
\]

\[
y_b=
\operatorname{Normalize}
\left(
[
\operatorname{Normalize}(g_b);
\operatorname{Flatten}(\operatorname{Normalize}(V_b))
]
\right)
\]

然后对描述子相似度：

\[
s_{ij}=y_i^Ty_j
\]

使用地点标签形成正负 pair：

\[
\mathcal P_i=\{j:\ell_j=\ell_i\},
\]

\[
\mathcal N_i=\{j:\ell_j\ne\ell_i\}.
\]

经过 hard pair mining 后，计算：

\[
\mathcal L_{\mathrm{MS}}
=
\frac1B\sum_i
\left[
\frac1\alpha
\log
\left(
1+\sum_{p\in\mathcal P_i}
e^{-\alpha(s_{ip}-\lambda)}
\right)
+
\frac1\beta
\log
\left(
1+\sum_{n\in\mathcal N_i}
e^{\beta(s_{in}-\lambda)}
\right)
\right].
\]

SALAD 训练的本质就是：

> **让局部特征提取、槽位分配和整图描述子共同适应视觉地点识别，而不是单独训练一个固定的“patch 有用性分类器”。**