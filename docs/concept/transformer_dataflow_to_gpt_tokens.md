# 从一组具体数据读懂 Transformer：PyTorch 编码器、注意力机制与 GPT Token 全流程

参考：ChatGPT

> 从三帧轨迹的逐层计算出发，把 `nn.TransformerEncoderLayer`、`nn.TransformerEncoder`、Q/K/V、多头注意力、残差、归一化、前馈网络、位置编码、掩码，以及 GPT 的分词、嵌入和下一 token 预测，连接成一条可以跟踪、计算和复现的数据流。

学习 Transformer 时，一个常见困难是：公式似乎都认识，模块名字也能背下来，但脑海里仍然没有“数据到底怎样经过这个网络”的画面。

比如，知道 Q 是 Query、K 是 Key、V 是 Value，并不等于知道第三帧究竟拿了哪些数字、与哪几帧比较、得到多少权重、又把什么数据混合到自己的表示中。

因此，本文先固定一份很小的数据，把数字逐步算出来；等看到每一步发生了什么，再给这些数据和操作贴上概念名称。最后把同一套理解迁移到 GPT：坐标不再是输入，文字先被分成 token，再变成向量；坐标回归不再是输出，模型改为给词表中的候选 token 打分。

## 目录

- [阅读约定与示例环境](#reading)
- [1. 先看整体：一个层与多个层](#sec-01)
- [2. 从三帧坐标到编码器输入](#sec-02)
- [3. 多头自注意力：第三帧怎样读取其他帧](#sec-03)
- [4. 残差、LayerNorm 与前馈网络](#sec-04)
- [5. 第二层、特征读出与预测输出](#sec-05)
- [6. 因果掩码、Padding 掩码与位置编码](#sec-06)
- [7. 把数据流对应到 Transformer 概念地图](#sec-07)
- [8. PyTorch 两个编码器模块的接口详解](#sec-08)
- [9. 完整训练案例：历史三维位置预测下一帧](#sec-09)
- [10. 与 GRU、LSTM、Decoder 的区别](#sec-10)
- [11. GPT 的文本怎样变成 token：Tokenizer 与 BPE](#sec-11)
- [12. 从 token ID 到下一个 token 的预测](#sec-12)
- [13. 真实分词、token 计数与 API 用量](#sec-13)
- [14. 易混淆点与完整知识回路](#sec-14)
- [附录 A：固定数字逐层计算的完整脚本](#appendix-a)
- [附录 B：两层网络全部中间数值](#appendix-b)
- [附录 C：接口测试与轨迹训练的完整脚本](#appendix-c)
- [附录 D：训练示例完整运行记录](#appendix-d)
- [附录 E：真实文本 token 检查脚本](#appendix-e)
- [附录 F：原始问题与内容对应关系](#appendix-f)
- [参考资料](#references)

<a id="reading"></a>
## 阅读约定与示例环境

本文有三个彼此独立、不要混用权重和数值的例子。

| 示例 | 输入 | 目的 | 权重状态 |
|---|---|---|---|
| 固定数字例子 | 三帧二维轨迹 `[[1,0],[0,1],[1,1]]` | 把每一步算出来 | 手工设定，未训练 |
| 可训练轨迹例子 | 随机生成的三维匀加速轨迹 | 展示完整训练流程 | 随训练更新 |
| GPT 教学例子 | `我喜欢机器人` | 展示分词、查表与生成 | 小词表、向量与 logits 均为教学假设 |

**几个边界提前说明：** 手工权重的输出只证明计算过程，不代表它学会了运动预测；三维训练数据是无量纲玩具数据，不是真实网球模型；GPT 章节讲公开实现能够说明的基本机制，不推断未公开模型的具体层数、宽度或所有架构细节。

---

<a id="sec-01"></a>
## 1. 先看整体：一个层与多个层

### 1.1 两个名字最直接的关系

`nn.TransformerEncoderLayer` 是一个完整的 Transformer 编码器层；`nn.TransformerEncoder` 把这样的层堆叠起来，形成多层编码器。它们默认输出一整段序列的特征，不直接替你完成分类、坐标预测或文本生成。

严格来说，它们是继承自 `nn.Module` 的**类**，而不是无状态的数学函数：

```python
layer = nn.TransformerEncoderLayer(...)  # 构建带参数的模块
output = layer(input)                    # 使用参数进行一次前向计算
```

一个最小可运行例子：

```python
import torch
from torch import nn

# 2 条序列，每条有 10 个时间步，每个时间步 64 维。
x = torch.randn(2, 10, 64)

layer = nn.TransformerEncoderLayer(
    d_model=64,
    nhead=4,
    dim_feedforward=128,
    dropout=0.0,
    batch_first=True,
)

encoder = nn.TransformerEncoder(
    encoder_layer=layer,
    num_layers=3,
    enable_nested_tensor=False,
)

y1 = layer(x)
y2 = encoder(x)

print(y1.shape)  # torch.Size([2, 10, 64])
print(y2.shape)  # torch.Size([2, 10, 64])
```

两条数据路径分别是：

```text
layer(x)：
[B,T,D] → 一层 EncoderLayer → [B,T,D]

encoder(x)：
[B,T,D] → 第1层 → 第2层 → 第3层 → [B,T,D]
```

**一层就处理整段序列，不是只处理一个时间步。** 它不是 `GRUCell` 那样的单步单元。三个层也不是分别处理第一、第二、第三帧，而是每层都处理当前输入的全部位置。

`num_layers=3` 不会把序列长度变成 3，也不会默认返回三层所有结果；编码器通常返回最后一层的输出。传进去的 `layer` 是创建各层的模板，所以三层编码器总共是三层，不是“模板一层再加三层”。[^pytorch-encoder]

### 1.2 统一形状：一行对应一个时间位置

本文统一使用 `batch_first=True`：

```text
[B, T, D]
 │  │  └─ 每个位置的特征维度
 │  └──── 序列长度，即时间位置/Token 数量
 └─────── 一个 batch 内的序列数量
```

省略 batch 维后，一个 `T×D` 矩阵的一行对应一个位置，一列对应一个特征通道。

**第三行始终是第三个位置的表示，但经过注意力以后，这一行的内容会融合其他行的信息。** 行的索引身份不变，不代表它只包含原来那一帧的测量值。

### 1.3 本文固定数值网络

接下来固定为：

```python
layer = nn.TransformerEncoderLayer(
    d_model=4,
    nhead=2,
    dim_feedforward=6,
    dropout=0.0,
    activation="relu",
    batch_first=True,
    norm_first=False,
)

encoder = nn.TransformerEncoder(
    layer,
    num_layers=2,
    norm=None,
    enable_nested_tensor=False,
)
```

完整通路为：

```text
三帧二维位置                         [1,3,2]
    ↓ 输入投影 2 → 4
每帧四维特征                         [1,3,4]
    ↓ 加位置编码
编码器输入                           [1,3,4]
    ↓ 第一个完整 EncoderLayer
第一层输出                           [1,3,4]
    ↓ 第二个完整 EncoderLayer
第二层输出                           [1,3,4]
    ↓ 取最后一个位置的特征
读出特征                             [1,4]
    ↓ 预测头 4 → 2
下一帧坐标预测                       [1,2]
```

输入投影、位置编码、最后的读出操作和预测头都是我们额外组合进去的，不是这两个编码器模块自动添加的。[^pytorch-source]

---

<a id="sec-02"></a>
## 2. 从三帧坐标到编码器输入

### 2.1 固定原始数据

```text
第1帧的位置：(1,0)
第2帧的位置：(0,1)
第3帧的位置：(1,1)
```

对应张量：

```python
positions = torch.tensor([
    [[1., 0.],
     [0., 1.],
     [1., 1.]]
])
# shape = [1,3,2]
```

后续省略 batch 维，写成：

$$
P_{\text{raw}}=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}.
$$

在这个任务里，我们选择把“一帧的两个坐标”作为一个整体，也就是一个序列单元、一个 Token。于是总共有三个 Token，不是六个。

### 2.2 输入投影：两个坐标变成四个特征

先用：

```python
input_proj = nn.Linear(2, 4, bias=False)
```

把权重设置成：

$$
W_{\text{in}}=
\begin{bmatrix}
1&0\\
0&1\\
1&0\\
0&1
\end{bmatrix}.
$$

PyTorch 的 `Linear` 存储权重形状为 `[out_features,in_features]`，因此对行向量的计算是 $y=xW^\top+b$。本例关闭偏置。

这个特殊权重只做一件事：

$$
[x,y]\longrightarrow[x,y,x,y].
$$

实际数据流：

```text
第1帧：[1,0] → [1,0,1,0]
第2帧：[0,1] → [0,1,0,1]
第3帧：[1,1] → [1,1,1,1]
```

所以：

$$
E=
\begin{bmatrix}
1&0&1&0\\
0&1&0&1\\
1&1&1&1
\end{bmatrix}.
$$

**为什么做这一步？** 因为后面的编码器使用 `d_model=4`，每个位置需要用四维特征表示。二维扩展成四维并不凭空增加测量信息，只是在模型使用的特征空间中重新表示输入。

一般的可学习权重可能产生：

```text
[x,y]
  ↓
[0.3x+0.7y, -0.2x+0.5y, 0.8x-0.1y, 0.4x+0.6y]
```

这些通道不必分别对应某个已知物理量。现在第三帧的 `[1,1,1,1]` 仍然只由第三帧的坐标生成，**帧与帧之间还没有交换信息**。

### 2.3 位置编码：为内容提供位置线索

给三个时间位置设置三个固定向量：

$$
P_{\text{pos}}=
\begin{bmatrix}
0&0&0&0\\
0&0&1&0\\
0&0&0&1
\end{bmatrix}.
$$

这里的位置向量是专门为演示选择的，并不是 PyTorch 自动生成的某种标准编码。

逐行相加：

```text
第1帧：[1,0,1,0] + [0,0,0,0] = [1,0,1,0]
第2帧：[0,1,0,1] + [0,0,1,0] = [0,1,1,1]
第3帧：[1,1,1,1] + [0,0,0,1] = [1,1,1,2]
```

得到真正送入编码器的矩阵：

$$
\boxed{X=E+P_{\text{pos}}=
\begin{bmatrix}
1&0&1&0\\
0&1&1&1\\
1&1&1&2
\end{bmatrix}.}
$$

同样的测量 `[1,1]`，放在不同位置时可以得到不同的模型输入：

```text
在第1个位置：
[1,1,1,1] + [0,0,0,0] = [1,1,1,1]

在第3个位置：
[1,1,1,1] + [0,0,0,1] = [1,1,1,2]
```

输入嵌入表达“测到了什么”，位置编码提供“位于序列哪个位置”的线索。加法位置编码之外还存在其他位置机制，不能把这里的固定加法向量当成所有现代语言模型的唯一做法。

到这一步，仍然是逐行处理，还没有把其他帧的测量内容混入当前行。

---

<a id="sec-03"></a>
## 3. 多头自注意力：第三帧怎样读取其他帧

### 3.1 Q、K、V 是同一输入的三种投影

对每行四维特征 `[a,b,c,d]`，本例手工设置三种投影：

```text
Q投影：[a,b,c,d] → [a,   b,   c,   d  ]
K投影：[a,b,c,d] → [a,   b,   d,   c  ]
V投影：[a,b,c,d] → [a+b, a-b, c+d, c-d]
```

采用 PyTorch 的行向量约定，可以写成：

$$
Q=XW_Q^\top,\qquad K=XW_K^\top,\qquad V=XW_V^\top.
$$

这里省略偏置，因为本例把相关偏置全部设为零。有些论文写 $Q=XW_Q$，只是对权重矩阵的形状采用了另一种约定，不是计算原理不同。

把输入矩阵代入：

```text
Q =
[
    [1,0,1,0],
    [0,1,1,1],
    [1,1,1,2]
]

K =
[
    [1,0,0,1],
    [0,1,1,1],
    [1,1,2,1]
]

V =
[
    [1, 1,1, 1],
    [1,-1,2, 0],
    [2, 0,3,-1]
]
```

特别看第三行：

```text
输入 X3 = [1,1,1,2]
Q3      = [1,1,1,2]
K3      = [1,1,2,1]
V3      = [2,0,3,-1]
```

为什么需要三份？因为后面分成两件不同的工作：

> 用当前位置的 Q 与各位置的 K 比较，决定权重如何分配；再按照这些权重，把各位置的 V 汇总成新的特征。

所以 Q/K 用来计算“怎样分配”，V 是“实际被搬运和组合的内容”。Q、K、V 都由同一段序列产生，因此是**自注意力**；来源相同不代表三个投影结果相同。

实际模块的调用近似为：

```python
attention_output, attention_weights = layer.self_attn(
    query=x,
    key=x,
    value=x,
)
```

这里传入同一个 `x`，不同投影在 `MultiheadAttention` 内部完成。注意：$W_Q,W_K,W_V$ 是参数，$Q,K,V$ 是本次前向计算产生的中间张量，不能混为一谈。

### 3.2 先投影，再分成两个头

本例：

$$
d_{\text{model}}=4,\qquad H=2,\qquad d_{\text{head}}=4/2=2.
$$

输入与投影形状是 `[B,T,D]`，分头之后是 `[B,H,T,d_head]`：

```text
[1,3,4] → [1,2,3,2]
```

第一个头使用投影结果的前两个数，第二个头使用后两个数。普通的可学习投影可以混合全部输入通道，因此**分头不是让每个头永远只能看原始输入的一半**。

### 3.3 第一个头拿到的数据

```text
Q_head1 = K_head1 =
[
    [1,0],
    [0,1],
    [1,1]
]

V_head1 =
[
    [1, 1],
    [1,-1],
    [2, 0]
]
```

继续只跟踪第三帧：

```text
q3 = [1,1]
```

与三个 Key 做点积：

```text
[1,1] · [1,0] = 1  ← 对第1帧的打分
[1,1] · [0,1] = 1  ← 对第2帧的打分
[1,1] · [1,1] = 2  ← 对第3帧的打分
```

得到 `[1,1,2]`。这三个数是“对三个位置的匹配打分”，不是第三帧新生成的三个特征。

### 3.4 缩放点积：为什么除以平方根

缩放点积注意力先将分数除以 $\sqrt{d_{\text{head}}}$：[^pytorch-sdpa]

```text
[1,1,2] / √2
    ↓
[0.7071,0.7071,1.4142]
```

缩放用于控制点积的数值量级。直观地说，维度越多，点积累加项越多；如果分数差异过大，Softmax 容易变得非常尖锐。这里除的是每个头的维度的平方根，不是序列长度的平方根。

单头的完整公式为：

$$
S=\frac{QK^\top}{\sqrt{d_{\text{head}}}},\qquad
A=\operatorname{softmax}(S+M),\qquad
O=AV.
$$

$M$ 是可选加性掩码，本节暂不加掩码。$S$ 的行索引表示“谁在查询”，列索引表示“查询哪个位置”。

### 3.5 Softmax：打分变成聚合权重

对第三行的三个打分计算：

$$
\operatorname{softmax}([s_1,s_2,s_3])
=\frac{[e^{s_1},e^{s_2},e^{s_3}]}{e^{s_1}+e^{s_2}+e^{s_3}}.
$$

具体数字：

```text
exp([0.7071,0.7071,1.4142])
    ↓
[2.0281,2.0281,4.1133]
    ↓ 除以三者之和
[0.2483,0.2483,0.5035]
```

这表示：

```text
从第1帧的 V 取约24.83%；
从第2帧的 V 取约24.83%；
从第3帧的 V 取约50.35%。
```

未舍入且没有注意力 Dropout 时，这一行权重之和为 1。这些是**内容聚合权重**，不能解释成“三帧测量各自正确的概率”。

对一般情况，第 $t$ 个位置的输出为：

$$
\mathbf o_t=\sum_{j=1}^{T}a_{tj}\mathbf v_j.
$$

例如，假设权重另为 `[0.1,0.3,0.6]`，那就是 $0.1\mathbf v_1+0.3\mathbf v_2+0.6\mathbf v_3$。这只是解释公式的假设，不能与本例实际算出的 `[0.2483,0.2483,0.5035]` 混用。

### 3.6 对 V 加权求和：信息真正跨帧流动的地方

第一个头的 Value：

```text
v1 = [1, 1]
v2 = [1,-1]
v3 = [2, 0]
```

第三帧的新特征：

$$
\begin{aligned}
\mathbf o^{(1)}_3
&=0.2483[1,1]+0.2483[1,-1]+0.5035[2,0]\\
&\approx[1.5035,0].
\end{aligned}
$$

拆开两个分量：

```text
第1个分量：0.2483×1 + 0.2483×1 + 0.5035×2 ≈ 1.5035
第2个分量：0.2483×1 + 0.2483×(-1) + 0.5035×0 = 0
```

**其他帧的数据正是在这里真正流入第三帧。** 前面的输入投影、位置相加、Q/K/V 投影都是逐行操作；`A @ V` 则在同一输出行中明确用到了不同输入行的 Value。

三帧一起计算时：

```text
A_head1 =
[
    [0.4011,0.1978,0.4011],
    [0.1978,0.4011,0.4011],
    [0.2483,0.2483,0.5035]
]

O_head1 = A_head1 @ V_head1 =
[
    [1.4011, 0.2033],
    [1.4011,-0.2033],
    [1.5035, 0.0000]
]
```

### 3.7 第二个头：换一套投影特征，重新分配关注权重

第二个头使用：

```text
Q_head2 =
[
    [1,0],
    [1,1],
    [1,2]
]

K_head2 =
[
    [0,1],
    [1,1],
    [2,1]
]

V_head2 =
[
    [1, 1],
    [2, 0],
    [3,-1]
]
```

第三帧的 Query 为 `[1,2]`：

```text
[1,2] · [0,1] = 2
[1,2] · [1,1] = 3
[1,2] · [2,1] = 4
```

于是：

```text
[2,3,4] / √2
    ↓
[1.4142,2.1213,2.8284]
    ↓ Softmax
[0.1400,0.2840,0.5760]
```

加权求和：

$$
\mathbf o^{(2)}_3
=0.1400[1,1]+0.2840[2,0]+0.5760[3,-1]
\approx[2.4359,-0.4359].
$$

现在两个头的差异非常具体：

| 第三帧所在的分支 | 聚合权重 | 输出 |
|---|---|---|
| 头 1 | `[0.2483,0.2483,0.5035]` | `[1.5035,0]` |
| 头 2 | `[0.1400,0.2840,0.5760]` | `[2.4359,-0.4359]` |

多头允许不同投影产生不同的匹配关系和内容表示。它并没有提前规定“某个头计算速度，另一个头计算加速度”；这些是否形成有用分工，需要由训练目标和数据决定。

本例第二个头对三帧恰好给出相同输出：

```text
O_head2 =
[
    [2.4359,-0.4359],
    [2.4359,-0.4359],
    [2.4359,-0.4359]
]
```

这是手工数据与权重造成的结果，不是多头注意力的要求。这里不同查询行的打分相差一个整体常数，而 Softmax 对整行加常数不变，因而出现相同权重。

### 3.8 多个头拼接，再进行输出投影

第三帧两个头的结果先**拼接**，不是平均：

```text
[1.5035,0] 与 [2.4359,-0.4359]
    ↓
[1.5035,0,2.4359,-0.4359]
```

随后进行注意力模块内部的输出投影。标准多头注意力就是各头结果拼接后，再经过一个线性变换。[^pytorch-mha]

本例设置为：

```text
[z1,z2,z3,z4] → [z1+z3,z2,z3,z4]
```

所以第三帧：

```text
[1.5035,0,2.4359,-0.4359]
    ↓
[3.9394,0,2.4359,-0.4359]
```

第一个分量中的：

```text
3.9394 = 1.5035 + 2.4359
           ↑        ↑
          头1      头2
```

直接展示了输出投影如何组合不同头的信息。

完整注意力模块输出记为 $M_{\text{attn}}$，避免与前面的掩码 $M$ 混淆：

```text
M_attn =
[
    [3.8371, 0.2033,2.4359,-0.4359],
    [3.8371,-0.2033,2.4359,-0.4359],
    [3.9394, 0.0000,2.4359,-0.4359]
]
```

到此才完成一个多头自注意力子模块，还没有完成整个 `TransformerEncoderLayer`。

---
<a id="sec-04"></a>
## 4. 残差、LayerNorm 与前馈网络

### 4.1 第一次残差连接：把输入直接加回来

本例采用 `norm_first=False`，即 Post-LN：注意力输出先与子层输入相加，再做 LayerNorm。[^pytorch-source]

第三帧输入：

```text
X3 = [1,1,1,2]
```

注意力输出：

```text
M_attn3 = [3.9394,0,2.4359,-0.4359]
```

逐元素相加：

```text
    [1.0000,1.0000,1.0000, 2.0000]
  + [3.9394,0.0000,2.4359,-0.4359]
  --------------------------------
    [4.9394,1.0000,3.4359, 1.5641]
```

这是 **Residual Connection / Skip Connection：残差连接、跳跃连接**，形式为：

$$
y=x+f(x).
$$

如果只保留注意力结果，原来的表示会被聚合结果替换；残差提供了一条直接参与后续计算的通路，让分支输出可以作为对原表示的更新。

这里的“残差”是网络连接方式的名称，不是预测坐标与真实坐标的误差，也不是 SLAM 中的重投影残差。它也不是把原始输入保存到某个额外隐状态里；这一步就是明确的逐元素加法。

### 4.2 LayerNorm：归一化这一行的四个特征

第三行现在是：

```text
[4.9394,1.0000,3.4359,1.5641]
```

先计算均值：

$$
\mu=\frac{1}{4}\sum_{i=1}^{4}x_i\approx2.7349.
$$

再计算方差：

$$
\sigma^2=\frac{1}{4}\sum_{i=1}^{4}(x_i-\mu)^2\approx2.4331.
$$

LayerNorm 的输出是：

$$
y_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
$$

本例设置：

```text
γ = [1,1,1,1]
β = [0,0,0,0]
ε = 0.00001
```

PyTorch 的 `LayerNorm(4)` 对最后四个特征计算统计量，方差对应 `unbiased=False`，不是除以 `4-1` 的样本方差。[^pytorch-ln]

得到：

```text
归一化之前：[4.9394, 1.0000,3.4359, 1.5641]
归一化之后：[1.4133,-1.1122,0.4495,-0.7506]
```

第一个分量可单独核对：

$$
\frac{4.9394-2.7349}{\sqrt{2.4331+0.00001}}\approx1.4133.
$$

完整结果：

```text
U =
[
    [1.2879,-0.8231,0.6496,-1.1144],
    [1.1288,-0.9157,0.8590,-1.0721],
    [1.4133,-1.1122,0.4495,-0.7506]
]
```

**为什么这么做？** 它调整当前隐藏向量的数值尺度，再交给后面的计算。这里归一化的是内部特征，不是物理坐标；也不是把数值压到 `[0,1]`。对于 `[B,T,D]`，`LayerNorm(D)` 不会把不同时间步或 batch 内的不同样本混起来计算均值。

### 4.3 前馈网络第一层：4 维变成 6 维

一个编码器层还包含逐位置前馈网络：

```text
Linear(4→6) → ReLU → Linear(6→4)
```

对应模块内部的：

```python
layer.linear1
layer.activation
layer.linear2
```

继续跟踪第三帧：

```text
u = [1.4133,-1.1122,0.4495,-0.7506]
```

把第一层 Linear 设置为：

```text
[u1,u2,u3,u4]
    ↓
[u1,u2,u3,u4,u1+u2,u3-u4]
```

于是：

```text
[1.4133,-1.1122,0.4495,-0.7506]
    ↓
[1.4133,-1.1122,0.4495,-0.7506,0.3011,1.2001]
```

最后两项来自：

```text
0.3011 = 1.4133 + (-1.1122)
1.2001 = 0.4495 - (-0.7506)
```

这一步并没有查看第一、第二帧，只是在加工第三帧当前已有的四个特征。它之所以间接包含其他帧的信息，是因为输入已经经过前面的注意力融合。

### 4.4 ReLU：为什么中间需要非线性

ReLU 的计算为：

$$
\operatorname{ReLU}(x)=\max(0,x).
$$

数据变化：

```text
之前：
[1.4133,-1.1122,0.4495,-0.7506,0.3011,1.2001]

之后：
[1.4133, 0.0000,0.4495, 0.0000,0.3011,1.2001]
```

为什么不直接串联两个 Linear？因为不加非线性时：

$$
W_2(W_1x)=(W_2W_1)x,
$$

本质上仍可合并成一个线性变换；考虑偏置时也仍然是一个仿射变换。加入 ReLU 后，哪些通道被保留取决于输入数值，不再能用一个固定线性变换概括。

负数变为零不是在说“负坐标不合理”。这里的负数是隐藏特征，不是原始空间坐标。

### 4.5 前馈网络第二层：6 维重新组合回 4 维

设 ReLU 输出为 `[r1,r2,r3,r4,r5,r6]`，第二个 Linear 执行：

```text
[r1,r2,r3,r4,r5,r6]
    ↓
[r1+r5,r2,r3+r6,r4]
```

代入数字：

```text
[1.4133,0,0.4495,0,0.3011,1.2001]
    ↓
[1.7145,0,1.6495,0]
```

回到四维，是因为层与层之间的表示宽度仍然是 `d_model=4`，接下来也需要与四维输入做残差相加。

忽略 Dropout，前馈网络的一般形式为：

$$
\operatorname{FFN}(u)=W_2\,\sigma(W_1u+b_1)+b_2,
$$

这里公式按列向量写；代码按 PyTorch 的行向量存储约定实现。该子模块对每个位置使用同一组参数，称为 **Position-wise Feed-Forward Network，逐位置前馈网络**，也常称 FFN 或 MLP。[^transformer-paper]

核心分工可以记成：

> 注意力负责不同位置之间的信息交换；FFN 负责每个位置内部的非线性特征加工。

### 4.6 第二次残差与归一化

前馈网络输入与输出相加：

```text
    [1.4133,-1.1122,0.4495,-0.7506]
  + [1.7145, 0.0000,1.6495, 0.0000]
  --------------------------------
    [3.1278,-1.1122,2.0990,-0.7506]
```

再经过第二个 LayerNorm：

```text
[1.2607,-1.0768,0.6935,-0.8775]
```

这才是第三帧经过一个完整 `TransformerEncoderLayer` 后的输出。

三帧的第一层结果为：

$$
\boxed{
Y^{(1)}\approx
\begin{bmatrix}
0.9931&-0.9263&1.0043&-1.0710\\
0.6945&-0.9414&1.2639&-1.0169\\
1.2607&-1.0768&0.6935&-0.8775
\end{bmatrix}.}
$$

### 4.7 一个完整层的公式与 Post-LN / Pre-LN

忽略 Dropout，刚才的 Post-LN 是：

$$
U=\operatorname{LN}_1\left(X+\operatorname{MHA}(X)\right),
$$

$$
Y=\operatorname{LN}_2\left(U+\operatorname{FFN}(U)\right).
$$

对应 `norm_first=False`。若设 `norm_first=True`，顺序改成 Pre-LN：

$$
U=X+\operatorname{MHA}\left(\operatorname{LN}_1(X)\right),
$$

$$
Y=U+\operatorname{FFN}\left(\operatorname{LN}_2(U)\right).
$$

两者改变的是归一化的先后位置，不改变输入输出的基本形状。后面的轨迹训练示例采用 Pre-LN，并在整个编码器堆叠末尾额外加一次 LayerNorm；固定数字示例采用 Post-LN，不能把两套数值交叉使用。[^pytorch-source]

经典结构图中的 **Add & Norm** 就是残差相加与归一化的合称，不是第三种新操作。

### 4.8 Dropout 在哪里、为什么本例没有看到它

Post-LN 层的完整示意为：

```text
输入 X
    ├───────────────────────────────┐
    ↓                               │
多头自注意力                        │
    ↓                               │
Dropout                             │
    ↓                               │
残差相加 ←──────────────────────────┘
    ↓
LayerNorm
    ├───────────────────────────────┐
    ↓                               │
Linear(D→F)                         │
    ↓                               │
激活函数                            │
    ↓                               │
Dropout                             │
    ↓                               │
Linear(F→D)                         │
    ↓                               │
Dropout                             │
    ↓                               │
残差相加 ←──────────────────────────┘
    ↓
LayerNorm
```

多头注意力内部还可对注意力权重应用 Dropout。本文固定数值和玩具训练均设 `dropout=0.0`，方便观察和验证；这不是实际任务必须采用的配置。[^pytorch-source]

例如，单独一个 `nn.Dropout(p=0.5)` 在训练时，对：

```text
[1.7145,0,1.6495,0]
```

一次可能输出：

```text
[3.4290,0,0,0]
```

随机置零一些元素，保留元素乘以 $1/(1-p)=2$。这是一个可能掩码的示意，不是固定数字主线中的实际结果。

Dropout 用于正则化；`nn.Dropout` 在 `.eval()` 后不进行这种随机丢弃。还要注意：直接调用底层 `F.scaled_dot_product_attention` 时，应根据评估状态自行把 `dropout_p` 设为零，不能把 `nn.Dropout` 的行为机械套到所有函数接口。[^pytorch-dropout][^pytorch-sdpa]

---

<a id="sec-05"></a>
## 5. 第二层、特征读出与预测输出

### 5.1 TransformerEncoder 把整层串起来

创建：

```python
encoder = nn.TransformerEncoder(layer, num_layers=2)
```

后，计算逻辑近似为：

```python
h1 = encoder.layers[0](x)
h2 = encoder.layers[1](h1)
```

如果配置了最终归一化，还会在所有层之后执行 `encoder.norm(h2)`。`encoder(x)` 默认返回最终输出，不自动返回 `[h1,h2]`。[^pytorch-encoder]

### 5.2 第二层的 Q/K/V 从新的输入重新产生

第二层输入就是第一层输出：

```text
[
    [0.9931,-0.9263,1.0043,-1.0710],
    [0.6945,-0.9414,1.2639,-1.0169],
    [1.2607,-1.0768,0.6935,-0.8775]
]
```

第二层不会直接复用第一层的注意力权重，而是从这些新的隐藏表示重新产生 Q、K、V、打分和权重。

例如，第三帧在第一个头里的聚合权重：

```text
第1层：[0.2483,0.2483,0.5035]
第2层：[0.3126,0.2423,0.4451]
```

第一层比较的是输入投影与位置编码后的特征；第二层比较的是已经完成一轮信息融合与非线性加工的特征。层数增加是继续处理更新后的表示，不是把同一结果复制几份。

### 5.3 第二层第三行的实际数值链

本例保留两个层创建时相同的初始参数数值，方便复算。第二层第三行的中间数据为：

```text
输入：
[1.260749,-1.076826,0.693542,-0.877465]

Q：
[1.260749,-1.076826,0.693542,-0.877465]

K：
[1.260749,-1.076826,-0.877465,0.693542]

V：
[0.183923,2.337575,-0.183923,1.571007]

头1的注意力权重：
[0.312624,0.242317,0.445059]

头2的注意力权重：
[0.311768,0.272524,0.415708]

两个头的输出：
[0.042892,2.036811] 与 [-0.029978,1.921665]

拼接：
[0.042892,2.036811,-0.029978,1.921665]

注意力输出投影：
[0.012914,2.036811,-0.029978,1.921665]

第一次残差：
[1.273663,0.959986,0.663564,1.044200]

第一次LayerNorm：
[1.320020,-0.116145,-1.473305,0.269430]

FFN第一层：
[1.320020,-0.116145,-1.473305,0.269430,1.203875,-1.742735]

ReLU：
[1.320020,0,0,0.269430,1.203875,0]

FFN第二层：
[2.523894,0,0,0.269430]

第二次残差：
[3.843914,-0.116145,-1.473305,0.538861]

第二次LayerNorm，即第二层输出：
[1.608392,-0.416456,-1.110396,-0.081540]
```

第二层所有行、所有头的矩阵都保留在附录 B。

三帧完整输出：

$$
\boxed{
Y^{(2)}\approx
\begin{bmatrix}
0.5132&0.9887&0.1521&-1.6541\\
-1.0628&0.0624&1.5879&-0.5875\\
1.6084&-0.4165&-1.1104&-0.0815
\end{bmatrix}.}
$$

### 5.4 初始数值相同，不等于共享参数

`TransformerEncoder` 创建的是各自独立的层参数，但默认是从模板复制，因此刚创建时对应参数的数值相同。官方也提示可以在创建后分别重新初始化。[^pytorch-encoder]

```python
w0 = encoder.layers[0].linear1.weight
w1 = encoder.layers[1].linear1.weight

print(torch.equal(w0, w1))            # True：初始数值相同
print(w0.data_ptr() == w1.data_ptr())  # False：不共享存储
```

“共享参数”是多层引用同一块参数；“初始数值相同”是各层各有参数，只是起点相同。训练时不能把这两者混为一谈。

固定数字例子特意保留相同初值；后面的训练例子会分别重新初始化每层的矩阵权重。

### 5.5 Readout：取最后一个位置的特征

编码器输出仍然是 `[1,3,4]`。我们要根据三帧历史预测第四帧，因此取最后一行：

```python
last_feature = encoder_output[:, -1, :]
# [1,4]
```

得到：

```text
[1.6084,-0.4165,-1.1104,-0.0815]
```

这一步称为 **Readout，特征读出**：决定使用哪些隐藏表示完成任务。本例使用最后一个位置；其他任务也可能使用有效位置的平均池化等方式。

### 5.6 Prediction Head：隐藏特征变成坐标

增加：

```python
head = nn.Linear(4, 2, bias=False)
```

设置它执行：

```text
[z1,z2,z3,z4] → [z1+0.5*z3, z2+0.5*z4]
```

代入：

```text
预测x = 1.6084 + 0.5×(-1.1104) ≈ 1.0532
预测y = -0.4165 + 0.5×(-0.0815) ≈ -0.4572
```

得到：

$$
\boxed{\hat{\mathbf p}_4=[1.0532,-0.4572].}
$$

这四个隐藏通道并不分别是位置、速度等物理量；预测头才将它们组合成任务需要的两个坐标。这里也可以称 **Regression Head，回归头**。

注意力模块内部的输出投影用于组合不同注意力头；这里的预测头用于生成最终任务结果。两者可能都使用 Linear，但不是同一个模块。

### 5.7 Loss：计算过程正确，不代表预测正确

假设数据集中第四帧标签是：

```text
target = [2,1]
```

这只是一个示意标签，并不是前三帧在没有额外运动假设时唯一能推导出的未来位置。

均方误差为：

$$
L=\frac{(1.0532-2)^2+(-0.4572-1)^2}{2}\approx1.5100.
$$

未舍入计算的结果为 `1.5099745861859648`。

训练将通过损失调整输入投影、Q/K/V 投影、注意力输出投影、FFN、LayerNorm 参数和预测头等参数。注意力权重 `[0.2483,0.2483,0.5035]` 本身不是网络中固定存储的一行参数，而是当前输入和当前参数共同生成的中间结果。

### 5.8 第三帧从头到尾的紧凑回放

```text
[1,1]                                       原始坐标
    ↓
[1,1,1,1]                                   输入投影
    ↓
[1,1,1,2]                                   加位置编码
    ↓
头1：[1.5035,0]；头2：[2.4359,-0.4359]       注意力聚合
    ↓
[1.5035,0,2.4359,-0.4359]                    拼接
    ↓
[3.9394,0,2.4359,-0.4359]                    输出投影
    ↓
[4.9394,1,3.4359,1.5641]                     第一次残差
    ↓
[1.4133,-1.1122,0.4495,-0.7506]              LayerNorm
    ↓
[1.4133,-1.1122,0.4495,-0.7506,0.3011,1.2001] Linear 4→6
    ↓
[1.4133,0,0.4495,0,0.3011,1.2001]            ReLU
    ↓
[1.7145,0,1.6495,0]                          Linear 6→4
    ↓
[3.1278,-1.1122,2.0990,-0.7506]              第二次残差
    ↓
[1.2607,-1.0768,0.6935,-0.8775]              第一层结束
    ↓ 与其他两行一起进入第二层
[1.6084,-0.4165,-1.1104,-0.0815]             第二层结束
    ↓
[1.0532,-0.4572]                             最终回归输出
```

---

<a id="sec-06"></a>
## 6. 因果掩码、Padding 掩码与位置编码

### 6.1 默认不加掩码：一个位置可以读取窗口内所有位置

在前面的例子里，目标是“已知三帧，预测第四帧”，因此允许这三帧历史互相交换信息。

但另一个任务可能要求：

```text
第1个输出只能使用第1帧；
第2个输出只能使用第1、2帧；
第3个输出只能使用第1、2、3帧。
```

这时需要限制每个位置的信息来源。

### 6.2 因果掩码直接改变 Softmax 之前的打分

第一层第一个头原来的打分矩阵：

```text
[
    [0.7071,0.0000,0.7071],
    [0.0000,0.7071,0.7071],
    [0.7071,0.7071,1.4142]
]
```

上三角、即未来位置，被置成负无穷：

```text
[
    [0.7071,  -∞,    -∞  ],
    [0.0000,0.7071,  -∞  ],
    [0.7071,0.7071,1.4142]
]
```

因为 $e^{-\infty}=0$，Softmax 后为：

```text
[
    [1.0000,0.0000,0.0000],
    [0.3302,0.6698,0.0000],
    [0.2483,0.2483,0.5035]
]
```

看第二行：

```text
不加掩码：[0.1978,0.4011,0.4011]
加因果掩码：[0.3302,0.6698,0]
```

信息汇总从：

$$
0.1978v_1+0.4011v_2+0.4011v_3
$$

改为：

$$
0.3302v_1+0.6698v_2.
$$

代入本例：

$$
0.3302[1,1]+0.6698[1,-1]=[1.0000,-0.3395].
$$

所以掩码规定的是“信息从哪些位置流来”，不是事后随意把某个最终输出置零。[^pytorch-sdpa]

### 6.3 对应代码与布尔含义

```python
mask = torch.tensor([
    [False, True,  True ],
    [False, False, True ],
    [False, False, False],
])

# 单层：
y1 = layer(x, src_mask=mask)

# 多层编码器：
y2 = encoder(x, mask=mask)
```

在这两个模块以及 `nn.MultiheadAttention` 的布尔掩码约定中：

```text
False：允许关注。
True：禁止关注。
```

但 `F.scaled_dot_product_attention` 的布尔 `attn_mask` 约定相反，`True` 表示允许参与注意力；从一个 API 迁移到另一个 API 时必须检查。[^pytorch-mha][^pytorch-sdpa]

长度为 5 时的因果掩码为：

```text
             被关注位置 j
             1  2  3  4  5
查询位置 i
    1        0  1  1  1  1
    2        0  0  1  1  1
    3        0  0  0  1  1
    4        0  0  0  0  1
    5        0  0  0  0  0

1 表示 True，即屏蔽。
```

通用生成方式：

```python
T = x.size(1)
causal_mask = torch.ones(
    T, T, dtype=torch.bool, device=x.device
).triu(diagonal=1)
```

注意力矩阵的行是 Query 位置，列是 Key 位置；上三角屏蔽的是 `j > i`，不是下三角。

### 6.4 第一层最后位置不变，不代表第二层也不变

在这个三帧例子里，最后一行无论是否使用因果掩码，都可以看前三帧，所以第一层最后一行输出相同。

但第二层读取的是第一层的输出。第一层的前两行已经因为加掩码而改变，因此第二层最后一行也会改变：

```text
不加因果掩码，第二层最后一行：
[1.608392,-0.416456,-1.110396,-0.081540]

加因果掩码，第二层最后一行：
[1.559217,0.172943,-1.005253,-0.726907]
```

这是本例数值计算直接得到的结果，也说明多层的信息依赖关系不能只检查当前层的最后一行掩码。

### 6.5 哪些任务真的需要因果掩码

假设任务只要求：

$$
[x_1,x_2,\ldots,x_{20}]\longrightarrow\hat x_{21},
$$

并且只对这个预测计算损失。此时窗口中的 `x1～x20` 都是已经得到的历史观测，历史窗口内部使用双向注意力，不一定构成未来泄漏。

但是，如果同一窗口中同时训练：

$$
\hat x_{t+1}=f(x_1,\ldots,x_t),
$$

则第 $t$ 个预测只能使用 `t` 及之前的信息。否则输入窗口中已有的 `x[t+1]` 可能直接被当前位置读取，形成标签泄漏。

**是否需要因果掩码，由每个预测被允许使用的信息决定，不由类名里有没有 `Encoder` 决定。**

### 6.6 Padding Mask：忽略用于补齐的无效位置

两条序列长度分别为 5 和 3：

```text
第1条：x1 x2 x3 x4  x5
第2条：x1 x2 x3 PAD PAD
```

代码：

```python
x = torch.randn(2, 5, 64)
padding_mask = torch.tensor([
    [False, False, False, False, False],
    [False, False, False, True,  True ],
], device=x.device)

y = encoder(x, src_key_padding_mask=padding_mask)
# y.shape == [2,5,64]
```

两种掩码的职责：

| 掩码 | 常用形状 | 含义 |
|---|---|---|
| `mask` / `src_mask` | `[T,T]` | 第 i 个位置能否关注第 j 个位置 |
| `src_key_padding_mask` | `[B,T]` | 每条序列哪些位置是补齐项 |

有 batch 时，Padding Mask 即使在 `batch_first=False` 下也仍然是 `[B,T]`。注意力掩码还可采用 `[B*nhead,T,T]`，用于给不同样本、不同头指定掩码。[^pytorch-mha]

二者可以同时传入：

```python
y = encoder(
    x,
    mask=causal_mask,
    src_key_padding_mask=padding_mask,
)
```

同时使用时保持类型一致、与计算兼容；本文都用布尔张量，并与输入放在相同设备上。

### 6.7 Padding 掩码不会自动把对应输出变零

Padding Mask 屏蔽的是作为信息来源的 Key 位置。Padding 所在的 Query 仍可能读取其他有效位置，再经过残差、偏置等得到非零输出。因此，做损失或池化时还要排除这些位置。[^pytorch-mha][^pytorch-building]

有效位置平均池化示例：

```python
valid = ~padding_mask
masked_y = y.masked_fill(padding_mask.unsqueeze(-1), 0.0)
pooled = masked_y.sum(dim=1) / valid.sum(
    dim=1, keepdim=True
).clamp_min(1)

print(pooled.shape)  # [B,D]
```

`clamp_min(1)` 只避免最后这次除法除以零，不代表整个模型已经能正确处理“没有任何有效 token”的样本；实际输入应至少保留可用位置。尤其在把因果与 padding 掩码叠加时，要留意是否出现某个 Query 的所有 Key 都被屏蔽。

### 6.8 位置编码不能由掩码替代

这个区别必须单独记住：

> 位置编码提供“当前是哪个位置”的线索；掩码规定“当前能读取哪些位置”。

没有位置编码、没有携带顺序信息的掩码、并关闭 Dropout 时，这种编码器具有排列等变性：

$$
F(PX)=PF(X),
$$

其中 $P$ 表示对序列行重新排列。

这不是“打乱输入后输出完全不变”，而是“输入怎么排列，输出也相应排列”。因为逐位置 Linear、LayerNorm、FFN 对每行规则相同，自注意力在同时重排 Query/Key/Value 后也对应重排输出，所以可以得到这个性质；附录 C 中有数值验证。

对于轨迹：

```text
位置1 → 位置2 → 位置3
位置3 → 位置2 → 位置1
```

运动含义不同，因此我们显式提供位置线索，而不只依赖内容。

一种可学习位置编码的写法：

```python
input_proj = nn.Linear(3, 64)
position_embedding = nn.Embedding(100, 64)

x = torch.randn(32, 20, 3)
time_index = torch.arange(x.size(1), device=x.device)

h = input_proj(x)                            # [32,20,64]
h = h + position_embedding(time_index)[None] # [32,20,64]
```

位置表支持索引 `0～99`，不是任意长度都可直接输入。正弦位置编码等是其他方案；本文两种编码器类都不会自动添加这些组件。[^pytorch-source][^pytorch-building]

---
<a id="sec-07"></a>
## 7. 把数据流对应到 Transformer 概念地图

### 7.1 新术语不一定意味着多了一步计算

Transformer 的术语有不同层级：有些在给数据命名，有些在给操作命名，还有些是在给一整组操作命名。

比如，“缩放点积注意力”“自注意力”“多头注意力”并不是三个一定前后串联的模块。它们分别说明计算方法、输入来源与并行组织方式。

### 7.2 数据、操作与模块的对应表

| 概念 | 类型 | 在固定数字例子中的对应 |
|---|---|---|
| Token | 序列单位 | 一帧二维坐标这个整体 |
| Sequence | 一组有位置关系的单位 | 三帧组成的序列 |
| Batch | 多条样本一起计算 | 本例 `B=1` |
| Input Projection | 操作/模块 | `Linear(2→4)` |
| Input Embedding | 数据表示 | 第三帧的 `[1,1,1,1]` |
| Positional Encoding | 位置信息机制 | 加上 `[0,0,0,1]` |
| Hidden Representation / Hidden State | 中间数据 | 第一层第三行 `[1.2607,-1.0768,0.6935,-0.8775]` |
| Query / Key / Value | 中间数据及其职责 | Q、K、V 三个投影结果 |
| Attention Scores | 中间打分 | 第一个头第三行 `[0.7071,0.7071,1.4142]` |
| Softmax | 操作 | 把一行打分归一化为聚合权重 |
| Attention Weights | 中间权重 | `[0.2483,0.2483,0.5035]` |
| Attention Output | 聚合后的特征 | 第一个头的 `[1.5035,0]` |
| Concatenation | 操作 | 将两个头的二维输出拼成四维 |
| Output Projection | 模块 | 将拼接后的头信息重新组合 |
| Residual / Skip Connection | 连接方式 | `x + sublayer(x)` |
| LayerNorm | 归一化操作 | 对一行四个特征归一化 |
| Add & Norm | 组合名称 | 残差相加与 LayerNorm |
| Activation Function | 非线性操作 | FFN 中的 ReLU |
| FFN / MLP | 子模块 | `Linear(4→6) → ReLU → Linear(6→4)` |
| Dropout | 随机正则化操作 | 本例关闭 |
| Encoder Layer / Block | 完整模块 | 注意力子层与 FFN 子层及各自残差、归一化 |
| TransformerEncoder | 模块堆叠 | 两个 Block 串联 |
| Readout | 读出方式 | 取最后一个位置的隐藏向量 |
| Prediction / Regression Head | 任务模块 | `Linear(4→2)`，输出坐标 |

“隐藏”不表示不能观察，而表示它是模型内部的表示；同一个位置的隐藏表示随层数变化。输入嵌入、带位置的表示和多层后的隐藏表示都可以是向量，但它们位于不同阶段，含义不能混用。

**Embedding 作为概念，不等于必须调用 `nn.Embedding`。** 连续坐标可以用 `nn.Linear` 产生输入嵌入；离散文字 ID 则常用 `nn.Embedding` 查表。[^pytorch-linear][^pytorch-embedding]

### 7.3 四种注意力名称分别回答什么问题

| 名称 | 回答的问题 | 本例解释 |
|---|---|---|
| Self-Attention，自注意力 | Q/K/V 从哪里来？ | 来自同一段序列 |
| Scaled Dot-Product Attention，缩放点积注意力 | 单个头如何计算？ | 点积、缩放、Softmax、加权汇总 |
| Multi-Head Attention，多头注意力 | 如何组织多个分支？ | 两个头并行，拼接后投影 |
| Causal Attention，因果注意力 | 一个位置允许读取哪里？ | 只读取当前位置及之前 |

所以一个模块可以同时是“采用缩放点积计算的、多头、因果自注意力”。这些词描述不同属性，不表示需要重复执行四次不同的注意力。[^pytorch-mha][^pytorch-sdpa]

“自”指整段序列关注自身的其他位置，并不是每帧只能看自己。第三帧读取第一帧仍然是自注意力。

### 7.4 两种 Head：注意力头与预测头

```text
Attention Head：
一层多头注意力内部的计算分支。

Prediction Head：
模型末尾把隐藏特征转换为任务结果的模块。
```

前者参与内部信息交互，后者决定输出坐标、类别或词表打分。名称中都包含 Head，不代表是同一层级的概念。

### 7.5 参数、中间张量与超参数

| 类别 | 示例 | 怎样变化 |
|---|---|---|
| 可学习参数 Parameter | 输入投影、Q/K/V 投影、FFN、LayerNorm 缩放与偏置、输出头权重 | 训练时由优化器更新 |
| 中间张量 Intermediate Tensor / Activation | Q/K/V、注意力权重、隐藏表示 | 每次前向根据输入与当前参数产生 |
| 超参数 Hyperparameter | `d_model`、`nhead`、`dim_feedforward`、`num_layers` | 由模型配置指定，不是反向传播直接更新的权重 |

位置编码也要看实现：本文固定二维例子使用手工常量；后面的训练模型使用 `nn.Parameter`，因此位置向量属于可学习参数。

### 7.6 前向传播、损失与反向传播

把数据从原始输入算到预测输出，叫 **Forward Pass，前向传播**；把预测与标签比较，得到 **Loss，损失**；沿计算关系求损失对参数的梯度，叫 **Backpropagation，反向传播**。优化器再依据这些梯度更新参数。[^pytorch-autograd]

在代码中分别对应：

```python
prediction = model(x)       # 前向传播
loss = criterion(prediction, target)
loss.backward()             # 计算梯度
optimizer.step()            # 更新参数
```

`loss.backward()` 不是把新的 token 或新的轨迹观测“倒着传回去重做输入”，而是在计算图上求导。

---

<a id="sec-08"></a>
## 8. PyTorch 两个编码器模块的接口详解

### 8.1 TransformerEncoderLayer 构造参数

常用构造形式：

```python
nn.TransformerEncoderLayer(
    d_model,
    nhead,
    dim_feedforward=2048,
    dropout=0.1,
    activation="relu",
    layer_norm_eps=1e-5,
    batch_first=False,
    norm_first=False,
    bias=True,
    device=None,
    dtype=None,
)
```

官方签名中激活函数的默认值写成 ReLU 函数；这里用等效的字符串 `"relu"` 表示。也可以用 `"gelu"` 或符合接口的可调用对象。[^pytorch-layer]

| 参数 | 含义 | 示例 |
|---|---|---|
| `d_model` | 每个位置进入该层与离开该层的特征维度 | `64` |
| `nhead` | 一层多头注意力的头数 | `4` |
| `dim_feedforward` | FFN 内部中间宽度 | `256` |
| `dropout` | Dropout 概率 | `0.1` |
| `activation` | FFN 中间激活函数 | `"gelu"` |
| `layer_norm_eps` | LayerNorm 分母中的稳定项 | `1e-5` |
| `batch_first` | 是否使用 `[B,T,D]` | 本文使用 `True` |
| `norm_first` | 是否在子层计算前归一化 | `True` 是 Pre-LN |
| `bias` | 是否使用相关 Linear 与 LayerNorm 的可学习加性偏置 | `True` |
| `device` | 参数设备 | CPU 或可用 CUDA 设备 |
| `dtype` | 参数数据类型 | `torch.float32` / `torch.float64` 等 |

### 8.2 d_model 不是时间长度，也不自动转换原始输入

假设三维位置输入为：

```text
[32,20,3]
```

其中 32 是 batch，20 是历史帧数，3 是坐标维数。设 `d_model=64`，需要显式投影：

```python
positions = torch.randn(32, 20, 3)
input_proj = nn.Linear(3, 64)
features = input_proj(positions)

layer = nn.TransformerEncoderLayer(
    d_model=64,
    nhead=4,
    batch_first=True,
)
output = layer(features)

print(features.shape)  # [32,20,64]
print(output.shape)    # [32,20,64]
```

`d_model=64` 不会自动把输入最后一维从 3 变成 64。该参数是在声明本层使用的表示宽度，而不是提供一个隐含的 `Linear(3,64)`。

### 8.3 nhead 与每头维度

需要满足：

```python
d_model % nhead == 0
```

例如：

```text
d_model=64，nhead=4
    ↓
每头维度=16
```

四个头的结果拼接后仍然是 64 维，不是 `64×4=256` 维。`nhead` 是一层内部的并行组织方式，`num_layers` 才是纵向堆叠层数。[^pytorch-mha]

### 8.4 dim_feedforward 只控制内部临时宽度

```text
d_model=64，dim_feedforward=256

[B,T,64]
    ↓ Linear
[B,T,256]
    ↓ 激活、Dropout、Linear
[B,T,64]
```

因此 FFN 中间设为 256，不会让整个编码器输出变成 256 维。

默认值是固定的 `2048`，不是自动设置成 `4*d_model`。后面的训练示例显式使用 `4*d_model`，属于我们选择的配置。[^pytorch-layer]

### 8.5 batch_first 错误可能不报错

```text
batch_first=True： [B,T,D]
batch_first=False：[T,B,D]
```

默认 `False`。若按 `[B,T,D]` 准备数据却忘了设置为 `True`，两个轴的大小通常都合法，程序可能运行，但会把 batch 和序列长度的语义理解反了。

本文始终用 `x[b,t,:]` 表示第 b 条序列第 t 个位置的特征。[^pytorch-layer]

### 8.6 TransformerEncoder 构造参数

```python
nn.TransformerEncoder(
    encoder_layer,
    num_layers,
    norm=None,
    enable_nested_tensor=True,
    mask_check=True,
)
```

其中：

| 参数 | 作用 |
|---|---|
| `encoder_layer` | 用于创建各层的模板 |
| `num_layers` | 堆叠完整层的数量 |
| `norm` | 全部层结束之后额外执行的归一化模块 |
| `enable_nested_tensor` | 允许满足条件时使用 NestedTensor 路径优化含 padding 的输入 |
| `mask_check` | 与 NestedTensor 转换路径中的掩码检查有关 |

`norm=None` 不是说每个层内部没有 LayerNorm，只是整个堆叠末尾不再额外做一次。`enable_nested_tensor=True` 也不保证所有形状、掩码组合都走优化路径，具体条件由实现决定；教学例子关闭它，便于直接观察普通稠密张量。`mask_check` 不应理解为对所有掩码正确性的全能验证开关。[^pytorch-encoder][^pytorch-source]

例如：

```python
encoder = nn.TransformerEncoder(
    encoder_layer=layer,
    num_layers=3,
    norm=nn.LayerNorm(64),
    enable_nested_tensor=False,
)
```

等价的主要逻辑：

```python
h = x
for block in encoder.layers:
    h = block(h)  # 实际实现还会逐层传递掩码
h = encoder.norm(h)
```

### 8.7 前向参数：src_mask 与 mask 不同名

单层：

```python
y = layer(
    src=x,
    src_mask=None,
    src_key_padding_mask=None,
    is_causal=False,
)
```

多层编码器：

```python
y = encoder(
    src=x,
    mask=None,
    src_key_padding_mask=None,
    is_causal=None,
)
```

记住：单层叫 `src_mask`，堆叠编码器叫 `mask`。它们都返回序列输出张量，不像 GRU 那样默认返回 `(output,h_n)`。[^pytorch-layer][^pytorch-encoder]

`is_causal` 带有向实现提示因果掩码性质的作用。不要不核对接口就把单独设置 `is_causal=True` 当成显式构造掩码的通用替代；错误提示可能导致错误执行。本文使用明确的上三角布尔掩码，最容易核对每条信息边是否被允许。[^pytorch-encoder]

### 8.8 怎样直接检查注意力权重

完整 EncoderLayer 默认不返回注意力权重。为了看清中间过程，本例直接调用内部的 `self_attn`，并要求保留各个头：

```python
out, weights = layer.self_attn(
    x, x, x,
    attn_mask=mask,
    need_weights=True,
    average_attn_weights=False,
)

# 本文有batch且是自注意力时：
# out:     [B,T,D]
# weights: [B,nhead,T,T]
```

这只得到注意力子模块结果，不包含后续残差、LayerNorm 和 FFN。附录 A 进一步把整层手工展开，再与官方层逐项比较。[^pytorch-mha]

---

<a id="sec-09"></a>
## 9. 完整训练案例：历史三维位置预测下一帧

前面已经知道数字怎样流动，但手工设置的权重没有训练。现在保留同样的数据组织思想，建立一套能训练的三维轨迹模型。

### 9.1 输入与标签的时间关系

网络结构：

```text
历史位置 [B,T,3]
    ↓ Linear(3→D)
输入特征 [B,T,D]
    ↓ 加可学习位置编码
    ↓ 带因果掩码的 TransformerEncoder
隐藏特征 [B,T,D]
    ↓ Linear(D→3)
逐位置的下一帧预测 [B,T,3]
```

输入和标签错开一帧：

```text
输入：p0 p1 p2 p3 ... p11
标签：p1 p2 p3 p4 ... p12
```

配合因果掩码：

```text
看到 p0             → 预测 p1
看到 p0,p1          → 预测 p2
看到 p0,p1,p2       → 预测 p3
...
看到 p0,p1,...,p11  → 预测 p12
```

`output[:,t]` 预测第 t 个输入之后的下一帧；`output[:,-1]` 才是根据完整历史窗口预测窗口之外的下一帧。

### 9.2 模型定义

下面是与附录 C 相同结构的模型定义。辅助函数在这里展开，便于直接看出初始化和掩码在哪里发生。

```python
import torch
from torch import nn


class TrajectoryPredictor(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        max_len: int = 64,
    ):
        super().__init__()
        if min(d_model, nhead, num_layers, max_len) <= 0:
            raise ValueError("模型尺寸必须为正数")
        if d_model % nhead != 0:
            raise ValueError("d_model 必须能被 nhead 整除")

        self.input_proj = nn.Linear(3, d_model)
        self.pos = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        # 在训练前分别初始化每层，避免随机矩阵权重初值完全一致。
        # 不要在加载已经训练好的权重之后再运行这段初始化。
        for block in self.encoder.layers:
            for name, p in block.named_parameters():
                if p.ndim > 1:
                    nn.init.xavier_uniform_(p)
                elif name.endswith("bias"):
                    nn.init.zeros_(p)
                # LayerNorm 的缩放参数保留默认值 1。

        self.head = nn.Linear(d_model, 3)

    def forward(
        self, x: torch.Tensor, causal: bool = True
    ) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != 3:
            raise ValueError("输入应为 [B,T,3]")
        T = x.size(1)
        if not 1 <= T <= self.pos.size(1):
            raise ValueError("序列为空或长度超过 max_len")

        h = self.input_proj(x)
        h = h + self.pos[:, :T]

        mask = None
        if causal:
            mask = torch.ones(
                T, T, dtype=torch.bool, device=x.device
            ).triu(1)

        h = self.encoder(h, mask=mask)
        return self.head(h)
```

调用：

```python
model = TrajectoryPredictor()
history = torch.randn(8, 12, 3)
predictions = model(history)

print(predictions.shape)        # [8,12,3]
print(predictions[:, -1].shape) # [8,3]
```

这里最后一个输出表示根据 12 帧历史预测第 13 帧，不是对第 12 帧的重新命名。

### 9.3 生成可学习的玩具轨迹

采用无量纲匀加速模型：

$$
\mathbf p(t)=\mathbf p_0+\mathbf v_0t+\frac{1}{2}\mathbf a t^2.
$$

每条轨迹的初始位置、速度和加速度随机生成：

```python
def make_batch(batch_size, seq_len, device):
    t = 0.1 * torch.arange(
        seq_len + 1, device=device
    ).view(1, -1, 1)

    p0 = torch.rand(batch_size, 1, 3, device=device) * 2 - 1
    v0 = torch.randn(batch_size, 1, 3, device=device) * 0.4
    a = torch.randn(batch_size, 1, 3, device=device) * 0.15

    positions = p0 + v0 * t + 0.5 * a * t.square()
    inputs = positions[:, :-1]
    targets = positions[:, 1:]
    return inputs, targets
```

需要生成 `T+1` 个位置，才能组成 `T` 组“当前前缀 → 下一帧”的训练目标。这不是空气阻力、旋转和弹跳都齐全的真实网球模型，只用于让网络机制能够直接运行。

### 9.4 训练过程逐步对应

```python
torch.manual_seed(42)
device = torch.device("cpu")
model = TrajectoryPredictor().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=2e-3, weight_decay=1e-4
)
criterion = nn.MSELoss()
test_x, test_y = make_batch(256, 12, device)

for step in range(1, 401):
    model.train()
    x, target = make_batch(64, 12, device)
    pred = model(x, causal=True)

    # 从已有三帧观测的位置开始监督。
    loss = criterion(pred[:, 2:], target[:, 2:])

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if step % 100 == 0:
        print(f"step={step}, loss={loss.item():.6f}")

model.eval()
with torch.no_grad():
    pred = model(test_x)
    test_loss = criterion(pred[:, 2:], test_y[:, 2:])
    next_position = pred[:, -1]

print("验证 MSE:", test_loss.item())
print("下一帧位置形状:", next_position.shape)  # [256,3]
```

为什么从 `[:,2:]` 开始？对于这个随机初始位置、速度和加速度的理想匀加速模型，在已知采样时间、无噪声条件下，一两帧不足以唯一确定完整运动参数；第三个输入位置开始，前缀已有三帧观测。这里不把不可由少量前缀唯一决定的标签强行作为同等监督。

各行训练代码的作用也可以直接对应：

```text
zero_grad：清除上一轮梯度。
backward：求当前损失对参数的梯度。
clip_grad_norm_：限制梯度的整体范数。
step：由优化器更新参数。
eval + no_grad：按评估模式运行且不记录求导图。
```

完整程序还包括输入校验、CPU/CUDA 选择、模型测试和完整验证输出，见附录 C。[^pytorch-autograd]

### 9.5 本次重新运行的结果

在 `PyTorch 2.10.0+cpu`、400 个训练步下，完整脚本得到：

| 项目 | 本次输出 |
|---|---:|
| 训练前固定验证集 MSE | `1.095346` |
| 第 100 步训练 batch MSE | `0.002395` |
| 第 200 步训练 batch MSE | `0.001197` |
| 第 300 步训练 batch MSE | `0.000768` |
| 第 400 步训练 batch MSE | `0.000739` |
| 训练后固定验证集 MSE | `0.000700` |
| 仅窗口最后一步的验证 MSE | `0.000918` |

训练 batch 每次随机生成，所以“第 400 步训练误差”和“固定验证集误差”不是同一个数据集上的数值。

一个验证样本的下一帧真实值与预测值：

```text
target:
[-1.1040921211, 0.4002150893, 0.3903486729]

prediction:
[-1.1495940685, 0.4160057008, 0.4028719962]
```

这些误差属于合成无量纲数据，不是米，也不能代表真实网球预测能力。随机数、设备、线程与软件实现差异还可能影响复现末位。

---

<a id="sec-10"></a>
## 10. 与 GRU、LSTM、Decoder 的区别

### 10.1 与 GRU/LSTM 对照理解接口

| 问题 | GRU/LSTM | 本文两个编码器模块 |
|---|---|---|
| 原始特征与内部宽度 | `input_size` 和 `hidden_size` 可以不同 | 输入最后一维必须匹配 `d_model`，必要时自己投影 |
| 层数在哪里设 | GRU/LSTM 构造中的 `num_layers` | `TransformerEncoder(...,num_layers=...)` |
| 历史怎样进入当前结果 | 通过递归状态传递 | 通过输入序列内部的注意力交互 |
| 返回什么 | 序列输出及最终状态 | 序列特征张量 |
| 跨调用状态怎么传 | 可显式传入前一次的状态 | 这两个类的标准接口没有递归状态或 KV 缓存参数 |

GRU 形式：

```python
sequence_output, hidden = gru(x, hidden)
```

LSTM 形式：

```python
sequence_output, (hidden, cell) = lstm(x, (hidden, cell))
```

编码器形式：

```python
sequence_output = encoder(x)
```

前两段表示接口关系，实际使用时需要创建相应模块与符合形状的初始状态。即便是 GRU/LSTM，是否跨调用延续状态也要由调用者明确组织，而不是假设每个模块自动记住所有上次调用。[^pytorch-gru][^pytorch-lstm]

因此，连续调用：

```python
encoder(frame_1)
encoder(frame_2)
encoder(frame_3)
```

不会让第三次调用自动拥有前两次输入。使用本文两个标准模块时，要自己组织历史窗口。不能由此推断“所有 Transformer 实现都不能缓存历史”；专门的自回归实现可以额外实现 KV cache。[^gpt2-sample]

### 10.2 Cross-Attention：Q 与 K/V 来自不同序列

自注意力：

```text
Q、K、V 都来自当前同一份序列表示。
```

交叉注意力：

```text
Q 来自一份表示；
K、V 来自另一份表示。
```

比如可以设计：

```text
未来时刻查询向量 → Q
已编码的历史轨迹 → K、V
```

这是“未来查询读取历史特征”的结构设想，并没有出现在前面的固定网络里。经典 Encoder–Decoder Transformer 的解码器就使用自己的表示产生 Query，读取来自编码器输出的 Key/Value。[^transformer-paper]

### 10.3 一个 Linear 预测头不是 TransformerDecoder

前面末尾的 `Linear(4,2)` 只是回归头，不是 `nn.TransformerDecoder`。

PyTorch 的 `TransformerDecoderLayer` 包含自注意力、读取编码器 memory 的交叉注意力和 FFN，主要输入包括 `tgt` 与 `memory`。[^pytorch-decoder]

所以前面模型的准确名称是：

```text
输入投影 + 位置编码 + TransformerEncoder + 回归预测头
```

不是“Encoder 接了一个 Transformer Decoder”。

### 10.4 GPT 的 decoder-only 也不等于直接调用 PyTorch 的 Decoder 类

公开 GPT 类自回归模型常采用因果自注意力堆叠来预测后续 token。它们不必包含经典编码器—解码器结构中的交叉注意力，也不能仅根据名称就认定内部等于 `nn.TransformerDecoderLayer`。[^gpt2-model][^gpt-oss]

前面的编码器例子提供的是注意力与特征加工的基础理解；加入因果约束可以表达只读取前缀的依赖关系，但不能据此宣称整个教学网络与某个实际 GPT 的全部内部细节相同。

接下来要迁移的是数据流的基本逻辑：**输入如何变成向量，向量怎样融合上下文，最终输出头怎样定义任务。**

---
<a id="sec-11"></a>
## 11. GPT 的文本怎样变成 token：Tokenizer 与 BPE

在轨迹例子里，一帧坐标天然可以成为一个序列单元。到了文字输入，必须先决定“哪些文字片段对应一个离散单元”。

GPT 类文本模型的基本通路可以表示为：

```text
原始文字
    ↓ Tokenizer 分词器
Token ID 序列
    ↓ Embedding 查表
每个 token 的输入向量
    ↓ 带位置信息的因果 Transformer
融合上下文的隐藏表示
    ↓ LM Head 语言模型输出头
对词表候选 token 的 logits
    ↓ 选择或采样
下一个 token ID
    ↓ Decode
文字或文字的字节片段
```

OpenAI 的 `tiktoken` 是公开的分词工具；嵌入、因果注意力、输出打分和生成过程可以在公开 GPT 实现中看到。[^tiktoken][^gpt2-model][^gpt2-sample]

### 11.1 “token 怎么计算”其实包含三个问题

```text
问题一：文字怎样切成 token？
    → Tokenizer 的编码规则。

问题二：模型怎样预测下一个 token？
    → Transformer + 词表输出头 + 选择或采样。

问题三：用了多少 token？
    → 编码结果的长度，以及完整请求的 usage 统计。
```

这三件事发生在不同阶段。Transformer 不是先理解问题难度，再决定把文字切成多少片；它先接收已经编码好的离散序列。

### 11.2 先定义一个六项的教学词表

输入：

```text
我喜欢机器人
```

为便于观察，假设教学词表是：

| Token ID | 文字片段 |
|---:|---|
| 0 | 我 |
| 1 | 喜欢 |
| 2 | 机器人 |
| 3 | 。 |
| 4 | 很 |
| 5 | 聪明 |

这个词表、编号与切分都是手工设定的，**不是这句话在真实 GPT 分词器中的输出**。

假设分词过程为：

```text
我喜欢机器人
    ↓
["我","喜欢","机器人"]
    ↓ 查编号
[0,1,2]
```

当前输入有三个 token，不是因为有六个汉字就一定是六个 token。

三个数量要分开：

```text
词表大小 V_vocab：6 种可选 token。
本次序列长度 T：3 个位置。
Token ID：0、1、2 是这些词表项的编号。
```

这里用 `V_vocab` 表示词表大小，避免与注意力里的 Value 矩阵 $V$ 混淆。

Token ID 是索引，不是连续数值特征。“机器人”的 ID 是 2，不表示它的语义是 ID 为 1 的“喜欢”的两倍。它只是让网络去嵌入表取第 2 行。

### 11.3 真实分词不是中文语法分析

真实 `tiktoken` 使用 **BPE，Byte Pair Encoding，字节对编码**。它不等于按中文“主语、动词、名词”分词，也不保证每个汉字或每个英文单词恰好一个 token。[^tiktoken]

可以先用一个 ASCII 单词看合并过程：

```text
lower
```

假设保存的合并规则和优先级是：

```text
l + o  → lo
lo + w → low
e + r  → er
```

那么：

```text
[l,o,w,e,r]
    ↓
[lo,w,e,r]
    ↓
[low,e,r]
    ↓
[low,er]
```

最终 `lower` 用两个片段表示。这里的规则仍然是教学假设，目的是展示“按照已有优先级不断合并相邻片段”的过程；OpenAI 的 BPE 教学实现采用的就是这样的基本机制。[^tiktoken-educational]

### 11.4 BPE 的规则怎样得到

准备分词器时，可以统计训练语料中哪些相邻片段最常出现，把高频组合合并为新词表项，然后继续统计与合并，直到达到目标词表规模。OpenAI 的教学实现完整展示了这类统计与合并过程。[^tiktoken-educational]

这是两种极端之间的折中：

```text
全按基础字节处理：
词表小，但一段文字可能需要很长的序列。

把所有完整词语都列入词表：
词表很大，未见过的新组合也不好覆盖。

采用可复用的子词或字节片段：
常见组合可以合并；陌生内容还可以拆成更小单位。
```

字节级基础片段有助于表示未在分词器训练中出现过的文本，并保持可逆恢复；不能因为一个新词不在完整词条中就认定无法编码。[^tiktoken]

### 11.5 使用时不会重新训练分词器

用户输入一段文字时，执行的是：

```text
使用已经保存的分词规则 → 得到 token ID 序列
```

而不是根据这次问题临时重训词表。`tiktoken` 的公开实现会先按正则规则分出片段，将片段转换成 UTF-8 字节，再按已保存的 BPE 合并优先级处理；特殊 token 还涉及相应的特殊规则。[^tiktoken-educational]

因此，分词结果取决于编码方案和输入文本本身，包括空格、大小写、标点和相邻内容，而不是模型先对语义做出的推理结果。

### 11.6 一个汉字为什么可能需要多个 token

例如：

```python
"你".encode("utf-8")
```

对应三个字节：

```text
E4 BD A0
```

假设某种编码把它分成两个已有片段：

```text
[E4 BD] + [A0]
```

那么需要两个 token，拼接起来才是完整字符。“拆成这两段”只是说明一种可能，不是在断言本文指定编码对“你”的真实结果。

反过来，常见的多个汉字也可能合并为一个 token。因而 token 边界可以在字符内部，也可以跨越多个字符；汉字数不能直接替代 token 数。[^tiktoken-cookbook]

---

<a id="sec-12"></a>
## 12. 从 token ID 到下一个 token 的预测

### 12.1 文本输入用查表，坐标输入用投影

轨迹输入是连续数值：

```text
二维坐标 → Linear → 四维特征
```

文本输入先变成离散编号：

```text
文字片段 → Token ID → Embedding 查表 → 四维特征
```

两者进入 Transformer 后都可以表示成 `[B,T,D]`，但输入端操作不同。公开 GPT 实现中可以看到通过 ID 从嵌入矩阵取向量。[^gpt2-model]

### 12.2 手工嵌入表与实际数据

假设教学词表的嵌入表中有：

```text
ID 0，“我”：
[0.1, 0.2, 0.3,0.4]

ID 1，“喜欢”：
[0.5,-0.2, 0.8,0.1]

ID 2，“机器人”：
[0.9, 0.3,-0.1,0.6]

ID 3～5：还有各自对应的四维向量。
```

输入 `[0,1,2]` 就取出这三行：

```text
[
    [0.1, 0.2, 0.3,0.4],
    [0.5,-0.2, 0.8,0.1],
    [0.9, 0.3,-0.1,0.6]
]
```

形状变化：

```text
[1,3] → Embedding → [1,3,4]
```

代码形式：

```python
embedding = nn.Embedding(
    num_embeddings=6,
    embedding_dim=4,
)

token_ids = torch.tensor([[0, 1, 2]])
x = embedding(token_ids)
print(x.shape)  # [1,3,4]
```

直接创建模块时，其初始参数不会自动等于上面的手工数字。若要逐值一致，需要显式赋值；这里主要展示查表与形状关系。

### 12.3 一个四维向量仍然只表示一个 token

```text
“机器人”
    ↓
[0.9,0.3,-0.1,0.6]
```

仍然是一个 token，用四个浮点数表示，不是四个 token：

$$
\boxed{T\neq D.}
$$

`T` 是序列位置数量，`D` 是每个位置的表示宽度，词表大小则是另外一个量 `V_vocab`。经过更多层也不会把一个输入 token 变成多个输入 token。

### 12.4 相同 Token ID 与不同上下文表示

同一个 ID 在同一嵌入表中查到相同初始向量。但加入位置信息、经过上下文交互后，隐藏表示可以不同。[^gpt2-model]

因此：

```text
Token ID：这个片段属于哪个词表项。
初始 Embedding：这个词表项对应的可学习向量。
上下文隐藏表示：这个位置结合前缀后形成的内部特征。
```

分词器负责文字与离散编号的转换；后面的神经网络负责在表示空间中学习上下文关系。

### 12.5 LM Head：从四维特征到六个候选打分

输入三个 token 经过因果 Transformer 后，得到 `[1,3,4]` 的隐藏表示。为了预测输入末尾之后的内容，取：

```python
last_hidden = hidden[:, -1, :]
# [1,4]
```

它是针对前缀“我喜欢机器人”的上下文表示，而不只是“机器人”原来的嵌入。

教学词表有六项，因此输出头可写成：

```python
lm_head = nn.Linear(4, 6)
logits = lm_head(last_hidden)
# [1,6]
```

语言模型输出头把隐藏维度映射到词表维度，为每个候选编号产生打分。公开实现可能使用独立输出矩阵，也可能与输入嵌入共享权重；本例用独立 Linear 说明基本数据流。[^gpt2-model][^gpt-oss]

假设输出为：

```text
logits = [0,0,1,3,0,-1]
```

| ID | 候选 token | 示例 logit |
|---:|---|---:|
| 0 | 我 | 0 |
| 1 | 喜欢 | 0 |
| 2 | 机器人 | 1 |
| 3 | 。 | 3 |
| 4 | 很 | 0 |
| 5 | 聪明 | -1 |

这组 logits 是教学假设，不是实际 GPT 对这句话的预测。也不是先输出一个浮点数，再将它四舍五入为 token ID；模型是在给每个候选分别打分。

### 12.6 Softmax 与选择：得到下一个 token

对六个 logits 计算 Softmax：

```text
[
    0.0382,  # 我
    0.0382,  # 喜欢
    0.1039,  # 机器人
    0.7675,  # 。
    0.0382,  # 很
    0.0141,  # 聪明
]
```

这里的 Softmax 沿词表候选维度计算，不是沿输入时间位置计算。

若采用最大项选择：

```text
选 ID 3 → 解码为“。”
```

原来的序列：

```text
[0,1,2]
```

变为：

```text
[0,1,2,3]
```

即：

```text
我喜欢机器人。
```

这一次新增的是一个输出 token。若采用采样，则根据候选概率分布随机选择，不一定永远选最大项；公开生成实现中可以看到这类选择过程。[^gpt2-sample][^gpt-oss]

注意，句号在这个教学词表中只是普通 token，并不天然等于生成结束标记。这里只展示生成一次的结果，是否继续取决于生成逻辑和停止条件。

### 12.7 Attention Softmax 与输出 Softmax 的区别

两处都用了 Softmax，但回答的问题不同：

| 位置 | 对谁归一化 | 得到什么 |
|---|---|---|
| 注意力内部 | 当前 Query 可以读取的各个 Key 位置 | 如何汇总历史 Value 的权重 |
| LM Head 之后 | 词表中的所有候选 token | 下一个 token 的候选概率 |

前者输出矩阵通常含 `T×T` 的位置关系；后者最后一维是 `V_vocab`。不要把 `[0.2483,0.2483,0.5035]` 的注意力权重，当成模型正在从三个文字候选中选择输出。

### 12.8 自回归生成与历史缓存

基本循环：

```text
已有前缀
    ↓
计算下一 token 的分数
    ↓
选择或采样一个 token
    ↓
追加到前缀
    ↓
继续生成
```

这叫 **Autoregressive Generation，自回归生成**。实际实现可缓存历史 Key/Value，避免每步把所有历史计算从头重复做；缓存是实现机制，不改变“按前缀生成下一个 token”的条件依赖关系。[^gpt2-sample]

也因此，不能把“本文 `nn.TransformerEncoder` 不提供跨调用缓存参数”误解成“GPT 每次一定从头计算、没有任何缓存”。

### 12.9 下一 token 训练：标签错开一位

教学文本：

```text
我喜欢机器人。
```

编码为：

```text
[0,1,2,3]
```

组成输入与标签：

```text
输入：[0,1,2]
标签：[1,2,3]
```

含义：

```text
看到“我”             → 预测“喜欢”
看到“我喜欢”         → 预测“机器人”
看到“我喜欢机器人”   → 预测“。”
```

因果掩码保证每个训练位置不能看到自己的未来标签。标准下一 token 目标把上下文位置的输出与后移一位的目标对应；具体训练实现可能在数据准备或损失函数内完成这个错位。[^gpt2-model]

最后一项真实标签为句号，预测概率约 `0.7675`，负对数损失：

$$
L=-\log p(\text{句号}\mid\text{我喜欢机器人})
\approx-\log(0.7675)\approx0.2647.
$$

未舍入结果约 `0.2646786053`。

嵌入表、注意力、FFN 与输出头等参数共同参与训练。Token ID 的整数大小本身没有这种语义；模型学习的是如何表示这些词表项、怎样根据前缀对下一项分配概率。

---

<a id="sec-13"></a>
## 13. 真实分词、token 计数与 API 用量

### 13.1 使用哪个编码，要对应具体模型

本文核对的 OpenAI `tiktoken` 公开映射中，`gpt-5`、`gpt-4.1`、`gpt-4o` 对应 `o200k_base`。但不能推断所有 GPT 都用相同编码，更不能推断所有模型的 token ID 一致；具体模型以工具映射与模型文档为准。[^tiktoken-model]

两种调用方式：

```python
# 明确指定一套编码，适合复现固定例子。
enc = tiktoken.get_encoding("o200k_base")

# 已知模型名时，查询该工具版本对应的编码。
enc = tiktoken.encoding_for_model("gpt-4o")
```

未知模型名可能触发异常；不要悄悄套用任意编码，然后声称得到了该模型的精确计数。

### 13.2 一个真实可核对的例子

对于纯文本：

```text
2 + 2 = 4
```

OpenAI 官方 Cookbook 给出的 `o200k_base` 编码为：[^tiktoken-cookbook]

```text
Token ID：
[17,659,220,17,314,220,19]

对应字节片段：
[b'2',b' +',b' ',b'2',b' =',b' ',b'4']
```

保留空格逐项看：

| 位置 | 文字片段 | ID |
|---:|---|---:|
| 1 | `"2"` | 17 |
| 2 | `" +"` | 659 |
| 3 | `" "` | 220 |
| 4 | `"2"` | 17 |
| 5 | `" ="` | 314 |
| 6 | `" "` | 220 |
| 7 | `"4"` | 19 |

因此：

```python
token_ids = [17, 659, 220, 17, 314, 220, 19]
count = len(token_ids)  # 7
```

这里显示了三个重要事实：空格有时与符号组合，有时单独成项；同一个 ID 出现两次，占两个位置；计数是列表长度，不是 ID 求和，也不是去重后的种类数。

### 13.3 查看任意中文的真实分词

安装：

```bash
pip install -U tiktoken
```

检查函数：

```python
import tiktoken


def inspect_tokens(text: str, encoding_name: str = "o200k_base") -> None:
    enc = tiktoken.get_encoding(encoding_name)
    token_ids = enc.encode(text)

    print("原文：", repr(text))
    print("编码：", encoding_name)
    print("Token ID：", token_ids)
    print("Token 数量：", len(token_ids))
    print()

    pieces = []
    for index, token_id in enumerate(token_ids):
        raw = enc.decode_single_token_bytes(token_id)
        pieces.append(raw)
        try:
            fragment = repr(raw.decode("utf-8"))
        except UnicodeDecodeError:
            fragment = "<不是完整的 UTF-8 字符>"
        print(
            f"位置={index:2d}  ID={token_id:6d}  "
            f"片段={fragment}  字节={raw!r}"
        )

    restored = b"".join(pieces).decode("utf-8")
    assert restored == text
    print("\n还原结果：", restored)


inspect_tokens("我喜欢机器人")
print("\n" + "=" * 60 + "\n")
inspect_tokens("2 + 2 = 4")
```

之所以使用 `decode_single_token_bytes()`，是因为一个 token 不一定包含完整 UTF-8 字符；单独把每个 token 强行解成字符串可能出现替换字符，字节形式更能反映真实边界。[^tiktoken-cookbook]

中文的真实 ID 与数量应以指定编码的运行结果为准，不要套用前面教学词表的 `[0,1,2]`。

**运行边界：** 本次整理环境能运行 PyTorch，但未安装 `tiktoken`，且安装所需网络连接不可用，因此没有执行本节真实分词代码。上面的算式编码采用官方 Cookbook 可核对的记录；没有编造中文运行结果。附录 E 保留带命令行参数和错误提示的版本，首次加载编码也可能需要下载词表数据。

### 13.4 纯文本计数不等于完整请求计数

```python
len(enc.encode(text))
```

只是在计算这段纯文本的 token 数。完整请求还可能包含实际提交的历史消息、角色与边界等结构、工具定义或 schemas，以及其他输入内容。[^openai-tokens]

所以“屏幕上这句话分成多少 token”和“这次请求使用多少输入 token”不是必然相同的数字。

### 13.5 输入、输出、缓存与推理 token

OpenAI 用量说明区分输入 token、输出 token、缓存输入 token 和推理 token。部分推理模型在可见答案之外还有内部推理 token，因此可见文字很短，也不代表总输出用量同样短。[^openai-tokens]

核对实际请求应查看返回中的 `usage`。常见字段如下，具体以所用 API 为准：

```text
Responses：
input_tokens、output_tokens、total_tokens

Chat Completions：
prompt_tokens、completion_tokens、total_tokens
```

这些统计不是把某个向量的维数乘进文本 token 数，也不是每经过一层就重复增加一轮输入 token。层数、宽度、序列长度和请求用量是不同层面的量。[^openai-tokens]

---

<a id="sec-14"></a>
## 14. 易混淆点与完整知识回路

### 14.1 最容易混用的概念，集中核对

| 常见误解 | 应当怎样理解 |
|---|---|
| 一个 EncoderLayer 只处理一帧 | 一层处理整个输入序列 |
| `num_layers=3` 就输出三个时间步 | 它控制纵向深度，不控制序列长度 |
| 传入模板一层后再创建三层，总共四层 | 模板用于复制，总共是指定的三层 |
| 四个注意力头让 64 维输出变成 256 维 | 通常拆分总模型维度，拼接后仍为 64 维 |
| `dim_feedforward` 决定编码器输出宽度 | 它是 FFN 内部中间宽度 |
| Q/K/V 是固定参数 | 它们是输入经过参数投影产生的中间数据 |
| 注意力权重就是输出特征 | 权重还要对 V 加权求和 |
| 自注意力只能看自己那个位置 | 它是同一序列内部的位置交互 |
| 各种注意力名称代表多个顺序模块 | 它们可能在描述同一模块的不同属性 |
| FFN 又跨帧混合了一次 | FFN 自身逐位置加工，跨帧内容来自之前的注意力 |
| 残差连接就是预测误差 | 它是 `x + f(x)` 的连接方式 |
| LayerNorm 沿整个时间轴求均值 | 本例沿每个位置的特征维求统计量 |
| `norm=None` 表示层内没有归一化 | 只是整个堆叠末尾不额外加 Norm |
| 各层初值相同就是共享参数 | 参数存储可以独立，只是起点数值相同 |
| 没有位置编码时打乱输入，输出完全不变 | 应区分排列不变与排列等变 |
| Padding Mask 会把对应输出自动设成零 | 它屏蔽信息来源，池化和损失仍需排除 padding |
| 因果掩码可以替代位置编码 | 可见范围与位置身份是不同问题 |
| 最后一个位置在第一层不变，所有后续层也不变 | 前面位置的变化可在更深层影响最后位置 |
| `Encoder` 不能做因果预测 | 任务是否泄漏由掩码、标签和可见信息共同决定 |
| Linear 输出头就是 Transformer Decoder | 回归头与多层解码器不是同一结构 |
| 标准 Encoder 多次调用会自动记忆上一帧 | 它不提供这样的跨调用递归状态接口 |
| 一汉字或一英文单词总等于一 token | 真实分词可能拆分或合并，需指定编码 |
| Token ID 的大小体现语义大小 | ID 是索引 |
| 四维 Embedding 就是四个 token | 一个 token 可以用四维或更高维向量表示 |
| token 数就是词表大小 | 一次输入长度与全部候选种类数不同 |
| 模型把浮点输出四舍五入成 token ID | 通常给各个候选打分，再选择或采样 |
| 一个句号一定表示停止生成 | 普通标点与停止条件要分开 |
| 可见回答文字数等于 API 总用量 | 完整输入结构与推理等可能影响用量 |

### 14.2 用同一个框架理解两个任务

```text
轨迹预测：

一帧位置
    ↓ 连续输入投影
一个特征向量
    ↓ 位置机制与 Transformer
包含上下文的隐藏表示
    ↓ 回归头
下一帧坐标
```

```text
文本生成：

文字片段
    ↓ Tokenizer
Token ID
    ↓ Embedding 查表
一个特征向量
    ↓ 位置机制与因果 Transformer
包含前缀信息的隐藏表示
    ↓ 词表输出头
候选 token 的 logits
    ↓ 选择或采样
下一个 token ID
    ↓ 解码
文字
```

输入端，坐标用连续投影，文字先分词再查表；输出端，轨迹模型回归坐标，语言模型给离散候选打分。中间的理解主线都是：一个序列位置携带向量，通过注意力交换信息，通过 FFN 加工特征，并在多层中持续更新。

### 14.3 最值得在调试器里盯住的问题

不妨持续观察第三行，问自己四个问题：它此刻包含什么数字？这些数字由哪一步得到？它是否已经使用其他行的数据？接下来的模块究竟在改变时间依赖关系，还是只在加工当前行的特征？

再看 GPT 时，把“第三帧”替换为“第三个 token”，把末尾的坐标回归头替换为词表打分头，就能沿着同一条思路继续追踪。

**最终可以把整套概念归纳为：Token 是单位，Embedding 是表示；位置机制提供位置线索，注意力决定跨位置的信息汇总，FFN 完成位置内部的非线性加工，残差与归一化组织每次更新；完整层反复堆叠，最后由任务输出头决定这些特征变成坐标、类别，还是下一个文字 token。**

---

<a id="appendix-a"></a>
## 附录 A：固定数字逐层计算的完整脚本

保存为 `transformer_numeric_flow.py`，执行：

```bash
python transformer_numeric_flow.py
```

这个脚本把本例的 Post-LN 层显式展开：Q/K/V 投影、分头、打分、Softmax、Value 聚合、拼接、输出投影、两次残差与归一化以及 FFN。它还将中间结果与官方 `MultiheadAttention`、`TransformerEncoderLayer` 及两层 `TransformerEncoder` 对比。

脚本仅为本例 `norm_first=False`、`dropout=0` 的结构编写，不是任意 Transformer 配置的通用替代实现。完整源码如下：

```python
"""固定数值逐步展开 TransformerEncoderLayer，再与 PyTorch 原模块核对。
运行：python transformer_numeric_flow.py
只依赖 PyTorch。未训练；输出用于展示计算，不代表预测精度。
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def make_model() -> tuple[nn.Linear, torch.Tensor, nn.TransformerEncoder, nn.Linear]:
    """手工设定投影、位置向量、编码器和输出头，方便复算。"""
    dtype = torch.float64
    projection = nn.Linear(2, 4, bias=False, dtype=dtype)
    position = torch.tensor(
        [[[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]], dtype=dtype
    )
    layer = nn.TransformerEncoderLayer(
        d_model=4, nhead=2, dim_feedforward=6,
        dropout=0.0, activation="relu", batch_first=True,
        norm_first=False, layer_norm_eps=1e-5, dtype=dtype,
    )
    head = nn.Linear(4, 2, bias=False, dtype=dtype)

    with torch.no_grad():
        projection.weight.copy_(torch.tensor(
            [[1, 0], [0, 1], [1, 0], [0, 1]], dtype=dtype))
        Wq = torch.eye(4, dtype=dtype)
        Wk = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0],
             [0, 0, 0, 1], [0, 0, 1, 0]], dtype=dtype)
        Wv = torch.tensor(
            [[1, 1, 0, 0], [1, -1, 0, 0],
             [0, 0, 1, 1], [0, 0, 1, -1]], dtype=dtype)
        layer.self_attn.in_proj_weight.copy_(torch.cat([Wq, Wk, Wv], dim=0))
        layer.self_attn.in_proj_bias.zero_()
        layer.self_attn.out_proj.weight.copy_(torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 0],
             [0, 0, 1, 0], [0, 0, 0, 1]], dtype=dtype))
        layer.self_attn.out_proj.bias.zero_()
        layer.linear1.weight.copy_(torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0],
             [0, 0, 1, 0], [0, 0, 0, 1],
             [1, 1, 0, 0], [0, 0, 1, -1]], dtype=dtype))
        layer.linear1.bias.zero_()
        layer.linear2.weight.copy_(torch.tensor(
            [[1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0]], dtype=dtype))
        layer.linear2.bias.zero_()
        for norm in (layer.norm1, layer.norm2):
            norm.weight.fill_(1.0)
            norm.bias.zero_()
        head.weight.copy_(torch.tensor(
            [[1, 0, 0.5, 0], [0, 1, 0, 0.5]], dtype=dtype))

    # 两个层参数存储独立，创建时数值相同；本例保留相同数值便于对比。
    encoder = nn.TransformerEncoder(layer, num_layers=2, norm=None,
                                    enable_nested_tensor=False)
    projection.eval()
    encoder.eval()
    head.eval()
    return projection, position, encoder, head


def inspect_layer(layer: nn.TransformerEncoderLayer, x: torch.Tensor,
                  mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """显式展开本例的 Post-LN 层。无 Dropout，支持布尔注意力掩码。"""
    if layer.norm_first or layer.dropout.p != 0.0 or layer.self_attn.dropout != 0.0:
        raise ValueError("此展示函数仅用于 norm_first=False、dropout=0 的层")
    if x.ndim != 3 or x.shape[-1] != layer.self_attn.embed_dim:
        raise ValueError("输入必须是 [batch, time, d_model]")
    B, T, D = x.shape
    heads = layer.self_attn.num_heads
    d_head = D // heads
    q, k, v = F.linear(x, layer.self_attn.in_proj_weight,
                      layer.self_attn.in_proj_bias).chunk(3, dim=-1)

    def split(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(B, T, heads, d_head).transpose(1, 2)

    qh, kh, vh = map(split, (q, k, v))
    scores = qh @ kh.transpose(-2, -1) / math.sqrt(d_head)
    masked_scores = scores
    if mask is not None:
        if mask.dtype != torch.bool or mask.shape != (T, T):
            raise ValueError("mask 必须为 [T,T] 的布尔张量，True 表示禁止关注")
        masked_scores = scores.masked_fill(mask.to(x.device)[None, None], -torch.inf)
    attention = masked_scores.softmax(dim=-1)
    per_head = attention @ vh
    concat = per_head.transpose(1, 2).contiguous().reshape(B, T, D)
    attn_output = layer.self_attn.out_proj(concat)
    residual1 = x + attn_output
    norm1 = layer.norm1(residual1)
    ff_pre = layer.linear1(norm1)
    ff_relu = layer.activation(ff_pre)
    ff_output = layer.linear2(ff_relu)
    residual2 = norm1 + ff_output
    output = layer.norm2(residual2)

    # 直接验证官方 MultiheadAttention 的输出与注意力权重。
    mha_output, mha_attention = layer.self_attn(
        x, x, x, attn_mask=mask, need_weights=True,
        average_attn_weights=False)
    torch.testing.assert_close(attn_output, mha_output, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(attention, mha_attention, rtol=1e-9, atol=1e-9)
    official = layer(x, src_mask=mask)
    torch.testing.assert_close(output, official, rtol=1e-9, atol=1e-9)
    return dict(input=x, Q=q, K=k, V=v, scores=scores,
                masked_scores=masked_scores, attention=attention,
                per_head=per_head, concat=concat, attn_output=attn_output,
                residual1=residual1, norm1=norm1, ff_pre=ff_pre,
                ff_relu=ff_relu, ff_output=ff_output,
                residual2=residual2, output=output)


@torch.no_grad()
def main() -> None:
    torch.set_printoptions(precision=6, linewidth=140, sci_mode=False)
    torch.set_num_threads(1)
    projection, position, encoder, head = make_model()
    raw = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]], dtype=torch.float64)
    embedding = projection(raw)
    x = embedding + position
    print("PyTorch:", torch.__version__)
    print("raw =", raw)
    print("embedding =", embedding)
    print("position =", position)
    print("encoder input =", x)

    traces = []
    current = x
    for i, layer in enumerate(encoder.layers, start=1):
        trace = inspect_layer(layer, current)
        traces.append(trace)
        print(f"\n{'=' * 20} Layer {i} {'=' * 20}")
        for name, tensor in trace.items():
            print(f"\n{name}  shape={tuple(tensor.shape)}\n{tensor[0]}")
        current = trace["output"]

    official_encoder = encoder(x)
    torch.testing.assert_close(current, official_encoder, rtol=1e-9, atol=1e-9)
    pred = head(current[:, -1])
    # 示意标签：并非前三帧唯一能推出的运动结果。
    target = torch.tensor([[2., 1.]], dtype=torch.float64)
    print("\nprediction =", pred)
    print("target =", target)
    print("MSE =", F.mse_loss(pred, target).item())
    row = traces[0]["residual1"][0, 2]
    print("\nLayer 1 residual1 row 3 mean =", row.mean().item())
    print("Layer 1 residual1 row 3 variance =", row.var(unbiased=False).item())

    mask = torch.ones(3, 3, dtype=torch.bool).triu(1)
    causal_first = inspect_layer(encoder.layers[0], x, mask)
    print("\ncausal mask =", mask)
    print("causal layer1 attention =", causal_first["attention"][0])
    print("causal layer1 output =", causal_first["output"][0])
    print("causal encoder output =", encoder(x, mask=mask)[0])

    # 真正的因果性测试：改未来输入，第 1、2 个位置的输出不应改变。
    modified = x.clone()
    modified[:, 2] += torch.tensor([10., -3., 7., 5.], dtype=x.dtype)
    causal_before = encoder(x, mask=mask)
    causal_after = encoder(modified, mask=mask)
    torch.testing.assert_close(causal_before[:, :2], causal_after[:, :2],
                               rtol=1e-9, atol=1e-9)
    a = encoder.layers[0].linear1.weight
    b = encoder.layers[1].linear1.weight
    print("\ninitial parameter values equal:", torch.equal(a, b))
    print("parameter storage shared:", a.data_ptr() == b.data_ptr())
    print("\nAll checks passed: MHA, individual layers, encoder stack, causal masking.")


if __name__ == "__main__":
    main()
```

---

<a id="appendix-b"></a>
## 附录 B：两层网络全部中间数值

以下是 `transformer_numeric_flow.py` 的完整输出，包含两层每个头的打分、掩码后打分、注意力权重、所有位置的中间张量、最终预测和因果掩码对照。本次重新运行与原对话附件中的记录逐字一致。

打印 `tensor[0]` 时省略了 batch 维，但每一项前面的 `shape` 仍然列出完整形状。

```text
PyTorch: 2.10.0+cpu
raw = tensor([[[1., 0.],
         [0., 1.],
         [1., 1.]]], dtype=torch.float64)
embedding = tensor([[[1., 0., 1., 0.],
         [0., 1., 0., 1.],
         [1., 1., 1., 1.]]], dtype=torch.float64)
position = tensor([[[0., 0., 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 0., 1.]]], dtype=torch.float64)
encoder input = tensor([[[1., 0., 1., 0.],
         [0., 1., 1., 1.],
         [1., 1., 1., 2.]]], dtype=torch.float64)

==================== Layer 1 ====================

input  shape=(1, 3, 4)
tensor([[1., 0., 1., 0.],
        [0., 1., 1., 1.],
        [1., 1., 1., 2.]], dtype=torch.float64)

Q  shape=(1, 3, 4)
tensor([[1., 0., 1., 0.],
        [0., 1., 1., 1.],
        [1., 1., 1., 2.]], dtype=torch.float64)

K  shape=(1, 3, 4)
tensor([[1., 0., 0., 1.],
        [0., 1., 1., 1.],
        [1., 1., 2., 1.]], dtype=torch.float64)

V  shape=(1, 3, 4)
tensor([[ 1.,  1.,  1.,  1.],
        [ 1., -1.,  2.,  0.],
        [ 2.,  0.,  3., -1.]], dtype=torch.float64)

scores  shape=(1, 2, 3, 3)
tensor([[[0.707107, 0.000000, 0.707107],
         [0.000000, 0.707107, 0.707107],
         [0.707107, 0.707107, 1.414214]],

        [[0.000000, 0.707107, 1.414214],
         [0.707107, 1.414214, 2.121320],
         [1.414214, 2.121320, 2.828427]]], dtype=torch.float64)

masked_scores  shape=(1, 2, 3, 3)
tensor([[[0.707107, 0.000000, 0.707107],
         [0.000000, 0.707107, 0.707107],
         [0.707107, 0.707107, 1.414214]],

        [[0.000000, 0.707107, 1.414214],
         [0.707107, 1.414214, 2.121320],
         [1.414214, 2.121320, 2.828427]]], dtype=torch.float64)

attention  shape=(1, 2, 3, 3)
tensor([[[0.401112, 0.197776, 0.401112],
         [0.197776, 0.401112, 0.401112],
         [0.248255, 0.248255, 0.503490]],

        [[0.140029, 0.283995, 0.575975],
         [0.140029, 0.283995, 0.575975],
         [0.140029, 0.283995, 0.575975]]], dtype=torch.float64)

per_head  shape=(1, 2, 3, 2)
tensor([[[ 1.401112,  0.203336],
         [ 1.401112, -0.203336],
         [ 1.503490,  0.000000]],

        [[ 2.435946, -0.435946],
         [ 2.435946, -0.435946],
         [ 2.435946, -0.435946]]], dtype=torch.float64)

concat  shape=(1, 3, 4)
tensor([[ 1.401112,  0.203336,  2.435946, -0.435946],
        [ 1.401112, -0.203336,  2.435946, -0.435946],
        [ 1.503490,  0.000000,  2.435946, -0.435946]], dtype=torch.float64)

attn_output  shape=(1, 3, 4)
tensor([[ 3.837058,  0.203336,  2.435946, -0.435946],
        [ 3.837058, -0.203336,  2.435946, -0.435946],
        [ 3.939436,  0.000000,  2.435946, -0.435946]], dtype=torch.float64)

residual1  shape=(1, 3, 4)
tensor([[ 4.837058,  0.203336,  3.435946, -0.435946],
        [ 3.837058,  0.796664,  3.435946,  0.564054],
        [ 4.939436,  1.000000,  3.435946,  1.564054]], dtype=torch.float64)

norm1  shape=(1, 3, 4)
tensor([[ 1.287902, -0.823122,  0.649585, -1.114365],
        [ 1.128764, -0.915696,  0.859043, -1.072110],
        [ 1.413347, -1.112212,  0.449465, -0.750599]], dtype=torch.float64)

ff_pre  shape=(1, 3, 6)
tensor([[ 1.287902, -0.823122,  0.649585, -1.114365,  0.464780,  1.763951],
        [ 1.128764, -0.915696,  0.859043, -1.072110,  0.213068,  1.931153],
        [ 1.413347, -1.112212,  0.449465, -0.750599,  0.301135,  1.200064]], dtype=torch.float64)

ff_relu  shape=(1, 3, 6)
tensor([[1.287902, 0.000000, 0.649585, 0.000000, 0.464780, 1.763951],
        [1.128764, 0.000000, 0.859043, 0.000000, 0.213068, 1.931153],
        [1.413347, 0.000000, 0.449465, 0.000000, 0.301135, 1.200064]], dtype=torch.float64)

ff_output  shape=(1, 3, 4)
tensor([[1.752682, 0.000000, 2.413536, 0.000000],
        [1.341831, 0.000000, 2.790196, 0.000000],
        [1.714482, 0.000000, 1.649528, 0.000000]], dtype=torch.float64)

residual2  shape=(1, 3, 4)
tensor([[ 3.040583, -0.823122,  3.063121, -1.114365],
        [ 2.470595, -0.915696,  3.649239, -1.072110],
        [ 3.127829, -1.112212,  2.098993, -0.750599]], dtype=torch.float64)

output  shape=(1, 3, 4)
tensor([[ 0.993084, -0.926339,  1.004280, -1.071024],
        [ 0.694472, -0.941382,  1.263853, -1.016943],
        [ 1.260749, -1.076826,  0.693542, -0.877465]], dtype=torch.float64)

==================== Layer 2 ====================

input  shape=(1, 3, 4)
tensor([[ 0.993084, -0.926339,  1.004280, -1.071024],
        [ 0.694472, -0.941382,  1.263853, -1.016943],
        [ 1.260749, -1.076826,  0.693542, -0.877465]], dtype=torch.float64)

Q  shape=(1, 3, 4)
tensor([[ 0.993084, -0.926339,  1.004280, -1.071024],
        [ 0.694472, -0.941382,  1.263853, -1.016943],
        [ 1.260749, -1.076826,  0.693542, -0.877465]], dtype=torch.float64)

K  shape=(1, 3, 4)
tensor([[ 0.993084, -0.926339, -1.071024,  1.004280],
        [ 0.694472, -0.941382, -1.016943,  1.263853],
        [ 1.260749, -1.076826, -0.877465,  0.693542]], dtype=torch.float64)

V  shape=(1, 3, 4)
tensor([[ 0.066744,  1.919423, -0.066744,  2.075304],
        [-0.246910,  1.635854,  0.246910,  2.280796],
        [ 0.183923,  2.337575, -0.183923,  1.571007]], dtype=torch.float64)

scores  shape=(1, 2, 3, 3)
tensor([[[ 1.304131,  1.104295,  1.590662],
         [ 1.104295,  0.967670,  1.335908],
         [ 1.590662,  1.335908,  1.943866]],

        [[-1.521139, -1.679317, -1.148356],
         [-1.679317, -1.817642, -1.282890],
         [-1.148356, -1.282890, -0.860633]]], dtype=torch.float64)

masked_scores  shape=(1, 2, 3, 3)
tensor([[[ 1.304131,  1.104295,  1.590662],
         [ 1.104295,  0.967670,  1.335908],
         [ 1.590662,  1.335908,  1.943866]],

        [[-1.521139, -1.679317, -1.148356],
         [-1.679317, -1.817642, -1.282890],
         [-1.148356, -1.282890, -0.860633]]], dtype=torch.float64)

attention  shape=(1, 2, 3, 3)
tensor([[[0.317393, 0.259902, 0.422704],
         [0.319190, 0.278429, 0.402381],
         [0.312624, 0.242317, 0.445059]],

        [[0.302529, 0.258268, 0.439202],
         [0.297857, 0.259378, 0.442765],
         [0.311768, 0.272524, 0.415708]]], dtype=torch.float64)

per_head  shape=(1, 2, 3, 2)
tensor([[[ 0.034757,  2.022477],
         [ 0.026564,  2.008726],
         [ 0.042892,  2.036811]],

        [[-0.037202,  1.906888],
         [-0.037272,  1.905319],
         [-0.029978,  1.921665]]], dtype=torch.float64)

concat  shape=(1, 3, 4)
tensor([[ 0.034757,  2.022477, -0.037202,  1.906888],
        [ 0.026564,  2.008726, -0.037272,  1.905319],
        [ 0.042892,  2.036811, -0.029978,  1.921665]], dtype=torch.float64)

attn_output  shape=(1, 3, 4)
tensor([[-0.002446,  2.022477, -0.037202,  1.906888],
        [-0.010707,  2.008726, -0.037272,  1.905319],
        [ 0.012914,  2.036811, -0.029978,  1.921665]], dtype=torch.float64)

residual1  shape=(1, 3, 4)
tensor([[0.990638, 1.096138, 0.967078, 0.835864],
        [0.683765, 1.067344, 1.226582, 0.888376],
        [1.273663, 0.959986, 0.663564, 1.044200]], dtype=torch.float64)

norm1  shape=(1, 3, 4)
tensor([[ 0.196478,  1.334875, -0.057748, -1.473605],
        [-1.396861,  0.498110,  1.284783, -0.386032],
        [ 1.320020, -0.116145, -1.473305,  0.269430]], dtype=torch.float64)

ff_pre  shape=(1, 3, 6)
tensor([[ 0.196478,  1.334875, -0.057748, -1.473605,  1.531353,  1.415857],
        [-1.396861,  0.498110,  1.284783, -0.386032, -0.898751,  1.670815],
        [ 1.320020, -0.116145, -1.473305,  0.269430,  1.203875, -1.742735]], dtype=torch.float64)

ff_relu  shape=(1, 3, 6)
tensor([[0.196478, 1.334875, 0.000000, 0.000000, 1.531353, 1.415857],
        [0.000000, 0.498110, 1.284783, 0.000000, 0.000000, 1.670815],
        [1.320020, 0.000000, 0.000000, 0.269430, 1.203875, 0.000000]], dtype=torch.float64)

ff_output  shape=(1, 3, 4)
tensor([[1.727832, 1.334875, 1.415857, 0.000000],
        [0.000000, 0.498110, 2.955599, 0.000000],
        [2.523894, 0.000000, 0.000000, 0.269430]], dtype=torch.float64)

residual2  shape=(1, 3, 4)
tensor([[ 1.924310,  2.669750,  1.358109, -1.473605],
        [-1.396861,  0.996220,  4.240382, -0.386032],
        [ 3.843914, -0.116145, -1.473305,  0.538861]], dtype=torch.float64)

output  shape=(1, 3, 4)
tensor([[ 0.513243,  0.988707,  0.152102, -1.654052],
        [-1.062811,  0.062440,  1.587879, -0.587509],
        [ 1.608392, -0.416456, -1.110396, -0.081540]], dtype=torch.float64)

prediction = tensor([[ 1.053194, -0.457226]], dtype=torch.float64)
target = tensor([[2., 1.]], dtype=torch.float64)
MSE = 1.5099745861859648

Layer 1 residual1 row 3 mean = 2.7348589859141343
Layer 1 residual1 row 3 variance = 2.4330507387498224

causal mask = tensor([[False,  True,  True],
        [False, False,  True],
        [False, False, False]])
causal layer1 attention = tensor([[[1.000000, 0.000000, 0.000000],
         [0.330238, 0.669762, 0.000000],
         [0.248255, 0.248255, 0.503490]],

        [[1.000000, 0.000000, 0.000000],
         [0.330238, 0.669762, 0.000000],
         [0.140029, 0.283995, 0.575975]]], dtype=torch.float64)
causal layer1 output = tensor([[ 1.414212, -0.942808,  0.471404, -0.942808],
        [ 0.549313, -1.148564,  1.348314, -0.749063],
        [ 1.260749, -1.076826,  0.693542, -0.877465]], dtype=torch.float64)
causal encoder output = tensor([[ 1.401806,  0.467269, -1.121445, -0.747630],
        [-1.027949, -0.381052,  1.654624, -0.245623],
        [ 1.559217,  0.172943, -1.005253, -0.726907]], dtype=torch.float64)

initial parameter values equal: True
parameter storage shared: False

All checks passed: MHA, individual layers, encoder stack, causal masking.
```

---

<a id="appendix-c"></a>
## 附录 C：接口测试与轨迹训练的完整脚本

保存为 `transformer_encoder_demo.py`。默认执行接口与掩码检查，再训练 400 步：

```bash
python transformer_encoder_demo.py
python transformer_encoder_demo.py --steps 400 --device cpu
```

有可用 CUDA 的环境也可选择 `--device cuda`；数值结果不保证与 CPU 逐位一致。这里保留原始完整脚本，包括数据生成、参数初始化、测试、训练和验证，不需要再从正文拼接零散片段。

```python
"""TransformerEncoderLayer / TransformerEncoder tutorial and runnable tests.

Run: python transformer_encoder_demo.py
Only dependency: PyTorch. Tested on PyTorch 2.10.0 CPU.
Synthetic data use dimensionless units; this is not a real tennis model.
"""
import argparse
import torch
from torch import nn


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """For nn.TransformerEncoder, True means an attention edge is blocked."""
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)


def reinitialize_encoder(encoder: nn.TransformerEncoder) -> None:
    """Initialize before training; never call this after loading learned weights."""
    for block in encoder.layers:
        for name, parameter in block.named_parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
            elif name.endswith("bias"):
                nn.init.zeros_(parameter)
            # LayerNorm's scale remains at its default value of one.


def demonstrate_layers_and_masks() -> None:
    torch.manual_seed(7)
    layer = nn.TransformerEncoderLayer(
        d_model=64, nhead=4, dim_feedforward=128,
        dropout=0.0, batch_first=True,
    )
    encoder = nn.TransformerEncoder(
        layer, num_layers=3, enable_nested_tensor=False,
    )
    layer.eval()
    encoder.eval()
    x = torch.randn(2, 10, 64)
    with torch.no_grad():
        one_layer = layer(x)
        three_layers = encoder(x)
    print("one layer:", tuple(one_layer.shape))
    print("three layers:", tuple(three_layers.shape))

    w0 = encoder.layers[0].linear1.weight
    w1 = encoder.layers[1].linear1.weight
    print("same initial values:", torch.equal(w0, w1))
    print("same storage:", w0.data_ptr() == w1.data_ptr())

    # Reconstruct the dense stack to show that it returns the final layer only.
    with torch.no_grad():
        explicit = x
        for block in encoder.layers:
            explicit = block(explicit)
        torch.testing.assert_close(three_layers, explicit)

    # Mutating the future must not affect earlier outputs under a causal mask.
    x_modified = x.clone()
    x_modified[:, 5:] = 5 * torch.randn_like(x_modified[:, 5:])
    mask = causal_mask(x.size(1), x.device)
    with torch.no_grad():
        h1 = encoder(x, mask=mask)
        h2 = encoder(x_modified, mask=mask)
        g1 = encoder(x)
        g2 = encoder(x_modified)
    masked_diff = (h1[:, :5] - h2[:, :5]).abs().max().item()
    unmasked_diff = (g1[:, :5] - g2[:, :5]).abs().max().item()
    torch.testing.assert_close(h1[:, :5], h2[:, :5], rtol=1e-5, atol=1e-5)
    print(f"past-output change, causal: {masked_diff:.8f}")
    print(f"past-output change, unmasked: {unmasked_diff:.8f}")

    # Right-padded sequences. Padding values must not affect valid outputs.
    padded = torch.randn(2, 5, 64)
    lengths = torch.tensor([5, 3])
    padding_mask = torch.arange(5)[None, :] >= lengths[:, None]
    changed_padding = padded.clone()
    changed_padding[padding_mask] = 20 * torch.randn_like(padded[padding_mask])
    with torch.no_grad():
        y1 = encoder(padded, src_key_padding_mask=padding_mask)
        y2 = encoder(changed_padding, src_key_padding_mask=padding_mask)
    torch.testing.assert_close(y1[~padding_mask], y2[~padding_mask])
    pooled = y1.masked_fill(padding_mask.unsqueeze(-1), 0).sum(dim=1)
    pooled = pooled / lengths.unsqueeze(-1)
    print("masked mean pooling:", tuple(pooled.shape))

    # No positional encoding and no asymmetric mask => permutation equivariance.
    permutation = torch.tensor([3, 0, 8, 2, 6, 9, 1, 7, 4, 5])
    with torch.no_grad():
        permuted_output = encoder(x[:, permutation])
        expected_output = encoder(x)[:, permutation]
    torch.testing.assert_close(permuted_output, expected_output, rtol=1e-5, atol=1e-5)
    print("shape, stack, causal, padding, and permutation tests: passed")


class TrajectoryPredictor(nn.Module):
    """Map [B,T,3] positions to [B,T,3] next-position predictions.

    With causal=True, output[:,t] uses inputs no later than t.
    A learned position table is used, so lengths beyond max_len are not allowed.
    """
    def __init__(
        self, d_model: int = 32, nhead: int = 4,
        num_layers: int = 2, max_len: int = 64,
    ) -> None:
        super().__init__()
        if min(d_model, nhead, num_layers, max_len) <= 0:
            raise ValueError("All model sizes must be positive.")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.input_proj = nn.Linear(3, d_model)
        self.pos = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        reinitialize_encoder(self.encoder)
        self.head = nn.Linear(d_model, 3)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != 3:
            raise ValueError("Expected input shape [B,T,3].")
        length = x.size(1)
        if not 1 <= length <= self.pos.size(1):
            raise ValueError("Sequence length is empty or exceeds max_len.")
        h = self.input_proj(x) + self.pos[:, :length]
        mask = causal_mask(length, x.device) if causal else None
        h = self.encoder(h, mask=mask)
        return self.head(h)


def make_batch(
    batch_size: int, seq_len: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Toy constant-acceleration trajectories; independent random batches."""
    time = 0.1 * torch.arange(seq_len + 1, device=device).view(1, -1, 1)
    p0 = torch.rand(batch_size, 1, 3, device=device) * 2 - 1
    velocity = torch.randn(batch_size, 1, 3, device=device) * 0.4
    acceleration = torch.randn(batch_size, 1, 3, device=device) * 0.15
    positions = p0 + velocity * time + 0.5 * acceleration * time.square()
    return positions[:, :-1], positions[:, 1:]


def train_example(steps: int, device: torch.device) -> None:
    torch.manual_seed(42)
    model = TrajectoryPredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    test_x, test_y = make_batch(256, 12, device)

    def evaluate() -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            prediction = model(test_x)
            loss = criterion(prediction[:, 2:], test_y[:, 2:]).item()
            last_loss = criterion(prediction[:, -1], test_y[:, -1]).item()
        return loss, last_loss

    initial, _ = evaluate()
    print("training device:", device)
    print(f"held-out MSE before training: {initial:.6f}")
    for step in range(1, steps + 1):
        model.train()
        x, target = make_batch(64, 12, device)
        prediction = model(x, causal=True)
        # At least three observed positions are needed to identify this toy model.
        loss = criterion(prediction[:, 2:], target[:, 2:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            print(f"step={step:4d}, training MSE={loss.item():.6f}")

    final, last = evaluate()
    print(f"held-out MSE after training: {final:.6f}")
    print(f"held-out final-step MSE: {last:.6f}")
    with torch.no_grad():
        pred = model(test_x[:1])[:, -1]
    print("history shape:", tuple(test_x[:1].shape))
    print("next-position shape:", tuple(pred.shape))
    print("target:", test_y[0, -1].cpu().tolist())
    print("prediction:", pred[0].cpu().tolist())
    print("Loss is in synthetic dimensionless units, not metres.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    # Small tensors can be slower with excessive CPU threads.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    print("PyTorch", torch.__version__)
    demonstrate_layers_and_masks()
    train_example(args.steps, torch.device(args.device))
```

---

<a id="appendix-d"></a>
## 附录 D：训练示例完整运行记录

以下是本次在 CPU 重新执行 400 步的完整标准输出。随机验证集在训练前固定，各训练 batch 独立生成；因此不要把训练 batch MSE 与验证 MSE 当成同一个统计量。

```text
PyTorch 2.10.0+cpu
one layer: (2, 10, 64)
three layers: (2, 10, 64)
same initial values: True
same storage: False
past-output change, causal: 0.00000000
past-output change, unmasked: 3.10651970
masked mean pooling: (2, 64)
shape, stack, causal, padding, and permutation tests: passed
training device: cpu
held-out MSE before training: 1.095346
step=   1, training MSE=1.022486
step= 100, training MSE=0.002395
step= 200, training MSE=0.001197
step= 300, training MSE=0.000768
step= 400, training MSE=0.000739
held-out MSE after training: 0.000700
held-out final-step MSE: 0.000918
history shape: (1, 12, 3)
next-position shape: (1, 3)
target: [-1.1040921211242676, 0.4002150893211365, 0.3903486728668213]
prediction: [-1.1495940685272217, 0.4160057008266449, 0.4028719961643219]
Loss is in synthetic dimensionless units, not metres.
```

---

<a id="appendix-e"></a>
## 附录 E：真实文本 token 检查脚本

保存为 `inspect_gpt_tokens.py`。它保留正文的逐 token 字节检查，并增加模型名/编码名选择、命令行输入和错误提示：

```bash
pip install -U tiktoken
python inspect_gpt_tokens.py
python inspect_gpt_tokens.py "我喜欢机器人" --encoding o200k_base
python inspect_gpt_tokens.py "2 + 2 = 4" --model gpt-4o
```

本次仅完成该脚本的语法检查，没有执行需要 `tiktoken` 和词表的真实分词。默认检查普通文本；若传入被分词器保留为特殊 token 的字面字符串，可能需要根据自己的任务明确配置特殊 token 处理策略。

```python
"""查看纯文本的 token ID、数量和逐 token 字节。

依赖：pip install -U tiktoken
运行：python inspect_gpt_tokens.py
      python inspect_gpt_tokens.py "我喜欢机器人" --encoding o200k_base
      python inspect_gpt_tokens.py "2 + 2 = 4" --model gpt-4o

本脚本只计算纯文本，不代表整个聊天请求的用量。
"""
from __future__ import annotations

import argparse
import sys


def inspect_tokens(text: str, encoding) -> None:
    token_ids = encoding.encode(text)
    print("原文：", repr(text))
    print("编码：", encoding.name)
    print("Token ID：", token_ids)
    print("Token 数量：", len(token_ids))
    print()

    pieces = []
    for index, token_id in enumerate(token_ids):
        raw = encoding.decode_single_token_bytes(token_id)
        pieces.append(raw)
        try:
            fragment = repr(raw.decode("utf-8"))
        except UnicodeDecodeError:
            fragment = "<不是完整的 UTF-8 字符>"
        print(
            f"位置={index:2d}  ID={token_id:6d}  "
            f"片段={fragment}  字节={raw!r}"
        )

    restored = b"".join(pieces).decode("utf-8")
    if restored != text:
        raise AssertionError("还原文本与原文不一致")
    print("\n还原结果：", restored)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texts", nargs="*", help="需要检查的纯文本")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--encoding", default=None, help="明确指定编码名")
    group.add_argument("--model", default=None, help="由模型名查询编码")
    args = parser.parse_args()

    try:
        import tiktoken
    except ImportError:
        print("缺少 tiktoken，请先运行：pip install -U tiktoken", file=sys.stderr)
        return 2

    try:
        if args.model:
            encoding = tiktoken.encoding_for_model(args.model)
        else:
            encoding = tiktoken.get_encoding(args.encoding or "o200k_base")
    except Exception as exc:
        print(
            f"无法加载指定编码：{exc}\n"
            "请检查模型名、tiktoken 版本及首次下载词表所需的网络。"
            "不会静默换用另一套编码。",
            file=sys.stderr,
        )
        return 2

    texts = args.texts or [
        "我喜欢机器人",
        "2 + 2 = 4",
        "在现在的大模型中，例如gpt，里面的token是怎么计算的",
    ]
    try:
        for index, text in enumerate(texts):
            if index:
                print("\n" + "=" * 60 + "\n")
            inspect_tokens(text, encoding)
    except Exception as exc:
        print(f"分词检查失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

