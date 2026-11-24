

接着前面的 DINOv2，**SALAD 最适合这样理解：DINOv2 负责把图像变成有意义的局部特征，SALAD 负责把这些局部特征组织成一条适合“判断是不是同一个地点”的整图描述子。**

它不是另一种 Transformer，也不是在两张图之间直接匹配特征点的网络，而是一个**可学习的特征聚合模块**。完整的 **DINOv2-SALAD** 系统，还会针对地点识别任务微调 DINOv2 的部分参数。SALAD 来自 CVPR 2024 论文 *Optimal Transport Aggregation for Visual Place Recognition*，全称是 **Sinkhorn Algorithm for Locally Aggregated Descriptors**。[Open Access CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Izquierdo_Optimal_Transport_Aggregation_for_Visual_Place_Recognition_CVPR_2024_paper.html?utm_source=chatgpt.com)

---

# 一、SALAD 要解决的，不是“这是什么”，而是“这是不是同一个地方”

假设机器人两次经过同一个建筑入口：

```text
第一次：
晴天，门口没有车，建筑正面比较完整。

第二次：
阴天，门口停着一辆公交车，相机视角稍有变化。
```

我们希望这两张图像的描述子仍然接近。

另一方面：

```text
第三张图：
另一栋建筑，也有玻璃幕墙、入口和公交车。
```

虽然第三张图在语义上也属于“建筑入口”，但我们希望它与前两张图区分开。

因此，地点识别需要兼顾两件事：

> **忽略不应影响地点判断的变化，同时保留能够区分具体地点的细节。**

DINOv2 的 CLS 特征当然可以用于图像检索，但它的通用预训练目标，并不直接等价于“让同一地点的不同拍摄结果接近，让不同地点远离”。DINOv2-SALAD 因而采用了针对视觉地点识别的监督训练。

## 1.1 为什么不直接把所有 patch 特征求平均？

先看一个纯数学例子。

图像 A 有两个局部特征：

\[
f_1=[2,0],\qquad f_2=[0,2]
\]

平均值为：

\[
\bar f_A=[1,1]
\]

图像 B 的两个局部特征为：

\[
f_1'=[1,1],\qquad f_2'=[1,1]
\]

平均值同样是：

\[
\bar f_B=[1,1]
\]

虽然两组局部特征的组成不同，简单平均却把它们变成了同一个结果。

这说明：

> **平均池化容易把不同类型的局部信息混在一起，并丢掉它们的组成差异。**

SALAD 的思路不是“所有东西混成一份”，而是：

```text
先把局部信息分配到多个聚合槽位
             ↓
每个槽位分别汇总一部分信息
             ↓
把这些汇总结果拼接起来
```

此外，它还允许一部分分配权重进入 **dustbin，即不进入最终局部描述子的弃用槽**。这正是 SALAD 聚合设计的重要部分。

---

# 二、先看真实模型：一张图片如何变成 8448 维描述子？

前面的 DINOv2 教学使用了 ViT-S/14。这里改用官方 DINOv2-SALAD 预训练模型的标准配置：

**DINOv2 ViT-B/14 + SALAD。**

其主干 token 维度为 768，SALAD 使用 64 个聚合槽位，每个槽位输出 128 维特征，CLS 分支输出 256 维。

## 2.1 DINOv2 输出什么？

以官方评估使用的：

\[
322\times322
\]

图像为例。由于 patch 大小为 \(14\times14\)，每边有：

\[
322/14=23
\]

个 patch，总数为：

\[
N=23\times23=529
\]

这里 \(N\) 表示图像块数量。

DINOv2 输出：

```text
局部特征：
[1, 768, 23, 23]

CLS 特征：
[1, 768]
```

其中第一个维度表示 batch 中只有一张图像。官方训练使用 \(224\times224\)，评估使用 \(322\times322\)；后者提供更多局部位置，但不改变最终描述子长度。

## 2.2 SALAD 有三个分支

局部特征分别进入两个分支，CLS 进入第三个分支。

### 分支 A：局部特征投影

每个 768 维 patch 特征经过：

\[
768\rightarrow512\rightarrow128
\]

得到真正用于后续加权汇总的局部特征。

尺寸变化：

```text
[1, 768, 23, 23]
        ↓
局部特征投影
[1, 128, 23, 23]
        ↓
展开空间维度
[1, 128, 529]
```

### 分支 B：分配分数预测

同一个 768 维 patch 特征，还经过另一个网络：

\[
768\rightarrow512\rightarrow64
\]

输出它与 64 个聚合槽位的匹配分数。

尺寸变化：

```text
[1, 768, 23, 23]
        ↓
分数预测
[1, 64, 23, 23]
        ↓
展开空间维度
[1, 64, 529]
```

### 分支 C：CLS 投影

CLS 特征经过：

\[
768\rightarrow512\rightarrow256
\]

得到：

```text
[1, 256]
```

官方局部分支使用 \(1\times1\) 卷积实现逐位置的特征变换，CLS 分支使用线性层；隐藏层包含 ReLU，局部两个分支在训练时还使用 dropout。

**这三个分支的作用不同：**

| 分支 | 回答的问题 |
|---|---|
| 局部特征投影 | 这个区域要向最终描述子贡献什么内容？ |
| 分配分数预测 | 这个区域的内容应该分到哪些槽位？ |
| CLS 投影 | 除了局部汇总，还保留哪些整图信息？ |

## 2.3 完整数据流

```text
图像 [1,3,322,322]
             ↓
          DINOv2
             │
       ┌─────┴─────────────────────────┐
       │                               │
局部特征 [1,768,23,23]          CLS [1,768]
       │                               │
  ┌────┴────────────┐                  │
  │                 │                  │
特征投影          分数预测            CLS 投影
  │                 │                  │
[1,128,529]     [1,64,529]          [1,256]
  │                 │                  │
  │            加入 dustbin            │
  │            [1,65,529]              │
  │                 │                  │
  │              Sinkhorn              │
  │                 │                  │
  │           去掉 dustbin 行          │
  │            [1,64,529]              │
  │                 │                  │
  └────── 加权聚合 ──┘                  │
             │                         │
       [1,128,64]                      │
             │                         │
    每个槽位归一化、展开                │
             │                         │
         [1,8192]                      │
             └──────── 拼接 ───────────┘
                         │
                    [1,8448]
                         │
                    整体归一化
```

因此：

\[
\boxed{
8448=64\times128+256
}
\]

接下来，把它缩小到可以手算的规模。

---

# 三、固定一个小例子：4 个 patch，2 个聚合槽位

我们假设 DINOv2 已经处理完图像，输出了 4 个局部 token。

为了手算，每个 token 只保留 3 维：

| 区域 | 内容标签，仅用于描述图片 | 教学 token |
|---|---|---|
| patch 1 | 建筑立面 | \(t_1=[1,0,0]^T\) |
| patch 2 | 入口结构 | \(t_2=[0,1,0]^T\) |
| patch 3 | 天空 | \(t_3=[0,0,0.2]^T\) |
| patch 4 | 公交车 | \(t_4=[0,0,1]^T\) |

这些维度不是实际 DINOv2 中的“建筑维”“天空维”。这里只是为了构造可以追踪的数字。

把 token 按列排列：

\[
T=
\begin{bmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&0.2&1
\end{bmatrix}
\]

现在我们设：

\[
N=4
\]

表示 4 个 patch；

\[
K=2
\]

表示 2 个普通聚合槽位；

\[
L=2
\]

表示每个局部特征投影后有 2 维。

## 3.1 局部特征分支：这些区域贡献什么内容？

为了简化算术，令各分支第一层为单位映射，且输入非负，所以 ReLU 不改变数值；下面直接展示第二层的投影矩阵，偏置取 0。

局部特征投影矩阵取：

\[
W_f=
\begin{bmatrix}
2&0&1\\
0&2&1
\end{bmatrix}
\]

因此：

\[
F=W_fT
\]

得到：

\[
\boxed{
F=
\begin{bmatrix}
2&0&0.2&1\\
0&2&0.2&1
\end{bmatrix}
}
\]

也就是：

\[
f_1=[2,0]^T
\]

\[
f_2=[0,2]^T
\]

\[
f_3=[0.2,0.2]^T
\]

\[
f_4=[1,1]^T
\]

这些就是之后要进行加权求和的内容。

## 3.2 分数分支：这些内容应该分给哪个槽位？

分数投影矩阵取：

\[
W_s=
\begin{bmatrix}
\ln8&0&0\\
0&\ln8&0
\end{bmatrix}
\]

其中：

\[
\ln8\approx2.0794
\]

于是：

\[
S=W_sT
\]

得到：

\[
\boxed{
S=
\begin{bmatrix}
\ln8&0&0&0\\
0&\ln8&0&0
\end{bmatrix}
}
\]

我们统一采用：

> **行表示聚合槽位，列表示 patch。**

因此：

\[
S_{k,i}
\]

表示“第 \(i\) 个 patch 分给第 \(k\) 个槽位的分数”。

这组数字表达了：

```text
槽位 1 更偏好 patch 1。

槽位 2 更偏好 patch 2。

patch 3、patch 4 对两个普通槽位都没有特别高的匹配分数。
```

但现在的 \(S\) **还不是分配权重，也不是概率**。

---

# 四、所谓“聚合槽位”，究竟是什么？

SALAD 论文称它们为 clusters，可以翻译为“簇”。为了避免把它直接理解成 K-means 聚类中心，这里主要使用“聚合槽位”。

**一个槽位，就是最终描述子中的一块固定输出空间。**

例如：

```text
槽位 1：最后输出一个 128 维向量。

槽位 2：最后输出另一个 128 维向量。

……

槽位 64：最后输出第 64 个 128 维向量。
```

它们不是：

```text
第一个槽位一定表示建筑。

第二个槽位一定表示树木。

第三个槽位一定表示汽车。
```

也不是每输入一张图，都重新做一次 K-means。

同一个模型中的所有图像，共用同一套分数预测网络，因此槽位编号具有统一的输出约定。具体各槽位学会偏好什么内容，由训练决定；SALAD 不要求使用显式的 K-means 中心，也不在聚合时减去中心计算残差。

## 4.1 为什么不直接对每个 patch 做 softmax？

对前面的分数矩阵，沿槽位方向做 softmax，可以得到：

\[
A=
\begin{bmatrix}
0.8889&0.1111&0.5&0.5\\
0.1111&0.8889&0.5&0.5
\end{bmatrix}
\]

例如 patch 1：

\[
\frac{[e^{\ln8},e^0]}{e^{\ln8}+e^0}
=
\frac{[8,1]}9
=
[0.8889,0.1111]
\]

这种计算保证：

\[
\sum_{k=1}^{K}A_{k,i}=1
\]

也就是：

> 每个 patch 的权重，全部分给普通槽位。

但它没有控制：

> 一个槽位从整张图中，最终接收多少总权重？

如果所有 patch 都偏好同一个槽位，大量权重就可能集中到那里。并且，即使某个 patch 对地点识别没什么帮助，它也必须完整地分给普通槽位。

SALAD 因而把问题改成：

> **既控制每个 patch 如何分配，也控制每个槽位接收多少，同时增加一个允许弃用信息的出口。**

这就是最优传输在这里的作用。

---

# 五、最优传输在这里“运输”的究竟是什么？

不是运输图像，不是移动像素，也不是估计物体运动。

这里运输的是：

\[
\boxed{\text{分配权重，也叫质量}}
\]

你可以把每个 patch 想象成拥有 1 单位的“贡献额度”。

这些额度需要分配到：

```text
普通槽位 1

普通槽位 2

dustbin
```

## 5.1 加入 dustbin

在分数矩阵下面添加一行：

\[
\bar S=
\begin{bmatrix}
\ln8&0&0&0\\
0&\ln8&0&0\\
0&0&0&0
\end{bmatrix}
\]

最后一行对应 dustbin。

教学中把它的分数取为 0。官方代码把 dustbin 分数表示为一个共享的可学习标量，对同一张图的所有 patch 使用相同数值。它不是一个为每个 patch 单独输出“无用概率”的分类器。[GitHub](https://raw.githubusercontent.com/serizba/salad/main/models/aggregators/salad.py)

## 5.2 第一类约束：每个 patch 一共分配 1 单位

用：

\[
\bar P\in\mathbb R^{(K+1)\times N}
\]

表示包含 dustbin 的分配矩阵。

要求每列之和为 1：

\[
\boxed{
\sum_{k=1}^{K+1}\bar P_{k,i}=1
}
\]

例如 patch 1 可以这样分：

```text
0.6 给槽位 1
0.1 给槽位 2
0.3 给 dustbin
```

总和：

\[
0.6+0.1+0.3=1
\]

## 5.3 第二类约束：每个普通槽位接收 1 单位

SALAD 的目标边缘约束是：

\[
\boxed{
\sum_{i=1}^{N}\bar P_{k,i}=1,\qquad k=1,\ldots,K
}
\]

也就是说，每个普通槽位的总接收量都是 1。

剩余质量进入 dustbin：

\[
\boxed{
\sum_{i=1}^{N}\bar P_{K+1,i}=N-K
}
\]

这组约束对应原论文的最优传输定义。[arXiv](https://arxiv.org/html/2311.15937v2)

对于我们的例子：

\[
N=4,\qquad K=2
\]

因此目标是：

| 接收位置 | 总接收量 |
|---|---:|
| 普通槽位 1 | 1 |
| 普通槽位 2 | 1 |
| dustbin | \(4-2=2\) |

输入总量：

\[
4
\]

输出总量：

\[
1+1+2=4
\]

两边守恒。

### 特别重要：这不等于“只保留两个 patch”

一个槽位可以接收：

```text
patch 1 的 0.6
patch 2 的 0.1
patch 3 的 0.1
patch 4 的 0.2
```

总和仍然是 1。

所以：

> **每个普通槽位接收 1 单位权重，不等于每个槽位只能选择 1 个 patch。**

对于真实的：

\[
N=529,\qquad K=64
\]

目标普通槽位总质量为 64，dustbin 总质量为：

\[
529-64=465
\]

这也不能解释成“硬删除 465 个 patch”。它是连续权重的分配，不是离散删除操作。

---

# 六、为什么用 Sinkhorn？它具体怎样算？

## 6.1 需要同时满足“匹配好”和“分配守恒”

一个合理的分配结果，应该优先把权重给分数高的组合：

\[
\sum_{k,i}\bar P_{k,i}\bar S_{k,i}
\]

越大越好。

但只追求这个目标，分配可能过于尖锐。因此加入熵正则，使分配保持一定平滑性：

\[
\boxed{
\max_{\bar P}
\left[
\sum_{k,i}\bar P_{k,i}\bar S_{k,i}
+
\varepsilon H(\bar P)
\right]
}
\]

同时满足前面的行、列总量约束。

其中：

\[
H(\bar P)
=
-\sum_{k,i}\bar P_{k,i}\log\bar P_{k,i}
\]

是熵项；

\[
\varepsilon>0
\]

控制平滑程度。

这里的 \(\bar S\) 是**分数，越大越好**。如果改写成“成本最小化”，则需要使用相应的负分数，不能把分数与成本的符号混在一起。

对这种熵正则问题，最优解可以写成：

\[
\bar P
=
\operatorname{diag}(u)
\exp(\bar S/\varepsilon)
\operatorname{diag}(v)
\]

其中 \(u\) 是逐行缩放系数，\(v\) 是逐列缩放系数。

因此，关键就变成：

> 找到适当的行、列缩放，使矩阵同时满足两类总量约束。

这就是 Sinkhorn 迭代的计算内容；SALAD 官方实现使用对数域运算，避免直接指数计算带来的数值问题。[GitHub](https://raw.githubusercontent.com/serizba/salad/main/models/aggregators/salad.py)

---

## 6.2 用前面的数字算第一轮

为便于手算，取：

\[
\varepsilon=1
\]

先对分数取指数：

\[
Q^{(0)}
=
\exp(\bar S)
=
\begin{bmatrix}
8&1&1&1\\
1&8&1&1\\
1&1&1&1
\end{bmatrix}
\]

注意：这个矩阵只是迭代起点，还不满足分配约束。

### 第一步：调整每一行的总量

前两行的和都是：

\[
8+1+1+1=11
\]

而目标是 1，因此分别除以 11。

最后一行的和是 4，目标是 2，因此乘以：

\[
2/4=0.5
\]

得到：

\[
Q^{(\mathrm{row})}
=
\begin{bmatrix}
0.7273&0.0909&0.0909&0.0909\\
0.0909&0.7273&0.0909&0.0909\\
0.5&0.5&0.5&0.5
\end{bmatrix}
\]

现在每行总量正确：

\[
[1,1,2]
\]

但是每列总量还不对。

### 第二步：调整每一列的总量

第一列、第二列之和都是：

\[
0.7273+0.0909+0.5
\approx1.3182
\]

第三列、第四列之和都是：

\[
0.0909+0.0909+0.5
\approx0.6818
\]

把每列分别除以自己的列和，得到：

\[
Q^{(1)}
=
\begin{bmatrix}
0.5517&0.0690&0.1333&0.1333\\
0.0690&0.5517&0.1333&0.1333\\
0.3793&0.3793&0.7333&0.7333
\end{bmatrix}
\]

现在每列之和都是 1。

但行和变成了：

\[
[0.8874,\ 0.8874,\ 2.2253]
\]

行约束又有些偏离。

所以继续：

```text
调整行总量
    ↓
调整列总量
    ↓
再次调整行总量
    ↓
再次调整列总量
    ↓
……
```

**每一步只调整一类约束，但通过反复迭代，使两类约束逐渐同时满足。**

---

## 6.3 三轮之后得到什么？

官方 SALAD 前向使用 3 轮 Sinkhorn 迭代。[GitHub](https://raw.githubusercontent.com/serizba/salad/main/models/aggregators/salad.py)

对我们的例子，3 轮后：

\[
\boxed{
\bar P\approx
\begin{bmatrix}
0.603276&0.075410&0.159721&0.159721\\
0.075410&0.603276&0.159721&0.159721\\
0.321314&0.321314&0.680558&0.680558
\end{bmatrix}
}
\]

对应关系为：

| 分配位置 | 建筑立面 | 入口结构 | 天空 | 公交车 |
|---|---:|---:|---:|---:|
| 槽位 1 | 0.6033 | 0.0754 | 0.1597 | 0.1597 |
| 槽位 2 | 0.0754 | 0.6033 | 0.1597 | 0.1597 |
| dustbin | 0.3213 | 0.3213 | 0.6806 | 0.6806 |

这时每列之和为 1，行和约为：

\[
[0.998128,\ 0.998128,\ 2.003744]
\]

已经接近目标：

\[
[1,1,2]
\]

但并非完全精确。

继续迭代至收敛，会得到约：

\[
\bar P_\infty\approx
\begin{bmatrix}
0.604109&0.075514&0.160189&0.160189\\
0.075514&0.604109&0.160189&0.160189\\
0.320377&0.320377&0.679623&0.679623
\end{bmatrix}
\]

因此，讲真实实现时要区分：

> **数学上的目标约束，与有限次迭代得到的近似结果。**

---

## 6.4 为什么相同的 dustbin 分数，却能产生不同的弃用权重？

你可能会问：

> dustbin 对所有 patch 的输入分数都是 0，为什么天空获得了更多 dustbin 权重？

因为最终分配不只取决于 dustbin 自己，而是取决于整个分数矩阵和容量竞争。

在这个例子中：

```text
建筑立面：
有一个普通槽位明显偏好它。

入口结构：
也有一个普通槽位明显偏好它。

天空、公交车：
没有得到普通槽位的高匹配分数。
```

普通槽位的接收量又有限，因此后两者有更多权重进入 dustbin。

不过，这不代表 SALAD 天生知道“天空和车必须丢弃”。

**这些偏好来自学习后的分数网络。我们在手算中直接设定了权重，只是为了展示已经形成这种偏好时，分配怎样发生。**

如果所有普通槽位对所有 patch 的分数完全一样，满足约束的平滑分配会趋向：

\[
\bar P_{k,i}=\frac1N
\]

对每个普通槽位都一样。

此时各槽位可能汇总出相同的内容。

所以：

> **Sinkhorn 提供分配规则，但不会凭空创造有辨别力的视觉知识。**

---

# 七、分配矩阵算出来之后，怎样变成最终描述子？

现在进入真正的“聚合”。

## 7.1 去掉 dustbin 行

从：

\[
\bar P\in\mathbb R^{3\times4}
\]

去掉最后一行，保留：

\[
P=
\begin{bmatrix}
0.603276&0.075410&0.159721&0.159721\\
0.075410&0.603276&0.159721&0.159721
\end{bmatrix}
\]

此时，**不能再把每一列重新归一化到 1**。

否则，原本分给 dustbin 的权重会被重新补回普通槽位，破坏弃用机制。官方实现是直接去掉 dustbin，然后进行加权汇总。[GitHub](https://raw.githubusercontent.com/serizba/salad/main/models/aggregators/salad.py)

例如，天空进入普通槽位的总权重只有：

\[
0.159721+0.159721
=
0.319442
\]

而建筑立面进入普通槽位的总权重为：

\[
0.603276+0.075410
=
0.678686
\]

---

## 7.2 每个槽位分别加权求和

前面得到的局部特征矩阵是：

\[
F=
\begin{bmatrix}
2&0&0.2&1\\
0&2&0.2&1
\end{bmatrix}
\]

聚合可以直接写成：

\[
\boxed{
V=FP^T
}
\]

其中：

\[
V\in\mathbb R^{L\times K}
\]

每一列是一个槽位的汇总特征。

### 槽位 1

\[
v_1=
0.603276f_1+
0.075410f_2+
0.159721f_3+
0.159721f_4
\]

代入：

\[
\begin{aligned}
v_1={}&
0.603276
\begin{bmatrix}2\\0\end{bmatrix}
+
0.075410
\begin{bmatrix}0\\2\end{bmatrix}\\
&+
0.159721
\begin{bmatrix}0.2\\0.2\end{bmatrix}
+
0.159721
\begin{bmatrix}1\\1\end{bmatrix}
\end{aligned}
\]

得到：

\[
v_1\approx
\begin{bmatrix}
1.398218\\
0.342484
\end{bmatrix}
\]

### 槽位 2

同理：

\[
v_2\approx
\begin{bmatrix}
0.342484\\
1.398218
\end{bmatrix}
\]

因此：

\[
\boxed{
V\approx
\begin{bmatrix}
1.398218&0.342484\\
0.342484&1.398218
\end{bmatrix}
}
\]

你可以看到，两个槽位没有把所有信息混成同一个平均值。

```text
槽位 1 的结果更受建筑立面影响。

槽位 2 的结果更受入口结构影响。
```

与此同时，天空和公交车的贡献减弱了，但没有完全消失。

---

## 7.3 为什么要对每个槽位分别归一化？

定义：

\[
\hat v_k=\frac{v_k}{\|v_k\|_2}
\]

这里 \(\|v_k\|_2\) 是向量长度。

我们的两个槽位长度都是约：

\[
1.439552
\]

因此：

\[
\hat v_1\approx
\begin{bmatrix}
0.971287\\
0.237910
\end{bmatrix}
\]

\[
\hat v_2\approx
\begin{bmatrix}
0.237910\\
0.971287
\end{bmatrix}
\]

单独归一化的效果是：

> **避免某个槽位仅仅因为特征数值幅度大，就压过其他槽位。**

它更强调每个槽位描述的方向，而不是原始幅值。

官方实现对每个槽位沿特征通道维进行 L2 归一化。[GitHub](https://raw.githubusercontent.com/serizba/salad/main/models/aggregators/salad.py)

---

## 7.4 再加入 CLS 分支

假设教学中的 CLS token 是：

\[
t_{\mathrm{CLS}}=[1,1,1]^T
\]

CLS 投影矩阵取：

\[
W_g=
\begin{bmatrix}
1&1&1\\
1&1&2
\end{bmatrix}
\]

因此：

\[
g=W_gt_{\mathrm{CLS}}
=
\begin{bmatrix}
3\\4
\end{bmatrix}
\]

归一化后：

\[
\hat g=
\begin{bmatrix}
0.6\\0.8
\end{bmatrix}
\]

这条分支保留一份整图信息，补充局部槽位聚合。它不是拿 CLS 替代局部特征，而是把两者结合。

---

## 7.5 拼接并整体归一化

按照官方实现的通道优先展开顺序，先把：

\[
\hat V=
\begin{bmatrix}
0.971287&0.237910\\
0.237910&0.971287
\end{bmatrix}
\]

展开，再与 \(\hat g\) 拼接：

\[
h=
[
0.6,\ 0.8,\
0.971287,\ 0.237910,\
0.237910,\ 0.971287
]
\]

最后：

\[
y=\frac{h}{\|h\|_2}
\]

因为 CLS 块和两个槽位块都已经归一化，所以：

\[
\|h\|_2=\sqrt3
\]

得到：

\[
\boxed{
y\approx
[
0.346410,\ 0.461880,\
0.560773,\ 0.137358,\
0.137358,\ 0.560773
]
}
\]

这就是我们这个小模型的**最终整图描述子**。

真实模型只是把：

```text
2 个槽位，每个 2 维，加 2 维 CLS
```

扩大为：

```text
64 个槽位，每个 128 维，加 256 维 CLS
```

计算逻辑相同。

---

# 八、这条描述子怎样用于地点识别？

数据库中的每张参考图像，都独立生成一条描述子：

\[
y_1,y_2,\ldots,y_M
\]

这里 \(M\) 是数据库图像数量。

当前查询图像生成：

\[
y_q
\]

因为这些描述子已经归一化，可以直接计算：

\[
s_j=y_q^Ty_j
\]

作为余弦相似度。

然后按 \(s_j\) 从大到小排序，返回候选参考图像。官方评估也是先提取参考图和查询图的描述子，再进行检索。[GitHub](https://raw.githubusercontent.com/serizba/salad/main/eval.py)

整个流程是：

```text
数据库图像
    ↓
DINOv2 + SALAD
    ↓
提前保存描述子

当前图像
    ↓
DINOv2 + SALAD
    ↓
查询描述子
    ↓
与数据库描述子比较
    ↓
返回相似地点候选
```

**最优传输发生在每张图像内部，不是在查询图与数据库图之间运行一次。**

这是 SALAD 很容易被误解的地方。

## 8.1 为什么它能容忍部分位置变化？

从聚合公式出发：

\[
V=FP^T
\]

假设局部内容仅仅发生了顺序重排，并且分数也跟随相同的内容重排，那么最终求和不会变化。

例如，把：

```text
[建筑立面, 入口, 天空, 公交车]
```

改为：

```text
[天空, 建筑立面, 公交车, 入口]
```

只要每个局部特征及其分配关系不变，聚合结果就不变。

这说明：

> **SALAD 不要求“同一个内容必须始终出现在相同的 patch 编号”。**

但这只是聚合层的性质。真实图像改变视角时，DINOv2 的位置编码、上下文和局部外观都会改变，因此不能据此推出“任意视角变化下描述子完全相同”。

---

# 九、最关键的问题：它怎么学会建筑重要、天空不重要？

到现在为止，我们只是解释了一个已经有合适权重的 SALAD 如何前向计算。

真正训练时，网络并没有我们手工设定的：

\[
W_f,\quad W_s,\quad W_g
\]

也没有人工给每个 patch 标注：

```text
你应该进槽位 1。

你应该进槽位 2。

你应该进 dustbin。
```

SALAD 使用的是**整张图像的地点关系监督**。

## 9.1 训练样本是什么？

例如：

```text
地点 A：
A1：晴天
A2：阴天
A3：有车辆遮挡
A4：另一拍摄时刻

地点 B：
B1、B2、B3、B4

……
```

训练要求：

```text
同一地点的描述子更接近。

不同地点的描述子更容易区分。
```

官方配置每个 batch 选取 60 个地点，每个地点 4 张图像，因此总共处理：

\[
60\times4=240
\]

张图像。训练采用 Multi-Similarity Loss 和相应的样本对挖掘方法。[GitHub](https://raw.githubusercontent.com/serizba/salad/main/main.py)

这里的“监督”主要是地点分组，不是像素级语义标注。

---

## 9.2 Multi-Similarity Loss 在推动什么？

先固定一张图像作为锚点，记为 \(a\)。

定义：

\[
s_{ap}=y_a^Ty_p
\]

为它与同地点图像 \(p\) 的相似度；

\[
s_{an}=y_a^Ty_n
\]

为它与不同地点图像 \(n\) 的相似度。

注意，这里的 \(a,p,n\) 都是**图像编号**，不是前面聚合时的 patch 编号。

对于经过样本挖掘保留下来的正样本集合 \(\mathcal P\) 和负样本集合 \(\mathcal N\)，单个锚点的损失可以写成：

\[
\mathcal L_a=
\frac1\alpha
\log
\left(
1+
\sum_{p\in\mathcal P}
e^{-\alpha(s_{ap}-b)}
\right)
+
\frac1\beta
\log
\left(
1+
\sum_{n\in\mathcal N}
e^{\beta(s_{an}-b)}
\right)
\]

其中：

- \(\alpha\) 调节正样本部分的权重变化。
- \(\beta\) 调节负样本部分的权重变化。
- \(b\) 是相似度的平移基准。

SALAD 官方损失配置使用：

\[
\alpha=1,\qquad\beta=50,\qquad b=0
\]

而不是直接照搬这个损失类的默认参数。[GitHub](https://raw.githubusercontent.com/serizba/salad/main/utils/losses.py)

### 用一对正样本和一对负样本计算

假设挖掘后保留：

\[
s_{ap}=0.6
\]

\[
s_{an}=0.7
\]

也就是说，不同地点反而比同地点更相似，当前表示有问题。

正样本损失：

\[
\mathcal L_+
=
\log(1+e^{-0.6})
\approx0.437488
\]

负样本损失：

\[
\mathcal L_-
=
\frac1{50}\log(1+e^{50\times0.7})
\approx0.7
\]

所以：

\[
\mathcal L_a\approx1.137488
\]

对这两个相似度求导：

\[
\frac{\partial\mathcal L_a}{\partial s_{ap}}
\approx-0.354344
\]

\[
\frac{\partial\mathcal L_a}{\partial s_{an}}
\approx1
\]

梯度下降因此会推动：

```text
同地点相似度 0.6：
向上提高。

不同地点相似度 0.7：
向下降低。
```

---

## 9.3 梯度如何传回 SALAD？

数据反向流动为：

```text
地点识别损失
      ↓
最终描述子
      ↓
整体归一化、拼接
      ├──────── CLS 投影分支
      │
      ↓
槽位归一化
      ↓
V = F Pᵀ
      ├──────── 局部内容 F
      │              ↓
      │         特征投影网络
      │
      └──────── 分配权重 P
                     ↓
                  Sinkhorn
                     ↓
                  分数矩阵 S
                     ↓
                 分数预测网络
```

Sinkhorn 中的迭代由可微的数值运算组成，因此可以沿着这条路径传递梯度。

如果车辆变化反复干扰同地点匹配，训练就可能通过调整特征和分配，让模型减少依赖这种不稳定信息；如果建筑细节有助于区分不同地点，训练就可能加强相应信息的贡献。

**这里的“可能”很重要：模型学习的是训练数据中有利于检索的规律，不是内置了一套绝对正确的“静态重要、动态无用”规则。**

## 9.4 DINOv2 本身也会更新吗？

会，但标准配置只微调最后 4 个 Transformer block，前面的 block 冻结；SALAD 的投影和聚合相关参数参与训练。[GitHub](https://raw.githubusercontent.com/serizba/salad/main/models/backbones/dinov2.py)

因此：

```text
DINOv2 的原始自监督预训练
          ↓
得到通用视觉主干
          ↓
加入 SALAD
          ↓
使用地点关系监督
微调主干后 4 层 + SALAD
```

**只训练后 4 层，不等于推理时只运行后 4 层。**完整的主干计算仍然需要执行，只是早期层在这个训练阶段不更新。

---

# 十、把几个容易混淆的关系彻底分开

## 10.1 SALAD 与 NetVLAD 的区别

NetVLAD 的典型聚合形式可以写成：

\[
v_k=\sum_i a_{k,i}(x_i-c_k)
\]

其中 \(x_i\) 是局部特征，\(c_k\) 是第 \(k\) 个中心，\(a_{k,i}\) 是分配权重。

SALAD 的形式是：

\[
v_k=\sum_i P_{k,i}f_i
\]

**它直接聚合投影后的特征，不减去一个显式中心。**

主要区别可以概括为：

| 环节 | NetVLAD 的典型形式 | SALAD |
|---|---|---|
| 分配 | 每个局部特征独立做 softmax | 同时考虑 patch 与槽位总量约束 |
| 弃用信息 | 普通分配没有 dustbin 出口 | 增加 dustbin |
| 聚合内容 | 特征相对中心的残差 | 投影后的特征本身 |
| 整图补充 | 取决于具体扩展 | 标准结构拼接 CLS 投影 |

这些差异是 SALAD 对局部特征分配与聚合所做的主要修改。[arXiv](https://arxiv.org/html/2311.15937v2)

## 10.2 DINOv2 和 SALAD 都有 Sinkhorn，但不是同一件事

在你前面学过的 DINOv2 中，Sinkhorn 可以用于**自监督训练时处理教师的原型分配目标**。

而 SALAD 中：

```text
输入：
一张图像内部的 patch—槽位分数。

输出：
用于聚合局部特征的权重矩阵。
```

SALAD 的 Sinkhorn 是聚合器前向计算的一部分，训练与推理都会执行。它不需要在使用时再运行一套 DINOv2 教师—学生训练过程。[GitHub](https://raw.githubusercontent.com/serizba/salad/main/models/aggregators/salad.py)

## 10.3 dustbin 不等于动态物体分割

从前面的数字已经可以看出：

```text
公交车：
仍有一部分权重进入普通槽位。
```

而且 DINOv2 的 token 已经经过上下文交互，某个区域的信息可能进入其他 token 和 CLS 中。

因此，即使某个 patch 主要进入 dustbin，也不能说：

> 这个区域的信息已经从整张图的表示中彻底删除。

同样，dustbin 也不能直接作为精确的动态物体掩码。

## 10.4 一个更细的数学事实：dustbin 的总弃用量不是模型自由决定的

由约束：

\[
\sum_i\bar P_{K+1,i}=N-K
\]

可知，目标弃用总质量由 \(N\) 和 \(K\) 决定。

甚至，如果仅把 dustbin 整行分数统一增加一个常数 \(z\)，在严格满足边缘约束的目标函数中，它只增加：

\[
z\sum_i\bar P_{K+1,i}
=
z(N-K)
\]

这是一个与具体分配方式无关的常数。

所以，不能把官方的共享 dustbin 标量理解成：

> 一个可以自由决定“今天删掉多少 patch”的阈值。

**真正关键的是普通槽位的相对分数、容量约束，以及由此形成的分配竞争。**这是从该分配公式直接得到的结论。

---

# 十一、用 PyTorch 复现刚才的完整数字

下面不需要下载 DINOv2 权重，只需要 PyTorch。

它复现的是：

```text
教学 token
    ↓
特征投影、分数预测
    ↓
3 轮 Sinkhorn
    ↓
加权聚合
    ↓
CLS 拼接和归一化
```

为了避免指数溢出，Sinkhorn 使用对数域计算。这里直接使用未缩放的目标总量 \([1,1,N-K]\)，与官方先缩放、最后恢复质量的写法，在本例中得到一致结果。

```python
import math

import torch
import torch.nn.functional as F


def sinkhorn_assignment(
    scores: torch.Tensor,
    dustbin_score: float = 0.0,
    steps: int = 3,
    epsilon: float = 1.0,
) -> torch.Tensor:
    """
    单张图像的 SALAD 式软分配。

    输入:
        scores: [K, N]，K 个普通槽位、N 个 patch。

    输出:
        assignment: [K + 1, N]，最后一行为 dustbin。

    目标边缘:
        每个 patch 的列和为 1；
        每个普通槽位的行和为 1；
        dustbin 的行和为 N - K。

    有限次迭代时，行约束通常只近似满足。
    """
    if scores.ndim != 2:
        raise ValueError("scores 必须是 [K, N] 二维张量。")

    k, n = scores.shape
    if not (n > k >= 1):
        raise ValueError("这里要求 N > K >= 1。")
    if steps < 1 or epsilon <= 0:
        raise ValueError("steps 必须为正整数，epsilon 必须大于 0。")
    if not scores.is_floating_point():
        raise TypeError("scores 必须是浮点张量。")
    if not torch.isfinite(scores).all():
        raise ValueError("scores 不能包含 NaN 或无穷大。")

    dustbin = torch.full(
        (1, n),
        dustbin_score,
        dtype=scores.dtype,
        device=scores.device,
    )

    # log(exp(score / epsilon)) 就是 score / epsilon。
    log_p = torch.cat([scores, dustbin], dim=0) / epsilon

    row_mass = torch.ones(
        k + 1, dtype=scores.dtype, device=scores.device
    )
    row_mass[-1] = n - k

    for _ in range(steps):
        # 将每一行调整到目标质量。
        log_p = (
            log_p
            + row_mass.log()[:, None]
            - torch.logsumexp(log_p, dim=1, keepdim=True)
        )

        # 将每一列调整到 1；log(1) = 0。
        log_p = (
            log_p
            - torch.logsumexp(log_p, dim=0, keepdim=True)
        )

    return log_p.exp()


torch.set_printoptions(precision=6, sci_mode=False)
dtype = torch.float64

# 每一列是一个 patch token。
tokens = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.2, 1.0],
], dtype=dtype)

feature_weight = torch.tensor([
    [2.0, 0.0, 1.0],
    [0.0, 2.0, 1.0],
], dtype=dtype)

score_weight = torch.tensor([
    [math.log(8.0), 0.0, 0.0],
    [0.0, math.log(8.0), 0.0],
], dtype=dtype)

# 两个局部分支。
local_features = feature_weight @ tokens   # [2, 4]
scores = score_weight @ tokens             # [2, 4]

# 包含 dustbin 的分配矩阵。
assignment = sinkhorn_assignment(scores, steps=3)

# 去掉 dustbin，不再重新归一化每个 patch。
p = assignment[:-1]                        # [2, 4]

# 每列对应一个聚合槽位。
pooled = local_features @ p.T              # [2, 2]
pooled_unit = F.normalize(pooled, p=2, dim=0)

# CLS 分支。
cls_token = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
global_weight = torch.tensor([
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 2.0],
], dtype=dtype)

global_feature = global_weight @ cls_token
global_unit = F.normalize(global_feature, p=2, dim=0)

# 与官方 [特征通道, 槽位] 的展开顺序一致。
descriptor = torch.cat([
    global_unit,
    pooled_unit.reshape(-1),
])
descriptor = F.normalize(descriptor, p=2, dim=0)

print("局部特征 F:\n", local_features)
print("\n分数 S:\n", scores)
print("\n分配矩阵（最后一行是 dustbin）:\n", assignment)
print("\n行和:\n", assignment.sum(dim=1))
print("\n列和:\n", assignment.sum(dim=0))
print("\n聚合结果 V:\n", pooled)
print("\n最终描述子:\n", descriptor)
print("\n描述子长度:\n", descriptor.norm())
```

最终输出应接近：

```text
分配矩阵：
[[0.603276, 0.075410, 0.159721, 0.159721],
 [0.075410, 0.603276, 0.159721, 0.159721],
 [0.321314, 0.321314, 0.680558, 0.680558]]

聚合结果：
[[1.398218, 0.342484],
 [0.342484, 1.398218]]

最终描述子：
[0.346410, 0.461880, 0.560773,
 0.137358, 0.137358, 0.560773]

描述子长度：
1
```

---

# 十二、真实模型中，怎样观察这几个分支？

安装官方仓库要求的依赖后，可以通过 PyTorch Hub 加载预训练模型。下面的预处理沿用官方评估代码的缩放和标准化方式；首次加载需要联网下载模型。[GitHub](https://raw.githubusercontent.com/serizba/salad/main/README.md)

```python
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


image_path = Path("building.jpg")
if not image_path.is_file():
    raise FileNotFoundError(f"找不到图片：{image_path.resolve()}")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = torch.hub.load(
    "serizba/salad",
    "dinov2_salad",
    trust_repo=True,
).to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize(
        (322, 322),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

with Image.open(image_path) as image:
    x = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

with torch.inference_mode():
    patch_map, cls_token = model.backbone(x)
    head = model.aggregator

    local_features = head.cluster_features(patch_map).flatten(2)
    scores = head.score(patch_map).flatten(2)
    global_feature = head.token_features(cls_token)

    descriptor = head((patch_map, cls_token))

print("DINOv2 局部输出：", tuple(patch_map.shape))
print("DINOv2 CLS：", tuple(cls_token.shape))
print("投影后的局部特征：", tuple(local_features.shape))
print("普通槽位分数：", tuple(scores.shape))
print("投影后的 CLS：", tuple(global_feature.shape))
print("最终描述子：", tuple(descriptor.shape))
print("最终描述子范数：", descriptor.norm(dim=-1).cpu())
```

对应形状为：

```text
DINOv2 局部输出：   (1, 768, 23, 23)
DINOv2 CLS：        (1, 768)
投影后的局部特征： (1, 128, 529)
普通槽位分数：     (1, 64, 529)
投影后的 CLS：     (1, 256)
最终描述子：       (1, 8448)
```

这些维度对应官方模型结构；上面的真实图片推理示例没有在本次对话中实际运行权重，具体特征值需要由输入图片计算得到。[GitHub](https://raw.githubusercontent.com/serizba/salad/main/models/aggregators/salad.py)

---

# 最后，把 SALAD 放回你的理解体系

从你熟悉的 SLAM 角度，一条合理的集成路径是：

```text
当前图像
    ↓
DINOv2-SALAD 整图描述子
    ↓
检索历史关键帧
    ↓
得到候选回环
    ↓
局部匹配与几何验证
    ↓
确认回环约束
```

这是根据它的输入输出设计出的使用方式。**SALAD 输出的是“用于相似地点检索的向量”，不是两帧之间的旋转、平移，也不是已经通过几何验证的回环。**

最值得记住的核心公式只有一组：

\[
\boxed{
\begin{aligned}
F&=\text{局部内容投影}(T)\\
S&=\text{槽位分数预测}(T)\\
P&=\text{Sinkhorn}(S+\text{dustbin})\text{，再去掉 dustbin}\\
V&=FP^T\\
y&=\operatorname{Normalize}
\left(
[\hat g;\operatorname{Flatten}(\hat V)]
\right)
\end{aligned}
}
\]

其中，**\(F\) 决定“汇总什么”，\(P\) 决定“怎样分配”，地点识别损失决定“哪些内容和分配方式值得学习”。**

这三层联系起来，才是 SALAD 的完整含义。