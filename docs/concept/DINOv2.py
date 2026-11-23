import copy

import numpy as np


# 从同一张原图取三份视图，教师和学生复用这些像素。
IMAGE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
G1 = IMAGE[:2, :2]
G2 = IMAGE[1:, 1:]
LOCAL = IMAGE[1:2, 1:2]
MASK_INDEX = 3


def softmax(scores, temperature=1.0):
    """把一行分数转成比例，减去最大值以避免指数溢出。"""
    scaled = np.asarray(scores, dtype=float) / temperature
    weights = np.exp(scaled - scaled.max())
    return weights / weights.sum()


def backbone(crop, scale, mask_index=None):
    """做一次均匀注意力和残差相加，同时算出各行特征对 w 的导数。"""
    values = np.asarray(crop, dtype=float).reshape(-1)
    # 输入为 [w × x, 1]，两维对 w 的导数分别是 x 和 0。
    tokens = np.column_stack([scale * values, np.ones_like(values)])
    derivatives = np.column_stack([values, np.zeros_like(values)])
    if mask_index is not None:
        tokens[mask_index] = [0.0, 0.0]
        derivatives[mask_index] = [0.0, 0.0]

    # 最前面放 CLS。每行加上所有行的平均值，导数也按同一规则相加。
    tokens = np.vstack([np.zeros(2), tokens])
    derivatives = np.vstack([np.zeros(2), derivatives])
    output = tokens + tokens.mean(axis=0)
    return output, derivatives + derivatives.mean(axis=0)


def student_features(student):
    """取学生三个裁剪的 CLS，以及遮挡位置 p3 的输出与导数。"""
    g1, dg1 = backbone(G1, student["w"], mask_index=MASK_INDEX)
    g2, dg2 = backbone(G2, student["w"])
    local, dlocal = backbone(LOCAL, student["w"])
    # 第 0 行留给 CLS。图像块编号从 0 开始，取 p3 的输出要加 1。
    row = MASK_INDEX + 1
    return {
        "G1": (g1[0], dg1[0]),
        "G2": (g2[0], dg2[0]),
        "L": (local[0], dlocal[0]),
        "P3": (g1[row], dg1[row]),
    }


def make_targets(teacher):
    """教师看完整的甲和乙，用两个训练头算出本轮固定的目标。"""
    g1, _ = backbone(G1, teacher["w"])
    g2, _ = backbone(G2, teacher["w"])
    return {
        "G1": softmax(teacher["dino"] @ g1[0]),
        "G2": softmax(teacher["dino"] @ g2[0]),
        "P3": softmax(teacher["ibot"] @ g1[MASK_INDEX + 1]),
    }


def loss_and_gradients(student, targets):
    """按五组配对汇总损失，逐项累加两个头和主干的解析梯度。"""
    features = student_features(student)
    losses = {"dino": 0.0, "ibot": 0.0}
    gradients = {
        "w": 0.0,
        "dino": np.zeros_like(student["dino"]),
        "ibot": np.zeros_like(student["ibot"]),
    }
    # 依次写出训练头、学生的行、教师的行和本项在总损失里的系数。
    pairs = [
        ("dino", "G1", "G2", 0.25),
        ("dino", "G2", "G1", 0.25),
        ("dino", "L", "G1", 0.25),
        ("dino", "L", "G2", 0.25),
        ("ibot", "P3", "P3", 0.5),
    ]
    comparisons = []
    for head, source, target_name, weight in pairs:
        h, dh_dw = features[source]
        prediction = softmax(student[head] @ h)
        target = targets[target_name]
        ce = -float(target @ np.log(prediction))
        losses[head] += weight * ce
        comparisons.append((head, source, target_name, ce))

        # 每个分数的梯度乘两个输入特征，得到该行两个权重的梯度。
        grad_scores = weight * (prediction - target)
        gradients[head] += np.outer(grad_scores, h)
        # 再经过旧头的权重传回特征，沿特征对 w 的导数传回主干。
        grad_feature = student[head].T @ grad_scores
        gradients["w"] += float(grad_feature @ dh_dw)

    return losses, gradients, comparisons


def main():
    """固定教师目标，更新学生并复算损失，最后对教师参数做 EMA。"""
    initial_head = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
    student = {"w": 1.0, "dino": initial_head.copy(), "ibot": initial_head.copy()}
    teacher = copy.deepcopy(student)

    targets = make_targets(teacher)
    losses, gradients, comparisons = loss_and_gradients(student, targets)
    for head, source, target_name, ce in comparisons:
        print(f"{head} 学生 {source} 配教师 {target_name} 的交叉熵 = {ce:.9f}")
    print(f'DINO 损失 = {losses["dino"]:.9f}')
    print(f'iBOT 损失 = {losses["ibot"]:.9f}')
    print(f"总损失 = {sum(losses.values()):.9f}")
    print(f'w 的总梯度 = {gradients["w"]:.9f}')

    # 所有参数都用同一组旧梯度更新，复算时继续使用上面的教师目标。
    learning_rate = 0.5
    updated = {
        name: value - learning_rate * gradients[name]
        for name, value in student.items()
    }
    new_losses, _, _ = loss_and_gradients(updated, targets)
    p3, _ = student_features(updated)["P3"]
    prediction = softmax(updated["ibot"] @ p3)
    print(f'新学生 w = {updated["w"]:.9f}')
    print(f"新学生 p3 分布 = {np.round(prediction, 6)}")
    print(f"更新学生后的总损失 = {sum(new_losses.values()):.9f}")

    # 分别对主干与两个训练头的对应参数做指数移动平均。
    teacher = {
        name: 0.99 * value + 0.01 * updated[name]
        for name, value in teacher.items()
    }
    print(f'新教师 w = {teacher["w"]:.9f}')


if __name__ == "__main__":
    main()