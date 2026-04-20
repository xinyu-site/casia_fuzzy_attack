# 3. 单 Agent 局部扰动与协同传播实验

## 3.1 实验目的

本实验作为整体 Trans 论文实验部分中的一个小节，用于验证 EGNN MARL 在单个智能体受到局部观测扰动时，是否能够减弱局部误差向团队协同的传播。与前面的结构化观测扰动实验不同，本实验不再关注所有 agent 同时受到扰动时的整体鲁棒性，而是专门分析“只扰动一个 agent，会不会影响其他未被扰动 agent 的动作和团队表现”。

该实验希望支撑一个简洁结论：EGNN 的几何关系建模不仅能提升扰动下的任务性能，也能降低单点观测误差对团队协同的放大效应。

## 3.2 实验设置

### 3.2.1 局部扰动设定

每次测试只选择一个 agent 作为 attacked agent，并只修改该 agent 的局部观测。其他 agent 的观测、环境真实状态、动作执行器和奖励函数保持不变。

被扰动的局部观测可以包括：

- attacked agent 看到的邻居相对位置；
- attacked agent 看到的目标相对位置；
- attacked agent 的局部坐标系方向；
- attacked agent 的局部速度、距离或角度特征。

这个设定可以区分直接观测扰动和协同传播效应：如果未被攻击 agent 的动作也发生明显变化，则说明局部误差已经通过策略耦合、图消息传递或联合决策机制传播到团队层面。

### 3.2.2 扰动类型

建议选择 2-3 类局部扰动即可，不需要全部展开：

- **Single-agent geometric noise**：只对 attacked agent 的局部几何观测加入高斯噪声。
- **Local relative position corruption**：只扰动 attacked agent 看到的邻居相对位置。
- **Local coordinate rotation**：只旋转 attacked agent 的局部坐标系。

其中，最小可执行版本可以只做 `single-agent relative position noise`。

### 3.2.3 扰动强度

建议使用三档扰动强度：

- Mild
- Medium
- Severe

如果需要连续曲线，可以设置：

- $\sigma \in \{0.01, 0.03, 0.05, 0.1\}$，用于局部高斯噪声；
- $\theta \in \{10^\circ, 20^\circ, 45^\circ, 90^\circ\}$，用于局部坐标旋转。

### 3.2.4 被攻击 Agent 选择

主实验中可以每个 episode 随机选择一个 attacked agent。若篇幅允许，可以补充一个简单对照：比较 random agent 和 central agent 两种情况，用于说明攻击关键节点是否更容易造成团队退化。

## 3.3 对比方法

本实验沿用整体实验方案中的主要方法，不额外扩展过多新 baseline：

- MLP / RNN MARL
- GraphSAGE / 普通 GNN MARL
- EGNN MARL
- EGNN MARL + symmetry consistency regularization

如果空间有限，表格中只保留这 4 类方法即可。额外鲁棒训练方法可以放到通用观测扰动对照实验中统一讨论。

## 3.4 评价指标

本实验只保留能直接回答“局部扰动是否传播”的关键指标。

### 3.4.1 团队性能下降

报告扰动后的团队任务性能，例如 Episodic Return、Success Rate 和 Performance Drop：

$$
\text{Drop}=\frac{J_{\text{clean}}-J_{\text{attack}}}{|J_{\text{clean}}|+\epsilon}.
$$

其中，$J_{\text{clean}}$ 表示无扰动测试性能，$J_{\text{attack}}$ 表示只扰动一个 agent 后的测试性能。

### 3.4.2 Non-attacked Agents Action Shift

该指标衡量未被攻击 agent 是否被局部扰动带偏：

$$
D_{\text{others}}=
\frac{1}{N-1}\sum_{j\neq i}
\left\|\pi_j(\tilde{o}^{(i)})-\pi_j(o)\right\|.
$$

其中，$i$ 是 attacked agent，$\tilde{o}^{(i)}$ 表示只扰动第 $i$ 个 agent 后的联合观测。该指标越低，说明局部扰动越不容易传播到其他 agent。

### 3.4.3 Team Action Divergence

该指标衡量团队整体动作偏移：

$$
D_{\text{team}}=
\frac{1}{N}\sum_{j=1}^{N}
\left\|\pi_j(\tilde{o}^{(i)})-\pi_j(o)\right\|.
$$

如果 EGNN 能稳定团队协同，则在相同攻击强度下应具有更低的 $D_{\text{team}}$。


## 3.5 输出图表

建议本小节最多放 2 张图和 1 张表。

### 图 1：扰动强度 vs. 团队性能

- 横轴：扰动强度。
- 纵轴：Episodic Return、Success Rate 或 Performance Drop。
- 曲线：不同方法。

该图用于展示单 agent 局部扰动下，EGNN MARL 的团队性能是否下降更慢。

### 图 2：扰动强度 vs. 协同传播指标

- 横轴：扰动强度。
- 纵轴：$D_{\text{others}}$ 或 $D_{\text{team}}$。
- 曲线：不同方法。

该图用于展示 EGNN 是否能降低局部扰动对未攻击 agent 和团队动作的影响。

### 表 1：单 Agent 局部扰动结果汇总

表格列建议包括：

- Method
- Clean Return
- Attack Return
- Performance Drop (%)
- $D_{\text{others}}$
- $D_{\text{team}}$

该表同时汇总任务性能和协同传播指标，足以支撑本小节的主要结论。

## 3.6 预期现象

本实验预期观察到以下现象：

1. 只扰动一个 agent 时，普通 MLP / GNN 方法的团队性能会明显下降。
2. 普通 GNN 可能因为消息传递机制放大局部错误，使未被攻击 agent 的动作也发生偏移。
3. EGNN MARL 在相同扰动强度下具有更低的 Performance Drop、$D_{\text{others}}$ 和 $D_{\text{team}}$。
4. 加入 symmetry consistency regularization 后，协同传播指标应进一步降低。

## 3.7 结果分析重点

写作时不要把本实验展开成独立大实验，只需要回答三个问题：

1. 单 agent 局部扰动是否会造成团队性能下降？
2. 未被攻击 agent 是否也出现动作偏移？
3. EGNN 是否比普通 MLP / GNN 更能减弱这种协同传播？

推荐表述为：单点局部扰动不仅影响 attacked agent，也会通过 MARL 协同机制影响团队决策；EGNN 的几何等变结构能够降低未攻击 agent 的动作偏移，从而减轻局部误差向全局协同失稳的传播。

## 3.8 Claim-Evidence 对齐

| 实验主张 | 对应证据 | 推荐图表 |
| --- | --- | --- |
| 单 agent 局部扰动会影响团队性能 | Attack Return 低于 Clean Return | 图 1；表 1 |
| 局部扰动会传播到未攻击 agent | $D_{\text{others}}$ 上升 | 图 2；表 1 |
| EGNN 能减弱协同传播 | $D_{\text{others}}$ 和 $D_{\text{team}}$ 更低 | 图 2；表 1 |
| 一致性正则进一步提升稳定性 | EGNN + regularization 优于 EGNN | 表 1 |
