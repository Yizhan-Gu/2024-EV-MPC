# EV Charging Forecast-MPC Study (Single Document, 0701 Focus)

## 0. 这份文档的定位

这是当前项目唯一保留的说明文档，整合了：
- 实验设计逻辑
- 代码运行说明
- 0701核心结果解释
- 图表选用建议（期刊向）
- 常见问题（包括 models 文件夹为空）

---

## 1. 研究问题与比较边界

本研究只比较一件事：
- 在同一个 MPC 框架下，替换不同预测方法，观察成本与削峰效果。

固定不变的部分：
- 同一个 rolling MPC 目标函数和约束
- 同一套电价参数
- 同一数据预处理规则

变化的部分：
- 预测模块（9种方法）
  - Perfect, Noforecast, Persistence, Statistic, GMM, LSTM, TCN, Transformer, iTransformer

比较对象：
- EV case（按EV会话建模）
- Charger case（按充电口聚合建模）

重要原则：
- 合理比较：同一天、同一case、同一MPC，仅方法不同。
- 不合理比较：跨天直接比较绝对成本（需求规模不同）。

---

## 2. 当前你要看的版本：0701 单日结果

你要求只看一天，所以当前图和解释全部聚焦：
- 日期：2023-07-01

核心对比维度：
1. Load curve（所有方法，不只Perfect）
2. MPC cost（方法×case）
3. Saving vs V0G（方法×case）
4. EV vs Charger 差值
5. 预测误差 vs 控制收益关系
6. 单日方法排名

---

## 3. 图表目录（仅0701）

图文件位置：
- figures/0701/

图清单：
1. 0701_fig01_load_curves_all_methods.png
2. 0701_fig02_cost_barh.png
3. 0701_fig03_saving_grouped.png
4. 0701_fig04_dumbbell_ev_vs_charger_cost.png
5. 0701_fig05_peak_cost_bubble.png
6. 0701_fig06_error_vs_saving.png
7. 0701_fig07_rank_card.png

配套数据：
- tables/result_0701_only.csv
- tables/figure_manifest_0701.csv

---

## 4. 预测方法学（Methodology, 逐方法详细）

### 4.1 统一预测目标与输出格式

所有预测器最终输出统一的 `forecast sessions`，字段一致：
- `AT_idx`: 预测到达时段（1..96）
- `DT_idx`: 预测离开时段（1..96）
- `ED`: 预测充电需求（kWh）
- `charger_id`: 预测所属充电口

统一输出的目的是保证公平比较：
- 预测器不同，但送入MPC的数据结构一致。
- MPC求解器不做任何“方法特化”。

中间层采用“日级画像”表示：
$$
\mathbf{x}_d=[a_1,\dots,a_{96},e_1,\dots,e_{96}]\in\mathbb{R}^{192}
$$
其中 $a_t$ 为时段到达会话数，$e_t$ 为该时段到达会话总能量。

### 4.2 Perfect（Oracle 上界）

- 定义：直接使用真实当日会话作为预测输入。
- 用途：理论上界，不代表可部署算法。
- 价值：用于衡量“预测误差导致的可达性能损失”。

### 4.3 Noforecast（无预测基线）

- 定义：不提供未来会话，仅靠滚动过程中已到达会话决策。
- 特点：最保守、信息最少。
- 意义：作为“没有预测能力”时的工程基线。

### 4.4 Persistence（同类日延续）

- 定义：取最近同类日（weekday/weekend）的会话模式，平移到目标日。
- 假设：短期行为在相邻同类日有时间稳定性。
- 风险：当出现异常出行日（活动、天气、考试周）时误差较大。

### 4.5 Statistic（经验分布采样）

- 定义：基于历史同类日对到达时段、会话时长、能量需求做经验分布采样。
- 实现：
  - 先估计预测总会话数
  - 再对 `AT_idx`、`duration`、`ED` 分别采样
  - 最后组合并做可行性裁剪
- 优点：可解释、计算快、鲁棒性较好。

### 4.6 GMM（多峰分布建模）

- 定义：对特征 $[AT_{idx}, duration, ED]$ 拟合高斯混合模型。
- 组件选择：BIC在 $k\in\{1,2,3,4,5\}$ 中自动选择。
- 采样：从拟合分布中采样会话，再映射回可行会话。
- 适用性：适合用户行为明显多峰（通勤/晚间两类人群）的场景。

### 4.7 LSTM（跨天时序建模）

- 输入：过去 `lookback_days` 天的日级画像序列。
- 输出：下一天日级画像预测。
- 架构：双层LSTM + 全连接头，损失为MSE。
- 特点：对“缓变趋势”敏感，但对突发异动可能滞后。

### 4.8 TCN（时序卷积）

- 输入输出同LSTM。
- 架构：因果卷积 + 膨胀卷积（多尺度感受野）。
- 特点：
  - 训练并行度高
  - 对局部结构变化捕捉较强
  - 在样本不大时通常比深层RNN更稳定

### 4.9 Transformer（注意力时序）

- 输入输出同LSTM。
- 架构：多头注意力 + 前馈层 + 残差归一化。
- 特点：善于建模长程依赖（例如跨时段耦合）。

### 4.10 iTransformer（变量维注意力）

- 思路：将“变量维度”视为token，强化变量间关系学习。
- 在本任务中，重点建模 `到达强度 ↔ 到达能量` 的耦合。
- 特点：对变量协同结构更敏感，但参数调节更依赖数据质量。

### 4.11 训练与推理协议（统一）

- 训练数据：目标日之前的历史数据。
- 日类型拆分：weekday / weekend 分别建模。
- 推理流程：
  1. 预测日级画像
  2. 画像反演为会话集合
  3. 统一可行性裁剪后送入MPC

### 4.12 预测误差评估（用于解释“预测-控制耦合”）

文中采用两个直观误差代理：
$$
Err_{energy}=\frac{|E^{pred}-E^{actual}|}{E^{actual}}\times 100\%
$$
$$
Err_{session}=\frac{|N^{pred}-N^{actual}|}{N^{actual}}\times 100\%
$$

它们不是唯一指标，但适合与控制收益（saving）做可解释关联分析。

---

## 5. MPC 是怎么运行的（Methodology, 运行机制）

### 5.1 滚动时域框架（Receding Horizon）

时域离散为 $N=96$，每步 $\Delta t=0.25$ 小时。对每个时段 $k=1..96$：
1. 形成优化输入：`已到达真实会话 + 未到达预测会话`
2. 在区间 $[k,N]$ 上求解一次优化
3. 仅执行当前时段控制量
4. 更新状态（已充电量、历史最大负荷）进入下一步

这保证了在线可执行性与对预测误差的逐步修正能力。

### 5.2 目标函数（成本最小化）

记 $L_t$ 为站级总负荷，$\gamma_{nc}$ 为全天最大负荷，$\gamma_{on}$ 为峰段最大负荷：
$$
\min J = r_{nc}\gamma_{nc}+r_{on}\gamma_{on}+\sum_{t=k}^{N} r_t L_t\Delta t
 + \alpha\Big(r_{nc}\gamma_{nc}+r_{on}\gamma_{on}+\sum_{t=k}^{N} r_t L_t\Delta t\Big)
 + \beta\sum_{t=k}^{N} L_t\Delta t
$$
其中：
- 第一项与第二项：需量费用（非峰+峰时）
- 第三项：电量费用（分时电价）
- 后两项：附加费项

### 5.3 EV case（个体级）

决策变量：每个会话每个时段功率 $P_{i,t}$。

主要约束：
1. 功率上限：$0\le P_{i,t}\le P_{max}$
2. 连接窗口约束：仅在 $AT_i\le t\le DT_i$ 可充电
3. 能量满足：
$$
\sum_{t=AT_i}^{DT_i} P_{i,t}\Delta t = E_i^{remain}
$$
4. 站级负荷耦合：
$$
L_t = \sum_i P_{i,t}
$$

### 5.4 Charger case（聚合级）

先按 `charger_id` 聚合会话，构建占用掩码 `occ_{j,t}`。

决策变量：每个充电口每时段聚合功率 $P_{j,t}$。

主要约束：
1. 占用约束：`occ_{j,t}=0` 时 $P_{j,t}=0$
2. 上限约束：`occ_{j,t}=1` 时 $0\le P_{j,t}\le P_{max}$
3. 聚合能量满足：
$$
\sum_t P_{j,t}\Delta t = E_j^{remain}
$$
4. 站级负荷：
$$
L_t = \sum_j P_{j,t}
$$

### 5.5 状态更新与可行性控制

每个滚动步执行后更新：
- EV case：每个会话已充电量
- Charger case：每个充电口累计已供能

若某子问题非最优，使用保守回退分配（fallback）保证流程不中断。

### 5.6 为什么这种运行方式合理

1. 与实际调度一致：控制器每15分钟重决策。
2. 误差可吸收：预测错误不会一次性固化到全天。
3. 比较公平：同一MPC骨架下，差异可归因到预测模块。

---

## 6. 每张图在论文中的用途

1) 0701_fig01_load_curves_all_methods.png
- 目的：展示所有方法在同一天的调度形状差异。
- 价值：这是机理图，证明不是只比一个数字。

2) 0701_fig02_cost_barh.png
- 目的：方法与case的绝对成本对比。
- 价值：主结果图之一，读者最先看。

3) 0701_fig03_saving_grouped.png
- 目的：相对 V0G 的节省率对比。
- 价值：把工程收益表达成百分比，更利于跨场景沟通。

4) 0701_fig04_dumbbell_ev_vs_charger_cost.png
- 目的：同一方法在EV和Charger之间的差值可视化。
- 价值：直接回答“换优化对象是否更好”。

5) 0701_fig05_peak_cost_bubble.png
- 目的：成本、峰值、节省三指标联合表达。
- 价值：给出 Pareto 风格 trade-off。

6) 0701_fig06_error_vs_saving.png
- 目的：预测质量和控制收益的关系。
- 价值：支撑“预测-控制耦合”结论。

7) 0701_fig07_rank_card.png
- 目的：单日在两个case下的方法排名。
- 价值：简明结论图，可放主文或补充材料。

---

## 7. 单日结果摘要（0701）

从当前图表可直接读出的稳定结论：
- Charger case 在大多数方法下成本更低。
- Perfect 仍是上界参考，但并非唯一可用方法。
- TCN/GMM/Transformer 在0701表现属于第二梯队。
- Noforecast 一致偏弱，说明预测模块有必要。

---

## 8. models 文件夹为什么是空的

这是你问的重点，答案如下：
- 目前 Notebook 里的深度模型是“训练后直接用于推理并出图”，默认不落盘。
- 因此 models/ 被保留为预留目录，但未写入 .keras 或 .h5 文件。

这不是错误，而是当前实现策略（避免文件膨胀、保持实验迭代快）。
如果你希望，我下一版可以把 LSTM/TCN/Transformer/iTransformer 全部保存到 models/ 并写入版本号与训练配置。

---

## 9. 复现方式（最简）

1. 打开 EV_Charging_Optimization_Research.ipynb。
2. 运行到 Part 6C（0701 Focused Journal Figures）。
3. 在 figures/0701 选图，在 tables/result_0701_only.csv 读取数值。

---

## 10. 当前目录精简策略

你要求避免杂乱，当前策略是：
- 文档只保留本文件。
- figures 只保留 0701 子目录。
- 其余跨天图已删除。

---

## 11. 下一步可选（如果你要冲顶刊）

可在不改变当前主逻辑的前提下追加：
1. 显著性检验（paired t-test + Wilcoxon）
2. 多随机种子误差条（提升统计可信度）
3. 统一图形模板（字体、线宽、标注规范）

