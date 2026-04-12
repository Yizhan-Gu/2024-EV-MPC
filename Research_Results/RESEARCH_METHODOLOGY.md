# 电动车充电预测与MPC交叉实验方法论（重构版）

## 1. 研究目标与原则

本项目严格遵循一个核心原则：
- **MPC求解逻辑保持不变**（同一目标函数、同一约束、同一滚动时域）
- **只替换预测模块**（不同预测方法输出 forecast sessions，再送入同一个MPC）

研究目标有两个：
1. 在同一MPC框架下比较多种预测方法优劣。
2. 在同一预测输入下比较两种优化对象：
   - EV个体级（EV case）
   - Charger聚合级（Charger case）

因此，本研究是一个严格的二维交叉实验：
- 预测方法 × 优化对象。

---

## 2. 与 MPC.jl 对齐的实现要点

本Notebook中的实现对齐了 [MPC.jl](../MPC.jl) 的关键流程：

1. 时间离散化：
- 24小时划分为 N=96 个 15分钟时段，$\Delta t=0.25$ 小时。

2. 会话参数：
- 每个会话抽取 `AT_idx`, `DT_idx`, `ED`。
- 使用与Julia一致的可行性过滤：
  - 若 `ED > P_max * Δt * (DT_idx - AT_idx + 1)`，则剔除。

3. MPC滚动优化：
- 在每个时段 $k=1..N$ 重新求解一次（rolling horizon）。
- 维护已充能量状态（EV状态或Charger状态）。
- 将“已到达真实会话 + 未到达预测会话”合并后求解。

4. 目标函数结构（保持一致）：
- 需量费用（非分时 + 峰时）
- 电量费用（峰时/非峰时分段）
- 附加费项

5. 对象切换：
- EV case：变量按会话（EV）展开。
- Charger case：先将会话按充电口聚合并构建占用掩码，再按充电口建模。

---

## 3. 数据预处理与特征工程

数据来源：
- [clean_charging_sessions.csv](../clean_charging_sessions.csv)

预处理步骤：
1. 自动识别并标准化列名（起止时间、能量、站点、端口）。
2. 构建 `charger_id = station_name|port`。
3. 提取 `day`, `day_type`（weekday/weekend）。
4. 计算 `AT_idx`, `DT_idx`（1..96）。
5. 执行可行性过滤（对应Julia的 overMaxPower 逻辑）。
6. 划分训练集/测试集：
   - 测试日：2023-07-01, 2023-07-02, 2023-07-03
   - 其余日期作为历史训练池。

---

## 4. 预测方法设计（9种）

### 4.1 基线方法
1. **Perfect**（上界Oracle）
- 直接使用真实当日会话作为预测。
- 作用：给出理论最佳参考。

2. **Noforecast**
- 不提供未来会话预测，仅依赖滚动过程中“已到达真实会话”。
- 作用：无预测能力下的纯在线策略。

3. **Persistence**
- 选取最近同类日（weekday/weekend）会话，平移到目标日。
- 作用：低成本的历史复用基线。

4. **Statistic**
- 按同类日经验分布采样（到达时段、时长、ED、charger分布）。
- 作用：统计学基线。

### 4.2 生成式统计方法
5. **GMM**
- 在同类日上对 `[AT_idx, duration, ED]` 拟合高斯混合模型。
- 组件数用 BIC 在 1~5 中选择。
- 采样得到会话，再做可行性裁剪。

### 4.3 深度学习方法（按日序列预测）
先构建“日级画像向量”：
- `arrivals[96] + arrival_energy[96]`，总维度 192。
- 用过去 `lookback_days=14` 天预测下一天画像，再回生成会话。

6. **LSTM**
- 两层循环结构，建模跨天时序依赖。

7. **TCN**
- 因果膨胀卷积，强调局部与多尺度时间结构。

8. **Transformer**
- 时序自注意力，建模长距离依赖。

9. **iTransformer**
- 采用“变量为token”的思路（输入转置后注意力），
- 更强调变量间关系（到达强度 vs 能量强度等）。

> 说明：深度模型都输出同一类型的日级画像，再通过同一回生成会话逻辑进入MPC，保证比较公平。

---

## 5. 统一MPC框架

### 5.1 EV case
每个时段 $k$：
1. 合并 `arrived_actual (AT<=k)` 与 `future_forecast (AT>k)`。
2. 约束每个会话在 `[AT,DT]` 内可充，且满足剩余能量。
3. 目标最小化未来区间总成本。
4. 仅执行当前时段动作（recourse），更新状态，进入下一时段。

### 5.2 Charger case
每个时段 $k$：
1. 对合并会话按 charger 聚合，构建占用掩码 `occ[t]`。
2. 对每个 charger 约束：
   - 非占用时段功率为0；占用时段功率上限为 `P_max`。
   - 满足聚合剩余能量。
3. 用相同费用模型求解并滚动执行。

---

## 6. 实验设置与质量控制

实验矩阵：
- 3个测试日 × 9种预测方法 × 2个case = **54次实验**。

质量控制：
- 所有结果都记录 `status_ratio_optimal`（滚动子问题最优率）。
- 所有方法使用同一套成本函数与参数。
- 所有预测输出都经过同一可行性裁剪，避免因非法样本导致假优势。

---

## 7. 本轮实测结果（已核验）

结果来源：
- [optimization_results.csv](optimization_results.csv)
- [summary_by_method_case.csv](tables/summary_by_method_case.csv)
- [daily_best_method_by_case.csv](tables/daily_best_method_by_case.csv)
- [ev_vs_charger_by_method.csv](tables/ev_vs_charger_by_method.csv)

### 7.1 EV case（按均值成本）
Top-3:
1. Perfect：mean cost = 4525.68，saving = 38.47%
2. GMM：mean cost = 5120.76，saving = 28.77%
3. Statistic：mean cost = 5223.87，saving = 27.24%

### 7.2 Charger case（按均值成本）
Top-3:
1. Perfect：mean cost = 4158.20，saving = 43.07%
2. TCN：mean cost = 4275.11，saving = 39.53%
3. Transformer：mean cost = 4358.87，saving = 39.03%

### 7.3 EV vs Charger 对比
本轮数据下，所有方法均表现为：
- `mean_cost_charger < mean_cost_ev`，即 Charger case 更优。

这说明在当前数据与参数下，聚合建模对预测误差更鲁棒。

---

## 8. 为什么这样搭建模型

1. **保持MPC核心不动**
- 这是保证“预测方法比较公平”的前提。
- 否则结果会混入控制器差异，不能归因到预测模块。

2. **深度方法使用日级画像**
- 会话级序列太稀疏且不齐整，直接端到端预测会话不稳定。
- 先预测“日级负荷结构”，再统一回生成会话，更稳健。

3. **加入GMM与统计方法**
- GMM能捕捉多峰随机性（用户随机到达行为）。
- 统计方法作为可解释基线，便于与深度模型比较收益。

4. **加入iTransformer**
- EV行为不仅有时间相关，还存在变量间相关（到达、时长、ED耦合）。
- iTransformer结构对这种“变量关系”更敏感。

---

## 9. 产出文件（全部在新目录）

主代码：
- [EV_Charging_Optimization_Research.ipynb](EV_Charging_Optimization_Research.ipynb)

主文档：
- [RESEARCH_METHODOLOGY.md](RESEARCH_METHODOLOGY.md)

关键表格：
- [optimization_results.csv](optimization_results.csv)
- [summary_by_method_case.csv](tables/summary_by_method_case.csv)
- [daily_best_method_by_case.csv](tables/daily_best_method_by_case.csv)
- [ev_vs_charger_by_method.csv](tables/ev_vs_charger_by_method.csv)
- [publication_main_table.csv](tables/publication_main_table.csv)

关键图：
- [saving_by_method_case.png](figures/saving_by_method_case.png)
- [cost_heatmap_ev.png](figures/cost_heatmap_ev.png)
- [cost_heatmap_charger.png](figures/cost_heatmap_charger.png)
- [ev_minus_charger_cost_delta.png](figures/ev_minus_charger_cost_delta.png)
- [load_profile_best_methods.png](figures/load_profile_best_methods.png)

---

## 10. 结论

这次重构后，项目已经满足你的核心要求：
- 同一个MPC框架；
- 仅替换预测模块；
- 对 EV / Charger 两个case做完整交叉实验；
- 给出可核验的图和表；
- 所有工作留在新目录，不改原仓库逻辑。

如果你认可，我下一步可以继续做“期刊级增强版”：
1. 给每个方法做多随机种子重复实验（置信区间）；
2. 加入显著性检验（paired t-test / Wilcoxon）；
3. 形成论文结果段（可直接粘贴到稿件）。
