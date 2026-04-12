# Research_Results（当前主入口）

这是本轮重构后的唯一主工作目录，遵循你的要求：
- 不改原仓库逻辑；
- 使用同一MPC框架；
- 只替换预测模块做交叉实验；
- 输出图表和表格全部在本目录。

## 1. 先看这两个文件

1) 主代码（完整可执行）
- [EV_Charging_Optimization_Research.ipynb](EV_Charging_Optimization_Research.ipynb)

2) 主文档（方法+思路+结果解释）
- [RESEARCH_METHODOLOGY.md](RESEARCH_METHODOLOGY.md)

## 2. 本轮实验范围

- 测试日：2023-07-01, 2023-07-02, 2023-07-03
- 方法：Perfect, Noforecast, Persistence, Statistic, GMM, LSTM, TCN, Transformer, iTransformer
- Case：EV / Charger
- 交叉实验总数：3 × 9 × 2 = 54

## 3. 关键结果文件

- 原始结果：
  - [optimization_results.csv](optimization_results.csv)

- 主要表格：
  - [summary_by_method_case.csv](tables/summary_by_method_case.csv)
  - [daily_best_method_by_case.csv](tables/daily_best_method_by_case.csv)
  - [ev_vs_charger_by_method.csv](tables/ev_vs_charger_by_method.csv)
  - [publication_main_table.csv](tables/publication_main_table.csv)

- 主要图像：
  - [saving_by_method_case.png](figures/saving_by_method_case.png)
  - [cost_heatmap_ev.png](figures/cost_heatmap_ev.png)
  - [cost_heatmap_charger.png](figures/cost_heatmap_charger.png)
  - [ev_minus_charger_cost_delta.png](figures/ev_minus_charger_cost_delta.png)
  - [load_profile_best_methods.png](figures/load_profile_best_methods.png)

## 4. 你关心的核心保证

- 同一MPC目标函数与约束框架；
- rolling horizon逻辑一致；
- 仅预测模块改变输入 forecast sessions；
- EV/Charger 两个对象在同一实验协议下比较。

## 5. 运行建议

如果你要复现：
1. 打开 [EV_Charging_Optimization_Research.ipynb](EV_Charging_Optimization_Research.ipynb)
2. 按单元格顺序从上到下运行
3. 最终结果会刷新 `tables/` 与 `figures/`

---
如果你希望下一步直接进入论文稿件阶段，我可以基于当前结果继续补：
- 显著性检验（paired t-test / Wilcoxon）
- 多随机种子置信区间
- 可直接粘贴到论文的 Results 段落草稿
