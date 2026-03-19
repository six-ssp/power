# 论文大纲与写作建议

## 1. 题目候选
- 面向异构光伏装置的场景自适应融合与标准化评测基准
- A Benchmark and Scene-Adaptive Hybrid for Ultra-Short-Term Photovoltaic Power Forecasting Across Heterogeneous Plants
- Standardized Benchmarking and Explainable Hybrid Fusion for Ultra-Short-Term PV Power Forecasting

## 2. 摘要写法
- 第一段：光伏短时预测的重要性，以及异构装置场景下误差模式差异明显。
- 第二段：现有研究常见问题，包括固定切分、单次运行、夜间样本稀释和固定全局权重融合。
- 第三段：本文构建标准化 benchmark，并提出按电站和辐照场景切换权重的 `Hybrid`。
- 第四段：在 daytime-only、多随机种子和 rolling-origin 评估下，`Hybrid` 稳定优于 `TFT`，`StackedXGB` 给出当前最佳精度。
- 第五段：总结方法的可解释性、复现性和工程价值。

## 3. 正文结构

### 3.1 引言
- 说明高频超短期光伏预测的应用价值。
- 强调异构装置下误差模式的差异。
- 引出需要 benchmark 和可解释融合方法。

### 3.2 相关工作
- 光伏短时预测中的机器学习和深度学习。
- 异构集成与场景自适应融合。
- 光伏预测中的物理约束与后修正。
- Benchmarking / evaluation protocol 在时序预测中的重要性。

### 3.3 数据与 benchmark
- 数据来源、英文字段、电站命名。
- 标准化特征工程。
- 主切分协议、gap 设置、daytime-only 评估、多随机种子、rolling-origin。

### 3.4 方法
- 基学习器：XGBoost、DNN、TFT。
- 主方法：`Hybrid`。
- 对照：AdaptiveBlend、StackedXGB。
- 物理后修正。

### 3.5 实验设计
- 总体指标：MAE、RMSE、R2。
- 主实验表。
- 消融表。
- Daytime-only 表。
- 多随机种子稳定性表。
- Rolling-origin 表。

### 3.6 结果分析
- `Hybrid` 为什么优于 `TFT`。
- 场景适配为什么是主增益来源。
- 为什么 `StackedXGB` 更强，但不替代 `Hybrid` 作为主方法。

### 3.7 讨论与局限
- 数据仍来自单站点、四个装置。
- 当前任务是未来一步天气已知的条件回归。
- 仍可继续扩展到多步预测和更大范围数据。