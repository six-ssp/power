# 论文大纲与写作建议

## 1. 题目候选
- 基于物理引导异构集成的澳大利亚光伏电站超短期功率预测方法
- 面向多装置场景的物理约束与自适应融合光伏功率预测
- Physics-Guided Adaptive Ensemble for Ultra-Short-Term Photovoltaic Power Forecasting Across Heterogeneous Plants

## 2. 摘要写法
- 第一段写问题背景：高频超短期光伏功率预测对调度和消纳很关键，但不同装置类型、跟踪方式和气象条件下误差模式差异明显。
- 第二段写现有问题：单模型在跨电站和跨场景下稳定性不足，固定权重融合难以适应阴晴变化、早晚低辐照和不同装置工况。
- 第三段写方法：构建包含树模型、深度网络、时序注意力模型的异构基学习器，并引入场景自适应融合与物理修正机制。
- 第四段写结果：在 Alice Springs 4 个光伏装置数据上，相比 TFT 与固定权重 Hybrid，所提方法在 MAE 和 RMSE 上取得进一步下降。
- 第五段写结论：说明该方法兼顾预测精度、跨装置泛化能力和工程可落地性。

## 3. 正文结构

### 3.1 引言
- 说明澳大利亚高渗透率光伏场景下超短期预测的意义。
- 点出现有方法三类不足：
  - 只关注单模型，难兼顾不同工况。
  - 固定权重融合忽略场景差异。
  - 纯数据驱动方法在低辐照/夜间回流场景下容易出现不稳定误差。
- 引出本文方法与贡献。

### 3.2 相关工作
- 传统机器学习：XGBoost、SVR、Random Forest 等。
- 深度学习：MLP、LSTM、Transformer、TFT。
- 光伏预测中的物理引导与混合集成方法。
- 最后指出空白：缺少针对多装置异构场景的场景自适应融合与物理后修正联合框架。

### 3.3 问题定义
- 输入：历史功率、历史气象、未来一步外生气象、时间周期特征、太阳几何特征、物理启发特征。
- 输出：下一时刻功率。
- 任务设置：5 min ahead 单步预测。
- 数据划分：每个电站按时间顺序 8:1:1。

### 3.4 方法
- 4.1 数据预处理与英文映射。
- 4.2 特征工程：
  - 时间周期特征
  - 太阳几何特征
  - 物理启发特征
  - 滞后与滚动统计特征
- 4.3 异构基学习器：
  - XGBoost
  - DNN
  - TFT
- 4.4 融合层设计：
  - 固定权重 Hybrid 作为基础对照
  - AdaptiveBlend 作为样本级动态加权融合
  - StackedXGB 作为二阶段场景融合器
- 4.5 物理修正：
  - 低辐照
  - 夜间太阳高度角约束
  - 正功率衰减修正

### 3.5 实验设计
- 数据集：Alice Springs 4 个不同光伏装置。
- 对比方法：Persistence、XGBoost、DNN、TFT、Hybrid。
- 主方法：AdaptiveBlend、StackedXGB。
- 指标：MAE、RMSE、R2。
- 消融：
  - 去物理修正
  - 等权重
  - 去掉 XGBoost / DNN / TFT

### 3.6 结果分析
- 总体结果：突出 StackedXGB 的最优性。
- 分电站结果：强调难站点 3A/4A 的提升更明显。
- 消融结果：说明固定权重限制了上限，XGBoost 和 TFT 在融合中都有价值。
- 分场景结果：用小时曲线、残差分布说明自适应融合和物理修正的作用。

### 3.7 讨论
- 为什么固定权重 Hybrid 不如自适应融合。
- 为什么 3A/4A 更能体现融合收益。
- 现阶段限制：
  - 单步预测
  - 未来一步外生变量为已知条件
  - 静态装置先验仍不完整

### 3.8 结论
- 总结本文提出的异构集成 + 场景自适应融合 + 物理修正框架。
- 强调精度收益与工程价值。
- 给出后续工作：多步预测、更严格外生变量设定、容量/倾角等静态先验。

## 4. 图表放置建议
- 图1：[method_framework.png](/e:/experiment/artifacts/paper_figures/method_framework.png)
- 图2：[baseline_overview.png](/e:/experiment/artifacts/paper_figures/baseline_overview.png)
- 图3：[ablation_overview.png](/e:/experiment/artifacts/paper_figures/ablation_overview.png)
- 图4：[plant_mae_heatmap.png](/e:/experiment/artifacts/paper_figures/plant_mae_heatmap.png)
- 图5：[scatter_comparison.png](/e:/experiment/artifacts/paper_figures/scatter_comparison.png)
- 图6：[residual_distribution.png](/e:/experiment/artifacts/paper_figures/residual_distribution.png)
- 图7：[hourly_mae_curve.png](/e:/experiment/artifacts/paper_figures/hourly_mae_curve.png)
- 图8：[plant_gain_over_tft.png](/e:/experiment/artifacts/paper_figures/plant_gain_over_tft.png)

## 5. 表格放置建议
- 表1：数据集与装置说明。
- 表2：总体 baseline 对比，引用 [baseline_metrics.csv](/e:/experiment/artifacts/metrics/baseline_metrics.csv)。
- 表3：消融实验，引用 [ablation_metrics.csv](/e:/experiment/artifacts/metrics/ablation_metrics.csv)。
- 表4：分电站结果，引用 [plant_level_metrics.csv](/e:/experiment/artifacts/metrics/plant_level_metrics.csv)。

## 6. 摘要与结论里建议保留的核心数字
- 最优模型：`StackedXGB`
- 总体指标：`MAE=0.019204`，`RMSE=0.059470`，`R2=0.997776`
- 相对当前 TFT：`MAE` 下降约 `4.10%`，`RMSE` 下降约 `3.49%`
- 相对固定权重 Hybrid：`MAE` 下降约 `9.49%`，`RMSE` 下降约 `11.66%`
