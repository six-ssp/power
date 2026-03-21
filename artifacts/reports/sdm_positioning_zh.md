# SDM 口径补丁
## 1. 推荐定位
- 问题类型：异构装置上的短时功率预测，可视作带已知未来外生变量的时间序列数据挖掘问题。
- 主方法：`Hybrid`，强调场景自适应、结构可解释和稳定收益。
- 强对照：`StackedXGB`，强调精度上界，不与主方法争夺叙事中心。

## 2. 推荐题目方向
- Scene-Aware Interpretable Fusion for Heterogeneous Photovoltaic Power Forecasting
- Knowledge-Guided Temporal Data Mining for Heterogeneous PV Power Forecasting
- Physics-Guided and Scene-Adaptive Fusion for Ultra-Short-Term PV Forecasting

## 3. 贡献点建议
- 构建统一实验流水线，覆盖标准化数据处理、时间切分、鲁棒性评估和图表导出。
- 提出按电站与辐照场景切换权重的 `Hybrid`，把融合结构显式化。
- 在固定切分、`daytime-only`、多随机种子和 `rolling-origin` 下验证 `Hybrid` 对 `TFT` 的稳定优势。

## 4. 最稳的实验叙事
- 固定切分：`Hybrid` MAE 从 `TFT` 的 `0.034981` 降到 `0.022353`，提升 `36.10%`。
- 白天子集：`Hybrid` MAE 从 `0.068453` 降到 `0.045330`，提升 `33.78%`。
- 多随机种子：`Hybrid` 平均 MAE=`0.021386`，优于 `TFT` 的 `0.034968`。
- rolling-origin：`Hybrid` 平均 MAE=`0.029571`，优于 `TFT` 的 `0.053056`。
- 精度上界：`StackedXGB` 固定切分 MAE=`0.021933`。

## 5. 不建议过度宣称的点
- 不要把当前工作硬写成通用行业 benchmark。
- 不要直接写成 constrained optimization，除非显式补出约束项、优化目标和 violation 指标。
- 更稳的表述是 `knowledge-guided`、`physics-guided` 或 `constraint-aware`。

## 6. 建议补进正文的方法描述
- `Hybrid` 的核心不是简单平均，而是在不同电站、不同辐照 regime 下选择不同的专家权重。
- 物理后修正只占次要增益，场景自适应才是主增益来源。
- `TFT` 现在已训练到与其他神经模型同一数量级的轮次，并恢复最佳验证轮次权重。