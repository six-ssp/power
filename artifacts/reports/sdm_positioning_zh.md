# SDM 口径说明

## 1. 推荐定位
- 问题类型：带已知未来外生变量的异构时间序列数据挖掘。
- 主方法：`Hybrid`，强调场景自适应、结构化可解释性和鲁棒性。
- 强对照：`TFT` 作为充分训练后的深度时序基线，`StackedXGB` 作为元学习性能对照。

## 2. 可以主打的贡献组织
- 方法贡献：`scene-aware interpretable fusion`。
- 评估贡献：`constraint-aware evaluation`，即在误差之外显式统计 physical violation。
- 系统贡献：统一数据处理、图表生成、训练配置记录和项目校验的可复现实验流水线。

## 3. 当前最能讲的证据
- 固定切分：`Hybrid` MAE=`0.021530`，优于 `TFT` 的 `0.026838`。
- 白天子集：`Hybrid` MAE=`0.043451`，优于 `TFT` 的 `0.053429`。
- 多随机种子：`Hybrid` 平均 MAE 低于 `TFT` 和 `StackedXGB`。
- rolling-origin：`Hybrid` 平均 MAE 最低。
- constraint-aware evaluation：`Hybrid` 的 physical violation rate 为 `0.000508`，明显低于 `DNN`，但高于 `TFT` 的 `0.000008` 与 `StackedXGB` 的 `0.000016`。

## 4. 推荐表述
- 用 `scene-aware interpretable fusion`、`heterogeneous temporal data mining`、`constraint-aware evaluation`。
- 不建议直接写成严格的 `constrained optimization`，除非后续显式补出优化目标、约束项和求解过程。
- 更稳的写法是：本文在异构时序数据上提出结构化、可解释的场景自适应融合，并通过 constraint-aware 指标补充传统误差评价。
