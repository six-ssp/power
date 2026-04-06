# 结果摘要

## 1. 固定切分主结果
| Model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| Persistence | 0.049670 | 0.175950 | 0.980560 |
| XGBoost | 0.042605 | 0.087974 | 0.995140 |
| DNN | 0.033542 | 0.068490 | 0.997054 |
| TFT | 0.026838 | 0.070332 | 0.996894 |
| Hybrid | 0.021530 | 0.064468 | 0.997390 |
| AdaptiveBlend | 0.024887 | 0.075229 | 0.996446 |
| StackedXGB | 0.029018 | 0.067131 | 0.997170 |

## 2. 白天子集结果
| Model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| Persistence | 0.103546 | 0.254142 | 0.969398 |
| XGBoost | 0.067180 | 0.125350 | 0.992555 |
| DNN | 0.049266 | 0.096565 | 0.995582 |
| TFT | 0.053429 | 0.101524 | 0.995117 |
| Hybrid | 0.043451 | 0.093083 | 0.995895 |
| AdaptiveBlend | 0.047765 | 0.108546 | 0.994418 |
| StackedXGB | 0.052496 | 0.096633 | 0.995576 |

## 3. Physical Violation
| Model | PhysicalViolationRate | NegativeRate | NightPositiveRateOnNight |
| --- | --- | --- | --- |
| Hybrid | 0.000508 | 0.000492 | 0.000031 |
| TFT | 0.000008 | 0.000008 | 0.000000 |
| StackedXGB | 0.000016 | 0.000000 | 0.000031 |
| DNN | 0.099240 | 0.099192 | 0.000093 |

## 4. BVP 指标
| Model | BVP | BVPMean | BVPMAE |
| --- | --- | --- | --- |
| Hybrid | 82.801440 | 0.001228 | 0.002170 |
| TFT | 138.271500 | 0.002051 | 0.002817 |
| StackedXGB | 537.819166 | 0.007976 | 0.007798 |
| DNN | 40.385780 | 0.000599 | 0.019131 |

## 5. Hybrid 消融
| Model | MAE | RMSE |
| --- | --- | --- |
| Full Hybrid | 0.021530 | 0.064468 |
| w/o Physics | 0.021750 | 0.064474 |
| w/o Plant Adaptation | 0.021745 | 0.064162 |
| w/o Scene Adaptation | 0.023153 | 0.066229 |

## 6. Hybrid BVP 消融
| Model | BVP | BVPMean | BVPMAE |
| --- | --- | --- | --- |
| Full Hybrid | 82.801440 | 0.001228 | 0.002170 |
| w/o Physics | 110.053723 | 0.001632 | 0.002575 |
| w/o Plant Adaptation | 80.853938 | 0.001199 | 0.002136 |
| w/o Scene Adaptation | 62.286833 | 0.000924 | 0.003651 |

## 7. 结论
- 固定切分下，`Hybrid` 的 MAE 为 `0.021530`，优于 `TFT` 的 `0.026838` 和 `StackedXGB` 的 `0.029018`。
- `daytime-only` 下，`Hybrid` 的 MAE 为 `0.043451`，同样优于 `TFT` 的 `0.053429` 与 `StackedXGB` 的 `0.052496`。
- physical violation rate 显示：`Hybrid` 不是约束一致性最优，但仍维持低违例水平，并显著好于 `DNN`。
- 在 `forecast_solar_elevation_deg <= 0` 或 `forecast_global_radiation <= 20.0` 的不可行域上，`Hybrid` 的 `BVP=82.801440`，低于 `w/o Physics` 的 `110.053723`，下降 `24.76%`。
- 对应不可行域上的 MAE 也从 `0.002575` 降到 `0.002170`，说明 Physics Adjustment 的收益主要体现在夜间/极低辐照样本。
- 当前最稳的主叙事是：`Hybrid` 提供了最强的综合预测表现，`TFT` 是更充分训练后的深度时序强基线，`StackedXGB` 是重要但不占主叙事中心的元学习对照。
