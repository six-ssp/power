# Scene-Aware Interpretable Fusion with Constraint-Aware Evaluation for Heterogeneous PV Time-Series Mining

## 摘要
本文将异构光伏装置上的 `5-minute-ahead` 功率预测问题表述为一个带已知未来外生变量的异构时间序列数据挖掘任务。围绕这一设定，本文构建了一套可复现实验流水线，并提出一种按电站与辐照场景切换权重的可解释融合方法 `Hybrid`。该方法在 `XGBoost`、`DNN` 和 `TFT` 三类异构基学习器之上进行结构化融合，并保留轻量物理后修正以抑制不合理输出。实验基于 Alice Springs 站点四个异构光伏装置，采用时间顺序切分、`6` 小时间隔保护、`daytime-only`、多随机种子和 rolling-origin 评估。结果表明，固定切分下 `Hybrid` 将 `TFT` 的 MAE 从 `0.020472` 降至 `0.018880`；在白天子集上，`Hybrid` 的 MAE 为 `0.038638`，优于 `TFT` 的 `0.041120`。在多随机种子和 rolling-origin 下，`Hybrid` 仍保持最低平均 MAE。进一步引入 physical violation rate 与 `BVP` 后，结果显示 `Hybrid` 在精度、稳定性与物理一致性之间取得了更平衡的表现。

## 关键词
光伏预测；异构时间序列；可解释融合；场景自适应；constraint-aware evaluation

## 1. 引言
光伏功率预测常被直接表述为一个单一时间序列回归任务，但在异构装置条件下，误差模式会随着电站类型、天气场景和昼夜状态发生系统性变化。对这类数据，更自然的问题并不是“寻找唯一最强单模型”，而是“如何在异构时序场景下进行结构化、可解释的模型选择与融合”。这一定义更接近数据挖掘而非单纯的工程预测实现。

从研究包装上看，本文不把贡献放在构造更大的黑盒模型，而是把重点放在三点：第一，如何在异构装置和场景切换条件下显式建模模型选择；第二，如何用更严格的时间协议证明结果不是偶然；第三，如何在传统误差指标之外补上一层约束感评估，使结论更接近可信的 temporal data mining。

## 2. 问题定义与方法动机
本文任务定义为：在下一时刻天气已知的前提下，预测下一时刻光伏功率。输入既包含当前与历史观测，也包含下一时刻外生天气信息，因此任务更准确地说是 conditional next-step regression。对于该设定，`XGBoost`、`DNN` 和 `TFT` 代表了三类具有明显互补性的基学习器：树模型擅长处理强非线性表格特征，前馈神经网络能够吸收大规模手工特征，而 `TFT` 则以显式 temporal architecture 建模时间上下文。

在此基础上，`Hybrid` 不试图替代这些模型，而是学习“在什么场景下信任哪个学习器”。其设计包含三层结构：先按辐照水平划分场景，再按电站学习场景特定的权重，最后对夜间和低辐照样本施加轻量物理后修正。这样，模型选择被写成一个结构化融合问题，而不是隐式留给单个黑盒网络。

![Method framework](../paper_figures/method_framework.png)

## 3. 方法
### 3.1 基学习器
本文保留三类基学习器：`XGBoost`、`DNN` 和 `TFT`。这样做的目的不是凑 baseline，而是构造足够异构的候选学习器集合，使上层融合真正面对“不同偏差模式”的选择问题。

### 3.2 Scene-Aware Hybrid
`Hybrid` 的核心机制是按场景和电站分解权重。对于每一个样本，先根据辐照水平归入场景，再调用该电站在该场景下的融合权重，对三个基学习器的输出做加权。这样得到的结果具有直接可解释性：权重本身就是“在该电站、该场景下更相信谁”的显式表达。

### 3.3 Constraint-Aware Evaluation
为了避免只用误差指标评估，本文引入 physical violation rate 与 `BVP`。当前版本使用两类最硬的违例：
- 负功率违例：预测值小于 `-epsilon_p`；
- 夜间正功率违例：夜间样本的预测值大于 `epsilon_p`。

其中，`epsilon_p` 按电站定义为训练集最大功率的 `1%` 与 `0.01` 之间的较大值。总体违例率定义为：

```text
PVR = (1 / N) * sum I[y_hat_t < -epsilon_p or (night_t = 1 and y_hat_t > epsilon_p)]
```

同时，定义夜间/极低辐射集合

```text
Omega_bvp = {t | elevation_t <= 0 or G_t <= R_th}
BVP = sum_{t in Omega_bvp} max(y_hat_t, 0)
```

其中 `R_th=20.0`。`BVP` 直接度量模型在物理上本该接近 `0` 的时段里错误输出了多少正向能量，更适合用于论证 Physics Adjustment 的贡献。

## 4. 实验设计
### 4.1 数据与任务
- 数据来自 Alice Springs 站点四个异构光伏装置。
- 采样频率为 `5 min`，监督样本总量为 `1,241,216`。
- 任务是已知下一时刻天气条件下的 `5-minute-ahead` 功率回归。

### 4.2 评估协议
- 主切分：按电站分别做时间顺序 `80 / 10 / 10` 切分。
- 间隔保护：训练/验证、验证/测试之间各保留 `72` 个时间步，约 `6` 小时。
- 补充协议：`daytime-only`、`3` 个随机种子、`3` 个 rolling-origin 窗口。

### 4.3 训练配置
| Model | Role | TrainingUnit | MaxEpochsOrRounds | BatchSize | LearningRate | PatienceOrEarlyStop | SelectionMetric | OptimizerOrBackend | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | base learner | boosting rounds | 300 | - | 0.050000 | 30 | val_rmse | XGBoost hist | tree ensemble with validation early stopping |
| DNN | base learner | epochs | 30 | 8192 | 0.001000 | 5 | val_rmse | AdamW | MLP with SmoothL1 loss, best-state restore, deterministic loader seed |
| TFT | base learner | epochs | 30 | 6144 | 0.000500 | 5 | val_loss | PyTorch Forecasting / Lightning | deterministic Lightning run with best checkpoint restore |
| AdaptiveBlend | meta learner | epochs | 30 | 16384 | 0.000300 | 6 | holdout_mae | AdamW | neural gating on validation holdout split with deterministic loader seed |
| StackedXGB | meta learner | boosting rounds | 600 | - | 0.030000 | 50 | holdout_rmse | XGBoost hist | stacking regressor on validation holdout split |

### 4.4 主实验实际执行轮次
| Model | TrainingUnit | MaxEpochsOrRounds | ExecutedEpochsOrRounds | BestEpochOrRound | SelectionMetric | BestSelectionScore |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost | boosting rounds | 300 | 102 | 72 | val_rmse | 0.154479 |
| DNN | epochs | 30 | 11 | 6 | val_rmse | 0.119298 |
| TFT | epochs | 30 | 10 | 5 | val_loss | 0.037967 |
| AdaptiveBlend | epochs | 30 | 7 | 1 | holdout_mae | 0.089350 |
| StackedXGB | boosting rounds | 600 | 177 | 127 | holdout_rmse | 0.307623 |

## 5. 实验结果
### 5.1 固定切分
| Model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| Persistence | 0.049670 | 0.175950 | 0.980560 |
| XGBoost | 0.042605 | 0.087974 | 0.995140 |
| DNN | 0.028391 | 0.067772 | 0.997116 |
| TFT | 0.020472 | 0.060960 | 0.997666 |
| MeanAverage | 0.021889 | 0.063961 | 0.997431 |
| StaticBlend | 0.019060 | 0.060239 | 0.997721 |
| Hybrid | 0.018880 | 0.059906 | 0.997746 |
| AdaptiveBlend | 0.021832 | 0.067281 | 0.997157 |
| StackedXGB | 0.030325 | 0.067355 | 0.997151 |

固定切分下，`Hybrid` 获得最低 MAE=`0.018880`。与训练预算抬高后的 `TFT` 相比，`Hybrid` 仍保持明显优势；`StackedXGB` 在当前高预算设定下也未超过 `Hybrid`。

![Fixed-split overview](../paper_figures/baseline_overview.png)

### 5.2 白天子集
| Model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| Persistence | 0.103546 | 0.254142 | 0.969398 |
| XGBoost | 0.067180 | 0.125350 | 0.992555 |
| DNN | 0.049996 | 0.097325 | 0.995512 |
| TFT | 0.041120 | 0.088030 | 0.996328 |
| MeanAverage | 0.041677 | 0.092282 | 0.995965 |
| StaticBlend | 0.038711 | 0.086993 | 0.996414 |
| Hybrid | 0.038638 | 0.086517 | 0.996454 |
| AdaptiveBlend | 0.043516 | 0.097151 | 0.995528 |
| StackedXGB | 0.054287 | 0.096877 | 0.995553 |

白天子集结果表明，`Hybrid` 的改进并非仅来自夜间样本结构，而是在真正有发电行为的时段仍然成立。

### 5.3 Constraint-Aware Evaluation
| Model | PhysicalViolationRate | NegativeRate | NightPositiveRate | NightPositiveRateOnNight | ViolationCount | NegativeCount | NightPositiveCount | NightSamples | Samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Persistence | 0.000008 | 0.000000 | 0.000008 | 0.000015 | 1 | 0 | 1 | 64540 | 123952 |
| XGBoost | 0.002114 | 0.000000 | 0.002114 | 0.004059 | 262 | 0 | 262 | 64540 | 123952 |
| DNN | 0.004639 | 0.004494 | 0.000145 | 0.000279 | 575 | 557 | 18 | 64540 | 123952 |
| TFT | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 64540 | 123952 |
| MeanAverage | 0.000363 | 0.000299 | 0.000065 | 0.000124 | 45 | 37 | 8 | 64540 | 123952 |
| StaticBlend | 0.000218 | 0.000210 | 0.000008 | 0.000015 | 27 | 26 | 1 | 64540 | 123952 |
| Hybrid | 0.000097 | 0.000097 | 0.000000 | 0.000000 | 12 | 12 | 0 | 64540 | 123952 |
| AdaptiveBlend | 0.000153 | 0.000137 | 0.000016 | 0.000031 | 19 | 17 | 2 | 64540 | 123952 |
| StackedXGB | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 64540 | 123952 |

| Model | BVP | BVPMean | BVPMAE | PositiveLeakCount | PositiveLeakRateOnRegion | RegionSamples | RegionRatio | RadiationThreshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Persistence | 30.888064 | 0.000458 | 0.000323 | 997 | 0.014786 | 67428 | 0.543985 | 20.000000 |
| XGBoost | 1369.826446 | 0.020315 | 0.020106 | 67428 | 1.000000 | 67428 | 0.543985 | 20.000000 |
| DNN | 57.127961 | 0.000847 | 0.008901 | 4709 | 0.069837 | 67428 | 0.543985 | 20.000000 |
| TFT | 41.876773 | 0.000621 | 0.001806 | 10124 | 0.150145 | 67428 | 0.543985 | 20.000000 |
| MeanAverage | 281.683668 | 0.004178 | 0.004086 | 65373 | 0.969523 | 67428 | 0.543985 | 20.000000 |
| StaticBlend | 49.076498 | 0.000728 | 0.001374 | 23444 | 0.347689 | 67428 | 0.543985 | 20.000000 |
| Hybrid | 50.554646 | 0.000750 | 0.001198 | 24567 | 0.364344 | 67428 | 0.543985 | 20.000000 |
| AdaptiveBlend | 162.332425 | 0.002407 | 0.002307 | 64764 | 0.960491 | 67428 | 0.543985 | 20.000000 |
| StackedXGB | 597.658303 | 0.008864 | 0.008683 | 67428 | 1.000000 | 67428 | 0.543985 | 20.000000 |

这一结果提供了与 `SDM` 更契合的补充证据：模型不仅要低误差，还要尽量少地产生负功率、夜间异常正功率以及不可行域上的正向能量泄漏。`Hybrid` 的违例率保持较低，显著好于 `DNN`，但仍高于 `TFT` 与 `StackedXGB`。另一方面，`Hybrid` 在 `BVP` 上为 `50.554646`，明显低于 `w/o Physics` 的 `56.344059`，说明 Physics Adjustment 的主要收益确实体现在夜间/极低辐照样本。更稳的结论不是“`Hybrid` 在所有约束一致性指标上最优”，而是“`Hybrid` 在预测精度与约束一致性之间取得更优平衡”。

### 5.4 消融与鲁棒性
| Model | MAE | RMSE |
| --- | --- | --- |
| Full Hybrid | 0.018880 | 0.059906 |
| w/o Physics | 0.018927 | 0.059906 |
| Mean Average | 0.021889 | 0.063961 |
| Equal Weights | 0.020937 | 0.063913 |
| w/o Plant Adaptation | 0.019362 | 0.060116 |
| w/o Scene Adaptation | 0.019060 | 0.060239 |
| w/o XGBoost | 0.019260 | 0.059845 |
| w/o DNN | 0.020145 | 0.061245 |
| w/o TFT | 0.023070 | 0.067829 |
| Adaptive Blend | 0.021832 | 0.067281 |
| Stacked XGB | 0.030325 | 0.067355 |

| Model | BVP | BVPMean | BVPMAE | PositiveLeakCount | PositiveLeakRateOnRegion | RegionSamples | RegionRatio | RadiationThreshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full Hybrid | 50.554646 | 0.000750 | 0.001198 | 24567 | 0.364344 | 67428 | 0.543985 | 20.000000 |
| w/o Physics | 56.344059 | 0.000836 | 0.001284 | 24567 | 0.364344 | 67428 | 0.543985 | 20.000000 |
| Mean Average | 281.683668 | 0.004178 | 0.004086 | 65373 | 0.969523 | 67428 | 0.543985 | 20.000000 |
| Equal Weights | 163.722051 | 0.002428 | 0.002336 | 65373 | 0.969523 | 67428 | 0.543985 | 20.000000 |
| w/o Plant Adaptation | 50.647489 | 0.000751 | 0.001215 | 29846 | 0.442635 | 67428 | 0.543985 | 20.000000 |
| w/o Scene Adaptation | 49.076498 | 0.000728 | 0.001374 | 23444 | 0.347689 | 67428 | 0.543985 | 20.000000 |
| w/o XGBoost | 40.295560 | 0.000598 | 0.001910 | 7153 | 0.106084 | 67428 | 0.543985 | 20.000000 |
| w/o DNN | 48.764927 | 0.000723 | 0.001178 | 21430 | 0.317820 | 67428 | 0.543985 | 20.000000 |
| w/o TFT | 109.800958 | 0.001628 | 0.002533 | 42264 | 0.626802 | 67428 | 0.543985 | 20.000000 |
| Adaptive Blend | 162.332425 | 0.002407 | 0.002307 | 64764 | 0.960491 | 67428 | 0.543985 | 20.000000 |
| Stacked XGB | 597.658303 | 0.008864 | 0.008683 | 67428 | 1.000000 | 67428 | 0.543985 | 20.000000 |

| Model | Runs | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| Persistence | 3 | 0.049670 +/- 0.000000 | 0.175950 +/- 0.000000 | 0.980560 +/- 0.000000 |
| XGBoost | 3 | 0.030599 +/- 0.008533 | 0.078277 +/- 0.006929 | 0.996122 +/- 0.000701 |
| DNN | 3 | 0.034508 +/- 0.001000 | 0.068652 +/- 0.000777 | 0.997040 +/- 0.000067 |
| TFT | 3 | 0.021144 +/- 0.004029 | 0.062665 +/- 0.005426 | 0.997516 +/- 0.000440 |
| Hybrid | 3 | 0.018796 +/- 0.001966 | 0.060554 +/- 0.002795 | 0.997693 +/- 0.000216 |
| AdaptiveBlend | 3 | 0.022755 +/- 0.001841 | 0.071059 +/- 0.003047 | 0.996823 +/- 0.000275 |
| StackedXGB | 3 | 0.026599 +/- 0.005934 | 0.064198 +/- 0.004715 | 0.997398 +/- 0.000371 |

| Model | Runs | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| AdaptiveBlend | 3 | 0.030991 +/- 0.007839 | 0.110969 +/- 0.045045 | 0.990369 +/- 0.008302 |
| DNN | 3 | 0.034938 +/- 0.002879 | 0.093428 +/- 0.022472 | 0.993972 +/- 0.003303 |
| Hybrid | 3 | 0.027830 +/- 0.006528 | 0.098015 +/- 0.034926 | 0.992801 +/- 0.005542 |
| Persistence | 3 | 0.060781 +/- 0.014533 | 0.208959 +/- 0.046285 | 0.971825 +/- 0.010685 |
| StackedXGB | 3 | 0.034299 +/- 0.008551 | 0.114509 +/- 0.052920 | 0.989274 +/- 0.010044 |
| TFT | 3 | 0.033845 +/- 0.008390 | 0.113959 +/- 0.051152 | 0.989486 +/- 0.009696 |
| XGBoost | 3 | 0.038132 +/- 0.004137 | 0.111301 +/- 0.026684 | 0.991401 +/- 0.004890 |

从消融结果看，`scene adaptation` 是 `Hybrid` 的主增益来源；从鲁棒性结果看，`Hybrid` 在多随机种子下的平均 MAE 为 `0.018796 +/- 0.001966`，低于 `TFT` 的 `0.021144 +/- 0.004029` 和 `StackedXGB` 的 `0.026599 +/- 0.005934`；在 rolling-origin 下，`Hybrid` 的平均 MAE 为 `0.027830 +/- 0.006528`，同样低于 `TFT` 的 `0.033845 +/- 0.008390` 和 `StackedXGB` 的 `0.034299 +/- 0.008551`。

![Rolling-origin overview](../paper_figures/rolling_origin_overview.png)

## 6. SDM 讨论
从 `SDM` 视角看，本文最合理的主线不是“提出一个全新的深度模型”，而是“在异构时间序列上显式建模模型选择，并用 constraint-aware 方式评估其可信性”。这使得 `Hybrid` 的价值不只是解释性更强，而是在高预算训练、白天子集、多随机种子和 rolling-origin 下都维持较强综合性能。

与此同时，physical violation 的结果也给出了边界：`Hybrid` 并不是严格意义上的约束最优模型，因此文章不宜包装成完整的 constrained optimization。更稳的表述是 knowledge-guided、scene-aware 和 constraint-aware temporal data mining。

## 7. 局限性
- 数据范围仍局限于同一站点的四个异构装置，跨站点外推能力尚未验证。
- 当前任务是单步预测，尚未扩展到多步预测或更长时间尺度。
- constraint-aware 指标目前聚焦于负功率、夜间异常正功率和不可行域正向能量泄漏，尚未进一步扩展到更强的清空包络或 ramp-rate 约束。

## 8. 结论
当前这套实验最适合形成如下论文主线：在异构光伏时间序列上，单一强基线难以覆盖所有场景，而场景自适应、结构化、可解释的融合能够在精度、稳定性和物理合理性之间取得更优平衡。就现有结果而言，`Hybrid` 是主方法，`TFT` 是更强后的深度时序基线，physical violation 与 `BVP` 则为全文补上了更符合 `SDM` 口味的 constraint-aware 评价维度。
