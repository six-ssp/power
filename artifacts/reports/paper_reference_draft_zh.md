# Scene-Aware Interpretable Fusion with Constraint-Aware Evaluation for Heterogeneous PV Time-Series Mining

## 摘要
本文将异构光伏装置上的 `5-minute-ahead` 功率预测问题表述为一个带已知未来外生变量的异构时间序列数据挖掘任务。围绕这一设定，本文构建了一套可复现实验流水线，并提出一种按电站与辐照场景切换权重的可解释融合方法 `Hybrid`。该方法在 `XGBoost`、`DNN` 和 `TFT` 三类异构基学习器之上进行结构化融合，并保留轻量物理后修正以抑制不合理输出。实验基于 Alice Springs 站点四个异构光伏装置，采用时间顺序切分、`6` 小时间隔保护、`daytime-only`、多随机种子和 rolling-origin 评估。结果表明，固定切分下 `Hybrid` 将 `TFT` 的 MAE 从 `0.026838` 降至 `0.021530`；在白天子集上，`Hybrid` 的 MAE 为 `0.043451`，优于 `TFT` 的 `0.053429`。在多随机种子和 rolling-origin 下，`Hybrid` 仍保持最低平均 MAE。进一步引入 physical violation rate 后，结果显示 `Hybrid` 在精度、稳定性与物理一致性之间取得了更平衡的表现。

## 1. 研究问题
光伏功率预测常被直接当作单一时序建模任务处理，但在异构装置条件下，误差模式会随着电站类型、天气场景和昼夜状态发生系统性变化。对这类数据，更自然的问题表述不是“寻找唯一最强单模型”，而是“如何在异构时序场景下进行结构化、可解释的模型选择与融合”。这一定义更接近数据挖掘而非单纯的工程预测实现。

## 2. 方法定位
本文保留三类互补的基学习器：`XGBoost`、`DNN` 和 `TFT`。在此基础上，本文提出 `Hybrid`：
- 先按辐照水平划分场景；
- 再按电站学习场景特定的模型权重；
- 最后对夜间与低辐照样本做轻量物理后修正。

这使得方法的核心贡献不在于再造一个更大的深度模型，而在于将“何时更信任哪一个学习器”显式建模为结构化融合问题。

## 3. 贡献点
1. 构建了一套面向四个异构光伏装置的可复现实验流水线，覆盖统一数据处理、时间切分、图表导出和校验机制。
2. 提出一种场景自适应、按电站分解的可解释融合方法 `Hybrid`，将异构时序场景下的模型选择显式化。
3. 在传统误差指标之外引入 physical violation rate，使评估从纯精度比较扩展到 constraint-aware evaluation。
4. 通过 `daytime-only`、多随机种子和 rolling-origin 三类鲁棒性协议，验证 `Hybrid` 的优势并不依赖单次固定切分。

## 4. 实验设计
### 4.1 数据与任务
- 数据来自 Alice Springs 站点四个异构光伏装置。
- 任务定义为：在下一时刻天气已知的前提下，预测下一时刻功率。
- 采样频率为 `5 min`，监督样本总量为 `1,241,216`。

### 4.2 训练配置
| Model | Role | TrainingUnit | MaxEpochsOrRounds | BatchSize | LearningRate | PatienceOrEarlyStop | SelectionMetric | OptimizerOrBackend | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | base learner | boosting rounds | 300 | - | 0.050000 | 30 | val_rmse | XGBoost hist | tree ensemble with validation early stopping |
| DNN | base learner | epochs | 30 | 8192 | 0.001000 | 5 | val_rmse | AdamW | MLP with SmoothL1 loss and best-state restore |
| TFT | base learner | epochs | 30 | 6144 | 0.000500 | 5 | val_loss | PyTorch Forecasting / Lightning | best checkpoint restored after early stopping |
| AdaptiveBlend | meta learner | epochs | 30 | 16384 | 0.000300 | 6 | holdout_mae | AdamW | neural gating on validation holdout split |
| StackedXGB | meta learner | boosting rounds | 600 | - | 0.030000 | 50 | holdout_rmse | XGBoost hist | stacking regressor on validation holdout split |

### 4.3 主实验实际执行轮次
| Model | TrainingUnit | MaxEpochsOrRounds | ExecutedEpochsOrRounds | BestEpochOrRound | SelectionMetric | BestSelectionScore |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost | boosting rounds | 300 | 102 | 72 | val_rmse | 0.154479 |
| DNN | epochs | 30 | 11 | 6 | val_rmse | 0.119298 |
| TFT | epochs | 30 | 10 | 5 | val_loss | 0.037967 |
| AdaptiveBlend | epochs | 30 | 7 | 1 | holdout_mae | 0.089350 |
| StackedXGB | boosting rounds | 600 | 177 | 127 | holdout_rmse | 0.307623 |

### 4.4 数学公式与约束

#### 4.4.1 目标函数与损失函数

**Blend Score (融合得分)** 用于优化基模型的权重组合，定义为：

$$\text{Score} = \text{MAE} + w_{\text{rmse}} \cdot \text{RMSE}$$

其中：
- $\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$
- $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$  
- $w_{\text{rmse}}$：RMSE权重因子，$w_{\text{rmse}} = 0.3$
- $N$：样本总数

**DNN损失函数**：采用 SmoothL1 Loss（也称 Huber Loss），对异常值鲁棒：

$$L_{\text{SmoothL1}} = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq 1 \\
|y - \hat{y}| - \frac{1}{2} & \text{otherwise}
\end{cases}$$

**TFT损失函数**：采用 Temporal Fusion Transformer 原生的分位数损失（Quantile Loss），支持多步预测的置信区间估计。

#### 4.4.2 模型融合公式

**线性加权融合**（用于 Hybrid、StackedXGB、AdaptiveBlend）：

$$\hat{y}_{\text{blend}} = \sum_{k=1}^{K} w_k \cdot \hat{y}_k$$

其中：
- $K$：基模型数量，当前 $K = 3$（XGBoost, DNN, TFT）
- $w_k$：第 $k$ 个基模型的权重，满足 $\sum_{k=1}^{K} w_k = 1$
- 权重约束：Hybrid 使用 $w_k \geq 0.0$（无下界），普通 Blend 使用 $w_k \geq 0.1$

**场景特定权重**（Hybrid方法核心）：权重随辐照场景和电站ID而变化：

$$\hat{y}_{\text{hybrid}} = \sum_{k=1}^{K} w_k^{(s,p)} \cdot \hat{y}_k$$

其中 $s \in \{\text{night, low\_rad, mid\_rad, high\_rad}\}$ 为辐照场景，$p$ 为电站ID。

权重搜索采用网格搜索（Grid Search）算法：
- 搜索步长 $\Delta w = 0.05$（参数 `hybrid_weight_step`）
- 对于3个基模型，权重组合遍历所有满足 $\sum w_k = 1$ 且 $w_k \geq 0.0$ 的点
- 每个权重组合在对应场景的数据上评估融合得分，选择得分最优的权重

**场景划分阈值**：

| 场景 | 条件 |
| --- | --- |
| Night (夜间) | $\text{forecast\_night\_flag} = 1$ |
| Low Radiation (低辐照) | $\text{forecast\_night\_flag} = 0$ AND $G_h < T_{\text{low}}$ |
| Mid Radiation (中辐照) | $T_{\text{low}} \leq G_h < T_{\text{high}}$ |
| High Radiation (高辐照) | $G_h \geq T_{\text{high}}$ |

其中：
- $G_h$：预报水平全辐照（Global Horizontal Irradiance），单位 W/m²
- $T_{\text{low}}$ 搜索范围：$\{50, 100, 150, 200\}$ W/m²（参数 `hybrid_low_radiation_candidates`）
- $T_{\text{high}}$ 搜索范围：$\{400, 500, 600, 700\}$ W/m²（参数 `hybrid_high_radiation_candidates`）
- 树档组合约束：$T_{\text{high}} > T_{\text{low}}$，总共有 $4 \times 4 = 16$ 个有效候选组合
- 每个 $(T_{\text{low}}, T_{\text{high}})$ 组合通过验证集上的融合得分选择最优阈值对

**AdaptiveBlend的软权重**（神经网络门控）：

$$w_k^{(t)} = \frac{\exp(f_k(x_t))}{\sum_{j=1}^{K} \exp(f_j(x_t))}$$

其中 $f_k$ 为神经网络的第 $k$ 个权重头输出，$x_t$ 为 $t$ 时刻的特征。

#### 4.4.3 物理约束调整

**Physics Adjustment (物理后修正)**：对融合预测进行减法调整，抑制不合理的低辐照和夜间正功率：

$$\hat{y}_{\text{adj}} = \hat{y}_{\text{blend}} - \alpha \cdot \max(0, \hat{y}_{\text{blend}}) \cdot r(G_h) \cdot n(h)$$

其中：
- $\alpha \in [0, 0.6]$：物理调整强度（通过在 $\{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6\}$ 上网格搜索选定），参数名 `night_alpha_grid`
- $r(G_h) = \min\left(1.0, \frac{T_{\text{threshold}} - G_h}{T_{\text{threshold}}}\right)$：辐照缩放因子，其中 $T_{\text{threshold}} = 20$ W/m²（参数名 `night_radiation_threshold`）
- $n(h) = \min\left(1.0, \frac{-h_s}{12°}\right)$：夜间强度因子，基于太阳高度角 $h_s$（单位为°）
- 当 $\hat{y}_{\text{blend}} \leq 0$ 时，调整不适用（`positive_part = max(0, ŷ)`）

物理约束的直观理解：
- 低辐照时 $(G_h < T_{\text{threshold}})$，预测的正功率受到缩减，缩减程度与辐照不足的比例成正比
- 夜间 $(h_s < 0°)$，夜间强度 $n(h)$ 随着太阳角度下降而线性增强，最大值为1.0
- 两个因子 $r(G_h)$ 和 $n(h)$ 的乘积确保约束的渐进性和相乘效应：只有在同时满足低辐照和接近夜间时，约束才会强烈

#### 4.4.4 物理违约评估指标

**Physical Violation Rate (物理违例率)**：

$$P_{\text{viol}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i < -m_p \text{ or } (\text{night}_i \text{ and } \hat{y}_i > m_p)]$$

其中：
- $m_p = \max\left(\text{train\_power\_max}_p \cdot \tau, m_{\min}\right)$：电站 $p$ 的违例阈值
- $\tau = 0.01$：容差比率
- $m_{\min} = 0.01$：最小絕對容差（kW）
- $\text{night}_i = 1$ 当第 $i$ 个样本为夜间预报

**Negative Rate (负率)**：

$$P_{\text{neg}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i < -m_p]$$

**Night Positive Rate (夜间正率)** / **NightPositiveRateOnNight**：

$$P_{\text{night\_pos}} = \frac{\sum_{i \in \text{Night}} \mathbb{1}[\hat{y}_i > m_p]}{|\text{Night}|}$$

这三个约束指标共同刻画模型的物理合理性：负率反映整体的负功率问题，夜间正率反映在夜间时对约束的违反程度。

#### 4.4.5 评估指标定义

**标准误差指标**：

| 指标 | 公式 | 含义 |
| --- | --- | --- |
| MAE | $\frac{1}{N}\sum\|y_i - \hat{y}_i\|$ | 平均绝对误差（主要指标） |
| RMSE | $\sqrt{\frac{1}{N}\sum(y_i-\hat{y}_i)^2}$ | 均方根误差（对大误差敏感） |
| R² | $1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | 决定系数，范围 $[-\infty, 1]$ |
| Bias | $\frac{1}{N}\sum(y_i-\hat{y}_i)$ | 系统偏差 |

其中 $\bar{y} = \frac{1}{N}\sum y_i$ 为目标值的均值。

**分层评估**：

- **Daytime-only**：仅在 $\text{forecast\_night\_flag} = 0$ 的样本上计算指标，反映真实发电时段的精度
- **Multi-seed**：在三个随机种子 $\{42, 52, 62\}$ 下独立训练，报告均值和标准差
- **Rolling-origin**：采用时间序列验证，滑动窗口为 $(0.6/0.7/0.8)$、$(0.7/0.8/0.9)$、$(0.8/0.9/1.0)$（分别表示训练/验证/测试集的数据比例），每个窗口报告独立指标并汇总。此方法验证模型在不同位置时间序列的稳健性

## 5. 实验结果
### 5.1 固定切分
| Model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| Persistence | 0.049670 | 0.175950 | 0.980560 |
| XGBoost | 0.042605 | 0.087974 | 0.995140 |
| DNN | 0.033542 | 0.068490 | 0.997054 |
| TFT | 0.026838 | 0.070332 | 0.996894 |
| Hybrid | 0.021530 | 0.064468 | 0.997390 |
| AdaptiveBlend | 0.024887 | 0.075229 | 0.996446 |
| StackedXGB | 0.029018 | 0.067131 | 0.997170 |

固定切分下，`Hybrid` 获得最低 MAE=`0.021530`。与更充分训练后的 `TFT` 相比，`Hybrid` 仍保持明显优势。

### 5.2 白天子集
| Model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| Persistence | 0.103546 | 0.254142 | 0.969398 |
| XGBoost | 0.067180 | 0.125350 | 0.992555 |
| DNN | 0.049266 | 0.096565 | 0.995582 |
| TFT | 0.053429 | 0.101524 | 0.995117 |
| Hybrid | 0.043451 | 0.093083 | 0.995895 |
| AdaptiveBlend | 0.047765 | 0.108546 | 0.994418 |
| StackedXGB | 0.052496 | 0.096633 | 0.995576 |

白天子集结果表明，`Hybrid` 的改进并非仅来自夜间样本结构，而是在真正有发电行为的时段仍然成立。

### 5.3 Physical Violation
| Model | PhysicalViolationRate | NegativeRate | NightPositiveRateOnNight |
| --- | --- | --- | --- |
| DNN | 0.099240 | 0.099192 | 0.000093 |
| TFT | 0.000008 | 0.000008 | 0.000000 |
| Hybrid | 0.000508 | 0.000492 | 0.000031 |
| StackedXGB | 0.000016 | 0.000000 | 0.000031 |

这一结果提供了与 `SDM` 更契合的补充证据：模型不仅要低误差，还要尽量少地产生负功率或夜间异常正功率。`Hybrid` 的违例率保持较低，显著好于 `DNN`，但仍高于 `TFT` 与 `StackedXGB`。因此更稳的结论不是“`Hybrid` 在所有约束一致性指标上最优”，而是“`Hybrid` 在预测精度与约束一致性之间取得更优平衡”。

### 5.4 多随机种子与 Rolling-Origin
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

`Hybrid` 在多随机种子和 rolling-origin 下均保持最低平均 MAE，说明其优势不依赖某一次随机初始化或某一个固定时间窗口。

## 6. 讨论
从固定切分结果看，`Hybrid` 已经超过 `TFT` 和 `StackedXGB`。从 SDM 叙事看，这一点非常关键：主方法不再只是“解释性更强但精度略弱”的折中方案，而是在更高训练预算下仍保持最强综合性能。与此同时，physical violation 指标提醒我们，`Hybrid` 不是纯约束一致性最优模型，因此文章不宜包装成严格的 constrained optimization；更合适的表述是 knowledge-guided、scene-aware 和 constraint-aware temporal data mining。

## 7. 结论
当前这套实验最适合形成如下论文主线：在异构光伏时间序列上，单一强基线难以覆盖所有场景，而场景自适应、结构化、可解释的融合能够在精度、稳定性和物理合理性之间取得更优平衡。就现有结果而言，`Hybrid` 是主方法，`TFT` 是更强后的深度时序基线，physical violation 则为全文补上了更符合 `SDM` 口味的 constraint-aware 评价维度。
