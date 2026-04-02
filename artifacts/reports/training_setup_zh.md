# 训练设置记录

## 1. 任务与切分
- 任务定义：`5-minute-ahead` 条件功率回归，下一时刻天气已知。
- 主切分：按电站分别做时间顺序 `80 / 10 / 10` 切分。
- 间隔保护：训练/验证、验证/测试之间各保留 `72` 个时间步，约 `6` 小时。
- 白天样本占比：`47.93%`。

## 2. 训练预算
| Model | Role | TrainingUnit | MaxEpochsOrRounds | BatchSize | LearningRate | PatienceOrEarlyStop | SelectionMetric | OptimizerOrBackend | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | base learner | boosting rounds | 300 | - | 0.050000 | 30 | val_rmse | XGBoost hist | tree ensemble with validation early stopping |
| DNN | base learner | epochs | 30 | 8192 | 0.001000 | 5 | val_rmse | AdamW | MLP with SmoothL1 loss, best-state restore, deterministic loader seed |
| TFT | base learner | epochs | 30 | 6144 | 0.000500 | 5 | val_loss | PyTorch Forecasting / Lightning | deterministic Lightning run with best checkpoint restore |
| AdaptiveBlend | meta learner | epochs | 30 | 16384 | 0.000300 | 6 | holdout_mae | AdamW | neural gating on validation holdout split with deterministic loader seed |
| StackedXGB | meta learner | boosting rounds | 600 | - | 0.030000 | 50 | holdout_rmse | XGBoost hist | stacking regressor on validation holdout split |

## 3. 主实验实际执行轮次
| Model | TrainingUnit | MaxEpochsOrRounds | ExecutedEpochsOrRounds | BestEpochOrRound | SelectionMetric | BestSelectionScore |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost | boosting rounds | 300 | 102 | 72 | val_rmse | 0.154479 |
| DNN | epochs | 30 | 11 | 6 | val_rmse | 0.119298 |
| TFT | epochs | 30 | 10 | 5 | val_loss | 0.037967 |
| AdaptiveBlend | epochs | 30 | 7 | 1 | holdout_mae | 0.089350 |
| StackedXGB | boosting rounds | 600 | 177 | 127 | holdout_rmse | 0.307623 |

## 4. 当前训练口径
- 神经模型统一提升到 `30` 轮预算量级，同时保留 early stopping 和 best checkpoint restore。
- 当前默认启用 deterministic training，并在复现模式下将 DataLoader worker 固定为单线程。
- `TFT` 采用 `bf16-mixed` 混合精度，以在 `8 GB` 级显存条件下支撑更高训练预算。
- 树模型继续使用 boosting rounds + early stopping，不强行改成固定轮数。
- 这套设置的重点是释放公平预算，而不是机械地让所有模型跑满相同轮数。
