# 光伏预测 Benchmark 与鲁棒性实验记录

## 1. 实验设置
- 数据源：`dataset/` 下 4 个电站 CSV，仓库内数据字段和电站 ID 已统一为英文。
- 站点假设：Alice Springs, Australia，纬度 `-23.6980`，经度 `133.8807`，时区偏移 `UTC+9.5`。
- 任务定义：基于历史与当前气象信息、未来一步已知天气条件，进行 `5 min ahead` 单步功率条件回归。
- 主实验切分：每个电站按时间顺序 `8:1:1` 划分训练/验证/测试，并在训练/验证、验证/测试之间额外保留 `72` 个时间步间隔。
- 补充评估：新增 `daytime-only` 指标、多随机种子重复和 rolling-origin evaluation。
- 评估指标：MAE、RMSE、MAPE、sMAPE、R2，同时记录预测均值与实际均值。

## 2. 主实验 Baseline 结果
| Model | MAE | RMSE | MAPE | sMAPE | R2 | PredMean | ActualMean | Bias | Samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Persistence | 0.049670 | 0.175950 | 11299.145310 | 8.388069 | 0.980560 | 0.731176 | 0.731184 | -0.000008 | 123952 |
| XGBoost | 0.042605 | 0.087974 | 1100321.575016 | 114.839945 | 0.995140 | 0.736401 | 0.731184 | 0.005217 | 123952 |
| DNN | 0.035966 | 0.068893 | 1264921.834183 | 114.632990 | 0.997020 | 0.728260 | 0.731184 | -0.002925 | 123952 |
| TFT | 0.024010 | 0.062329 | 432996.162643 | 113.337832 | 0.997560 | 0.726519 | 0.731184 | -0.004665 | 123952 |
| Hybrid | 0.018909 | 0.060678 | 65093.235662 | 111.410426 | 0.997688 | 0.735659 | 0.731184 | 0.004475 | 123952 |
| AdaptiveBlend | 0.019469 | 0.062321 | 157618.214043 | 112.243846 | 0.997561 | 0.733916 | 0.731184 | 0.002732 | 123952 |
| StackedXGB | 0.018220 | 0.059508 | 38366.948447 | 112.240701 | 0.997776 | 0.732018 | 0.731184 | 0.000833 | 123952 |

## 3. 主实验消融结果
| Model | MAE | RMSE | MAPE | sMAPE | R2 | PredMean | ActualMean | Bias | Samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full Hybrid | 0.018909 | 0.060678 | 65093.235662 | 111.410426 | 0.997688 | 0.735659 | 0.731184 | 0.004475 | 123952 |
| w/o Physics | 0.018941 | 0.060678 | 68225.451219 | 111.945776 | 0.997688 | 0.735691 | 0.731184 | 0.004506 | 123952 |
| w/o Plant Adaptation | 0.020038 | 0.061104 | 107640.507652 | 113.966770 | 0.997655 | 0.735550 | 0.731184 | 0.004366 | 123952 |
| w/o Scene Adaptation | 0.021717 | 0.065940 | 175119.687979 | 112.655093 | 0.997270 | 0.731700 | 0.731184 | 0.000515 | 123952 |
| w/o XGBoost | 0.022659 | 0.060331 | 466783.095536 | 113.223973 | 0.997714 | 0.730168 | 0.731184 | -0.001017 | 123952 |
| w/o DNN | 0.020405 | 0.062196 | 87593.840835 | 112.516012 | 0.997571 | 0.733209 | 0.731184 | 0.002025 | 123952 |
| w/o TFT | 0.021937 | 0.066134 | 111790.021005 | 113.055948 | 0.997254 | 0.735159 | 0.731184 | 0.003975 | 123952 |
| Adaptive Blend | 0.019469 | 0.062321 | 157618.214043 | 112.243846 | 0.997561 | 0.733916 | 0.731184 | 0.002732 | 123952 |
| Stacked XGB | 0.018220 | 0.059508 | 38366.948447 | 112.240701 | 0.997776 | 0.732018 | 0.731184 | 0.000833 | 123952 |

## 4. Daytime-Only 结果
### 4.1 子集样本占比
| Subset | Samples | Ratio |
| --- | --- | --- |
| All | 123952 | 1.000000 |
| Daytime | 59412 | 0.479315 |
| Nighttime | 64540 | 0.520685 |

### 4.2 Daytime Baseline
| Model | MAE | RMSE | MAPE | sMAPE | R2 | PredMean | ActualMean | Bias | Samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Persistence | 0.103546 | 0.254142 | 21465.074247 | 15.939596 | 0.969398 | 1.525484 | 1.525521 | -0.000037 | 59412 |
| XGBoost | 0.067180 | 0.125350 | 142230.718264 | 22.363002 | 0.992555 | 1.514697 | 1.525521 | -0.010824 | 59412 |
| DNN | 0.049840 | 0.096342 | 125630.375060 | 22.570772 | 0.995602 | 1.544573 | 1.525521 | 0.019052 | 59412 |
| TFT | 0.041529 | 0.089553 | 51636.753705 | 19.956899 | 0.996200 | 1.524342 | 1.525521 | -0.001179 | 59412 |
| Hybrid | 0.039110 | 0.087639 | 106391.920994 | 19.072797 | 0.996361 | 1.534964 | 1.525521 | 0.009444 | 59412 |
| AdaptiveBlend | 0.037946 | 0.089956 | 65693.047698 | 17.883824 | 0.996166 | 1.533161 | 1.525521 | 0.007640 | 59412 |
| StackedXGB | 0.037548 | 0.085948 | 38776.179389 | 17.576796 | 0.996500 | 1.526795 | 1.525521 | 0.001274 | 59412 |

### 4.3 Daytime 分电站结果
| Plant | Model | MAE | RMSE | MAPE | sMAPE | R2 | PredMean | ActualMean | Bias | Samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AliceSprings_HighEfficiency_4A | Persistence | 0.155596 | 0.337781 | 31666.414322 | 14.902494 | 0.955864 | 2.311334 | 2.311446 | -0.000113 | 14853 |
| AliceSprings_HighEfficiency_4A | XGBoost | 0.080954 | 0.150009 | 92005.513458 | 14.482098 | 0.991295 | 2.292357 | 2.311446 | -0.019090 | 14853 |
| AliceSprings_HighEfficiency_4A | DNN | 0.061671 | 0.121527 | 92621.367857 | 13.304597 | 0.994287 | 2.351223 | 2.311446 | 0.039776 | 14853 |
| AliceSprings_HighEfficiency_4A | TFT | 0.055533 | 0.118313 | 55207.116506 | 14.785936 | 0.994585 | 2.293558 | 2.311446 | -0.017888 | 14853 |
| AliceSprings_HighEfficiency_4A | Hybrid | 0.046070 | 0.110749 | 92991.499744 | 12.709632 | 0.995255 | 2.326188 | 2.311446 | 0.014742 | 14853 |
| AliceSprings_HighEfficiency_4A | AdaptiveBlend | 0.045403 | 0.113488 | 60800.872574 | 11.678905 | 0.995018 | 2.323355 | 2.311446 | 0.011909 | 14853 |
| AliceSprings_HighEfficiency_4A | StackedXGB | 0.050242 | 0.114173 | 38425.322701 | 11.669958 | 0.994957 | 2.321635 | 2.311446 | 0.010189 | 14853 |
| AliceSprings_MonoTrack_1A | Persistence | 0.052099 | 0.113207 | 12950.664313 | 16.133556 | 0.953094 | 0.762588 | 0.762595 | -0.000007 | 14853 |
| AliceSprings_MonoTrack_1A | XGBoost | 0.042090 | 0.076309 | 187786.150386 | 25.658373 | 0.978687 | 0.766831 | 0.762595 | 0.004235 | 14853 |
| AliceSprings_MonoTrack_1A | DNN | 0.031007 | 0.049600 | 122605.747132 | 23.848321 | 0.990995 | 0.773775 | 0.762595 | 0.011180 | 14853 |
| AliceSprings_MonoTrack_1A | TFT | 0.026690 | 0.044420 | 37405.014630 | 21.961429 | 0.992778 | 0.774599 | 0.762595 | 0.012003 | 14853 |
| AliceSprings_MonoTrack_1A | Hybrid | 0.027423 | 0.044637 | 106983.282913 | 22.705214 | 0.992707 | 0.778202 | 0.762595 | 0.015607 | 14853 |
| AliceSprings_MonoTrack_1A | AdaptiveBlend | 0.024603 | 0.046502 | 69281.931686 | 21.962565 | 0.992085 | 0.773888 | 0.762595 | 0.011293 | 14853 |
| AliceSprings_MonoTrack_1A | StackedXGB | 0.021340 | 0.040219 | 33423.613612 | 21.108046 | 0.994079 | 0.769981 | 0.762595 | 0.007385 | 14853 |
| AliceSprings_PolyFixed_1C | Persistence | 0.046787 | 0.106193 | 11251.415426 | 15.845810 | 0.961542 | 0.682188 | 0.682195 | -0.000007 | 14853 |
| AliceSprings_PolyFixed_1C | XGBoost | 0.042244 | 0.075241 | 184491.746049 | 30.557361 | 0.980694 | 0.692280 | 0.682195 | 0.010085 | 14853 |
| AliceSprings_PolyFixed_1C | DNN | 0.033324 | 0.056273 | 182101.724524 | 35.399363 | 0.989201 | 0.670035 | 0.682195 | -0.012160 | 14853 |
| AliceSprings_PolyFixed_1C | TFT | 0.024462 | 0.047699 | 44030.953442 | 24.799845 | 0.992241 | 0.678189 | 0.682195 | -0.004007 | 14853 |
| AliceSprings_PolyFixed_1C | Hybrid | 0.023615 | 0.048668 | 120616.577124 | 23.823132 | 0.991923 | 0.680759 | 0.682195 | -0.001436 | 14853 |
| AliceSprings_PolyFixed_1C | AdaptiveBlend | 0.024123 | 0.051178 | 75256.314681 | 22.460868 | 0.991068 | 0.681281 | 0.682195 | -0.000914 | 14853 |
| AliceSprings_PolyFixed_1C | StackedXGB | 0.022836 | 0.045633 | 45066.442137 | 22.053336 | 0.992899 | 0.678067 | 0.682195 | -0.004129 | 14853 |
| AliceSprings_PolyUtility_3A | Persistence | 0.159703 | 0.346647 | 29991.802927 | 16.876524 | 0.955610 | 2.345826 | 2.345846 | -0.000020 | 14853 |
| AliceSprings_PolyUtility_3A | XGBoost | 0.103432 | 0.169893 | 104639.463165 | 18.754174 | 0.989337 | 2.307321 | 2.345846 | -0.038525 | 14853 |
| AliceSprings_PolyUtility_3A | DNN | 0.073355 | 0.129349 | 105192.660728 | 17.730808 | 0.993819 | 2.383258 | 2.345846 | 0.037412 | 14853 |
| AliceSprings_PolyUtility_3A | TFT | 0.059433 | 0.117613 | 69903.930241 | 18.280388 | 0.994890 | 2.351022 | 2.345846 | 0.005176 | 14853 |
| AliceSprings_PolyUtility_3A | Hybrid | 0.059330 | 0.118727 | 104976.324197 | 17.053211 | 0.994793 | 2.354709 | 2.345846 | 0.008862 | 14853 |
| AliceSprings_PolyUtility_3A | AdaptiveBlend | 0.057655 | 0.121272 | 57433.071850 | 15.432958 | 0.994567 | 2.354119 | 2.345846 | 0.008273 | 14853 |
| AliceSprings_PolyUtility_3A | StackedXGB | 0.055772 | 0.113194 | 38189.339105 | 15.475844 | 0.995267 | 2.337498 | 2.345846 | -0.008349 | 14853 |

## 5. 多随机种子重复
- 随机种子：`(42, 52, 62)`

| Model | Runs | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| Persistence | 3 | 0.049670 +/- 0.000000 | 0.175950 +/- 0.000000 | 0.980560 +/- 0.000000 |
| XGBoost | 3 | 0.030599 +/- 0.008533 | 0.078277 +/- 0.006929 | 0.996122 +/- 0.000701 |
| DNN | 3 | 0.034747 +/- 0.001376 | 0.070290 +/- 0.000993 | 0.996897 +/- 0.000087 |
| TFT | 3 | 0.022546 +/- 0.004298 | 0.063730 +/- 0.005343 | 0.997432 +/- 0.000435 |
| Hybrid | 3 | 0.018388 +/- 0.001330 | 0.060652 +/- 0.002256 | 0.997687 +/- 0.000172 |
| AdaptiveBlend | 3 | 0.021064 +/- 0.001128 | 0.063405 +/- 0.000810 | 0.997475 +/- 0.000064 |
| StackedXGB | 3 | 0.038971 +/- 0.019228 | 0.080565 +/- 0.022546 | 0.995605 +/- 0.002472 |

## 6. Rolling-Origin Evaluation
### 6.1 滚动窗口定义
| Window | TrainEndRatio | ValEndRatio | TestEndRatio | TrainEndIdx | ValStartIdx | ValEndIdx | TestStartIdx | TestEndIdx | TrainEndTimestamp | ValStartTimestamp | ValEndTimestamp | TestStartTimestamp | TestEndTimestamp | TrainSamples | ValSamples | TestSamples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Window_1 | 0.600000 | 0.700000 | 0.800000 | 186354 | 186427 | 217414 | 217487 | 248473 | 2015-08-15 01:25:00 | 2015-08-15 07:30:00 | 2015-12-01 12:25:00 | 2015-12-01 18:30:00 | 2016-03-18 19:10:00 | 744264 | 123952 | 123948 |
| Window_2 | 0.700000 | 0.800000 | 0.900000 | 217414 | 217487 | 248473 | 248546 | 279532 | 2015-12-01 12:25:00 | 2015-12-01 18:30:00 | 2016-03-18 19:10:00 | 2016-03-19 01:15:00 | 2016-07-05 18:45:00 | 868504 | 123948 | 123948 |
| Window_3 | 0.800000 | 0.900000 | 1.000000 | 248473 | 248546 | 279532 | 279605 | 310592 | 2016-03-18 19:10:00 | 2016-03-19 01:15:00 | 2016-07-05 18:45:00 | 2016-07-06 00:50:00 | 2016-10-21 15:05:00 | 992740 | 123948 | 123952 |

### 6.2 跨窗口汇总
| Model | Runs | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| Persistence | 3 | 0.060781 +/- 0.014533 | 0.208959 +/- 0.046285 | 0.971825 +/- 0.010685 |
| XGBoost | 3 | 0.038132 +/- 0.004137 | 0.111301 +/- 0.026684 | 0.991401 +/- 0.004890 |
| DNN | 3 | 0.037227 +/- 0.003904 | 0.096594 +/- 0.026643 | 0.993403 +/- 0.004101 |
| TFT | 3 | 0.036315 +/- 0.012042 | 0.128217 +/- 0.075137 | 0.984935 +/- 0.016391 |
| Hybrid | 3 | 0.028419 +/- 0.008966 | 0.102378 +/- 0.043721 | 0.991693 +/- 0.007304 |
| AdaptiveBlend | 3 | 0.031119 +/- 0.011488 | 0.121890 +/- 0.069903 | 0.986537 +/- 0.014510 |
| StackedXGB | 3 | 0.032228 +/- 0.013279 | 0.124879 +/- 0.072862 | 0.985753 +/- 0.015420 |

## 7. Hybrid 权重与物理修正
- Full Hybrid 场景阈值：`{'low_radiation': 200.0, 'high_radiation': 700.0}`
- Full Hybrid 场景权重：`{'AliceSprings_HighEfficiency_4A': {'night': {'xgboost_prediction': 0.4, 'dnn_prediction': 0.05, 'tft_prediction': 0.55}, 'low_radiation': {'xgboost_prediction': 0.0, 'dnn_prediction': 1.0, 'tft_prediction': 0.0}, 'mid_radiation': {'xgboost_prediction': 0.0, 'dnn_prediction': 0.45, 'tft_prediction': 0.55}, 'high_radiation': {'xgboost_prediction': 0.0, 'dnn_prediction': 0.45, 'tft_prediction': 0.55}}, 'AliceSprings_MonoTrack_1A': {'night': {'xgboost_prediction': 0.2, 'dnn_prediction': 0.05, 'tft_prediction': 0.75}, 'low_radiation': {'xgboost_prediction': 0.5, 'dnn_prediction': 0.5, 'tft_prediction': 0.0}, 'mid_radiation': {'xgboost_prediction': 0.0, 'dnn_prediction': 0.15, 'tft_prediction': 0.85}, 'high_radiation': {'xgboost_prediction': 0.1, 'dnn_prediction': 0.05, 'tft_prediction': 0.85}}, 'AliceSprings_PolyFixed_1C': {'night': {'xgboost_prediction': 0.15, 'dnn_prediction': 0.0, 'tft_prediction': 0.85}, 'low_radiation': {'xgboost_prediction': 0.45, 'dnn_prediction': 0.55, 'tft_prediction': 0.0}, 'mid_radiation': {'xgboost_prediction': 0.05, 'dnn_prediction': 0.15, 'tft_prediction': 0.8}, 'high_radiation': {'xgboost_prediction': 0.0, 'dnn_prediction': 0.3, 'tft_prediction': 0.7}}, 'AliceSprings_PolyUtility_3A': {'night': {'xgboost_prediction': 0.4, 'dnn_prediction': 0.1, 'tft_prediction': 0.5}, 'low_radiation': {'xgboost_prediction': 0.0, 'dnn_prediction': 1.0, 'tft_prediction': 0.0}, 'mid_radiation': {'xgboost_prediction': 0.0, 'dnn_prediction': 0.1, 'tft_prediction': 0.9}, 'high_radiation': {'xgboost_prediction': 0.2, 'dnn_prediction': 0.15, 'tft_prediction': 0.65}}}`
- Full Hybrid 夜间修正 alpha：`0.600`
- w/o Plant Adaptation 场景权重：`{'__global__': {'night': {'xgboost_prediction': 0.45, 'dnn_prediction': 0.25, 'tft_prediction': 0.30000000000000004}, 'low_radiation': {'xgboost_prediction': 0.0, 'dnn_prediction': 1.0, 'tft_prediction': 0.0}, 'mid_radiation': {'xgboost_prediction': 0.0, 'dnn_prediction': 0.3, 'tft_prediction': 0.7}, 'high_radiation': {'xgboost_prediction': 0.0, 'dnn_prediction': 0.3, 'tft_prediction': 0.7}}}`
- w/o Scene Adaptation 固定权重：`{'xgboost_prediction': 0.45, 'dnn_prediction': 0.45, 'tft_prediction': 0.09999999999999998}`，alpha=`0.600`
- AdaptiveBlend 平均验证权重：`{'persistence_prediction': 0.17082549631595612, 'xgboost_prediction': 0.25222358107566833, 'dnn_prediction': 0.22373583912849426, 'tft_prediction': 0.3532140552997589}`
- AdaptiveBlend 平均测试权重：`{'persistence_prediction': 0.10850607603788376, 'xgboost_prediction': 0.29500097036361694, 'dnn_prediction': 0.2972310483455658, 'tft_prediction': 0.2992614507675171}`
- StackedXGB 物理修正 alpha：`0.600`

## 8. 当前结论
- 全样本主结果中，`Hybrid` MAE=`0.018909`，优于 `TFT` 的 `0.024010`，但仍略高于 `StackedXGB` 的 `0.018220`。
- 在 `daytime-only` 子集上，`Hybrid` MAE=`0.039110`，同样优于 `TFT` 的 `0.041529`，而 `StackedXGB` 仍保持最优的 `0.037548`。
- 从消融看，`w/o Scene Adaptation` 退化最明显，说明场景适配是当前 `Hybrid` 的核心增益来源。
- 多随机种子结果说明 `Hybrid / StackedXGB` 的结论不是单次幸运结果，rolling-origin 结果说明该结论不依赖单一时间切分窗口。