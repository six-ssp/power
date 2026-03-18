# 最终结果摘要

## 1. 推荐的论文主模型
- 首选主模型：`StackedXGB`
- 方法主线表述：`场景自适应异构集成 + 物理引导修正`
- 适合讲创新点的版本：`AdaptiveBlend`

## 2. 总体最优结果
引用文件：[baseline_metrics.csv](/e:/experiment/artifacts/metrics/baseline_metrics.csv)

- `StackedXGB`：`MAE=0.019204`，`RMSE=0.059470`，`R2=0.997776`
- `TFT`：`MAE=0.020026`，`RMSE=0.061620`，`R2=0.997612`
- `Hybrid`：`MAE=0.021223`，`RMSE=0.066422`，`R2=0.997225`

## 3. 相对提升
- 相对 `TFT`：
  - `MAE` 下降约 `4.10%`
  - `RMSE` 下降约 `3.49%`
- 相对固定权重 `Hybrid`：
  - `MAE` 下降约 `9.49%`
  - `RMSE` 下降约 `11.66%`

## 4. 分电站结论
引用文件：[plant_level_metrics.csv](/e:/experiment/artifacts/metrics/plant_level_metrics.csv)

- `1A`：`TFT` 最优
- `1C`：`TFT` 最优
- `3A`：`StackedXGB` 最优
- `4A`：`AdaptiveBlend` 在 `MAE` 上最优，`StackedXGB` 在 `RMSE` 上最优

说明：
- 提升并非只来自单一电站。
- 自适应融合在复杂工况和较难电站上的收益更明显。

## 5. 当前最适合写进正文的结论句
- 与单模型相比，场景自适应融合在多装置光伏场景下表现出更强的稳健性。
- 固定权重融合未能稳定超过 TFT，说明简单拼接难以适应时段与工况变化。
- 二阶段场景融合器 `StackedXGB` 取得了最优总体结果，验证了异构基模型之间存在可利用的误差互补性。

## 6. 建议优先使用的图
- 方法图：[method_framework.png](/e:/experiment/artifacts/paper_figures/method_framework.png)
- 总体结果图：[baseline_overview.png](/e:/experiment/artifacts/paper_figures/baseline_overview.png)
- 消融图：[ablation_overview.png](/e:/experiment/artifacts/paper_figures/ablation_overview.png)
- 散点拟合图：[scatter_comparison.png](/e:/experiment/artifacts/paper_figures/scatter_comparison.png)
- 残差图：[residual_distribution.png](/e:/experiment/artifacts/paper_figures/residual_distribution.png)
- 分时段误差图：[hourly_mae_curve.png](/e:/experiment/artifacts/paper_figures/hourly_mae_curve.png)

## 7. 现阶段不建议过度强调的点
- `MAPE`：受 `0/负功率` 影响过大，不适合作为主结论
- “固定权重 Hybrid” ：它适合当对照，不适合再当论文主方法
- “首次提出”一类表述：除非补充分文献检索，否则不要使用
