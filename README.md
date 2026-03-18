﻿# Alice Springs 光伏短时预测实验

本项目面向澳大利亚 Alice Springs 站点的 4 个不同光伏装置，完成了一个可复现的短时功率预测实验闭环：

- 统一英文代码与字段映射
- `8:1:1` 时间顺序划分
- `Persistence / XGBoost / DNN / TFT / Hybrid / AdaptiveBlend / StackedXGB` baseline 对比
- 组件与物理修正消融实验
- 自动生成指标表、预测效果图、训练损失图、中文实验记录
- 自动生成论文图、论文大纲和方法说明文档

## 1. 数据说明

原始数据位于 `dataset/`：

- `data_1A.csv`
- `data_1C.csv`
- `data_3A.csv`
- `dtat_4A.csv`

代码内部会将中文列名映射为英文列名，原始 CSV 文件不会被修改。

当前实验使用的站点假设：

- 位置：`Alice Springs, Australia`
- 纬度：`-23.6980`
- 经度：`133.8807`
- 时区偏移：`UTC+9.5`

## 2. 任务定义

当前版本统一做 **5 分钟 ahead 的单步短时预测**：

- 输入：历史功率、历史气象、未来一步已知气象、时间周期特征、太阳几何特征、物理启发特征
- 输出：下一时刻 `power`
- 划分：每个电站按时间顺序 `8:1:1`

之所以先统一成单步，是为了让 `XGBoost / DNN / TFT` 能在同一口径下直接公平对比，便于先完成基线和消融。

## 3. 特征设计

当前特征分为 5 组：

1. 原始气象特征：`irradiance / temperature / humidity / wind_speed / direct_radiation / global_radiation`
2. 时间周期特征：小时、年内天数、月份、星期的正余弦编码
3. 地理与太阳几何特征：太阳赤纬、时差、时角、太阳高度角、天顶角、`cos(zenith)`、晴空辐照代理等
4. 物理启发特征：温度降额、有效辐射、辐照-温度比、夜间标记等
5. 历史统计特征：多阶滞后、滚动均值、滚动标准差

## 4. 模型与实验设置

### Baseline

- `Persistence`：当前功率直接作为下一时刻预测
- `XGBoost`：树模型，验证集早停
- `DNN`：MLP，`SmoothL1Loss`
- `TFT`：多电站联合训练，`QuantileLoss`
- `Hybrid`：带正权重约束的全局加权融合，并加夜间低辐照物理修正
- `AdaptiveBlend`：基于 `Persistence / XGBoost / DNN / TFT` 的样本级动态权重融合
- `StackedXGB`：将基模型预测与气象/太阳几何上下文一起输入二阶段 XGBoost，并加低辐照物理修正

### 消融

- `w/o Physics`
- `Equal Weights`
- `w/o XGBoost`
- `w/o DNN`
- `w/o TFT`

### 固定权重 Hybrid 约束

对旧版 `Hybrid`，融合时强制：

- `XGBoost / DNN / TFT` 都必须有占比
- 当前最优权重：`XGBoost=0.70, DNN=0.20, TFT=0.10`
- 夜间物理修正系数：`alpha=0.6`

## 5. 当前实验结果

### Baseline 总表

| 模型 | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| Persistence | 0.049555 | 0.175746 | 0.980575 |
| XGBoost | 0.023927 | 0.072653 | 0.996680 |
| DNN | 0.037083 | 0.072354 | 0.996708 |
| TFT | 0.020026 | 0.061620 | 0.997612 |
| Hybrid | 0.021223 | 0.066422 | 0.997225 |
| AdaptiveBlend | 0.019451 | 0.062294 | 0.997559 |
| StackedXGB | 0.019204 | 0.059470 | 0.997776 |

### 消融总表

| 模型 | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| Full Hybrid | 0.021223 | 0.066422 | 0.997225 |
| w/o Physics | 0.021241 | 0.066422 | 0.997225 |
| Equal Weights | 0.020899 | 0.061008 | 0.997659 |
| w/o XGBoost | 0.035509 | 0.070710 | 0.996855 |
| w/o DNN | 0.022596 | 0.071181 | 0.996813 |
| w/o TFT | 0.022376 | 0.069026 | 0.997003 |
| Adaptive Blend | 0.019451 | 0.062294 | 0.997559 |
| Stacked XGB | 0.019204 | 0.059470 | 0.997776 |

### 结果解读

当前这组结果最需要正视的结论有三点：

1. **StackedXGB 已经超过当前 TFT 与固定权重 Hybrid**。
2. **AdaptiveBlend 在 MAE 上也优于当前 TFT**，说明“样本级动态融合”方向是有效的。
3. 旧版 **全局固定正权重 Hybrid 仍然没有稳定优于 TFT**，说明固定权重约束本身限制了融合上限。

这意味着：

- 现在这套代码已经不只是 baseline，而是有一版可以继续往论文主方法推进的 `StackedXGB` 融合结果。
- 如果要继续冲更强结果，重点应该放在“场景自适应融合 + 更强物理与静态先验”，而不是继续卡在固定权重 Hybrid 上。

## 6. 为什么 MAPE 很大

这套数据存在大量：

- `0` 功率
- 小幅负功率回流

因此 `MAPE` 会因为分母接近 `0` 而被放大，在本数据集上不适合作为主指标。当前更建议主看：

- `MAE`
- `RMSE`
- `R2`
- 预测均值与实际均值偏差

## 7. 运行方式

### 7.1 创建环境

建议使用项目内 `.venv`。

如果要重装 GPU 版 PyTorch：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

### 7.2 运行完整实验

```powershell
.\.venv\Scripts\python run_experiments.py
```

### 7.3 做项目检查

```powershell
.\.venv\Scripts\python tools\verify_project.py
```

### 7.4 生成论文图

```powershell
.\.venv\Scripts\python tools\generate_paper_figures.py
```

## 8. 输出目录

运行完成后会生成：

- `artifacts/metrics/baseline_metrics.csv`
- `artifacts/metrics/ablation_metrics.csv`
- `artifacts/metrics/plant_level_metrics.csv`
- `artifacts/metrics/validation_predictions.csv`
- `artifacts/metrics/test_predictions.csv`
- `artifacts/reports/training_log_zh.md`
- `artifacts/reports/paper_outline_zh.md`
- `artifacts/reports/method_story_zh.md`
- `artifacts/reports/result_summary_zh.md`
- `artifacts/checks/project_check_zh.md`
- `artifacts/paper_figures/`
- `artifacts/plots/baseline_mae_rmse.png`
- `artifacts/plots/forecast_examples.png`
- `artifacts/plots/training_curves.png`

当前 `artifacts/paper_figures/` 中重点建议直接用于论文的图包括：

- `method_framework.png`
- `baseline_overview.png`
- `ablation_overview.png`
- `scatter_comparison.png`
- `residual_distribution.png`
- `hourly_mae_curve.png`
- `plant_gain_over_tft.png`

## 9. 代码结构

- `run_experiments.py`：实验主入口
- `tools/verify_project.py`：项目检查与可复现性确认
- `tools/generate_paper_figures.py`：基于已有结果生成论文图
- `pvbench/config.py`：配置与站点/装置元数据
- `pvbench/data.py`：数据读取、英文映射、特征工程、划分
- `pvbench/models.py`：XGBoost、DNN、TFT、融合与物理修正
- `pvbench/reporting.py`：指标、图表、报告输出

## 10. 后续优化建议

如果目标是往 CCF-B 论文推进，优先建议按下面顺序继续：

1. 把任务从单步扩展到 `12~24` 步多步预测
2. 将 `AdaptiveBlend / StackedXGB` 扩展到更系统的场景自适应融合
3. 增加更明确的物理约束：夜间回流、温升降额、晴空上界、装置类型差异
4. 引入更强的站点静态信息：容量、倾角、朝向、是否跟踪
5. 对 `XGBoost / DNN / TFT` 分别做系统化超参搜索，而不是只做手工微调

## 11. 当前局限

- 固定权重 `Hybrid` 还没超过 `TFT`
- `StackedXGB` 虽然效果最好，但当前仍属于经验型二阶段融合，还可以继续增强物理可解释性
- `TFT` 样本量大，即使用 GPU 训练时间仍然偏长
- 站点静态信息目前主要来自装置名称解析，缺少容量/倾角等更强先验
- 当前实验为单步预测，多步预测结果还没有补齐
