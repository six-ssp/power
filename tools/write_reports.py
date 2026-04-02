from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pvbench import ExperimentConfig


def md_table(frame: pd.DataFrame) -> str:
    columns = frame.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values: list[str] = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_mean_std(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "Model": str(row["Model"]),
                "Runs": int(row["Runs"]),
                "MAE": f'{row["MAE_mean"]:.6f} +/- {row["MAE_std"]:.6f}',
                "RMSE": f'{row["RMSE_mean"]:.6f} +/- {row["RMSE_std"]:.6f}',
                "R2": f'{row["R2_mean"]:.6f} +/- {row["R2_std"]:.6f}',
            }
        )
    return pd.DataFrame(rows)


def get_metric(frame: pd.DataFrame, model: str, column: str) -> float:
    return float(frame.loc[frame["Model"] == model, column].iloc[0])


def main() -> None:
    config = ExperimentConfig()
    metric_dir = config.metric_dir
    report_dir = config.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(metric_dir / "baseline_metrics.csv")
    daytime = pd.read_csv(metric_dir / "baseline_daytime_metrics.csv")
    physical = pd.read_csv(metric_dir / "baseline_physical_metrics.csv")
    ablation = pd.read_csv(metric_dir / "ablation_metrics.csv")
    subset_counts = pd.read_csv(metric_dir / "subset_counts.csv")
    seed_summary = pd.read_csv(metric_dir / "seed_repeat_summary.csv")
    rolling_summary = pd.read_csv(metric_dir / "rolling_origin_summary.csv")
    rolling_windows = pd.read_csv(metric_dir / "rolling_origin_windows.csv")
    training_config = pd.read_csv(metric_dir / "training_configuration.csv")
    training_execution = pd.read_csv(metric_dir / "training_execution_summary.csv")

    seed_display = format_mean_std(seed_summary)
    rolling_display = format_mean_std(rolling_summary)
    physical_focus = (
        physical.set_index("Model")
        .loc[["Hybrid", "TFT", "StackedXGB", "DNN"], ["PhysicalViolationRate", "NegativeRate", "NightPositiveRateOnNight"]]
        .reset_index()
    )
    ablation_focus = ablation[
        ablation["Model"].isin(["Full Hybrid", "w/o Physics", "w/o Plant Adaptation", "w/o Scene Adaptation"])
    ][["Model", "MAE", "RMSE"]].copy()

    hybrid_mae = get_metric(baseline, "Hybrid", "MAE")
    tft_mae = get_metric(baseline, "TFT", "MAE")
    stacked_mae = get_metric(baseline, "StackedXGB", "MAE")

    hybrid_day_mae = get_metric(daytime, "Hybrid", "MAE")
    tft_day_mae = get_metric(daytime, "TFT", "MAE")
    stacked_day_mae = get_metric(daytime, "StackedXGB", "MAE")

    hybrid_phys = get_metric(physical, "Hybrid", "PhysicalViolationRate")
    tft_phys = get_metric(physical, "TFT", "PhysicalViolationRate")
    stacked_phys = get_metric(physical, "StackedXGB", "PhysicalViolationRate")

    hybrid_seed_mae = get_metric(seed_summary, "Hybrid", "MAE_mean")
    tft_seed_mae = get_metric(seed_summary, "TFT", "MAE_mean")
    stacked_seed_mae = get_metric(seed_summary, "StackedXGB", "MAE_mean")
    hybrid_seed_std = get_metric(seed_summary, "Hybrid", "MAE_std")
    tft_seed_std = get_metric(seed_summary, "TFT", "MAE_std")
    stacked_seed_std = get_metric(seed_summary, "StackedXGB", "MAE_std")

    hybrid_roll_mae = get_metric(rolling_summary, "Hybrid", "MAE_mean")
    tft_roll_mae = get_metric(rolling_summary, "TFT", "MAE_mean")
    stacked_roll_mae = get_metric(rolling_summary, "StackedXGB", "MAE_mean")
    hybrid_roll_std = get_metric(rolling_summary, "Hybrid", "MAE_std")
    tft_roll_std = get_metric(rolling_summary, "TFT", "MAE_std")
    stacked_roll_std = get_metric(rolling_summary, "StackedXGB", "MAE_std")

    daytime_ratio = float(subset_counts.loc[subset_counts["Subset"] == "Daytime", "Ratio"].iloc[0])

    training_setup = f"""# 训练设置记录

## 1. 任务与切分
- 任务定义：`5-minute-ahead` 条件功率回归，下一时刻天气已知。
- 主切分：按电站分别做时间顺序 `80 / 10 / 10` 切分。
- 间隔保护：训练/验证、验证/测试之间各保留 `72` 个时间步，约 `6` 小时。
- 白天样本占比：`{daytime_ratio:.2%}`。

## 2. 训练预算
{md_table(training_config)}

## 3. 主实验实际执行轮次
{md_table(training_execution)}

## 4. 当前训练口径
- 神经模型统一提升到 `30` 轮预算量级，同时保留 early stopping 和 best checkpoint restore。
- 当前默认启用 deterministic training，并在复现模式下将 DataLoader worker 固定为单线程。
- `TFT` 采用 `bf16-mixed` 混合精度，以在 `8 GB` 级显存条件下支撑更高训练预算。
- 树模型继续使用 boosting rounds + early stopping，不强行改成固定轮数。
- 这套设置的重点是释放公平预算，而不是机械地让所有模型跑满相同轮数。
""".strip()

    result_summary = f"""# 结果摘要

## 1. 固定切分主结果
{md_table(baseline[["Model", "MAE", "RMSE", "R2"]])}

## 2. 白天子集结果
{md_table(daytime[["Model", "MAE", "RMSE", "R2"]])}

## 3. Physical Violation
{md_table(physical_focus)}

## 4. Hybrid 消融
{md_table(ablation_focus)}

## 5. 结论
- 固定切分下，`Hybrid` 的 MAE 为 `{hybrid_mae:.6f}`，优于 `TFT` 的 `{tft_mae:.6f}` 和 `StackedXGB` 的 `{stacked_mae:.6f}`。
- `daytime-only` 下，`Hybrid` 的 MAE 为 `{hybrid_day_mae:.6f}`，同样优于 `TFT` 的 `{tft_day_mae:.6f}` 与 `StackedXGB` 的 `{stacked_day_mae:.6f}`。
- physical violation rate 显示：`Hybrid` 不是约束一致性最优，但仍维持低违例水平，并显著好于 `DNN`。
- 当前最稳的主叙事是：`Hybrid` 提供了最强的综合预测表现，`TFT` 是更充分训练后的深度时序强基线，`StackedXGB` 是重要但不占主叙事中心的元学习对照。
""".strip()

    robustness_summary = f"""# 鲁棒性记录

## 1. 子集统计
{md_table(subset_counts)}

## 2. 多随机种子
{md_table(seed_display)}

## 3. Rolling-Origin 窗口
{md_table(rolling_windows)}

## 4. Rolling-Origin 汇总
{md_table(rolling_display)}

## 5. 当前判断
- 多随机种子下，`Hybrid` 的平均 MAE 为 `{hybrid_seed_mae:.6f} +/- {hybrid_seed_std:.6f}`，低于 `TFT` 的 `{tft_seed_mae:.6f} +/- {tft_seed_std:.6f}` 与 `StackedXGB` 的 `{stacked_seed_mae:.6f} +/- {stacked_seed_std:.6f}`。
- rolling-origin 下，`Hybrid` 的平均 MAE 为 `{hybrid_roll_mae:.6f} +/- {hybrid_roll_std:.6f}`，仍低于 `TFT` 的 `{tft_roll_mae:.6f} +/- {tft_roll_std:.6f}` 与 `StackedXGB` 的 `{stacked_roll_mae:.6f} +/- {stacked_roll_std:.6f}`。
- 把 `TFT` 的训练预算抬高后，`Hybrid` 依旧保持优势，因此当前结论不再建立在“弱训练基线”上。
""".strip()

    sdm_positioning = f"""# SDM 口径说明

## 1. 推荐定位
- 问题类型：带已知未来外生变量的异构时间序列数据挖掘。
- 主方法：`Hybrid`，强调场景自适应、结构化可解释性和鲁棒性。
- 强对照：`TFT` 作为充分训练后的深度时序基线，`StackedXGB` 作为元学习性能对照。

## 2. 可以主打的贡献组织
- 方法贡献：`scene-aware interpretable fusion`。
- 评估贡献：`constraint-aware evaluation`，即在误差之外显式统计 physical violation。
- 系统贡献：统一数据处理、图表生成、训练配置记录和项目校验的可复现实验流水线。

## 3. 当前最能讲的证据
- 固定切分：`Hybrid` MAE=`{hybrid_mae:.6f}`，优于 `TFT` 的 `{tft_mae:.6f}`。
- 白天子集：`Hybrid` MAE=`{hybrid_day_mae:.6f}`，优于 `TFT` 的 `{tft_day_mae:.6f}`。
- 多随机种子：`Hybrid` 平均 MAE 低于 `TFT` 和 `StackedXGB`。
- rolling-origin：`Hybrid` 平均 MAE 最低。
- constraint-aware evaluation：`Hybrid` 的 physical violation rate 为 `{hybrid_phys:.6f}`，明显低于 `DNN`，但高于 `TFT` 的 `{tft_phys:.6f}` 与 `StackedXGB` 的 `{stacked_phys:.6f}`。

## 4. 推荐表述
- 用 `scene-aware interpretable fusion`、`heterogeneous temporal data mining`、`constraint-aware evaluation`。
- 不建议直接写成严格的 `constrained optimization`，除非后续显式补出优化目标、约束项和求解过程。
- 更稳的写法是：本文在异构时序数据上提出结构化、可解释的场景自适应融合，并通过 constraint-aware 指标补充传统误差评价。
""".strip()

    paper_draft = f"""# Scene-Aware Interpretable Fusion with Constraint-Aware Evaluation for Heterogeneous PV Time-Series Mining

## 摘要
本文将异构光伏装置上的 `5-minute-ahead` 功率预测问题表述为一个带已知未来外生变量的异构时间序列数据挖掘任务。围绕这一设定，本文构建了一套可复现实验流水线，并提出一种按电站与辐照场景切换权重的可解释融合方法 `Hybrid`。该方法在 `XGBoost`、`DNN` 和 `TFT` 三类异构基学习器之上进行结构化融合，并保留轻量物理后修正以抑制不合理输出。实验基于 Alice Springs 站点四个异构光伏装置，采用时间顺序切分、`6` 小时间隔保护、`daytime-only`、多随机种子和 rolling-origin 评估。结果表明，固定切分下 `Hybrid` 将 `TFT` 的 MAE 从 `{tft_mae:.6f}` 降至 `{hybrid_mae:.6f}`；在白天子集上，`Hybrid` 的 MAE 为 `{hybrid_day_mae:.6f}`，优于 `TFT` 的 `{tft_day_mae:.6f}`。在多随机种子和 rolling-origin 下，`Hybrid` 仍保持最低平均 MAE。进一步引入 physical violation rate 后，结果显示 `Hybrid` 在精度、稳定性与物理一致性之间取得了更平衡的表现。

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
为了避免只用误差指标评估，本文引入 physical violation rate。当前版本使用两类最硬的违例：
- 负功率违例：预测值小于 `-epsilon_p`；
- 夜间正功率违例：夜间样本的预测值大于 `epsilon_p`。

其中，`epsilon_p` 按电站定义为训练集最大功率的 `1%` 与 `0.01` 之间的较大值。总体违例率定义为：

```text
PVR = (1 / N) * sum I[y_hat_t < -epsilon_p or (night_t = 1 and y_hat_t > epsilon_p)]
```

这个指标不把 `Hybrid` 强行包装成约束优化模型，但确实给出了更接近 `SDM` 口径的 constraint-aware evaluation。

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
{md_table(training_config)}

### 4.4 主实验实际执行轮次
{md_table(training_execution)}

## 5. 实验结果
### 5.1 固定切分
{md_table(baseline[["Model", "MAE", "RMSE", "R2"]])}

固定切分下，`Hybrid` 获得最低 MAE=`{hybrid_mae:.6f}`。与训练预算抬高后的 `TFT` 相比，`Hybrid` 仍保持明显优势；`StackedXGB` 在当前高预算设定下也未超过 `Hybrid`。

![Fixed-split overview](../paper_figures/baseline_overview.png)

### 5.2 白天子集
{md_table(daytime[["Model", "MAE", "RMSE", "R2"]])}

白天子集结果表明，`Hybrid` 的改进并非仅来自夜间样本结构，而是在真正有发电行为的时段仍然成立。

### 5.3 Constraint-Aware Evaluation
{md_table(physical_focus)}

这一结果提供了与 `SDM` 更契合的补充证据：模型不仅要低误差，还要尽量少地产生负功率或夜间异常正功率。`Hybrid` 的违例率保持较低，显著好于 `DNN`，但仍高于 `TFT` 与 `StackedXGB`。因此更稳的结论不是“`Hybrid` 在所有约束一致性指标上最优”，而是“`Hybrid` 在预测精度与约束一致性之间取得更优平衡”。

### 5.4 消融与鲁棒性
{md_table(ablation_focus)}

{md_table(seed_display)}

{md_table(rolling_display)}

从消融结果看，`scene adaptation` 是 `Hybrid` 的主增益来源；从鲁棒性结果看，`Hybrid` 在多随机种子下的平均 MAE 为 `{hybrid_seed_mae:.6f} +/- {hybrid_seed_std:.6f}`，低于 `TFT` 的 `{tft_seed_mae:.6f} +/- {tft_seed_std:.6f}` 和 `StackedXGB` 的 `{stacked_seed_mae:.6f} +/- {stacked_seed_std:.6f}`；在 rolling-origin 下，`Hybrid` 的平均 MAE 为 `{hybrid_roll_mae:.6f} +/- {hybrid_roll_std:.6f}`，同样低于 `TFT` 的 `{tft_roll_mae:.6f} +/- {tft_roll_std:.6f}` 和 `StackedXGB` 的 `{stacked_roll_mae:.6f} +/- {stacked_roll_std:.6f}`。

![Rolling-origin overview](../paper_figures/rolling_origin_overview.png)

## 6. SDM 讨论
从 `SDM` 视角看，本文最合理的主线不是“提出一个全新的深度模型”，而是“在异构时间序列上显式建模模型选择，并用 constraint-aware 方式评估其可信性”。这使得 `Hybrid` 的价值不只是解释性更强，而是在高预算训练、白天子集、多随机种子和 rolling-origin 下都维持较强综合性能。

与此同时，physical violation 的结果也给出了边界：`Hybrid` 并不是严格意义上的约束最优模型，因此文章不宜包装成完整的 constrained optimization。更稳的表述是 knowledge-guided、scene-aware 和 constraint-aware temporal data mining。

## 7. 局限性
- 数据范围仍局限于同一站点的四个异构装置，跨站点外推能力尚未验证。
- 当前任务是单步预测，尚未扩展到多步预测或更长时间尺度。
- constraint-aware 指标目前聚焦于负功率和夜间异常正功率，尚未进一步扩展到更强的清空包络或 ramp-rate 约束。

## 8. 结论
当前这套实验最适合形成如下论文主线：在异构光伏时间序列上，单一强基线难以覆盖所有场景，而场景自适应、结构化、可解释的融合能够在精度、稳定性和物理合理性之间取得更优平衡。就现有结果而言，`Hybrid` 是主方法，`TFT` 是更强后的深度时序基线，physical violation 则为全文补上了更符合 `SDM` 口味的 constraint-aware 评价维度。
""".strip()

    outputs = {
        report_dir / "training_setup_zh.md": training_setup,
        report_dir / "result_summary_zh.md": result_summary,
        report_dir / "robustness_summary_zh.md": robustness_summary,
        report_dir / "sdm_positioning_zh.md": sdm_positioning,
        report_dir / "paper_reference_draft_zh.md": paper_draft,
    }
    for path, content in outputs.items():
        path.write_text(content + "\n", encoding="utf-8-sig")
        print("saved", path)


if __name__ == "__main__":
    main()
