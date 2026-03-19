from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pvbench import ExperimentConfig
from pvbench.reporting import PLANT_TITLE_MAP

plt.switch_backend("Agg")

COLORS = {
    "Persistence": "#9c755f",
    "XGBoost": "#4c78a8",
    "DNN": "#72b7b2",
    "TFT": "#f58518",
    "Hybrid": "#e45756",
    "AdaptiveBlend": "#54a24b",
    "StackedXGB": "#b279a2",
}

PREDICTION_COLUMN_MAP = {
    "XGBoost": "xgboost_prediction",
    "TFT": "tft_prediction",
    "Hybrid": "hybrid_prediction",
    "AdaptiveBlend": "adaptive_blend_prediction",
    "StackedXGB": "stacked_xgboost_prediction",
}


def plot_baseline_overview(baseline: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        axis.bar(
            baseline["Model"],
            baseline[metric],
            color=[COLORS.get(name, "#4c78a8") for name in baseline["Model"]],
        )
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=22)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_daytime_overview(daytime_baseline: pd.DataFrame, output: Path) -> None:
    plot_baseline_overview(daytime_baseline, output)


def plot_ablation_overview(ablation: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, metric in zip(axes, ["MAE", "RMSE"]):
        axis.bar(ablation["Model"], ablation[metric], color="#4c78a8")
        axis.set_title(f"Ablation {metric}")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_plant_heatmap(plant_table: pd.DataFrame, metric: str, output: Path) -> None:
    pivot = plant_table.pivot(index="Plant", columns="Model", values=metric).copy()
    pivot.index = [PLANT_TITLE_MAP.get(name, name) for name in pivot.index]

    figure, axis = plt.subplots(figsize=(9, 5.2))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")
    axis.set_xticks(np.arange(len(pivot.columns)))
    axis.set_xticklabels(pivot.columns, rotation=22)
    axis.set_yticks(np.arange(len(pivot.index)))
    axis.set_yticklabels(pivot.index)
    axis.set_title(f"{metric} Heatmap by Plant and Model")

    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            axis.text(col, row, f"{pivot.iloc[row, col]:.3f}", ha="center", va="center", color="black", fontsize=8)

    figure.colorbar(image, ax=axis, shrink=0.88)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_relative_improvement(baseline: pd.DataFrame, output: Path) -> None:
    persistence_mae = float(baseline.loc[baseline["Model"] == "Persistence", "MAE"].iloc[0])
    plot_frame = baseline[baseline["Model"] != "Persistence"].copy()
    plot_frame["MAE_Improvement_%"] = (persistence_mae - plot_frame["MAE"]) / persistence_mae * 100.0

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        plot_frame["Model"],
        plot_frame["MAE_Improvement_%"],
        color=[COLORS.get(name, "#4c78a8") for name in plot_frame["Model"]],
    )
    axis.set_title("Relative MAE Improvement vs Persistence")
    axis.set_ylabel("Improvement (%)")
    axis.tick_params(axis="x", rotation=22)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_scatter_comparison(test_predictions: pd.DataFrame, output: Path) -> None:
    sample_size = min(10000, len(test_predictions))
    sample = test_predictions.sample(n=sample_size, random_state=42) if len(test_predictions) > sample_size else test_predictions

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    min_value = float(
        min(
            sample["target_power"].min(),
            sample["tft_prediction"].min(),
            sample["stacked_xgboost_prediction"].min(),
        )
    )
    max_value = float(
        max(
            sample["target_power"].max(),
            sample["tft_prediction"].max(),
            sample["stacked_xgboost_prediction"].max(),
        )
    )

    for axis, column, title, color in [
        (axes[0], "tft_prediction", "TFT", COLORS["TFT"]),
        (axes[1], "stacked_xgboost_prediction", "StackedXGB", COLORS["StackedXGB"]),
    ]:
        axis.scatter(sample["target_power"], sample[column], s=6, alpha=0.18, color=color, edgecolors="none")
        axis.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black", linewidth=1.0)
        axis.set_title(f"Actual vs {title}")
        axis.set_xlabel("Actual Power")
        axis.set_ylabel("Predicted Power")
        axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_residual_distribution(test_predictions: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    residual_columns = [
        ("tft_prediction", "TFT", COLORS["TFT"]),
        ("hybrid_prediction", "Hybrid", COLORS["Hybrid"]),
        ("adaptive_blend_prediction", "AdaptiveBlend", COLORS["AdaptiveBlend"]),
        ("stacked_xgboost_prediction", "StackedXGB", COLORS["StackedXGB"]),
    ]

    for column, label, color in residual_columns:
        residual = test_predictions[column].to_numpy(dtype=float) - test_predictions["target_power"].to_numpy(dtype=float)
        axes[0].hist(
            residual,
            bins=80,
            density=True,
            histtype="step",
            linewidth=1.6,
            color=color,
            label=label,
        )
    axes[0].set_title("Residual Density")
    axes[0].set_xlabel("Prediction Error")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    rmse_frame = []
    for column, label, _ in residual_columns:
        residual = test_predictions[column].to_numpy(dtype=float) - test_predictions["target_power"].to_numpy(dtype=float)
        rmse_frame.append({"Model": label, "ResidualStd": float(np.std(residual))})
    rmse_table = pd.DataFrame(rmse_frame)
    axes[1].bar(
        rmse_table["Model"],
        rmse_table["ResidualStd"],
        color=[COLORS.get(name, "#4c78a8") for name in rmse_table["Model"]],
    )
    axes[1].set_title("Residual Standard Deviation")
    axes[1].tick_params(axis="x", rotation=22)
    axes[1].grid(axis="y", alpha=0.2)

    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_hourly_mae_curve(test_predictions: pd.DataFrame, output: Path) -> None:
    frame = test_predictions.copy()
    frame["forecast_timestamp"] = pd.to_datetime(frame["forecast_timestamp"])
    frame["hour"] = frame["forecast_timestamp"].dt.hour

    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    for model_name in ["XGBoost", "TFT", "AdaptiveBlend", "StackedXGB"]:
        column = PREDICTION_COLUMN_MAP[model_name]
        mae_by_hour = (
            frame.assign(abs_error=np.abs(frame[column] - frame["target_power"]))
            .groupby("hour")["abs_error"]
            .mean()
            .reset_index()
        )
        axis.plot(
            mae_by_hour["hour"],
            mae_by_hour["abs_error"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=model_name,
            color=COLORS[model_name],
        )
    axis.set_title("Hourly MAE Curve")
    axis.set_xlabel("Hour of Day")
    axis.set_ylabel("MAE")
    axis.set_xticks(range(0, 24, 2))
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_gain_by_plant(plant_table: pd.DataFrame, output: Path) -> None:
    pivot = plant_table.pivot(index="Plant", columns="Model", values="MAE")
    gain_frame = pd.DataFrame(
        {
            "Plant": [PLANT_TITLE_MAP.get(name, name) for name in pivot.index],
            "StackedXGB_vs_TFT_%": (pivot["TFT"] - pivot["StackedXGB"]) / pivot["TFT"] * 100.0,
            "AdaptiveBlend_vs_TFT_%": (pivot["TFT"] - pivot["AdaptiveBlend"]) / pivot["TFT"] * 100.0,
        }
    )

    x_positions = np.arange(len(gain_frame))
    width = 0.35

    figure, axis = plt.subplots(figsize=(9, 4.6))
    axis.bar(
        x_positions - width / 2,
        gain_frame["AdaptiveBlend_vs_TFT_%"],
        width=width,
        color=COLORS["AdaptiveBlend"],
        label="AdaptiveBlend vs TFT",
    )
    axis.bar(
        x_positions + width / 2,
        gain_frame["StackedXGB_vs_TFT_%"],
        width=width,
        color=COLORS["StackedXGB"],
        label="StackedXGB vs TFT",
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(gain_frame["Plant"], rotation=18)
    axis.set_ylabel("MAE Improvement (%)")
    axis.set_title("Per-Plant Gain over TFT")
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_seed_stability(seed_summary: pd.DataFrame, output: Path) -> None:
    focus_models = ["DNN", "TFT", "Hybrid", "AdaptiveBlend", "StackedXGB"]
    plot_frame = seed_summary[seed_summary["Model"].isin(focus_models)].copy()

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for axis, metric in zip(axes, ["MAE", "RMSE"]):
        axis.bar(
            plot_frame["Model"],
            plot_frame[f"{metric}_mean"],
            yerr=plot_frame[f"{metric}_std"],
            capsize=4,
            color=[COLORS.get(name, "#4c78a8") for name in plot_frame["Model"]],
        )
        axis.set_title(f"Seed Stability: {metric}")
        axis.tick_params(axis="x", rotation=22)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_rolling_origin_overview(rolling_metrics: pd.DataFrame, output: Path) -> None:
    focus_models = ["TFT", "Hybrid", "StackedXGB"]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for axis, metric in zip(axes, ["MAE", "RMSE"]):
        for model_name in focus_models:
            model_frame = rolling_metrics[rolling_metrics["Model"] == model_name].copy()
            axis.plot(
                model_frame["Window"],
                model_frame[metric],
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=model_name,
                color=COLORS[model_name],
            )
        axis.set_title(f"Rolling-Origin {metric}")
        axis.set_xlabel("Window")
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_method_framework(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    boxes = [
        (0.04, 0.65, 0.22, 0.2, "#d7eaf3", "Input Features\nHistorical power\nWeather\nSolar geometry\nPhysics cues"),
        (0.33, 0.73, 0.16, 0.12, "#d9f0d3", "XGBoost"),
        (0.33, 0.55, 0.16, 0.12, "#d9f0d3", "DNN"),
        (0.33, 0.37, 0.16, 0.12, "#d9f0d3", "TFT"),
        (0.57, 0.60, 0.2, 0.16, "#fde0c5", "Adaptive Fusion / Stacking\nScene-aware combination"),
        (0.57, 0.34, 0.2, 0.14, "#f8d8e1", "Physics Correction\nLow-radiation / night adjustment"),
        (0.83, 0.52, 0.12, 0.16, "#e2d8f3", "Final Power\nForecast"),
    ]

    for x_pos, y_pos, width, height, color, text in boxes:
        rectangle = patches.FancyBboxPatch(
            (x_pos, y_pos),
            width,
            height,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor="#333333",
            facecolor=color,
        )
        axis.add_patch(rectangle)
        axis.text(x_pos + width / 2, y_pos + height / 2, text, ha="center", va="center", fontsize=11)

    arrow_specs = [
        ((0.26, 0.75), (0.33, 0.79)),
        ((0.26, 0.75), (0.33, 0.61)),
        ((0.26, 0.75), (0.33, 0.43)),
        ((0.49, 0.79), (0.57, 0.68)),
        ((0.49, 0.61), (0.57, 0.68)),
        ((0.49, 0.43), (0.57, 0.68)),
        ((0.67, 0.60), (0.67, 0.48)),
        ((0.77, 0.60), (0.83, 0.60)),
        ((0.77, 0.41), (0.83, 0.56)),
    ]
    for start, end in arrow_specs:
        axis.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", linewidth=1.8, color="#444444"))

    axis.text(0.34, 0.89, "Innovation 1: heterogeneous base learners", fontsize=10, color="#125b50")
    axis.text(0.56, 0.80, "Innovation 2: scene-adaptive fusion", fontsize=10, color="#8a4f00")
    axis.text(0.56, 0.26, "Innovation 3: physics-guided post-adjustment", fontsize=10, color="#8f2048")

    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_catalog(output: Path) -> None:
    content = "\n".join(
        [
            "# 论文图目录",
            "",
            "## 图1 baseline_overview.png",
            "- Baseline 模型在 MAE / RMSE / R2 上的总体比较。",
            "",
            "## 图2 daytime_baseline_overview.png",
            "- Daytime-only 子集上的总体比较。",
            "",
            "## 图3 ablation_overview.png",
            "- 完整模型与各消融设置在 MAE / RMSE 上的比较。",
            "",
            "## 图4 plant_mae_heatmap.png",
            "- 不同电站与不同模型的 MAE 热力图。",
            "",
            "## 图5 plant_rmse_heatmap.png",
            "- 不同电站与不同模型的 RMSE 热力图。",
            "",
            "## 图6 relative_improvement.png",
            "- 各模型相对 Persistence 的 MAE 提升比例。",
            "",
            "## 图7 scatter_comparison.png",
            "- TFT 与 StackedXGB 的真实值-预测值散点拟合图。",
            "",
            "## 图8 residual_distribution.png",
            "- 主要模型残差分布与残差标准差比较。",
            "",
            "## 图9 hourly_mae_curve.png",
            "- 不同小时段的 MAE 曲线，可用于说明场景差异。",
            "",
            "## 图10 plant_gain_over_tft.png",
            "- AdaptiveBlend / StackedXGB 相对 TFT 的分电站收益。",
            "",
            "## 图11 seed_stability.png",
            "- 多随机种子重复下 MAE / RMSE 的均值和标准差。",
            "",
            "## 图12 rolling_origin_overview.png",
            "- rolling-origin 各窗口下 TFT / Hybrid / StackedXGB 的指标曲线。",
            "",
            "## 图13 method_framework.png",
            "- 方法框架图，可直接放到论文方法部分。",
            "",
            "## 已有图复用建议",
            "- `forecast_examples.png`：预测效果示意图。",
            "- `training_curves.png`：训练收敛过程图。",
            "- `baseline_mae_rmse.png`：简化版 baseline 柱状图。",
        ]
    )
    output.write_text(content, encoding="utf-8-sig")


def main() -> None:
    config = ExperimentConfig()
    metrics_dir = config.metric_dir
    paper_dir = config.paper_dir
    paper_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(metrics_dir / "baseline_metrics.csv")
    baseline_daytime = pd.read_csv(metrics_dir / "baseline_daytime_metrics.csv")
    ablation = pd.read_csv(metrics_dir / "ablation_metrics.csv")
    plant_table = pd.read_csv(metrics_dir / "plant_level_metrics.csv")
    seed_summary = pd.read_csv(metrics_dir / "seed_repeat_summary.csv")
    rolling_metrics = pd.read_csv(metrics_dir / "rolling_origin_metrics.csv")
    test_predictions = pd.read_csv(metrics_dir / "test_predictions.csv")

    plot_baseline_overview(baseline, paper_dir / "baseline_overview.png")
    plot_daytime_overview(baseline_daytime, paper_dir / "daytime_baseline_overview.png")
    plot_ablation_overview(ablation, paper_dir / "ablation_overview.png")
    plot_plant_heatmap(plant_table, "MAE", paper_dir / "plant_mae_heatmap.png")
    plot_plant_heatmap(plant_table, "RMSE", paper_dir / "plant_rmse_heatmap.png")
    plot_relative_improvement(baseline, paper_dir / "relative_improvement.png")
    plot_scatter_comparison(test_predictions, paper_dir / "scatter_comparison.png")
    plot_residual_distribution(test_predictions, paper_dir / "residual_distribution.png")
    plot_hourly_mae_curve(test_predictions, paper_dir / "hourly_mae_curve.png")
    plot_gain_by_plant(plant_table, paper_dir / "plant_gain_over_tft.png")
    plot_seed_stability(seed_summary, paper_dir / "seed_stability.png")
    plot_rolling_origin_overview(rolling_metrics, paper_dir / "rolling_origin_overview.png")
    plot_method_framework(paper_dir / "method_framework.png")
    write_catalog(paper_dir / "paper_figure_catalog_zh.md")
    print("saved", paper_dir)


if __name__ == "__main__":
    main()
