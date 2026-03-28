from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pvbench import ExperimentConfig
from pvbench.reporting import PLANT_TITLE_MAP

plt.switch_backend("Agg")

BACKGROUND = "#ffffff"
PANEL = "#ffffff"
TEXT = "#2b2723"
MUTED = "#736b63"
GRID = "#d6d0c3"
EDGE = "#fdfcf9"

COLORS = {
    "Persistence": "#8DA0CB",
    "XGBoost": "#56B4E9",
    "DNN": "#66C2A5",
    "TFT": "#FC8D62",
    "Hybrid": "#D55E00",
    "AdaptiveBlend": "#A6D854",
    "StackedXGB": "#CC79A7",
}

DISPLAY_LABELS = {
    "Persistence": "Pers.",
    "XGBoost": "XGB",
    "DNN": "DNN",
    "TFT": "TFT",
    "Hybrid": "Hybrid",
    "AdaptiveBlend": "AdaBlend",
    "StackedXGB": "StackXGB",
}

HEATMAP_LABELS = {
    "Persistence": "Persist",
    "XGBoost": "XGB",
    "DNN": "DNN",
    "TFT": "TFT",
    "Hybrid": "Hybrid",
    "AdaptiveBlend": "AdaBlend",
    "StackedXGB": "Stacked",
}

PREDICTION_COLUMN_MAP = {
    "XGBoost": "xgboost_prediction",
    "TFT": "tft_prediction",
    "Hybrid": "hybrid_prediction",
    "AdaptiveBlend": "adaptive_blend_prediction",
    "StackedXGB": "stacked_xgboost_prediction",
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "paper_heat",
    ["#fff8d8", "#F0E442", "#E69F00", "#D55E00"],
)

plt.rcParams.update(
    {
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BACKGROUND,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "grid.color": GRID,
        "grid.alpha": 0.45,
        "grid.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Serif",
        "font.size": 12,
        "axes.titleweight": "semibold",
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "legend.frameon": True,
        "legend.facecolor": "#ffffff",
        "legend.edgecolor": "#d6cfbf",
    }
)


def style_axis(
    axis: plt.Axes,
    title: str,
    ylabel: str | None = None,
    xlabel: str | None = None,
    grid_axis: str = "y",
    tick_rotation: float = 0,
) -> None:
    axis.set_title(title, color=TEXT, pad=12)
    if ylabel:
        axis.set_ylabel(ylabel)
    if xlabel:
        axis.set_xlabel(xlabel)
    axis.set_axisbelow(True)
    axis.grid(axis=grid_axis, linestyle="-", linewidth=0.8, alpha=0.45)
    axis.tick_params(axis="x", rotation=tick_rotation)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(MUTED)
        axis.spines[spine].set_linewidth(0.9)


def order_models(frame: pd.DataFrame, column: str = "Model") -> pd.DataFrame:
    order = list(DISPLAY_LABELS.keys())
    return frame.assign(_order=frame[column].map({name: i for i, name in enumerate(order)})).sort_values("_order").drop(columns="_order")


def add_bar_labels(axis: plt.Axes, bars, fmt: str = "{:.3f}") -> None:
    ymin, ymax = axis.get_ylim()
    offset = (ymax - ymin) * 0.018
    for bar in bars:
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            color=MUTED,
        )


def model_colors(models: pd.Series) -> list[str]:
    return [COLORS.get(name, "#3a5f78") for name in models]


def model_edges(models: pd.Series) -> list[str]:
    return ["#8a3d00" if name == "Hybrid" else EDGE for name in models]


def plot_baseline_overview(baseline: pd.DataFrame, output: Path) -> None:
    plot_frame = order_models(baseline).copy()
    labels = [DISPLAY_LABELS[name] for name in plot_frame["Model"]]

    figure, axes = plt.subplots(1, 3, figsize=(16.4, 5.0))
    for axis, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        bars = axis.bar(
            labels,
            plot_frame[metric],
            color=model_colors(plot_frame["Model"]),
            edgecolor=model_edges(plot_frame["Model"]),
            linewidth=1.2,
            width=0.78,
        )
        style_axis(axis, metric, grid_axis="y", tick_rotation=0)
        axis.tick_params(axis="x", labelsize=11)
        axis.margins(x=0.05)
        if metric == "R2":
            lower = max(0.975, float(plot_frame[metric].min()) - 0.0015)
            axis.set_ylim(lower, 1.0006)
            add_bar_labels(axis, bars, "{:.4f}")
        else:
            axis.set_ylim(0, float(plot_frame[metric].max()) * 1.14)
            add_bar_labels(axis, bars, "{:.3f}")
    figure.tight_layout(w_pad=2.0)
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def plot_daytime_overview(daytime_baseline: pd.DataFrame, output: Path) -> None:
    plot_baseline_overview(daytime_baseline, output)


def plot_ablation_overview(ablation: pd.DataFrame, output: Path) -> None:
    rename_map = {
        "Full Hybrid": "Full\nHybrid",
        "w/o Physics": "w/o\nPhysics",
        "w/o Plant Adaptation": "w/o Plant\nAdaptation",
        "w/o Scene Adaptation": "w/o Scene\nAdaptation",
    }
    plot_frame = ablation[ablation["Model"].isin(rename_map.keys())].copy()
    plot_frame["VariantDisplay"] = plot_frame["Model"].map(rename_map)
    colors = ["#D55E00", "#FFD92F", "#8DA0CB", "#CC79A7"]

    figure, axes = plt.subplots(1, 2, figsize=(13.8, 4.8))
    for axis, metric in zip(axes, ["MAE", "RMSE"]):
        bars = axis.bar(
            plot_frame["VariantDisplay"],
            plot_frame[metric],
            color=colors,
            edgecolor=EDGE,
            linewidth=1.2,
            width=0.78,
        )
        style_axis(axis, f"Hybrid Ablation: {metric}", grid_axis="y", tick_rotation=0)
        axis.tick_params(axis="x", labelsize=11)
        axis.set_ylim(0, float(plot_frame[metric].max()) * 1.14)
        add_bar_labels(axis, bars, "{:.3f}")
    figure.tight_layout(w_pad=2.0)
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def plot_plant_heatmap(plant_table: pd.DataFrame, metric: str, output: Path) -> None:
    pivot = plant_table.pivot(index="Plant", columns="Model", values=metric).copy()
    pivot = pivot[[name for name in DISPLAY_LABELS if name in pivot.columns]]
    pivot.index = [PLANT_TITLE_MAP.get(name, name) for name in pivot.index]
    pivot.columns = [HEATMAP_LABELS.get(name, name) for name in pivot.columns]

    figure, axis = plt.subplots(figsize=(10.6, 5.4))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap=HEATMAP_CMAP)
    axis.set_xticks(np.arange(len(pivot.columns)))
    axis.set_xticklabels(pivot.columns, rotation=0)
    axis.set_yticks(np.arange(len(pivot.index)))
    axis.set_yticklabels(pivot.index)
    style_axis(axis, f"{metric} by Plant and Model", grid_axis="both", tick_rotation=0)
    axis.tick_params(axis="x", labelsize=11)
    axis.grid(False)

    data = pivot.to_numpy(dtype=float)
    threshold = np.nanmin(data) + (np.nanmax(data) - np.nanmin(data)) * 0.55
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col]
            axis.text(
                col,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="#fffdf9" if value >= threshold else TEXT,
                fontsize=9,
                fontweight="semibold" if value == np.nanmin(data[row]) else "normal",
            )

    cbar = figure.colorbar(image, ax=axis, shrink=0.88, pad=0.02)
    cbar.outline.set_edgecolor("#cbbfae")
    cbar.ax.tick_params(colors=TEXT)
    figure.tight_layout()
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def plot_relative_improvement(baseline: pd.DataFrame, output: Path) -> None:
    persistence_mae = float(baseline.loc[baseline["Model"] == "Persistence", "MAE"].iloc[0])
    plot_frame = baseline[baseline["Model"] != "Persistence"].copy()
    plot_frame["MAE_Improvement_%"] = (persistence_mae - plot_frame["MAE"]) / persistence_mae * 100.0
    plot_frame = order_models(plot_frame)
    labels = [DISPLAY_LABELS[name] for name in plot_frame["Model"]]

    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    bars = axis.bar(
        labels,
        plot_frame["MAE_Improvement_%"],
        color=model_colors(plot_frame["Model"]),
        edgecolor=model_edges(plot_frame["Model"]),
        linewidth=1.2,
        width=0.78,
    )
    style_axis(axis, "Relative MAE Improvement over Persistence", ylabel="Improvement (%)", grid_axis="y", tick_rotation=0)
    axis.tick_params(axis="x", labelsize=11)
    axis.set_ylim(0, float(plot_frame["MAE_Improvement_%"].max()) * 1.16)
    add_bar_labels(axis, bars, "{:.1f}")
    figure.tight_layout()
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def plot_scatter_comparison(test_predictions: pd.DataFrame, output: Path) -> None:
    sample_size = min(12000, len(test_predictions))
    sample = test_predictions.sample(n=sample_size, random_state=42) if len(test_predictions) > sample_size else test_predictions

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), sharex=True, sharey=True)
    min_value = float(min(sample["target_power"].min(), sample["tft_prediction"].min(), sample["hybrid_prediction"].min()))
    max_value = float(max(sample["target_power"].max(), sample["tft_prediction"].max(), sample["hybrid_prediction"].max()))

    for axis, column, title, color in [
        (axes[0], "tft_prediction", "TFT", COLORS["TFT"]),
        (axes[1], "hybrid_prediction", "Hybrid", COLORS["Hybrid"]),
    ]:
        axis.scatter(sample["target_power"], sample[column], s=7, alpha=0.18, color=color, edgecolors="none")
        axis.plot([min_value, max_value], [min_value, max_value], linestyle="--", color=MUTED, linewidth=1.1)
        style_axis(axis, f"Actual vs {title}", xlabel="Actual Power", ylabel="Predicted Power", grid_axis="both")
        axis.set_xlim(min_value, max_value)
        axis.set_ylim(min_value, max_value)

    figure.tight_layout(w_pad=1.8)
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def plot_residual_distribution(test_predictions: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    residual_columns = [
        ("tft_prediction", "TFT", COLORS["TFT"]),
        ("hybrid_prediction", "Hybrid", COLORS["Hybrid"]),
        ("stacked_xgboost_prediction", "StackedXGB", COLORS["StackedXGB"]),
    ]

    for column, label, color in residual_columns:
        residual = test_predictions[column].to_numpy(dtype=float) - test_predictions["target_power"].to_numpy(dtype=float)
        axes[0].hist(
            residual,
            bins=90,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=label,
        )
    axes[0].axvline(0.0, color=MUTED, linewidth=1.0, linestyle="--")
    style_axis(axes[0], "Residual Density", xlabel="Prediction Error", ylabel="Density", grid_axis="both")
    axes[0].legend(loc="upper right")

    residual_std_rows = []
    for column, label, _ in residual_columns:
        residual = test_predictions[column].to_numpy(dtype=float) - test_predictions["target_power"].to_numpy(dtype=float)
        residual_std_rows.append({"Model": label, "ResidualStd": float(np.std(residual))})
    rmse_table = pd.DataFrame(residual_std_rows)
    bars = axes[1].bar(
        [DISPLAY_LABELS.get(name, name) for name in rmse_table["Model"]],
        rmse_table["ResidualStd"],
        color=[COLORS.get(name, "#3a5f78") for name in rmse_table["Model"]],
        edgecolor=["#8a3d00" if name == "Hybrid" else EDGE for name in rmse_table["Model"]],
        linewidth=1.2,
        width=0.74,
    )
    style_axis(axes[1], "Residual Spread", ylabel="Residual Standard Deviation", grid_axis="y", tick_rotation=0)
    axes[1].tick_params(axis="x", labelsize=11)
    axes[1].set_ylim(0, float(rmse_table["ResidualStd"].max()) * 1.18)
    add_bar_labels(axes[1], bars, "{:.3f}")

    figure.tight_layout(w_pad=2.0)
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def plot_hourly_mae_curve(test_predictions: pd.DataFrame, output: Path) -> None:
    frame = test_predictions.copy()
    frame["forecast_timestamp"] = pd.to_datetime(frame["forecast_timestamp"])
    frame["hour"] = frame["forecast_timestamp"].dt.hour

    figure, axis = plt.subplots(figsize=(10, 5.0))
    for model_name in ["XGBoost", "TFT", "Hybrid", "StackedXGB"]:
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
            linewidth=2.5 if model_name == "Hybrid" else 2.0,
            markersize=4.5,
            label=model_name,
            color=COLORS[model_name],
        )
    style_axis(axis, "Hour-of-Day MAE Profile", xlabel="Hour of Day", ylabel="MAE", grid_axis="both")
    axis.set_xticks(range(0, 24, 2))
    axis.legend(ncol=2, loc="upper right")
    figure.tight_layout()
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def plot_gain_by_plant(plant_table: pd.DataFrame, output: Path) -> None:
    pivot = plant_table.pivot(index="Plant", columns="Model", values="MAE")
    gain_frame = pd.DataFrame(
        {
            "Plant": [PLANT_TITLE_MAP.get(name, name) for name in pivot.index],
            "Hybrid_vs_TFT_%": (pivot["TFT"] - pivot["Hybrid"]) / pivot["TFT"] * 100.0,
            "StackedXGB_vs_TFT_%": (pivot["TFT"] - pivot["StackedXGB"]) / pivot["TFT"] * 100.0,
        }
    )

    x_positions = np.arange(len(gain_frame))
    width = 0.34

    figure, axis = plt.subplots(figsize=(9.8, 4.9))
    hybrid_bars = axis.bar(
        x_positions - width / 2,
        gain_frame["Hybrid_vs_TFT_%"],
        width=width,
        color=COLORS["Hybrid"],
        edgecolor="#8a3d00",
        linewidth=1.2,
        label="Hybrid vs TFT",
    )
    stacked_bars = axis.bar(
        x_positions + width / 2,
        gain_frame["StackedXGB_vs_TFT_%"],
        width=width,
        color=COLORS["StackedXGB"],
        edgecolor=EDGE,
        linewidth=1.2,
        label="StackedXGB vs TFT",
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(gain_frame["Plant"], rotation=16)
    axis.axhline(0.0, color=MUTED, linewidth=1.0)
    style_axis(axis, "Per-Plant Gain over TFT", ylabel="MAE Improvement (%)", grid_axis="y", tick_rotation=10)
    ymax = max(float(gain_frame["Hybrid_vs_TFT_%"].max()), float(gain_frame["StackedXGB_vs_TFT_%"].max()))
    ymin = min(float(gain_frame["Hybrid_vs_TFT_%"].min()), float(gain_frame["StackedXGB_vs_TFT_%"].min()))
    axis.set_ylim(min(-2.0, ymin - 2.0), ymax + 3.0)
    add_bar_labels(axis, hybrid_bars, "{:.1f}")
    add_bar_labels(axis, stacked_bars, "{:.1f}")
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def plot_seed_stability(seed_summary: pd.DataFrame, output: Path) -> None:
    focus_models = ["DNN", "TFT", "Hybrid", "AdaptiveBlend", "StackedXGB"]
    plot_frame = seed_summary[seed_summary["Model"].isin(focus_models)].copy()
    plot_frame = order_models(plot_frame)
    labels = [DISPLAY_LABELS[name] for name in plot_frame["Model"]]

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))
    for axis, metric in zip(axes, ["MAE", "RMSE"]):
        bars = axis.bar(
            labels,
            plot_frame[f"{metric}_mean"],
            yerr=plot_frame[f"{metric}_std"],
            capsize=4,
            color=model_colors(plot_frame["Model"]),
            edgecolor=model_edges(plot_frame["Model"]),
            linewidth=1.2,
            width=0.76,
        )
        style_axis(axis, f"Seed Stability: {metric}", grid_axis="y", tick_rotation=0)
        axis.tick_params(axis="x", labelsize=11)
        axis.set_ylim(0, float(plot_frame[f"{metric}_mean"].max()) * 1.20)
        add_bar_labels(axis, bars, "{:.3f}")
    figure.tight_layout(w_pad=2.0)
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def plot_rolling_origin_overview(rolling_metrics: pd.DataFrame, output: Path) -> None:
    focus_models = ["TFT", "Hybrid", "StackedXGB"]
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))

    for axis, metric in zip(axes, ["MAE", "RMSE"]):
        for model_name in focus_models:
            model_frame = rolling_metrics[rolling_metrics["Model"] == model_name].copy()
            axis.plot(
                model_frame["Window"],
                model_frame[metric],
                marker="o",
                linewidth=2.6 if model_name == "Hybrid" else 2.1,
                markersize=6 if model_name == "Hybrid" else 5,
                label=model_name,
                color=COLORS[model_name],
            )
        style_axis(axis, f"Rolling-Origin {metric}", xlabel="Window", ylabel=metric, grid_axis="both", tick_rotation=12)
    axes[1].legend(loc="upper right")
    figure.tight_layout(w_pad=2.0)
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def plot_method_framework(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(12.5, 6.2))
    figure.patch.set_facecolor(BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    boxes = [
        (0.04, 0.64, 0.22, 0.21, "#f8f4c4", "Input Features\nHistorical power\nWeather\nSolar geometry\nPhysics cues"),
        (0.33, 0.75, 0.16, 0.10, "#d7edf8", "XGBoost"),
        (0.33, 0.58, 0.16, 0.10, "#d7f0e6", "DNN"),
        (0.33, 0.41, 0.16, 0.10, "#fde0d2", "TFT"),
        (0.58, 0.58, 0.21, 0.16, "#f8d7c7", "Scene-Aware Fusion\nplant-aware weights\nregime-specific trust"),
        (0.58, 0.33, 0.21, 0.13, "#efd7e8", "Physics Adjustment\nnight and low-radiation\ncorrection"),
        (0.84, 0.51, 0.12, 0.15, "#e5dcf2", "Final Power\nForecast"),
    ]

    for x_pos, y_pos, width, height, color, text in boxes:
        rectangle = patches.FancyBboxPatch(
            (x_pos, y_pos),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.4,
            edgecolor="#6b6259",
            facecolor=color,
        )
        axis.add_patch(rectangle)
        axis.text(x_pos + width / 2, y_pos + height / 2, text, ha="center", va="center", fontsize=11, color=TEXT)

    arrow_specs = [
        ((0.26, 0.74), (0.33, 0.80)),
        ((0.26, 0.74), (0.33, 0.63)),
        ((0.26, 0.74), (0.33, 0.46)),
        ((0.49, 0.80), (0.58, 0.66)),
        ((0.49, 0.63), (0.58, 0.66)),
        ((0.49, 0.46), (0.58, 0.66)),
        ((0.69, 0.58), (0.69, 0.46)),
        ((0.79, 0.66), (0.84, 0.58)),
        ((0.79, 0.40), (0.84, 0.58)),
    ]
    for start, end in arrow_specs:
        axis.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", linewidth=1.8, color="#5f584f"))

    axis.text(0.04, 0.92, "Scene-Aware Interpretable Fusion", fontsize=16, fontweight="semibold", color=TEXT)
    axis.text(0.35, 0.895, "Heterogeneous base learners", fontsize=9.8, color="#2f7fb6")
    axis.text(0.60, 0.79, "Adaptive fusion under changing regimes", fontsize=9.8, color="#D55E00")
    axis.text(0.60, 0.25, "Constraint-aware post-adjustment", fontsize=9.8, color="#B05A8C")

    figure.tight_layout()
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def write_catalog_clean(output: Path) -> None:
    content = "\n".join(
        [
            "# Figure Catalog",
            "",
            "## baseline_overview.png",
            "- Overall fixed-split comparison on MAE, RMSE, and R2.",
            "",
            "## daytime_baseline_overview.png",
            "- Daytime-only comparison on MAE, RMSE, and R2.",
            "",
            "## ablation_overview.png",
            "- Hybrid ablation comparison on MAE and RMSE.",
            "",
            "## plant_mae_heatmap.png",
            "- Per-plant MAE heatmap across models.",
            "",
            "## plant_rmse_heatmap.png",
            "- Per-plant RMSE heatmap across models.",
            "",
            "## relative_improvement.png",
            "- Relative MAE improvement over Persistence.",
            "",
            "## scatter_comparison.png",
            "- Actual-versus-predicted scatter plots for TFT and Hybrid.",
            "",
            "## residual_distribution.png",
            "- Residual density and residual spread comparison for the main methods.",
            "",
            "## hourly_mae_curve.png",
            "- Hour-of-day MAE profile across representative models.",
            "",
            "## plant_gain_over_tft.png",
            "- Per-plant MAE gain over TFT for Hybrid and StackedXGB.",
            "",
            "## seed_stability.png",
            "- Mean and standard deviation under repeated random seeds.",
            "",
            "## rolling_origin_overview.png",
            "- Rolling-origin MAE and RMSE across evaluation windows.",
            "",
            "## method_framework.png",
            "- Method diagram for the forecasting pipeline and Hybrid fusion.",
            "",
            "## Additional Reusable Plots",
            "- `forecast_examples.png`: example forecasting traces.",
            "- `training_curves.png`: training convergence curves.",
            "- `baseline_mae_rmse.png`: simplified baseline MAE/RMSE figure.",
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
    write_catalog_clean(paper_dir / "paper_figure_catalog_zh.md")
    print("saved", paper_dir)


if __name__ == "__main__":
    main()
