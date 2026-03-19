from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


plt.switch_backend("Agg")

PLANT_TITLE_MAP = {
    "AliceSprings_MonoTrack_1A": "MonoTrack 1A",
    "AliceSprings_PolyFixed_1C": "PolyFixed 1C",
    "AliceSprings_PolyUtility_3A": "PolyUtility 3A",
    "AliceSprings_HighEfficiency_4A": "HighEfficiency 4A",
    "Plant_1A_单晶双轴": "MonoTrack 1A",
    "Plant_1C_多晶固定": "PolyFixed 1C",
    "Plant_3A_多晶大型": "PolyUtility 3A",
    "Plant_4A_高效对比": "HighEfficiency 4A",
}


def compute_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    y_true = frame["target_power"].to_numpy(dtype=float)
    y_pred = frame[prediction_column].to_numpy(dtype=float)
    error = y_pred - y_true
    abs_true = np.clip(np.abs(y_true), 1e-6, None)

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mape = float(np.mean(np.abs(error) / abs_true) * 100.0)
    smape = float(np.mean(2.0 * np.abs(error) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "R2": r2,
        "PredMean": float(np.mean(y_pred)),
        "ActualMean": float(np.mean(y_true)),
        "Bias": float(np.mean(error)),
        "Samples": int(len(frame)),
    }


def filter_daytime_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "forecast_night_flag" not in frame.columns:
        raise KeyError("Expected 'forecast_night_flag' in prediction frame to build daytime-only metrics.")
    return frame[frame["forecast_night_flag"] == 0].copy()


def summarize_repeated_metrics(metric_frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    metric_columns = ["MAE", "RMSE", "MAPE", "sMAPE", "R2", "PredMean", "ActualMean", "Bias", "Samples"]
    rows: list[dict[str, float | str | int]] = []

    for model_name, model_frame in metric_frame.groupby("Model", sort=False):
        row: dict[str, float | str | int] = {"Model": model_name, "Runs": int(model_frame[id_column].nunique())}
        for column in metric_columns:
            row[f"{column}_mean"] = float(model_frame[column].mean())
            row[f"{column}_std"] = float(model_frame[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def save_metrics_table(rows: list[dict[str, float | str]], path: Path) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return table


def plot_metric_bars(metric_table: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(metric_table["Model"], metric_table["MAE"], color="#4c78a8")
    axes[0].set_title("MAE Comparison")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(metric_table["Model"], metric_table["RMSE"], color="#f58518")
    axes[1].set_title("RMSE Comparison")
    axes[1].tick_params(axis="x", rotation=20)

    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_training_curves(
    dnn_history: pd.DataFrame,
    tft_history: pd.DataFrame,
    adaptive_history: pd.DataFrame,
    stacked_history: pd.DataFrame,
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    if not dnn_history.empty:
        axes[0].plot(dnn_history["epoch"], dnn_history["train_loss"], label="train_loss")
        axes[0].plot(dnn_history["epoch"], dnn_history["val_loss"], label="val_loss")
        axes[0].set_title("DNN Loss")
        axes[0].legend()
    else:
        axes[0].set_title("DNN Loss (empty)")

    if not tft_history.empty:
        tft_epoch = tft_history.dropna(subset=["epoch"]).copy()
        if "train_loss_epoch" in tft_epoch.columns:
            axes[1].plot(tft_epoch["epoch"], tft_epoch["train_loss_epoch"], label="train_loss_epoch")
        if "val_loss" in tft_epoch.columns:
            val_rows = tft_epoch.dropna(subset=["val_loss"])
            axes[1].plot(val_rows["epoch"], val_rows["val_loss"], label="val_loss")
        axes[1].set_title("TFT Loss")
        axes[1].legend()
    else:
        axes[1].set_title("TFT Loss (empty)")

    if not adaptive_history.empty:
        axes[2].plot(adaptive_history["epoch"], adaptive_history["train_loss"], label="train_loss")
        axes[2].plot(adaptive_history["epoch"], adaptive_history["holdout_loss"], label="holdout_loss")
        axes[2].plot(adaptive_history["epoch"], adaptive_history["holdout_mae"], label="holdout_mae")
        axes[2].set_title("Adaptive Blend")
        axes[2].legend()
    else:
        axes[2].set_title("Adaptive Blend (empty)")

    if not stacked_history.empty:
        axes[3].plot(stacked_history["round"], stacked_history["train_rmse"], label="train_rmse")
        axes[3].plot(stacked_history["round"], stacked_history["holdout_rmse"], label="holdout_rmse")
        axes[3].set_title("Stacked XGB")
        axes[3].legend()
    else:
        axes[3].set_title("Stacked XGB (empty)")

    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_forecast_examples(frame: pd.DataFrame, plants: list[str], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    axes = axes.flatten()

    for axis, plant_id in zip(axes, plants):
        plant_frame = frame[frame["plant_id"] == plant_id].sort_values("forecast_timestamp").head(288)
        axis.plot(plant_frame["forecast_timestamp"], plant_frame["target_power"], label="Actual", linewidth=1.8)
        for column, label, color in [
            ("xgboost_prediction", "XGBoost", "#4c78a8"),
            ("tft_prediction", "TFT", "#f58518"),
            ("hybrid_prediction", "Hybrid", "#72b7b2"),
            ("adaptive_blend_prediction", "AdaptiveBlend", "#54a24b"),
            ("stacked_xgboost_prediction", "StackedXGB", "#e45756"),
        ]:
            if column in plant_frame.columns:
                axis.plot(plant_frame["forecast_timestamp"], plant_frame[column], label=label, alpha=0.9, linewidth=1.1, color=color)
        axis.set_title(PLANT_TITLE_MAP.get(plant_id, plant_id))
        axis.tick_params(axis="x", rotation=30)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=5)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_markdown_report(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8-sig")
