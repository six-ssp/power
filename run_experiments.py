from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch

from pvbench import ExperimentConfig, load_and_prepare_data
from pvbench.data import build_tabular_matrices
from pvbench.models import (
    apply_physics_adjustment,
    fit_adaptive_blend,
    fit_dnn,
    fit_stacked_xgboost,
    fit_tft,
    fit_xgboost,
    tune_blend_weights,
    tune_physics_alpha,
)
from pvbench.reporting import (
    compute_metrics,
    plot_forecast_examples,
    plot_metric_bars,
    plot_training_curves,
    save_metrics_table,
    write_markdown_report,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_prediction_frame(frame: pd.DataFrame, prediction_name: str, prediction: np.ndarray) -> pd.DataFrame:
    result = frame[
        [
            "plant_id",
            "plant_name_en",
            "module_type",
            "mounting_type",
            "forecast_time_idx",
            "forecast_timestamp",
            "target_power",
            "power_now",
            "forecast_irradiance",
            "forecast_temperature",
            "forecast_humidity",
            "forecast_wind_speed",
            "forecast_direct_radiation",
            "forecast_global_radiation",
            "forecast_solar_elevation_deg",
            "forecast_cos_zenith",
            "forecast_clear_sky_proxy",
            "forecast_night_flag",
            "forecast_hour_sin",
            "forecast_hour_cos",
            "forecast_doy_sin",
            "forecast_doy_cos",
        ]
    ].copy()
    result[prediction_name] = prediction.astype(float)
    return result


def merge_predictions(base_frame: pd.DataFrame, tft_frame: pd.DataFrame, frame_name: str) -> pd.DataFrame:
    merged = base_frame.merge(
        tft_frame[["plant_id", "forecast_time_idx", "tft_prediction"]],
        on=["plant_id", "forecast_time_idx"],
        how="inner",
    )
    if len(merged) != len(base_frame):
        print(f"[{frame_name}] dropped {len(base_frame) - len(merged)} rows because TFT prediction keys were missing.")
    return merged


def build_blend(
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    prediction_columns: list[str],
    config: ExperimentConfig,
) -> tuple[dict[str, float], float, np.ndarray, np.ndarray]:
    weights, blended_val, _ = tune_blend_weights(
        validation_frame=val_frame,
        prediction_columns=prediction_columns,
        min_weight=config.min_blend_weight if len(prediction_columns) == 3 else 0.0,
        step=config.blend_step,
    )
    alpha, adjusted_val = tune_physics_alpha(val_frame, blended_val, config)

    blended_test = np.zeros(len(test_frame), dtype=float)
    for column, weight in weights.items():
        blended_test += test_frame[column].to_numpy(dtype=float) * weight
    adjusted_test = apply_physics_adjustment(test_frame, blended_test, alpha, config.night_radiation_threshold)
    return weights, alpha, adjusted_val, adjusted_test


def build_markdown_table(frame: pd.DataFrame) -> str:
    columns = frame.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def normalize_weight_dict(weight_dict: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in weight_dict.items()}


def main() -> None:
    config = ExperimentConfig()
    set_seed(config.random_seed)
    prepared_data = load_and_prepare_data(config)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    train_x, val_x, test_x, model_columns = build_tabular_matrices(
        prepared_data.train_frame,
        prepared_data.val_frame,
        prepared_data.test_frame,
        prepared_data.feature_columns,
        prepared_data.categorical_columns,
    )
    train_y = prepared_data.train_frame["target_power"].to_numpy(dtype=float)
    val_y = prepared_data.val_frame["target_power"].to_numpy(dtype=float)
    print("Train/Val/Test rows:", len(train_x), len(val_x), len(test_x))
    print("Feature count:", len(model_columns))

    persistence_val = prepared_data.val_frame["power_now"].to_numpy(dtype=float)
    persistence_test = prepared_data.test_frame["power_now"].to_numpy(dtype=float)

    xgb_result = fit_xgboost(train_x, train_y, val_x, val_y, test_x, config)
    dnn_result = fit_dnn(train_x, train_y, val_x, val_y, test_x, config)
    tft_result = fit_tft(prepared_data, config)

    val_base = make_prediction_frame(prepared_data.val_frame, "persistence_prediction", persistence_val)
    test_base = make_prediction_frame(prepared_data.test_frame, "persistence_prediction", persistence_test)
    val_base["xgboost_prediction"] = xgb_result.val_predictions
    val_base["dnn_prediction"] = dnn_result.val_predictions
    test_base["xgboost_prediction"] = xgb_result.test_predictions
    test_base["dnn_prediction"] = dnn_result.test_predictions

    val_all = merge_predictions(val_base, tft_result.val_predictions, "val")
    test_all = merge_predictions(test_base, tft_result.test_predictions, "test")

    full_weights, full_alpha, _, hybrid_test = build_blend(
        val_frame=val_all,
        test_frame=test_all,
        prediction_columns=["xgboost_prediction", "dnn_prediction", "tft_prediction"],
        config=config,
    )
    full_weights = normalize_weight_dict(full_weights)
    raw_blend_test = np.zeros(len(test_all), dtype=float)
    for column, weight in full_weights.items():
        raw_blend_test += test_all[column].to_numpy(dtype=float) * weight
    test_all["hybrid_prediction"] = hybrid_test
    test_all["hybrid_no_physics_prediction"] = raw_blend_test

    equal_weights = {"xgboost_prediction": 1.0 / 3.0, "dnn_prediction": 1.0 / 3.0, "tft_prediction": 1.0 / 3.0}
    equal_val = sum(val_all[column].to_numpy(dtype=float) * weight for column, weight in equal_weights.items())
    equal_alpha, _ = tune_physics_alpha(val_all, equal_val, config)
    equal_test = sum(test_all[column].to_numpy(dtype=float) * weight for column, weight in equal_weights.items())
    test_all["hybrid_equal_weight_prediction"] = apply_physics_adjustment(
        test_all,
        equal_test,
        equal_alpha,
        config.night_radiation_threshold,
    )

    component_ablation_specs = {
        "hybrid_without_xgboost": ["dnn_prediction", "tft_prediction"],
        "hybrid_without_dnn": ["xgboost_prediction", "tft_prediction"],
        "hybrid_without_tft": ["xgboost_prediction", "dnn_prediction"],
    }
    ablation_weights: dict[str, dict[str, float]] = {}
    ablation_alphas: dict[str, float] = {}
    for name, columns in component_ablation_specs.items():
        weights, alpha, _, adjusted_test = build_blend(val_all, test_all, columns, config)
        test_all[name] = adjusted_test
        ablation_weights[name] = normalize_weight_dict(weights)
        ablation_alphas[name] = alpha

    optimized_prediction_columns = [
        "persistence_prediction",
        "xgboost_prediction",
        "dnn_prediction",
        "tft_prediction",
    ]
    adaptive_result = fit_adaptive_blend(val_all, test_all, optimized_prediction_columns, config)
    stacked_result = fit_stacked_xgboost(val_all, test_all, optimized_prediction_columns, config)
    val_all["adaptive_blend_prediction"] = adaptive_result.val_predictions
    test_all["adaptive_blend_prediction"] = adaptive_result.test_predictions
    val_all["stacked_xgboost_prediction"] = stacked_result.val_predictions
    test_all["stacked_xgboost_prediction"] = stacked_result.test_predictions

    baseline_rows = []
    baseline_specs = {
        "Persistence": "persistence_prediction",
        "XGBoost": "xgboost_prediction",
        "DNN": "dnn_prediction",
        "TFT": "tft_prediction",
        "Hybrid": "hybrid_prediction",
        "AdaptiveBlend": "adaptive_blend_prediction",
        "StackedXGB": "stacked_xgboost_prediction",
    }
    for model_name, column in baseline_specs.items():
        metrics = compute_metrics(test_all, column)
        baseline_rows.append({"Model": model_name, **metrics})
    baseline_table = save_metrics_table(baseline_rows, config.metric_dir / "baseline_metrics.csv")

    ablation_rows = []
    ablation_specs = {
        "Full Hybrid": "hybrid_prediction",
        "w/o Physics": "hybrid_no_physics_prediction",
        "Equal Weights": "hybrid_equal_weight_prediction",
        "w/o XGBoost": "hybrid_without_xgboost",
        "w/o DNN": "hybrid_without_dnn",
        "w/o TFT": "hybrid_without_tft",
        "Adaptive Blend": "adaptive_blend_prediction",
        "Stacked XGB": "stacked_xgboost_prediction",
    }
    for model_name, column in ablation_specs.items():
        metrics = compute_metrics(test_all, column)
        ablation_rows.append({"Model": model_name, **metrics})
    ablation_table = save_metrics_table(ablation_rows, config.metric_dir / "ablation_metrics.csv")

    plant_rows = []
    for plant_id in sorted(test_all["plant_id"].unique()):
        plant_frame = test_all[test_all["plant_id"] == plant_id]
        for model_name, column in baseline_specs.items():
            metrics = compute_metrics(plant_frame, column)
            plant_rows.append({"Plant": plant_id, "Model": model_name, **metrics})
    plant_table = save_metrics_table(plant_rows, config.metric_dir / "plant_level_metrics.csv")
    val_all.to_csv(config.metric_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    test_all.to_csv(config.metric_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    baseline_table.to_csv(config.metric_dir / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    ablation_table.to_csv(config.metric_dir / "ablation_metrics.csv", index=False, encoding="utf-8-sig")
    plant_table.to_csv(config.metric_dir / "plant_level_metrics.csv", index=False, encoding="utf-8-sig")

    plot_metric_bars(baseline_table, config.plot_dir / "baseline_mae_rmse.png")
    plot_training_curves(
        dnn_result.history,
        tft_result.history,
        adaptive_result.history,
        stacked_result.history,
        config.plot_dir / "training_curves.png",
    )
    plot_forecast_examples(test_all, sorted(test_all["plant_id"].unique()), config.plot_dir / "forecast_examples.png")

    training_record = f"""
# 光伏基线与消融实验记录

## 1. 实验设置
- 数据源：`dataset/` 下 4 个电站 CSV，代码内统一映射为英文列名，原始数据文件未改动。
- 站点假设：Alice Springs, Australia，纬度 `-23.6980`，经度 `133.8807`，时区偏移 `UTC+9.5`。
- 任务定义：基于历史与当前气象信息进行 `5 min ahead` 单步短时功率预测。
- 划分方式：每个电站按时间顺序 `8:1:1` 划分训练/验证/测试。
- 评估指标：MAE、RMSE、MAPE、sMAPE、R2，同时记录预测均值与实际均值。

## 2. 特征工程
- 时间周期特征：小时、年内天数、月份、星期的正余弦编码。
- 地理与太阳几何特征：太阳赤纬、时差、时角、太阳高度角、天顶角、cos(zenith)、空气质量代理、晴空辐照代理。
- 物理启发特征：温度降额、有效总辐射、有效直辐射、辐照-温度比、湿度-风速交互、夜间标记。
- 历史特征：功率/辐射/温度/风速的多阶滞后与滚动统计量。
- 未来已知特征：下一时刻气象量与太阳高度代理，用于统一 XGBoost、DNN、TFT 的输入口径。

## 3. 模型设置
- Persistence：用当前时刻功率直接预测下一时刻。
- XGBoost：`hist` 树方法，带验证集早停。
- DNN：三层 MLP，损失为 SmoothL1Loss，验证 RMSE 早停。
- TFT：多电站联合训练，损失为 QuantileLoss，记录 MAE/RMSE。
- Hybrid：验证集搜索正权重，保证 `XGBoost/DNN/TFT` 都占比；再叠加夜间低辐照物理修正。
- AdaptiveBlend：基于 `Persistence/XGBoost/DNN/TFT` 的样本级动态权重融合，并在低辐照场景做物理后处理。
- StackedXGB：把基模型预测与气象/几何上下文一起输入二阶段 XGBoost，再做低辐照物理后处理。

## 4. Baseline 结果
{build_markdown_table(baseline_table)}

## 5. 消融结果
{build_markdown_table(ablation_table)}

## 6. 融合权重与物理修正
- Full Hybrid 权重：`{full_weights}`
- Full Hybrid 夜间修正 alpha：`{full_alpha:.3f}`
- Equal Weight 夜间修正 alpha：`{equal_alpha:.3f}`
- w/o XGBoost 权重：`{ablation_weights['hybrid_without_xgboost']}`，alpha=`{ablation_alphas['hybrid_without_xgboost']:.3f}`
- w/o DNN 权重：`{ablation_weights['hybrid_without_dnn']}`，alpha=`{ablation_alphas['hybrid_without_dnn']:.3f}`
- w/o TFT 权重：`{ablation_weights['hybrid_without_tft']}`，alpha=`{ablation_alphas['hybrid_without_tft']:.3f}`
- AdaptiveBlend 平均验证权重：`{adaptive_result.avg_val_weights}`
- AdaptiveBlend 平均测试权重：`{adaptive_result.avg_test_weights}`
- AdaptiveBlend 物理修正 alpha：`{adaptive_result.physics_alpha:.3f}`
- StackedXGB 物理修正 alpha：`{stacked_result.physics_alpha:.3f}`

## 7. 结果观察
- 融合模型如果比单模型稳定下降，说明三类模型在误差结构上存在互补性。
- `w/o Physics` 与 Full Hybrid 的差异主要反映夜间低辐照场景的修正价值。
- `w/o TFT / w/o DNN / w/o XGBoost` 可以直接判断三类模型在融合中的边际贡献。
- 当前新增的 `AdaptiveBlend` 与 `StackedXGB` 用于验证“固定全局权重是否限制融合上限”。
- 如果后续要继续冲更强结果，优先建议扩展到多步预测、加站点容量/安装角度元数据、再做更系统的超参搜索。
""".strip()
    write_markdown_report(config.report_dir / "training_log_zh.md", training_record)

    summary_payload = pd.DataFrame(
        [
            {
                "name": "full_blend_weights",
                "value": str(full_weights),
            },
            {
                "name": "full_physics_alpha",
                "value": f"{full_alpha:.6f}",
            },
            {
                "name": "equal_weight_physics_alpha",
                "value": f"{equal_alpha:.6f}",
            },
            {
                "name": "adaptive_blend_avg_val_weights",
                "value": str(adaptive_result.avg_val_weights),
            },
            {
                "name": "adaptive_blend_avg_test_weights",
                "value": str(adaptive_result.avg_test_weights),
            },
            {
                "name": "adaptive_blend_physics_alpha",
                "value": f"{adaptive_result.physics_alpha:.6f}",
            },
            {
                "name": "stacked_xgboost_physics_alpha",
                "value": f"{stacked_result.physics_alpha:.6f}",
            },
        ]
    )
    summary_payload.to_csv(config.metric_dir / "blend_summary.csv", index=False, encoding="utf-8-sig")

    print("Baseline metrics saved to:", config.metric_dir / "baseline_metrics.csv")
    print("Ablation metrics saved to:", config.metric_dir / "ablation_metrics.csv")
    print("Training log saved to:", config.report_dir / "training_log_zh.md")


if __name__ == "__main__":
    main()
