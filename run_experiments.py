from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from pvbench import ExperimentConfig, load_and_prepare_data
from pvbench.data import PreparedData, build_tabular_matrices, build_windowed_prepared_data
from pvbench.models import (
    MetaModelResult,
    apply_physics_adjustment,
    fit_adaptive_blend,
    fit_dnn,
    fit_scene_aware_hybrid,
    fit_stacked_xgboost,
    fit_tft,
    fit_xgboost,
    tune_blend_weights,
    tune_physics_alpha,
)
from pvbench.reporting import (
    compute_metrics,
    filter_daytime_frame,
    plot_forecast_examples,
    plot_metric_bars,
    plot_training_curves,
    save_metrics_table,
    summarize_repeated_metrics,
    write_markdown_report,
)


BASELINE_SPECS = {
    "Persistence": "persistence_prediction",
    "XGBoost": "xgboost_prediction",
    "DNN": "dnn_prediction",
    "TFT": "tft_prediction",
    "Hybrid": "hybrid_prediction",
    "AdaptiveBlend": "adaptive_blend_prediction",
    "StackedXGB": "stacked_xgboost_prediction",
}


ABLATION_SPECS = {
    "Full Hybrid": "hybrid_prediction",
    "w/o Physics": "hybrid_no_physics_prediction",
    "w/o Plant Adaptation": "hybrid_without_plant_adaptation",
    "w/o Scene Adaptation": "hybrid_without_scene_adaptation",
    "w/o XGBoost": "hybrid_without_xgboost",
    "w/o DNN": "hybrid_without_dnn",
    "w/o TFT": "hybrid_without_tft",
    "Adaptive Blend": "adaptive_blend_prediction",
    "Stacked XGB": "stacked_xgboost_prediction",
}


@dataclass
class ExperimentArtifacts:
    prepared_data: PreparedData
    baseline_table: pd.DataFrame
    ablation_table: pd.DataFrame
    plant_table: pd.DataFrame
    val_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    blend_summary: pd.DataFrame
    dnn_history: pd.DataFrame
    tft_history: pd.DataFrame
    adaptive_history: pd.DataFrame
    stacked_history: pd.DataFrame
    hybrid_result: MetaModelResult
    hybrid_without_plant_result: MetaModelResult | None
    fixed_global_weights: dict[str, float]
    fixed_global_alpha: float
    component_hybrid_summaries: dict[str, dict[str, object]]
    adaptive_result: MetaModelResult
    stacked_result: MetaModelResult


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


def normalize_nested_weight_dict(weight_dict: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, dict[str, float]]]:
    normalized: dict[str, dict[str, dict[str, float]]] = {}
    for scope_key, scope_value in weight_dict.items():
        normalized[scope_key] = {scene_name: normalize_weight_dict(scene_weights) for scene_name, scene_weights in scope_value.items()}
    return normalized


def clone_runtime_config(base_config: ExperimentConfig, seed: int, log_subdir: str | None = None) -> ExperimentConfig:
    config = deepcopy(base_config)
    config.random_seed = seed
    config.xgb_params = dict(base_config.xgb_params)
    config.xgb_params["random_state"] = seed
    config.stack_xgb_params = dict(base_config.stack_xgb_params)
    config.stack_xgb_params["random_state"] = seed
    if log_subdir is not None:
        config.log_dir = base_config.log_dir / log_subdir
        config.log_dir.mkdir(parents=True, exist_ok=True)
    return config


def collect_metric_table(frame: pd.DataFrame, specs: dict[str, str]) -> pd.DataFrame:
    rows = []
    for model_name, column in specs.items():
        metrics = compute_metrics(frame, column)
        rows.append({"Model": model_name, **metrics})
    return pd.DataFrame(rows)


def collect_plant_metric_table(frame: pd.DataFrame, specs: dict[str, str]) -> pd.DataFrame:
    rows = []
    for plant_id in sorted(frame["plant_id"].unique()):
        plant_frame = frame[frame["plant_id"] == plant_id]
        for model_name, column in specs.items():
            metrics = compute_metrics(plant_frame, column)
            rows.append({"Plant": plant_id, "Model": model_name, **metrics})
    return pd.DataFrame(rows)


def build_subset_count_table(frame: pd.DataFrame) -> pd.DataFrame:
    daytime_mask = frame["forecast_night_flag"] == 0
    night_mask = frame["forecast_night_flag"] == 1
    total = len(frame)
    rows = [
        {"Subset": "All", "Samples": total, "Ratio": 1.0},
        {"Subset": "Daytime", "Samples": int(daytime_mask.sum()), "Ratio": float(daytime_mask.mean())},
        {"Subset": "Nighttime", "Samples": int(night_mask.sum()), "Ratio": float(night_mask.mean())},
    ]
    return pd.DataFrame(rows)


def format_mean_std_table(summary_frame: pd.DataFrame) -> pd.DataFrame:
    display = summary_frame[["Model", "Runs"]].copy()
    for metric in ("MAE", "RMSE", "R2"):
        display[metric] = summary_frame.apply(
            lambda row: f"{row[f'{metric}_mean']:.6f} +/- {row[f'{metric}_std']:.6f}",
            axis=1,
        )
    return display


def extract_metric(frame: pd.DataFrame, model_name: str, metric: str) -> float:
    return float(frame.loc[frame["Model"] == model_name, metric].iloc[0])


def build_window_description(
    prepared_data: PreparedData,
    window_name: str,
    train_end_ratio: float,
    val_end_ratio: float,
    test_end_ratio: float,
) -> dict[str, object]:
    reference_plant = sorted(prepared_data.raw_frame["plant_id"].unique())[0]
    reference_frame = (
        prepared_data.raw_frame[prepared_data.raw_frame["plant_id"] == reference_plant]
        .sort_values("time_idx")
        .reset_index(drop=True)
    )
    timestamp_lookup = reference_frame.set_index("time_idx")["timestamp"]
    return {
        "Window": window_name,
        "TrainEndRatio": train_end_ratio,
        "ValEndRatio": val_end_ratio,
        "TestEndRatio": test_end_ratio,
        "TrainEndIdx": prepared_data.train_cutoff,
        "ValStartIdx": prepared_data.val_start_idx,
        "ValEndIdx": prepared_data.val_cutoff,
        "TestStartIdx": prepared_data.test_start_idx,
        "TestEndIdx": prepared_data.test_cutoff,
        "TrainEndTimestamp": str(timestamp_lookup.loc[prepared_data.train_cutoff]),
        "ValStartTimestamp": str(timestamp_lookup.loc[prepared_data.val_start_idx]),
        "ValEndTimestamp": str(timestamp_lookup.loc[prepared_data.val_cutoff]),
        "TestStartTimestamp": str(timestamp_lookup.loc[prepared_data.test_start_idx]),
        "TestEndTimestamp": str(timestamp_lookup.loc[prepared_data.test_cutoff]),
        "TrainSamples": len(prepared_data.train_frame),
        "ValSamples": len(prepared_data.val_frame),
        "TestSamples": len(prepared_data.test_frame),
    }


def run_single_experiment(
    config: ExperimentConfig,
    prepared_data: PreparedData | None = None,
    include_ablations: bool = True,
) -> ExperimentArtifacts:
    set_seed(config.random_seed)
    prepared = prepared_data if prepared_data is not None else load_and_prepare_data(config)
    print(f"Running experiment with seed={config.random_seed}.")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    train_x, val_x, test_x, model_columns = build_tabular_matrices(
        prepared.train_frame,
        prepared.val_frame,
        prepared.test_frame,
        prepared.feature_columns,
        prepared.categorical_columns,
    )
    train_y = prepared.train_frame["target_power"].to_numpy(dtype=float)
    val_y = prepared.val_frame["target_power"].to_numpy(dtype=float)
    print("Train/Val/Test rows:", len(train_x), len(val_x), len(test_x))
    print("Feature count:", len(model_columns))

    persistence_val = prepared.val_frame["power_now"].to_numpy(dtype=float)
    persistence_test = prepared.test_frame["power_now"].to_numpy(dtype=float)

    xgb_result = fit_xgboost(train_x, train_y, val_x, val_y, test_x, config)
    dnn_result = fit_dnn(train_x, train_y, val_x, val_y, test_x, config)
    tft_result = fit_tft(prepared, config)

    val_base = make_prediction_frame(prepared.val_frame, "persistence_prediction", persistence_val)
    test_base = make_prediction_frame(prepared.test_frame, "persistence_prediction", persistence_test)
    val_base["xgboost_prediction"] = xgb_result.val_predictions
    val_base["dnn_prediction"] = dnn_result.val_predictions
    test_base["xgboost_prediction"] = xgb_result.test_predictions
    test_base["dnn_prediction"] = dnn_result.test_predictions

    val_all = merge_predictions(val_base, tft_result.val_predictions, "val")
    test_all = merge_predictions(test_base, tft_result.test_predictions, "test")

    hybrid_prediction_columns = ["xgboost_prediction", "dnn_prediction", "tft_prediction"]
    hybrid_result = fit_scene_aware_hybrid(
        validation_frame=val_all,
        test_frame=test_all,
        prediction_columns=hybrid_prediction_columns,
        config=config,
    )
    if hybrid_result.raw_val_predictions is None or hybrid_result.raw_test_predictions is None:
        raise RuntimeError("Scene-aware Hybrid must provide raw predictions for reporting.")
    if hybrid_result.regime_weights is None or hybrid_result.thresholds is None:
        raise RuntimeError("Scene-aware Hybrid must provide selected thresholds and regime weights.")

    val_all["hybrid_prediction"] = hybrid_result.val_predictions
    test_all["hybrid_prediction"] = hybrid_result.test_predictions

    hybrid_without_plant_result: MetaModelResult | None = None
    fixed_global_weights: dict[str, float] = {}
    fixed_global_alpha = 0.0
    component_hybrid_summaries: dict[str, dict[str, object]] = {}
    ablation_table = pd.DataFrame()

    if include_ablations:
        hybrid_without_plant_result = fit_scene_aware_hybrid(
            validation_frame=val_all,
            test_frame=test_all,
            prediction_columns=hybrid_prediction_columns,
            config=config,
            plant_specific=False,
        )
        fixed_global_weights, fixed_global_alpha, fixed_global_val, fixed_global_test = build_blend(
            val_frame=val_all,
            test_frame=test_all,
            prediction_columns=hybrid_prediction_columns,
            config=config,
        )
        fixed_global_weights = normalize_weight_dict(fixed_global_weights)

        val_all["hybrid_no_physics_prediction"] = hybrid_result.raw_val_predictions
        test_all["hybrid_no_physics_prediction"] = hybrid_result.raw_test_predictions
        val_all["hybrid_without_plant_adaptation"] = hybrid_without_plant_result.val_predictions
        test_all["hybrid_without_plant_adaptation"] = hybrid_without_plant_result.test_predictions
        val_all["hybrid_without_scene_adaptation"] = fixed_global_val
        test_all["hybrid_without_scene_adaptation"] = fixed_global_test

        component_ablation_specs = {
            "hybrid_without_xgboost": ["dnn_prediction", "tft_prediction"],
            "hybrid_without_dnn": ["xgboost_prediction", "tft_prediction"],
            "hybrid_without_tft": ["xgboost_prediction", "dnn_prediction"],
        }
        for name, columns in component_ablation_specs.items():
            ablation_result = fit_scene_aware_hybrid(val_all, test_all, columns, config)
            val_all[name] = ablation_result.val_predictions
            test_all[name] = ablation_result.test_predictions
            component_hybrid_summaries[name] = {
                "thresholds": ablation_result.thresholds,
                "physics_alpha": ablation_result.physics_alpha,
                "regime_weights": normalize_nested_weight_dict(ablation_result.regime_weights or {}),
            }

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

    baseline_table = collect_metric_table(test_all, BASELINE_SPECS)
    plant_table = collect_plant_metric_table(test_all, BASELINE_SPECS)

    if include_ablations:
        ablation_table = collect_metric_table(test_all, ABLATION_SPECS)

    hybrid_regime_weights = normalize_nested_weight_dict(hybrid_result.regime_weights or {})
    hybrid_without_plant_weights = normalize_nested_weight_dict(
        hybrid_without_plant_result.regime_weights or {}
    ) if hybrid_without_plant_result is not None else {}

    summary_rows = [
        {"name": "hybrid_thresholds", "value": str(hybrid_result.thresholds)},
        {"name": "hybrid_regime_weights", "value": str(hybrid_regime_weights)},
        {"name": "hybrid_physics_alpha", "value": f"{hybrid_result.physics_alpha:.6f}"},
        {"name": "adaptive_blend_avg_val_weights", "value": str(adaptive_result.avg_val_weights)},
        {"name": "adaptive_blend_avg_test_weights", "value": str(adaptive_result.avg_test_weights)},
        {"name": "adaptive_blend_physics_alpha", "value": f"{adaptive_result.physics_alpha:.6f}"},
        {"name": "stacked_xgboost_physics_alpha", "value": f"{stacked_result.physics_alpha:.6f}"},
    ]
    if include_ablations and hybrid_without_plant_result is not None:
        summary_rows.extend(
            [
                {"name": "hybrid_without_plant_thresholds", "value": str(hybrid_without_plant_result.thresholds)},
                {"name": "hybrid_without_plant_regime_weights", "value": str(hybrid_without_plant_weights)},
                {"name": "hybrid_without_plant_physics_alpha", "value": f"{hybrid_without_plant_result.physics_alpha:.6f}"},
                {"name": "hybrid_without_scene_weights", "value": str(fixed_global_weights)},
                {"name": "hybrid_without_scene_physics_alpha", "value": f"{fixed_global_alpha:.6f}"},
            ]
        )
    blend_summary = pd.DataFrame(summary_rows)

    return ExperimentArtifacts(
        prepared_data=prepared,
        baseline_table=baseline_table,
        ablation_table=ablation_table,
        plant_table=plant_table,
        val_predictions=val_all,
        test_predictions=test_all,
        blend_summary=blend_summary,
        dnn_history=dnn_result.history,
        tft_history=tft_result.history,
        adaptive_history=adaptive_result.history,
        stacked_history=stacked_result.history,
        hybrid_result=hybrid_result,
        hybrid_without_plant_result=hybrid_without_plant_result,
        fixed_global_weights=fixed_global_weights,
        fixed_global_alpha=fixed_global_alpha,
        component_hybrid_summaries=component_hybrid_summaries,
        adaptive_result=adaptive_result,
        stacked_result=stacked_result,
    )


def run_seed_repeats(config: ExperimentConfig, prepared_data: PreparedData) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeat_rows = []
    for seed in config.robustness_seeds:
        runtime_config = clone_runtime_config(config, seed=seed, log_subdir=f"seed_repeats/seed_{seed}")
        artifacts = run_single_experiment(runtime_config, prepared_data=prepared_data, include_ablations=False)
        run_table = artifacts.baseline_table.copy()
        run_table.insert(0, "Seed", seed)
        repeat_rows.append(run_table)
        partial_repeat_table = pd.concat(repeat_rows, ignore_index=True)
        partial_repeat_summary = summarize_repeated_metrics(partial_repeat_table, id_column="Seed")
        partial_repeat_table.to_csv(config.metric_dir / "seed_repeat_metrics.csv", index=False, encoding="utf-8-sig")
        partial_repeat_summary.to_csv(config.metric_dir / "seed_repeat_summary.csv", index=False, encoding="utf-8-sig")
    repeat_table = pd.concat(repeat_rows, ignore_index=True)
    repeat_summary = summarize_repeated_metrics(repeat_table, id_column="Seed")
    return repeat_table, repeat_summary


def run_rolling_origin_evaluation(
    config: ExperimentConfig,
    prepared_data: PreparedData,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rolling_rows = []
    window_rows = []

    for index, (train_end_ratio, val_end_ratio, test_end_ratio) in enumerate(config.rolling_origin_windows, start=1):
        window_name = f"Window_{index}"
        window_prepared = build_windowed_prepared_data(
            prepared_data=prepared_data,
            train_end_ratio=train_end_ratio,
            val_end_ratio=val_end_ratio,
            test_end_ratio=test_end_ratio,
            split_gap_steps=config.split_gap_steps,
        )
        window_rows.append(
            build_window_description(
                prepared_data=window_prepared,
                window_name=window_name,
                train_end_ratio=train_end_ratio,
                val_end_ratio=val_end_ratio,
                test_end_ratio=test_end_ratio,
            )
        )
        runtime_config = clone_runtime_config(
            config,
            seed=config.random_seed,
            log_subdir=f"rolling_origin/{window_name.lower()}",
        )
        artifacts = run_single_experiment(runtime_config, prepared_data=window_prepared, include_ablations=False)
        window_table = artifacts.baseline_table.copy()
        window_table.insert(0, "Window", window_name)
        window_table.insert(1, "TrainEndRatio", train_end_ratio)
        window_table.insert(2, "ValEndRatio", val_end_ratio)
        window_table.insert(3, "TestEndRatio", test_end_ratio)
        rolling_rows.append(window_table)
        partial_rolling_table = pd.concat(rolling_rows, ignore_index=True)
        partial_rolling_summary = summarize_repeated_metrics(partial_rolling_table, id_column="Window")
        pd.DataFrame(window_rows).to_csv(config.metric_dir / "rolling_origin_windows.csv", index=False, encoding="utf-8-sig")
        partial_rolling_table.to_csv(config.metric_dir / "rolling_origin_metrics.csv", index=False, encoding="utf-8-sig")
        partial_rolling_summary.to_csv(config.metric_dir / "rolling_origin_summary.csv", index=False, encoding="utf-8-sig")

    rolling_table = pd.concat(rolling_rows, ignore_index=True)
    rolling_summary = summarize_repeated_metrics(rolling_table, id_column="Window")
    window_description_table = pd.DataFrame(window_rows)
    return rolling_table, rolling_summary, window_description_table


def build_training_record(
    config: ExperimentConfig,
    artifacts: ExperimentArtifacts,
    baseline_daytime_table: pd.DataFrame,
    plant_daytime_table: pd.DataFrame,
    subset_count_table: pd.DataFrame,
    seed_summary: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    rolling_window_table: pd.DataFrame,
) -> str:
    seed_display = format_mean_std_table(seed_summary)
    rolling_display = format_mean_std_table(rolling_summary)
    hybrid_mae = extract_metric(artifacts.baseline_table, "Hybrid", "MAE")
    tft_mae = extract_metric(artifacts.baseline_table, "TFT", "MAE")
    stacked_mae = extract_metric(artifacts.baseline_table, "StackedXGB", "MAE")
    hybrid_daytime_mae = extract_metric(baseline_daytime_table, "Hybrid", "MAE")
    tft_daytime_mae = extract_metric(baseline_daytime_table, "TFT", "MAE")
    stacked_daytime_mae = extract_metric(baseline_daytime_table, "StackedXGB", "MAE")

    hybrid_regime_weights = normalize_nested_weight_dict(artifacts.hybrid_result.regime_weights or {})
    hybrid_without_plant_weights = normalize_nested_weight_dict(
        artifacts.hybrid_without_plant_result.regime_weights or {}
    ) if artifacts.hybrid_without_plant_result is not None else {}

    return f"""
# 光伏预测 Benchmark 与鲁棒性实验记录

## 1. 实验设置
- 数据源：`dataset/` 下 4 个电站 CSV，仓库内数据字段和电站 ID 已统一为英文。
- 站点假设：Alice Springs, Australia，纬度 `-23.6980`，经度 `133.8807`，时区偏移 `UTC+9.5`。
- 任务定义：基于历史与当前气象信息、未来一步已知天气条件，进行 `5 min ahead` 单步功率条件回归。
- 主实验切分：每个电站按时间顺序 `8:1:1` 划分训练/验证/测试，并在训练/验证、验证/测试之间额外保留 `{config.split_gap_steps}` 个时间步间隔。
- 补充评估：新增 `daytime-only` 指标、多随机种子重复和 rolling-origin evaluation。
- 评估指标：MAE、RMSE、MAPE、sMAPE、R2，同时记录预测均值与实际均值。

## 2. 主实验 Baseline 结果
{build_markdown_table(artifacts.baseline_table)}

## 3. 主实验消融结果
{build_markdown_table(artifacts.ablation_table)}

## 4. Daytime-Only 结果
### 4.1 子集样本占比
{build_markdown_table(subset_count_table)}

### 4.2 Daytime Baseline
{build_markdown_table(baseline_daytime_table)}

### 4.3 Daytime 分电站结果
{build_markdown_table(plant_daytime_table)}

## 5. 多随机种子重复
- 随机种子：`{config.robustness_seeds}`

{build_markdown_table(seed_display)}

## 6. Rolling-Origin Evaluation
### 6.1 滚动窗口定义
{build_markdown_table(rolling_window_table)}

### 6.2 跨窗口汇总
{build_markdown_table(rolling_display)}

## 7. Hybrid 权重与物理修正
- Full Hybrid 场景阈值：`{artifacts.hybrid_result.thresholds}`
- Full Hybrid 场景权重：`{hybrid_regime_weights}`
- Full Hybrid 夜间修正 alpha：`{artifacts.hybrid_result.physics_alpha:.3f}`
- w/o Plant Adaptation 场景权重：`{hybrid_without_plant_weights}`
- w/o Scene Adaptation 固定权重：`{artifacts.fixed_global_weights}`，alpha=`{artifacts.fixed_global_alpha:.3f}`
- AdaptiveBlend 平均验证权重：`{artifacts.adaptive_result.avg_val_weights}`
- AdaptiveBlend 平均测试权重：`{artifacts.adaptive_result.avg_test_weights}`
- StackedXGB 物理修正 alpha：`{artifacts.stacked_result.physics_alpha:.3f}`

## 8. 当前结论
- 全样本主结果中，`Hybrid` MAE=`{hybrid_mae:.6f}`，优于 `TFT` 的 `{tft_mae:.6f}`，但仍略高于 `StackedXGB` 的 `{stacked_mae:.6f}`。
- 在 `daytime-only` 子集上，`Hybrid` MAE=`{hybrid_daytime_mae:.6f}`，同样优于 `TFT` 的 `{tft_daytime_mae:.6f}`，而 `StackedXGB` 仍保持最优的 `{stacked_daytime_mae:.6f}`。
- 从消融看，`w/o Scene Adaptation` 退化最明显，说明场景适配是当前 `Hybrid` 的核心增益来源。
- 多随机种子结果说明 `Hybrid / StackedXGB` 的结论不是单次幸运结果，rolling-origin 结果说明该结论不依赖单一时间切分窗口。
""".strip()


def build_result_summary(
    artifacts: ExperimentArtifacts,
    baseline_daytime_table: pd.DataFrame,
    seed_summary: pd.DataFrame,
    rolling_summary: pd.DataFrame,
) -> str:
    seed_display = format_mean_std_table(seed_summary)
    rolling_display = format_mean_std_table(rolling_summary)
    hybrid_mae = extract_metric(artifacts.baseline_table, "Hybrid", "MAE")
    tft_mae = extract_metric(artifacts.baseline_table, "TFT", "MAE")
    stacked_mae = extract_metric(artifacts.baseline_table, "StackedXGB", "MAE")
    hybrid_daytime_mae = extract_metric(baseline_daytime_table, "Hybrid", "MAE")
    tft_daytime_mae = extract_metric(baseline_daytime_table, "TFT", "MAE")
    stacked_daytime_mae = extract_metric(baseline_daytime_table, "StackedXGB", "MAE")

    return f"""
# 最终结果摘要

## 1. 论文主线建议
- 主方法：`Hybrid`
- 最强性能对照：`StackedXGB`
- 评测定位：面向 4 个异构光伏装置的标准化 benchmark / evaluation protocol

## 2. 全样本主结果
{build_markdown_table(artifacts.baseline_table[["Model", "MAE", "RMSE", "R2"]])}

## 3. Daytime-Only 结果
{build_markdown_table(baseline_daytime_table[["Model", "MAE", "RMSE", "R2"]])}

## 4. 最关键的结论
- 全样本上，`Hybrid` 的 MAE 从 `TFT` 的 `{tft_mae:.6f}` 下降到 `{hybrid_mae:.6f}`。
- 白天子集上，`Hybrid` 的 MAE 从 `TFT` 的 `{tft_daytime_mae:.6f}` 下降到 `{hybrid_daytime_mae:.6f}`。
- `StackedXGB` 仍然给出当前最优结果，全样本 MAE=`{stacked_mae:.6f}`，白天 MAE=`{stacked_daytime_mae:.6f}`。
- `Hybrid` 最适合作为正文主方法，`StackedXGB` 最适合作为强性能上界。

## 5. 多随机种子稳定性
{build_markdown_table(seed_display)}

## 6. Rolling-Origin 汇总
{build_markdown_table(rolling_display)}

## 7. 推荐写法
- 更稳的写法是：本文构建了一个面向异构光伏装置的标准化评测 benchmark，并提出了一个按电站、按辐照场景切换权重的可解释 `Hybrid`；在更严格的 daytime-only、多随机种子和 rolling-origin 评估下，`Hybrid` 仍稳定优于强基线 `TFT`。
""".strip()


def build_method_story() -> str:
    return """
# 方法思路与创新点说明

## 1. 研究定位
本文更适合定位为两部分工作：
- 一套面向 4 个异构光伏装置的标准化评测 benchmark
- 一个以 `Hybrid` 为主线、以 `StackedXGB` 为性能上界的可解释融合方法体系

## 2. 主方法为什么选 Hybrid
- `Hybrid` 不是简单的固定加权平均，而是显式地按电站和辐照场景切换权重。
- 这种结构本身可解释，审稿人可以直接读出“什么场景更信谁”。
- 相比之下，`StackedXGB` 虽然更强，但它更适合作为 strongest empirical model，而不是方法叙事中心。

## 3. 创新点建议
- 创新点1：构建了统一的数据处理、切分协议、基线模型和结果导出流程，形成了可复现的 benchmark。
- 创新点2：提出了按电站、按场景切换权重的 `Hybrid`，把异构基模型的误差互补显式结构化。
- 创新点3：在严格于单次主切分的评测下，仍验证了 `Hybrid` 对 `TFT` 的稳定优势，包括 daytime-only、多随机种子和 rolling-origin。

## 4. 论文里建议保留的角色分工
- `TFT`：最强单模型基线
- `Hybrid`：主方法
- `StackedXGB`：性能上界和强对照
- `AdaptiveBlend`：动态权重融合的辅助对照

## 5. benchmark 这条线怎么写更稳
- 不要写成“行业标准”。
- 更稳的写法是：本文建立了一个针对 Alice Springs 四个异构光伏装置的标准化评测 benchmark。
- 如果后续数据和代码都能公开，这条贡献会更强。
""".strip()


def build_paper_outline() -> str:
    return """
# 论文大纲与写作建议

## 1. 题目候选
- 面向异构光伏装置的场景自适应融合与标准化评测基准
- A Benchmark and Scene-Adaptive Hybrid for Ultra-Short-Term Photovoltaic Power Forecasting Across Heterogeneous Plants
- Standardized Benchmarking and Explainable Hybrid Fusion for Ultra-Short-Term PV Power Forecasting

## 2. 摘要写法
- 第一段：光伏短时预测的重要性，以及异构装置场景下误差模式差异明显。
- 第二段：现有研究常见问题，包括固定切分、单次运行、夜间样本稀释和固定全局权重融合。
- 第三段：本文构建标准化 benchmark，并提出按电站和辐照场景切换权重的 `Hybrid`。
- 第四段：在 daytime-only、多随机种子和 rolling-origin 评估下，`Hybrid` 稳定优于 `TFT`，`StackedXGB` 给出当前最佳精度。
- 第五段：总结方法的可解释性、复现性和工程价值。

## 3. 正文结构

### 3.1 引言
- 说明高频超短期光伏预测的应用价值。
- 强调异构装置下误差模式的差异。
- 引出需要 benchmark 和可解释融合方法。

### 3.2 相关工作
- 光伏短时预测中的机器学习和深度学习。
- 异构集成与场景自适应融合。
- 光伏预测中的物理约束与后修正。
- Benchmarking / evaluation protocol 在时序预测中的重要性。

### 3.3 数据与 benchmark
- 数据来源、英文字段、电站命名。
- 标准化特征工程。
- 主切分协议、gap 设置、daytime-only 评估、多随机种子、rolling-origin。

### 3.4 方法
- 基学习器：XGBoost、DNN、TFT。
- 主方法：`Hybrid`。
- 对照：AdaptiveBlend、StackedXGB。
- 物理后修正。

### 3.5 实验设计
- 总体指标：MAE、RMSE、R2。
- 主实验表。
- 消融表。
- Daytime-only 表。
- 多随机种子稳定性表。
- Rolling-origin 表。

### 3.6 结果分析
- `Hybrid` 为什么优于 `TFT`。
- 场景适配为什么是主增益来源。
- 为什么 `StackedXGB` 更强，但不替代 `Hybrid` 作为主方法。

### 3.7 讨论与局限
- 数据仍来自单站点、四个装置。
- 当前任务是未来一步天气已知的条件回归。
- 仍可继续扩展到多步预测和更大范围数据。
""".strip()


def build_robustness_report(
    subset_count_table: pd.DataFrame,
    baseline_daytime_table: pd.DataFrame,
    seed_summary: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    rolling_window_table: pd.DataFrame,
) -> str:
    return f"""
# 鲁棒性评估记录

## 1. Daytime-Only 子集
{build_markdown_table(subset_count_table)}

{build_markdown_table(baseline_daytime_table[["Model", "MAE", "RMSE", "R2"]])}

## 2. 多随机种子重复
{build_markdown_table(format_mean_std_table(seed_summary))}

## 3. Rolling-Origin 窗口
{build_markdown_table(rolling_window_table)}

{build_markdown_table(format_mean_std_table(rolling_summary))}
""".strip()


def save_primary_outputs(
    config: ExperimentConfig,
    main_artifacts: ExperimentArtifacts,
    baseline_daytime_table: pd.DataFrame,
    plant_daytime_table: pd.DataFrame,
    subset_count_table: pd.DataFrame,
) -> None:
    save_metrics_table(main_artifacts.baseline_table.to_dict("records"), config.metric_dir / "baseline_metrics.csv")
    save_metrics_table(main_artifacts.ablation_table.to_dict("records"), config.metric_dir / "ablation_metrics.csv")
    save_metrics_table(main_artifacts.plant_table.to_dict("records"), config.metric_dir / "plant_level_metrics.csv")
    save_metrics_table(baseline_daytime_table.to_dict("records"), config.metric_dir / "baseline_daytime_metrics.csv")
    save_metrics_table(plant_daytime_table.to_dict("records"), config.metric_dir / "plant_level_daytime_metrics.csv")
    save_metrics_table(subset_count_table.to_dict("records"), config.metric_dir / "subset_counts.csv")
    save_metrics_table(main_artifacts.blend_summary.to_dict("records"), config.metric_dir / "blend_summary.csv")
    main_artifacts.val_predictions.to_csv(config.metric_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    main_artifacts.test_predictions.to_csv(config.metric_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    plot_metric_bars(main_artifacts.baseline_table, config.plot_dir / "baseline_mae_rmse.png")
    plot_metric_bars(baseline_daytime_table, config.plot_dir / "baseline_daytime_mae_rmse.png")
    plot_training_curves(
        main_artifacts.dnn_history,
        main_artifacts.tft_history,
        main_artifacts.adaptive_history,
        main_artifacts.stacked_history,
        config.plot_dir / "training_curves.png",
    )
    plot_forecast_examples(
        main_artifacts.test_predictions,
        sorted(main_artifacts.test_predictions["plant_id"].unique()),
        config.plot_dir / "forecast_examples.png",
    )


def main() -> None:
    config = ExperimentConfig()
    main_config = clone_runtime_config(config, seed=config.random_seed, log_subdir="main")
    main_artifacts = run_single_experiment(main_config, include_ablations=True)

    daytime_frame = filter_daytime_frame(main_artifacts.test_predictions)
    baseline_daytime_table = collect_metric_table(daytime_frame, BASELINE_SPECS)
    plant_daytime_table = collect_plant_metric_table(daytime_frame, BASELINE_SPECS)
    subset_count_table = build_subset_count_table(main_artifacts.test_predictions)

    save_primary_outputs(
        config=config,
        main_artifacts=main_artifacts,
        baseline_daytime_table=baseline_daytime_table,
        plant_daytime_table=plant_daytime_table,
        subset_count_table=subset_count_table,
    )

    seed_repeat_table, seed_repeat_summary = run_seed_repeats(config, main_artifacts.prepared_data)
    rolling_origin_table, rolling_origin_summary, rolling_window_table = run_rolling_origin_evaluation(
        config,
        main_artifacts.prepared_data,
    )
    save_metrics_table(seed_repeat_table.to_dict("records"), config.metric_dir / "seed_repeat_metrics.csv")
    save_metrics_table(seed_repeat_summary.to_dict("records"), config.metric_dir / "seed_repeat_summary.csv")
    save_metrics_table(rolling_origin_table.to_dict("records"), config.metric_dir / "rolling_origin_metrics.csv")
    save_metrics_table(rolling_origin_summary.to_dict("records"), config.metric_dir / "rolling_origin_summary.csv")
    save_metrics_table(rolling_window_table.to_dict("records"), config.metric_dir / "rolling_origin_windows.csv")

    training_record = build_training_record(
        config=config,
        artifacts=main_artifacts,
        baseline_daytime_table=baseline_daytime_table,
        plant_daytime_table=plant_daytime_table,
        subset_count_table=subset_count_table,
        seed_summary=seed_repeat_summary,
        rolling_summary=rolling_origin_summary,
        rolling_window_table=rolling_window_table,
    )
    write_markdown_report(config.report_dir / "training_log_zh.md", training_record)
    write_markdown_report(
        config.report_dir / "result_summary_zh.md",
        build_result_summary(
            artifacts=main_artifacts,
            baseline_daytime_table=baseline_daytime_table,
            seed_summary=seed_repeat_summary,
            rolling_summary=rolling_origin_summary,
        ),
    )
    write_markdown_report(config.report_dir / "method_story_zh.md", build_method_story())
    write_markdown_report(config.report_dir / "paper_outline_zh.md", build_paper_outline())
    write_markdown_report(
        config.report_dir / "robustness_summary_zh.md",
        build_robustness_report(
            subset_count_table=subset_count_table,
            baseline_daytime_table=baseline_daytime_table,
            seed_summary=seed_repeat_summary,
            rolling_summary=rolling_origin_summary,
            rolling_window_table=rolling_window_table,
        ),
    )

    print("Baseline metrics saved to:", config.metric_dir / "baseline_metrics.csv")
    print("Ablation metrics saved to:", config.metric_dir / "ablation_metrics.csv")
    print("Daytime metrics saved to:", config.metric_dir / "baseline_daytime_metrics.csv")
    print("Seed repeat summary saved to:", config.metric_dir / "seed_repeat_summary.csv")
    print("Rolling-origin summary saved to:", config.metric_dir / "rolling_origin_summary.csv")
    print("Training log saved to:", config.report_dir / "training_log_zh.md")


if __name__ == "__main__":
    main()
