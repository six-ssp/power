from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pvbench import ExperimentConfig, load_and_prepare_data
from pvbench.data import build_windowed_prepared_data
from pvbench.reporting import build_physical_violation_margin_map
from run_experiments import (
    BASELINE_SPECS,
    build_subset_count_table,
    build_training_configuration_table,
    build_training_execution_table,
    build_window_description,
    clone_runtime_config,
    collect_metric_table,
    collect_physical_metric_table,
    collect_plant_metric_table,
    collect_plant_physical_metric_table,
    filter_daytime_frame,
    log_progress,
    run_single_experiment,
    save_metrics_table,
    save_primary_outputs,
)


def upsert_by_column(existing: pd.DataFrame, incoming: pd.DataFrame, key: str) -> pd.DataFrame:
    if existing.empty:
        return incoming.copy()
    if incoming.empty:
        return existing.copy()
    remaining = existing[~existing[key].isin(incoming[key])].copy()
    combined = pd.concat([remaining, incoming], ignore_index=True)
    return combined


def run_main_stage(config: ExperimentConfig) -> None:
    log_progress("Stage runner: main stage started.")
    main_config = clone_runtime_config(config, seed=config.random_seed, log_subdir="main_stage")
    artifacts = run_single_experiment(main_config, include_ablations=True)
    training_config_table = build_training_configuration_table(config)
    training_execution_table = build_training_execution_table(config, artifacts)

    daytime_frame = filter_daytime_frame(artifacts.test_predictions)
    baseline_daytime_table = collect_metric_table(daytime_frame, BASELINE_SPECS)
    plant_daytime_table = collect_plant_metric_table(daytime_frame, BASELINE_SPECS)
    physical_margin_map = build_physical_violation_margin_map(
        artifacts.prepared_data.train_frame,
        tolerance_ratio=config.physical_violation_tolerance_ratio,
        min_margin=config.physical_violation_min_margin,
    )
    baseline_physical_table = collect_physical_metric_table(
        artifacts.test_predictions,
        BASELINE_SPECS,
        margin_map=physical_margin_map,
    )
    plant_physical_table = collect_plant_physical_metric_table(
        artifacts.test_predictions,
        BASELINE_SPECS,
        margin_map=physical_margin_map,
    )
    subset_count_table = build_subset_count_table(artifacts.test_predictions)

    save_primary_outputs(
        config=config,
        main_artifacts=artifacts,
        baseline_daytime_table=baseline_daytime_table,
        plant_daytime_table=plant_daytime_table,
        subset_count_table=subset_count_table,
        baseline_physical_table=baseline_physical_table,
        plant_physical_table=plant_physical_table,
    )
    save_metrics_table(training_config_table.to_dict("records"), config.metric_dir / "training_configuration.csv")
    save_metrics_table(training_execution_table.to_dict("records"), config.metric_dir / "training_execution_summary.csv")
    log_progress("Stage runner: main stage finished.")


def run_rolling_window_stage(config: ExperimentConfig, window_index: int) -> None:
    if window_index < 1 or window_index > len(config.rolling_origin_windows):
        raise ValueError(f"window_index must be between 1 and {len(config.rolling_origin_windows)}.")

    prepared_data = load_and_prepare_data(config)
    train_end_ratio, val_end_ratio, test_end_ratio = config.rolling_origin_windows[window_index - 1]
    window_name = f"Window_{window_index}"

    log_progress(f"Stage runner: rolling {window_name} started.")
    window_prepared = build_windowed_prepared_data(
        prepared_data=prepared_data,
        train_end_ratio=train_end_ratio,
        val_end_ratio=val_end_ratio,
        test_end_ratio=test_end_ratio,
        split_gap_steps=config.split_gap_steps,
    )
    runtime_config = clone_runtime_config(
        config,
        seed=config.random_seed,
        log_subdir=f"rolling_origin/{window_name.lower()}_stage",
    )
    artifacts = run_single_experiment(runtime_config, prepared_data=window_prepared, include_ablations=False)

    window_table = artifacts.baseline_table.copy()
    window_table.insert(0, "Window", window_name)
    window_table.insert(1, "TrainEndRatio", train_end_ratio)
    window_table.insert(2, "ValEndRatio", val_end_ratio)
    window_table.insert(3, "TestEndRatio", test_end_ratio)
    window_description = pd.DataFrame(
        [
            build_window_description(
                prepared_data=window_prepared,
                window_name=window_name,
                train_end_ratio=train_end_ratio,
                val_end_ratio=val_end_ratio,
                test_end_ratio=test_end_ratio,
            )
        ]
    )

    rolling_metrics_path = config.metric_dir / "rolling_origin_metrics.csv"
    rolling_summary_path = config.metric_dir / "rolling_origin_summary.csv"
    rolling_windows_path = config.metric_dir / "rolling_origin_windows.csv"

    existing_metrics = pd.read_csv(rolling_metrics_path) if rolling_metrics_path.exists() else pd.DataFrame()
    existing_windows = pd.read_csv(rolling_windows_path) if rolling_windows_path.exists() else pd.DataFrame()

    updated_metrics = upsert_by_column(existing_metrics, window_table, key="Window")
    updated_windows = upsert_by_column(existing_windows, window_description, key="Window")
    updated_metrics = updated_metrics.sort_values(["Window", "Model"]).reset_index(drop=True)
    updated_windows = updated_windows.sort_values("Window").reset_index(drop=True)

    save_metrics_table(updated_metrics.to_dict("records"), rolling_metrics_path)
    save_metrics_table(updated_windows.to_dict("records"), rolling_windows_path)

    summary_rows: list[dict[str, float | str | int]] = []
    for model_name, model_frame in updated_metrics.groupby("Model", sort=False):
        summary_rows.append(
            {
                "Model": model_name,
                "Runs": int(model_frame["Window"].nunique()),
                "MAE_mean": float(model_frame["MAE"].mean()),
                "MAE_std": float(model_frame["MAE"].std(ddof=0)),
                "RMSE_mean": float(model_frame["RMSE"].mean()),
                "RMSE_std": float(model_frame["RMSE"].std(ddof=0)),
                "MAPE_mean": float(model_frame["MAPE"].mean()),
                "MAPE_std": float(model_frame["MAPE"].std(ddof=0)),
                "sMAPE_mean": float(model_frame["sMAPE"].mean()),
                "sMAPE_std": float(model_frame["sMAPE"].std(ddof=0)),
                "R2_mean": float(model_frame["R2"].mean()),
                "R2_std": float(model_frame["R2"].std(ddof=0)),
                "PredMean_mean": float(model_frame["PredMean"].mean()),
                "PredMean_std": float(model_frame["PredMean"].std(ddof=0)),
                "ActualMean_mean": float(model_frame["ActualMean"].mean()),
                "ActualMean_std": float(model_frame["ActualMean"].std(ddof=0)),
                "Bias_mean": float(model_frame["Bias"].mean()),
                "Bias_std": float(model_frame["Bias"].std(ddof=0)),
                "Samples_mean": float(model_frame["Samples"].mean()),
                "Samples_std": float(model_frame["Samples"].std(ddof=0)),
            }
        )
    save_metrics_table(summary_rows, rolling_summary_path)
    log_progress(f"Stage runner: rolling {window_name} finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run long experiment stages separately.")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    subparsers.add_parser("main", help="Run and save only the main experiment outputs.")
    rolling_parser = subparsers.add_parser("rolling-window", help="Run and save one rolling-origin window.")
    rolling_parser.add_argument("--index", type=int, required=True, help="1-based rolling window index.")

    args = parser.parse_args()
    config = ExperimentConfig()

    if args.stage == "main":
        run_main_stage(config)
    elif args.stage == "rolling-window":
        run_rolling_window_stage(config, args.index)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
