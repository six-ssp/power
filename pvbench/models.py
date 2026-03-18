from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, RMSE, QuantileLoss

from .config import WEATHER_COLUMNS, ExperimentConfig
from .data import GEO_FEATURES, PHYSICS_FEATURES, PreparedData


@dataclass
class TabularModelResult:
    model_name: str
    val_predictions: np.ndarray
    test_predictions: np.ndarray
    history: pd.DataFrame


@dataclass
class TFTResult:
    val_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    history: pd.DataFrame


@dataclass
class MetaModelResult:
    model_name: str
    val_predictions: np.ndarray
    test_predictions: np.ndarray
    history: pd.DataFrame
    physics_alpha: float = 0.0
    avg_val_weights: dict[str, float] | None = None
    avg_test_weights: dict[str, float] | None = None


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class AdaptiveBlendRegressor(nn.Module):
    def __init__(self, input_dim: int, num_models: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.weight_head = nn.Linear(prev_dim, num_models)

    def forward(self, features: torch.Tensor, base_predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        weights = torch.softmax(self.weight_head(hidden), dim=1)
        prediction = torch.sum(weights * base_predictions, dim=1)
        return prediction, weights


class TFTLightningWrapper:
    @staticmethod
    def known_reals() -> list[str]:
        return [
            "time_idx",
            *WEATHER_COLUMNS,
            "hour_sin",
            "hour_cos",
            "doy_sin",
            "doy_cos",
            "month_sin",
            "month_cos",
            "weekday_sin",
            "weekday_cos",
            *GEO_FEATURES,
            *PHYSICS_FEATURES,
        ]


class TFTPredictionHelper:
    @staticmethod
    def to_prediction_frame(
        prediction_package,
        decoded_index: pd.DataFrame,
        raw_frame: pd.DataFrame,
        prediction_column: str,
    ) -> pd.DataFrame:
        predictions = prediction_package.output.detach().cpu().reshape(-1).numpy()
        actuals = prediction_package.y[0].detach().cpu().reshape(-1).numpy()
        decoded = decoded_index.reset_index(drop=True).copy()

        forecast_column = "time_idx_first_prediction"
        if forecast_column not in decoded.columns:
            matches = [column for column in decoded.columns if "prediction" in column and "time_idx" in column]
            if not matches:
                raise KeyError(f"Cannot locate forecast index column in decoded index: {decoded.columns.tolist()}")
            forecast_column = matches[0]

        prediction_frame = pd.DataFrame(
            {
                "plant_id": decoded["plant_id"].astype(str).to_numpy(),
                "forecast_time_idx": decoded[forecast_column].to_numpy(dtype=int),
                prediction_column: predictions,
                "target_power": actuals,
            }
        )
        timestamp_lookup = raw_frame[["plant_id", "time_idx", "timestamp"]].rename(
            columns={"time_idx": "forecast_time_idx", "timestamp": "forecast_timestamp"}
        )
        prediction_frame = prediction_frame.merge(timestamp_lookup, on=["plant_id", "forecast_time_idx"], how="left")
        return prediction_frame


META_CONTEXT_COLUMNS = [
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


META_CATEGORICAL_COLUMNS = ["plant_id", "plant_name_en", "module_type", "mounting_type"]


def fit_xgboost(
    train_matrix: pd.DataFrame,
    y_train: np.ndarray,
    val_matrix: pd.DataFrame,
    y_val: np.ndarray,
    test_matrix: pd.DataFrame,
    config: ExperimentConfig,
) -> TabularModelResult:
    xgb_params = dict(config.xgb_params)
    if torch.cuda.is_available():
        xgb_params["device"] = "cuda"
    model = XGBRegressor(**xgb_params, early_stopping_rounds=30)
    model.fit(
        train_matrix,
        y_train,
        eval_set=[(train_matrix, y_train), (val_matrix, y_val)],
        verbose=False,
    )
    evals = model.evals_result()
    if config.xgb_predict_on_cpu:
        model.set_params(device="cpu")
    history = pd.DataFrame(
        {
            "round": np.arange(len(evals["validation_0"]["rmse"])),
            "train_rmse": evals["validation_0"]["rmse"],
            "val_rmse": evals["validation_1"]["rmse"],
        }
    )
    return TabularModelResult(
        model_name="xgboost",
        val_predictions=model.predict(val_matrix),
        test_predictions=model.predict(test_matrix),
        history=history,
    )


def fit_dnn(
    train_matrix: pd.DataFrame,
    y_train: np.ndarray,
    val_matrix: pd.DataFrame,
    y_val: np.ndarray,
    test_matrix: pd.DataFrame,
    config: ExperimentConfig,
) -> TabularModelResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_matrix).astype(np.float32)
    x_val = scaler.transform(val_matrix).astype(np.float32)
    x_test = scaler.transform(test_matrix).astype(np.float32)

    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)

    model = MLPRegressor(x_train.shape[1], config.dnn_hidden_dims, config.dnn_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.dnn_learning_rate, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=config.dnn_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=config.dnn_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )

    best_state = None
    best_rmse = float("inf")
    patience_left = config.dnn_patience
    history_rows: list[dict[str, float]] = []

    for epoch in range(1, config.dnn_epochs + 1):
        model.train()
        train_losses = []
        for features, target in train_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        val_predictions = []
        with torch.no_grad():
            for features, target in val_loader:
                features = features.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                prediction = model(features)
                loss = criterion(prediction, target)
                val_losses.append(float(loss.detach().cpu()))
                val_predictions.append(prediction.cpu().numpy())
        val_pred = np.concatenate(val_predictions)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": float(np.mean(val_losses)),
                "val_rmse": val_rmse,
            }
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_left = config.dnn_patience
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is None:
        raise RuntimeError("DNN training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_predictions = model(torch.from_numpy(x_val).to(device)).cpu().numpy()
        test_predictions = model(torch.from_numpy(x_test).to(device)).cpu().numpy()

    return TabularModelResult(
        model_name="dnn",
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        history=pd.DataFrame(history_rows),
    )


def fit_tft(prepared_data: PreparedData, config: ExperimentConfig) -> TFTResult:
    raw_frame = prepared_data.raw_frame.copy()
    training_frame = raw_frame[raw_frame["time_idx"] <= prepared_data.train_cutoff]
    validation_frame = raw_frame[raw_frame["time_idx"] <= prepared_data.val_cutoff]

    training_dataset = TimeSeriesDataSet(
        training_frame,
        time_idx="time_idx",
        target="power",
        group_ids=["plant_id"],
        static_categoricals=["plant_id", "plant_name_en", "module_type", "mounting_type"],
        time_varying_known_reals=TFTLightningWrapper.known_reals(),
        time_varying_unknown_reals=["power"],
        max_encoder_length=config.tft_encoder_length,
        min_encoder_length=config.tft_encoder_length,
        max_prediction_length=config.horizon,
        min_prediction_length=config.horizon,
        target_normalizer=GroupNormalizer(groups=["plant_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    val_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        validation_frame,
        min_prediction_idx=prepared_data.train_cutoff + 1,
        stop_randomization=True,
        predict=False,
    )
    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        raw_frame,
        min_prediction_idx=prepared_data.val_cutoff + 1,
        stop_randomization=True,
        predict=False,
    )

    train_loader = training_dataset.to_dataloader(
        train=True,
        batch_size=config.tft_batch_size,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = val_dataset.to_dataloader(
        train=False,
        batch_size=config.tft_batch_size * 2,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )
    test_loader = test_dataset.to_dataloader(
        train=False,
        batch_size=config.tft_batch_size * 2,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )

    logger = CSVLogger(save_dir=str(config.log_dir), name="tft")
    early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=config.tft_patience, mode="min")
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=1e-3,
        hidden_size=config.tft_hidden_size,
        attention_head_size=config.tft_attention_heads,
        hidden_continuous_size=config.tft_hidden_continuous_size,
        dropout=0.1,
        output_size=7,
        loss=QuantileLoss(),
        logging_metrics=nn.ModuleList([MAE(), RMSE()]),
        reduce_on_plateau_patience=1,
        log_interval=100,
    )

    trainer = pl.Trainer(
        max_epochs=config.tft_max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        gradient_clip_val=0.1,
        callbacks=[early_stop],
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    val_predictions = model.predict(
        val_loader,
        mode="prediction",
        return_y=True,
        trainer_kwargs={"logger": False, "enable_model_summary": False},
    )
    test_predictions = model.predict(
        test_loader,
        mode="prediction",
        return_y=True,
        trainer_kwargs={"logger": False, "enable_model_summary": False},
    )

    metrics_path = Path(logger.log_dir) / "metrics.csv"
    history = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    val_prediction_frame = TFTPredictionHelper.to_prediction_frame(
        val_predictions,
        val_dataset.decoded_index,
        raw_frame,
        prediction_column="tft_prediction",
    )
    test_prediction_frame = TFTPredictionHelper.to_prediction_frame(
        test_predictions,
        test_dataset.decoded_index,
        raw_frame,
        prediction_column="tft_prediction",
    )
    return TFTResult(
        val_predictions=val_prediction_frame,
        test_predictions=test_prediction_frame,
        history=history,
    )


def tune_blend_weights(
    validation_frame: pd.DataFrame,
    prediction_columns: list[str],
    min_weight: float,
    step: float,
) -> tuple[dict[str, float], np.ndarray, float]:
    weight_grid = np.round(np.arange(min_weight, 1.0, step), 4)
    best_weights: dict[str, float] | None = None
    best_score = float("inf")
    best_prediction = None

    for values in itertools.product(weight_grid, repeat=len(prediction_columns) - 1):
        remainder = 1.0 - float(sum(values))
        if remainder < min_weight - 1e-9:
            continue
        weights = list(values) + [remainder]
        if any(weight < min_weight - 1e-9 for weight in weights):
            continue
        blended = np.zeros(len(validation_frame), dtype=float)
        for column, weight in zip(prediction_columns, weights):
            blended += validation_frame[column].to_numpy(dtype=float) * weight
        mae = float(np.mean(np.abs(blended - validation_frame["target_power"].to_numpy(dtype=float))))
        if mae < best_score:
            best_score = mae
            best_weights = {column: weight for column, weight in zip(prediction_columns, weights)}
            best_prediction = blended

    if best_weights is None or best_prediction is None:
        raise RuntimeError("Blend weight search did not find a valid solution.")
    return best_weights, best_prediction, best_score


def apply_physics_adjustment(frame: pd.DataFrame, prediction: np.ndarray, alpha: float, threshold: float) -> np.ndarray:
    positive_part = np.clip(prediction, a_min=0.0, a_max=None)
    low_radiation = np.clip((threshold - frame["forecast_global_radiation"].to_numpy(dtype=float)) / threshold, 0.0, 1.0)
    night_strength = np.clip((-frame["forecast_solar_elevation_deg"].to_numpy(dtype=float)) / 12.0, 0.0, 1.0)
    reduction = alpha * low_radiation * night_strength * positive_part
    return prediction - reduction


def tune_physics_alpha(frame: pd.DataFrame, blended_prediction: np.ndarray, config: ExperimentConfig) -> tuple[float, np.ndarray]:
    best_alpha = 0.0
    best_prediction = blended_prediction.copy()
    best_mae = float(np.mean(np.abs(blended_prediction - frame["target_power"].to_numpy(dtype=float))))

    for alpha in config.night_alpha_grid:
        candidate = apply_physics_adjustment(frame, blended_prediction, alpha, config.night_radiation_threshold)
        mae = float(np.mean(np.abs(candidate - frame["target_power"].to_numpy(dtype=float))))
        if mae < best_mae:
            best_alpha = alpha
            best_prediction = candidate
            best_mae = mae
    return best_alpha, best_prediction


def split_meta_validation_frame(frame: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    holdout_parts: list[pd.DataFrame] = []

    for _, plant_frame in frame.groupby("plant_id", sort=False):
        plant_frame = plant_frame.sort_values("forecast_time_idx").reset_index(drop=True)
        split_index = int(len(plant_frame) * train_ratio)
        split_index = min(max(split_index, 1), len(plant_frame) - 1)
        train_parts.append(plant_frame.iloc[:split_index].copy())
        holdout_parts.append(plant_frame.iloc[split_index:].copy())

    return pd.concat(train_parts, ignore_index=True), pd.concat(holdout_parts, ignore_index=True)


def build_meta_feature_frame(frame: pd.DataFrame, prediction_columns: list[str]) -> pd.DataFrame:
    meta_frame = pd.DataFrame(index=frame.index)

    for column in prediction_columns:
        meta_frame[column] = frame[column].to_numpy(dtype=float)
    for column in META_CONTEXT_COLUMNS:
        if column in frame.columns:
            meta_frame[column] = frame[column].to_numpy(dtype=float)
    for column in META_CATEGORICAL_COLUMNS:
        if column in frame.columns:
            meta_frame[column] = frame[column].astype(str).to_numpy()

    prediction_matrix = frame[prediction_columns].to_numpy(dtype=float)
    meta_frame["prediction_mean"] = prediction_matrix.mean(axis=1)
    meta_frame["prediction_std"] = prediction_matrix.std(axis=1)
    meta_frame["prediction_min"] = prediction_matrix.min(axis=1)
    meta_frame["prediction_max"] = prediction_matrix.max(axis=1)
    meta_frame["prediction_range"] = meta_frame["prediction_max"] - meta_frame["prediction_min"]

    if "persistence_prediction" in frame.columns and "tft_prediction" in frame.columns:
        meta_frame["persistence_tft_gap"] = (
            frame["persistence_prediction"].to_numpy(dtype=float) - frame["tft_prediction"].to_numpy(dtype=float)
        )
    if "xgboost_prediction" in frame.columns and "tft_prediction" in frame.columns:
        meta_frame["xgboost_tft_gap"] = (
            frame["xgboost_prediction"].to_numpy(dtype=float) - frame["tft_prediction"].to_numpy(dtype=float)
        )
    if "dnn_prediction" in frame.columns and "tft_prediction" in frame.columns:
        meta_frame["dnn_tft_gap"] = (
            frame["dnn_prediction"].to_numpy(dtype=float) - frame["tft_prediction"].to_numpy(dtype=float)
        )
    if "xgboost_prediction" in frame.columns and "dnn_prediction" in frame.columns:
        meta_frame["xgboost_dnn_gap"] = (
            frame["xgboost_prediction"].to_numpy(dtype=float) - frame["dnn_prediction"].to_numpy(dtype=float)
        )
    return meta_frame


def build_meta_design_matrices(
    train_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    prediction_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_meta = build_meta_feature_frame(train_frame, prediction_columns)
    holdout_meta = build_meta_feature_frame(holdout_frame, prediction_columns)
    val_meta = build_meta_feature_frame(val_frame, prediction_columns)
    test_meta = build_meta_feature_frame(test_frame, prediction_columns)

    categorical_columns = [column for column in META_CATEGORICAL_COLUMNS if column in train_meta.columns]
    encoded = pd.get_dummies(
        pd.concat(
            {
                "train": train_meta,
                "holdout": holdout_meta,
                "val": val_meta,
                "test": test_meta,
            },
            axis=0,
        ),
        columns=categorical_columns,
        dtype=float,
    )
    return encoded.xs("train"), encoded.xs("holdout"), encoded.xs("val"), encoded.xs("test")


def fit_stacked_xgboost(
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    prediction_columns: list[str],
    config: ExperimentConfig,
) -> MetaModelResult:
    meta_train_frame, meta_holdout_frame = split_meta_validation_frame(validation_frame, config.meta_train_ratio)
    train_matrix, holdout_matrix, val_matrix, test_matrix = build_meta_design_matrices(
        meta_train_frame,
        meta_holdout_frame,
        validation_frame,
        test_frame,
        prediction_columns,
    )

    y_train = meta_train_frame["target_power"].to_numpy(dtype=float)
    y_holdout = meta_holdout_frame["target_power"].to_numpy(dtype=float)

    stack_params = dict(config.stack_xgb_params)
    if torch.cuda.is_available():
        stack_params["device"] = "cuda"

    model = XGBRegressor(**stack_params, early_stopping_rounds=50)
    model.fit(
        train_matrix,
        y_train,
        eval_set=[(train_matrix, y_train), (holdout_matrix, y_holdout)],
        verbose=False,
    )

    if config.xgb_predict_on_cpu:
        model.set_params(device="cpu")

    holdout_prediction = model.predict(holdout_matrix)
    physics_alpha, _ = tune_physics_alpha(meta_holdout_frame, holdout_prediction, config)

    evals = model.evals_result()
    history = pd.DataFrame(
        {
            "round": np.arange(len(evals["validation_0"]["rmse"])),
            "train_rmse": evals["validation_0"]["rmse"],
            "holdout_rmse": evals["validation_1"]["rmse"],
        }
    )

    val_prediction = model.predict(val_matrix)
    test_prediction = model.predict(test_matrix)
    return MetaModelResult(
        model_name="stacked_xgboost",
        val_predictions=apply_physics_adjustment(
            validation_frame,
            val_prediction,
            physics_alpha,
            config.night_radiation_threshold,
        ),
        test_predictions=apply_physics_adjustment(
            test_frame,
            test_prediction,
            physics_alpha,
            config.night_radiation_threshold,
        ),
        history=history,
        physics_alpha=physics_alpha,
    )


def fit_adaptive_blend(
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    prediction_columns: list[str],
    config: ExperimentConfig,
) -> MetaModelResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_train_frame, meta_holdout_frame = split_meta_validation_frame(validation_frame, config.meta_train_ratio)
    train_matrix, holdout_matrix, val_matrix, test_matrix = build_meta_design_matrices(
        meta_train_frame,
        meta_holdout_frame,
        validation_frame,
        test_frame,
        prediction_columns,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_matrix).astype(np.float32)
    x_holdout = scaler.transform(holdout_matrix).astype(np.float32)
    x_val = scaler.transform(val_matrix).astype(np.float32)
    x_test = scaler.transform(test_matrix).astype(np.float32)

    base_train = meta_train_frame[prediction_columns].to_numpy(dtype=np.float32)
    base_holdout = meta_holdout_frame[prediction_columns].to_numpy(dtype=np.float32)
    base_val = validation_frame[prediction_columns].to_numpy(dtype=np.float32)
    base_test = test_frame[prediction_columns].to_numpy(dtype=np.float32)

    y_train = meta_train_frame["target_power"].to_numpy(dtype=np.float32)
    y_holdout = meta_holdout_frame["target_power"].to_numpy(dtype=np.float32)

    model = AdaptiveBlendRegressor(
        input_dim=x_train.shape[1],
        num_models=len(prediction_columns),
        hidden_dims=config.adaptive_hidden_dims,
        dropout=config.adaptive_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.adaptive_learning_rate, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(base_train),
            torch.from_numpy(y_train),
        ),
        batch_size=config.adaptive_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    holdout_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_holdout),
            torch.from_numpy(base_holdout),
            torch.from_numpy(y_holdout),
        ),
        batch_size=config.adaptive_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )

    best_state = None
    best_holdout_mae = float("inf")
    patience_left = config.adaptive_patience
    history_rows: list[dict[str, float]] = []

    for epoch in range(1, config.adaptive_epochs + 1):
        model.train()
        train_losses = []
        for features, base_predictions, target in train_loader:
            features = features.to(device, non_blocking=True)
            base_predictions = base_predictions.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(features, base_predictions)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        holdout_losses = []
        holdout_predictions = []
        with torch.no_grad():
            for features, base_predictions, target in holdout_loader:
                features = features.to(device, non_blocking=True)
                base_predictions = base_predictions.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                prediction, _ = model(features, base_predictions)
                loss = criterion(prediction, target)
                holdout_losses.append(float(loss.detach().cpu()))
                holdout_predictions.append(prediction.cpu().numpy())

        holdout_prediction = np.concatenate(holdout_predictions)
        holdout_mae = float(
            np.mean(np.abs(holdout_prediction - meta_holdout_frame["target_power"].to_numpy(dtype=float)))
        )
        holdout_rmse = float(
            np.sqrt(np.mean(np.square(holdout_prediction - meta_holdout_frame["target_power"].to_numpy(dtype=float))))
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "holdout_loss": float(np.mean(holdout_losses)),
                "holdout_mae": holdout_mae,
                "holdout_rmse": holdout_rmse,
            }
        )

        if holdout_mae < best_holdout_mae:
            best_holdout_mae = holdout_mae
            patience_left = config.adaptive_patience
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is None:
        raise RuntimeError("Adaptive blend training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        holdout_tensor = torch.from_numpy(x_holdout).to(device)
        holdout_base_tensor = torch.from_numpy(base_holdout).to(device)
        holdout_prediction, _ = model(holdout_tensor, holdout_base_tensor)
        holdout_prediction_np = holdout_prediction.cpu().numpy()

        val_tensor = torch.from_numpy(x_val).to(device)
        val_base_tensor = torch.from_numpy(base_val).to(device)
        val_prediction, val_weights = model(val_tensor, val_base_tensor)

        test_tensor = torch.from_numpy(x_test).to(device)
        test_base_tensor = torch.from_numpy(base_test).to(device)
        test_prediction, test_weights = model(test_tensor, test_base_tensor)

    physics_alpha, _ = tune_physics_alpha(meta_holdout_frame, holdout_prediction_np, config)

    avg_val_weights = {
        column: float(weight)
        for column, weight in zip(prediction_columns, val_weights.cpu().numpy().mean(axis=0).tolist())
    }
    avg_test_weights = {
        column: float(weight)
        for column, weight in zip(prediction_columns, test_weights.cpu().numpy().mean(axis=0).tolist())
    }

    return MetaModelResult(
        model_name="adaptive_blend",
        val_predictions=apply_physics_adjustment(
            validation_frame,
            val_prediction.cpu().numpy(),
            physics_alpha,
            config.night_radiation_threshold,
        ),
        test_predictions=apply_physics_adjustment(
            test_frame,
            test_prediction.cpu().numpy(),
            physics_alpha,
            config.night_radiation_threshold,
        ),
        history=pd.DataFrame(history_rows),
        physics_alpha=physics_alpha,
        avg_val_weights=avg_val_weights,
        avg_test_weights=avg_test_weights,
    )
