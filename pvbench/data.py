from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COLUMN_MAP, PLANT_METADATA, SITE_METADATA, WEATHER_COLUMNS, ExperimentConfig


@dataclass
class PreparedData:
    raw_frame: pd.DataFrame
    supervised_frame: pd.DataFrame
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    feature_columns: list[str]
    categorical_columns: list[str]
    geo_feature_columns: list[str]
    physics_feature_columns: list[str]
    train_cutoff: int
    val_cutoff: int


TIME_FEATURES = [
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
]

GEO_FEATURES = [
    "solar_declination_rad",
    "equation_of_time_min",
    "hour_angle_rad",
    "solar_elevation_deg",
    "solar_zenith_deg",
    "cos_zenith",
    "air_mass_proxy",
    "clear_sky_proxy",
]

PHYSICS_FEATURES = [
    "temperature_derate",
    "effective_global_radiation",
    "effective_direct_radiation",
    "irradiance_temp_ratio",
    "humidity_wind_interaction",
    "night_flag",
]

CATEGORICAL_FEATURES = ["plant_id", "plant_name_en", "module_type", "mounting_type"]


def load_and_prepare_data(config: ExperimentConfig) -> PreparedData:
    raw_frame = load_raw_frame(config.data_dir)
    raw_frame = add_metadata(raw_frame)
    raw_frame = add_time_features(raw_frame)
    raw_frame = add_solar_geometry(raw_frame)
    raw_frame = add_physics_features(raw_frame)
    raw_frame = raw_frame.sort_values(["plant_id", "timestamp"]).reset_index(drop=True)
    raw_frame[WEATHER_COLUMNS + ["power"]] = raw_frame[WEATHER_COLUMNS + ["power"]].astype("float32")
    raw_frame["time_idx"] = raw_frame.groupby("plant_id").cumcount()

    group_sizes = raw_frame.groupby("plant_id").size().tolist()
    if len(set(group_sizes)) != 1:
        raise ValueError("All plants must have the same number of timestamps for this experiment setup.")

    total_steps = group_sizes[0]
    train_cutoff = int(total_steps * config.train_ratio) - 1
    val_cutoff = int(total_steps * (config.train_ratio + config.val_ratio)) - 1

    supervised_frame, feature_columns = build_supervised_frame(raw_frame, config)
    train_frame = supervised_frame[supervised_frame["forecast_time_idx"] <= train_cutoff].copy()
    val_frame = supervised_frame[
        (supervised_frame["forecast_time_idx"] > train_cutoff)
        & (supervised_frame["forecast_time_idx"] <= val_cutoff)
    ].copy()
    test_frame = supervised_frame[supervised_frame["forecast_time_idx"] > val_cutoff].copy()

    return PreparedData(
        raw_frame=raw_frame,
        supervised_frame=supervised_frame,
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        feature_columns=feature_columns,
        categorical_columns=CATEGORICAL_FEATURES,
        geo_feature_columns=GEO_FEATURES,
        physics_feature_columns=PHYSICS_FEATURES,
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
    )


def load_raw_frame(data_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        frame = pd.read_csv(csv_path).rename(columns=COLUMN_MAP)
        frame["source_file"] = csv_path.name
        frames.append(frame)
    raw_frame = pd.concat(frames, ignore_index=True)
    raw_frame["timestamp"] = pd.to_datetime(raw_frame["timestamp"])
    return raw_frame


def add_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["latitude"] = SITE_METADATA["latitude"]
    frame["longitude"] = SITE_METADATA["longitude"]
    frame["site_name"] = SITE_METADATA["site_name"]
    frame["timezone_offset_hours"] = SITE_METADATA["timezone_offset_hours"]

    frame["plant_name_en"] = frame["plant_id"].map(lambda value: PLANT_METADATA[value]["plant_name_en"])
    frame["module_type"] = frame["plant_id"].map(lambda value: PLANT_METADATA[value]["module_type"])
    frame["mounting_type"] = frame["plant_id"].map(lambda value: PLANT_METADATA[value]["mounting_type"])
    return frame


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    hour_float = frame["timestamp"].dt.hour + frame["timestamp"].dt.minute / 60.0
    day_of_year = frame["timestamp"].dt.dayofyear
    month = frame["timestamp"].dt.month
    weekday = frame["timestamp"].dt.weekday

    frame["hour_sin"] = np.sin(2.0 * np.pi * hour_float / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * hour_float / 24.0)
    frame["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.0)
    frame["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.0)
    frame["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    frame["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    frame["weekday_sin"] = np.sin(2.0 * np.pi * weekday / 7.0)
    frame["weekday_cos"] = np.cos(2.0 * np.pi * weekday / 7.0)
    return frame


def add_solar_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    lat_rad = np.deg2rad(frame["latitude"].to_numpy(dtype=float))
    lon_deg = frame["longitude"].to_numpy(dtype=float)
    tz_offset = frame["timezone_offset_hours"].to_numpy(dtype=float)

    timestamps = frame["timestamp"]
    minute_of_day = (
        timestamps.dt.hour.to_numpy(dtype=float) * 60.0
        + timestamps.dt.minute.to_numpy(dtype=float)
        + timestamps.dt.second.to_numpy(dtype=float) / 60.0
    )
    day_of_year = timestamps.dt.dayofyear.to_numpy(dtype=float)
    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1.0 + (minute_of_day / 60.0 - 12.0) / 24.0)

    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    time_offset = equation_of_time + 4.0 * lon_deg - 60.0 * tz_offset
    true_solar_time = minute_of_day + time_offset
    hour_angle_deg = true_solar_time / 4.0 - 180.0
    hour_angle_rad = np.deg2rad(hour_angle_deg)

    cos_zenith = (
        np.sin(lat_rad) * np.sin(declination)
        + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle_rad)
    )
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith_rad = np.arccos(cos_zenith)
    elevation_rad = np.pi / 2.0 - zenith_rad
    solar_elevation_deg = np.rad2deg(elevation_rad)
    solar_zenith_deg = np.rad2deg(zenith_rad)

    safe_cos = np.clip(cos_zenith, 1e-4, None)
    air_mass_proxy = np.where(cos_zenith > 0.0, 1.0 / safe_cos, 0.0)
    extraterrestrial = 1367.0 * (1.0 + 0.033 * np.cos(2.0 * np.pi * day_of_year / 365.0))
    clear_sky_proxy = extraterrestrial * np.clip(cos_zenith, 0.0, None)

    frame["solar_declination_rad"] = declination
    frame["equation_of_time_min"] = equation_of_time
    frame["hour_angle_rad"] = hour_angle_rad
    frame["solar_elevation_deg"] = solar_elevation_deg
    frame["solar_zenith_deg"] = solar_zenith_deg
    frame["cos_zenith"] = np.clip(cos_zenith, 0.0, None)
    frame["air_mass_proxy"] = air_mass_proxy
    frame["clear_sky_proxy"] = clear_sky_proxy
    return frame


def add_physics_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["temperature_derate"] = np.clip(frame["temperature"] - 25.0, a_min=0.0, a_max=None)
    frame["effective_global_radiation"] = frame["global_radiation"] * frame["cos_zenith"]
    frame["effective_direct_radiation"] = frame["direct_radiation"] * frame["cos_zenith"]
    frame["irradiance_temp_ratio"] = frame["global_radiation"] / (frame["temperature_derate"] + 1.0)
    frame["humidity_wind_interaction"] = frame["humidity"] * frame["wind_speed"]
    frame["night_flag"] = (frame["solar_elevation_deg"] < 0.0).astype(int)
    return frame


def build_supervised_frame(frame: pd.DataFrame, config: ExperimentConfig) -> tuple[pd.DataFrame, list[str]]:
    work_frame = frame.copy()
    group = work_frame.groupby("plant_id", group_keys=False)

    work_frame["power_now"] = work_frame["power"]
    work_frame["target_power"] = group["power"].shift(-config.horizon)
    work_frame["forecast_time_idx"] = work_frame["time_idx"] + config.horizon
    work_frame["forecast_timestamp"] = group["timestamp"].shift(-config.horizon)

    future_columns: list[str] = []
    future_base_columns = WEATHER_COLUMNS + [
        "solar_elevation_deg",
        "cos_zenith",
        "clear_sky_proxy",
        "night_flag",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]
    for column in future_base_columns:
        future_name = f"forecast_{column}"
        work_frame[future_name] = group[column].shift(-config.horizon)
        future_columns.append(future_name)

    lag_columns: list[str] = []
    lag_sources = ["power", "global_radiation", "direct_radiation", "temperature", "wind_speed"]
    for lag in config.lags:
        for column in lag_sources:
            feature_name = f"{column}_lag_{lag}"
            work_frame[feature_name] = group[column].shift(lag)
            lag_columns.append(feature_name)

    rolling_columns: list[str] = []
    for window in config.rolling_windows:
        for column in ["power", "global_radiation", "direct_radiation"]:
            mean_name = f"{column}_rolling_mean_{window}"
            std_name = f"{column}_rolling_std_{window}"
            shifted = group[column].shift(1)
            grouped_shifted = shifted.groupby(work_frame["plant_id"])
            work_frame[mean_name] = grouped_shifted.rolling(window).mean().reset_index(level=0, drop=True)
            work_frame[std_name] = grouped_shifted.rolling(window).std().reset_index(level=0, drop=True)
            rolling_columns.extend([mean_name, std_name])

    base_columns = [
        "power_now",
        *WEATHER_COLUMNS,
        *TIME_FEATURES,
        *GEO_FEATURES,
        *PHYSICS_FEATURES,
        *future_columns,
        *lag_columns,
        *rolling_columns,
    ]
    feature_columns = base_columns.copy()

    required_columns = feature_columns + CATEGORICAL_FEATURES + ["target_power", "forecast_timestamp"]
    supervised_frame = work_frame.dropna(subset=required_columns).copy()
    return supervised_frame, feature_columns


def build_tabular_matrices(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train_features = train_frame[feature_columns + categorical_columns].copy()
    val_features = val_frame[feature_columns + categorical_columns].copy()
    test_features = test_frame[feature_columns + categorical_columns].copy()

    merged = pd.concat(
        {
            "train": train_features,
            "val": val_features,
            "test": test_features,
        },
        axis=0,
    )
    encoded = pd.get_dummies(merged, columns=categorical_columns, dtype=float)

    train_matrix = encoded.xs("train")
    val_matrix = encoded.xs("val")
    test_matrix = encoded.xs("test")
    model_columns = train_matrix.columns.tolist()
    return train_matrix, val_matrix, test_matrix, model_columns
