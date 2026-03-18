from dataclasses import dataclass, field
from pathlib import Path

COLUMN_MAP = {
    "时间": "timestamp",
    "电站ID": "plant_id",
    "辐照度": "irradiance",
    "温度": "temperature",
    "湿度": "humidity",
    "风速": "wind_speed",
    "直辐射": "direct_radiation",
    "总辐射": "global_radiation",
    "实际功率": "power",
}

WEATHER_COLUMNS = [
    "irradiance",
    "temperature",
    "humidity",
    "wind_speed",
    "direct_radiation",
    "global_radiation",
]

PLANT_METADATA = {
    "Plant_1A_单晶双轴": {
        "plant_name_en": "plant_1a_dual_axis",
        "module_type": "mono",
        "mounting_type": "dual_axis_tracking",
    },
    "Plant_1C_多晶固定": {
        "plant_name_en": "plant_1c_fixed",
        "module_type": "poly",
        "mounting_type": "fixed_tilt",
    },
    "Plant_3A_多晶大型": {
        "plant_name_en": "plant_3a_utility",
        "module_type": "poly",
        "mounting_type": "utility_scale",
    },
    "Plant_4A_高效对比": {
        "plant_name_en": "plant_4a_high_efficiency",
        "module_type": "advanced",
        "mounting_type": "comparison_array",
    },
}

SITE_METADATA = {
    "site_name": "Alice Springs, Australia",
    "latitude": -23.6980,
    "longitude": 133.8807,
    "timezone_offset_hours": 9.5,
}


@dataclass
class ExperimentConfig:
    random_seed: int = 42
    horizon: int = 1
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24, 288)
    rolling_windows: tuple[int, ...] = (3, 12, 24)
    dnn_hidden_dims: tuple[int, ...] = (256, 128, 64)
    dnn_dropout: float = 0.2
    dnn_batch_size: int = 4096
    dnn_epochs: int = 20
    dnn_patience: int = 4
    dnn_learning_rate: float = 1e-3
    tft_encoder_length: int = 72
    tft_batch_size: int = 512
    tft_max_epochs: int = 2
    tft_patience: int = 1
    tft_hidden_size: int = 24
    tft_attention_heads: int = 4
    tft_hidden_continuous_size: int = 16
    min_blend_weight: float = 0.1
    blend_step: float = 0.05
    night_radiation_threshold: float = 20.0
    night_alpha_grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    meta_train_ratio: float = 0.8
    adaptive_hidden_dims: tuple[int, ...] = (128, 64)
    adaptive_dropout: float = 0.1
    adaptive_batch_size: int = 8192
    adaptive_epochs: int = 30
    adaptive_patience: int = 5
    adaptive_learning_rate: float = 5e-4
    num_workers: int = 4
    xgb_predict_on_cpu: bool = True
    stack_xgb_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 600,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "min_child_weight": 1.0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": 8,
        }
    )
    xgb_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "min_child_weight": 2.0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": 8,
        }
    )
    project_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    artifact_dir: Path = field(init=False)
    metric_dir: Path = field(init=False)
    plot_dir: Path = field(init=False)
    paper_dir: Path = field(init=False)
    report_dir: Path = field(init=False)
    check_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = self.project_dir / "dataset"
        self.artifact_dir = self.project_dir / "artifacts"
        self.metric_dir = self.artifact_dir / "metrics"
        self.plot_dir = self.artifact_dir / "plots"
        self.paper_dir = self.artifact_dir / "paper_figures"
        self.report_dir = self.artifact_dir / "reports"
        self.check_dir = self.artifact_dir / "checks"
        self.model_dir = self.artifact_dir / "models"
        self.log_dir = self.artifact_dir / "logs"
        for path in [
            self.artifact_dir,
            self.metric_dir,
            self.plot_dir,
            self.paper_dir,
            self.report_dir,
            self.check_dir,
            self.model_dir,
            self.log_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
