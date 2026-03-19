# Alice Springs PV Power Forecasting

This repository provides a reproducible experimental pipeline for `5-minute-ahead` photovoltaic power forecasting on four heterogeneous PV assets at the Alice Springs site. It standardizes data ingestion, feature engineering, model training, evaluation, and figure/report generation, with `Hybrid` as the main interpretable fusion method and `StackedXGB` as a strong empirical upper bound on the fixed split.

<p align="center">
  <img src="artifacts/paper_figures/method_framework.png" width="88%" alt="Method framework">
</p>
<p align="center">
  Forecasting pipeline: standardized features, heterogeneous base learners, scene-aware fusion, and physics-guided adjustment.
</p>

## Highlights

- Unified comparison of `Persistence`, `XGBoost`, `DNN`, `TFT`, `Hybrid`, `AdaptiveBlend`, and `StackedXGB`
- Standardized English dataset schema and automated preprocessing pipeline
- Time-ordered evaluation with split gap, `daytime-only`, multi-seed, and `rolling-origin` protocols
- Automatic export of metrics, predictions, plots, paper figures, and Chinese experiment records
- Codebase organized for repeated paper experiments rather than one-off notebook runs

## Project At A Glance

| Item | Value |
| --- | --- |
| Site | Alice Springs, Australia |
| Data files | `dataset/data_1A.csv`, `dataset/data_1C.csv`, `dataset/data_3A.csv`, `dataset/data_4A.csv` |
| Assets | 4 heterogeneous PV assets |
| Time span | `2013-04-23 08:35:00` to `2016-10-21 15:05:00` |
| Resolution | `5 min` |
| Task | Next-step power regression under known next-step weather |
| Raw samples | `1,242,372` |
| Supervised samples | `1,241,216` |
| Continuous features | `96` |
| Encoded features | `111` |
| Main metrics | `MAE`, `RMSE`, `R2` |

## Data Schema

All dataset files in this repository use English column names:

| Column | Description |
| --- | --- |
| `timestamp` | observation timestamp |
| `plant_id` | asset identifier |
| `irradiance` | irradiance-related sensor input |
| `temperature` | ambient/module temperature signal |
| `humidity` | humidity signal |
| `wind_speed` | wind speed signal |
| `direct_radiation` | direct radiation |
| `global_radiation` | global radiation |
| `power` | target PV output |

The current task formulation assumes the next-step weather is available and performs conditional regression for the next-step power output. This is an explicit task definition in the repository, not an accidental information leak.

## Evaluation Protocol

| Component | Setting |
| --- | --- |
| Main split | per-asset chronological `80 / 10 / 10` |
| Split gap | `72` steps between train-val and val-test, about `6 hours` |
| Daytime-only | filter with `forecast_night_flag == 0`, ratio about `47.93%` |
| Multi-seed | `42`, `52`, `62` |
| Rolling-origin windows | `60/70/80`, `70/80/90`, `80/90/100` |
| Stored metrics | `MAE`, `RMSE`, `MAPE`, `sMAPE`, `R2`, `Bias`, `Samples` |

`MAPE` is exported for completeness, but `MAE`, `RMSE`, and `R2` are the primary metrics because nighttime power is often close to zero.

## Repository Layout

```text
.
|-- dataset/                 # normalized CSV files
|-- pvbench/
|   |-- config.py            # experiment configuration
|   |-- data.py              # loading, normalization, feature engineering, splits
|   |-- models.py            # baselines, Hybrid, AdaptiveBlend, StackedXGB
|   `-- reporting.py         # metrics, plots, report utilities
|-- tools/
|   |-- generate_paper_figures.py
|   `-- verify_project.py
|-- artifacts/
|   |-- metrics/
|   |-- plots/
|   |-- paper_figures/
|   |-- reports/
|   `-- checks/
|-- run_experiments.py       # end-to-end experiment entry
`-- requirements.txt
```

## Quick Start

### 1. Create the environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

### 2. Run the full experiment

```powershell
.\.venv\Scripts\python run_experiments.py
```

### 3. Verify generated artifacts

```powershell
.\.venv\Scripts\python tools\verify_project.py
```

### 4. Generate paper figures

```powershell
.\.venv\Scripts\python tools\generate_paper_figures.py
```

## Main Results

### Fixed-Split Results

| Model | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| Persistence | 0.049670 | 0.175950 | 0.980560 |
| XGBoost | 0.042605 | 0.087974 | 0.995140 |
| DNN | 0.035966 | 0.068893 | 0.997020 |
| TFT | 0.024010 | 0.062329 | 0.997560 |
| Hybrid | 0.018909 | 0.060678 | 0.997688 |
| AdaptiveBlend | 0.019469 | 0.062321 | 0.997561 |
| StackedXGB | 0.018220 | 0.059508 | 0.997776 |

### Robustness Summary

| Setting | TFT | Hybrid | StackedXGB |
| --- | ---: | ---: | ---: |
| Fixed split MAE | 0.024010 | 0.018909 | 0.018220 |
| Daytime-only MAE | 0.041529 | 0.039110 | 0.037548 |
| Multi-seed MAE | 0.022546 +/- 0.004298 | 0.018388 +/- 0.001330 | 0.038971 +/- 0.019228 |
| Rolling-origin MAE | 0.036315 +/- 0.012042 | 0.028419 +/- 0.008966 | 0.032228 +/- 0.013279 |

Interpretation:

- `StackedXGB` is the best model on the fixed split and on the daytime-only fixed split.
- `Hybrid` is the most stable primary method across the stricter robustness protocols.
- `Hybrid` consistently improves over `TFT` under fixed split, daytime-only, multi-seed, and `rolling-origin` evaluation.

### Hybrid Ablation

| Variant | MAE | RMSE |
| --- | ---: | ---: |
| Full Hybrid | 0.018909 | 0.060678 |
| w/o Physics | 0.018941 | 0.060678 |
| w/o Plant Adaptation | 0.020038 | 0.061104 |
| w/o Scene Adaptation | 0.021717 | 0.065940 |

The ablation indicates that scene adaptation is the main source of improvement, while the physics correction is a smaller but still measurable refinement.

## Figures

<table>
  <tr>
    <td><img src="artifacts/paper_figures/baseline_overview.png" width="100%" alt="Baseline overview"></td>
    <td><img src="artifacts/paper_figures/daytime_baseline_overview.png" width="100%" alt="Daytime results"></td>
  </tr>
  <tr>
    <td align="center">Fixed-split results</td>
    <td align="center">Daytime-only results</td>
  </tr>
  <tr>
    <td><img src="artifacts/paper_figures/seed_stability.png" width="100%" alt="Seed stability"></td>
    <td><img src="artifacts/paper_figures/rolling_origin_overview.png" width="100%" alt="Rolling-origin overview"></td>
  </tr>
  <tr>
    <td align="center">Multi-seed stability</td>
    <td align="center">Rolling-origin evaluation</td>
  </tr>
</table>

## Generated Artifacts

| Path | Description |
| --- | --- |
| `artifacts/metrics/` | result tables, ablations, daytime metrics, multi-seed summaries, rolling-origin summaries, predictions |
| `artifacts/plots/` | general experiment plots |
| `artifacts/paper_figures/` | paper-ready figures |
| `artifacts/reports/` | Chinese training log, result summary, method story, robustness summary |
| `artifacts/checks/` | project verification outputs |

Useful files for paper writing:

- `artifacts/reports/result_summary_zh.md`
- `artifacts/reports/robustness_summary_zh.md`
- `artifacts/metrics/baseline_metrics.csv`
- `artifacts/metrics/baseline_daytime_metrics.csv`
- `artifacts/metrics/seed_repeat_summary.csv`
- `artifacts/metrics/rolling_origin_summary.csv`

## Reproducibility Notes

- Verified environment: `Python 3.12`, CUDA available, GPU PyTorch runtime
- Main entry: `run_experiments.py`
- Project verification: `tools/verify_project.py`
- Figure generation: `tools/generate_paper_figures.py`
- Current README reflects the latest generated artifacts in `artifacts/`
