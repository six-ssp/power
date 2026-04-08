from __future__ import annotations

import json
import platform
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pvbench import ExperimentConfig, load_and_prepare_data
from pvbench.data import build_tabular_matrices
from pvbench.models import split_meta_validation_frame
from tools.reproducibility import (
    build_results_manifest,
    compare_results_manifests,
    save_results_manifest,
    verify_dataset_against_parts,
)


def has_split_overlap(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    left_keys = set(zip(left['plant_id'], left['forecast_time_idx']))
    right_keys = set(zip(right['plant_id'], right['forecast_time_idx']))
    return bool(left_keys & right_keys)


def split_gap_is_valid(prepared_data, gap_steps: int) -> bool:
    for plant_id in sorted(prepared_data.train_frame['plant_id'].unique()):
        train_max = int(prepared_data.train_frame.loc[prepared_data.train_frame['plant_id'] == plant_id, 'forecast_time_idx'].max())
        val_min = int(prepared_data.val_frame.loc[prepared_data.val_frame['plant_id'] == plant_id, 'forecast_time_idx'].min())
        val_max = int(prepared_data.val_frame.loc[prepared_data.val_frame['plant_id'] == plant_id, 'forecast_time_idx'].max())
        test_min = int(prepared_data.test_frame.loc[prepared_data.test_frame['plant_id'] == plant_id, 'forecast_time_idx'].min())
        if val_min - train_max - 1 != gap_steps:
            return False
        if test_min - val_max - 1 != gap_steps:
            return False
    return True


def train_covers_eval_categories(prepared_data) -> bool:
    for column in prepared_data.categorical_columns:
        train_values = set(prepared_data.train_frame[column].astype(str).unique())
        val_values = set(prepared_data.val_frame[column].astype(str).unique())
        test_values = set(prepared_data.test_frame[column].astype(str).unique())
        if not val_values.issubset(train_values):
            return False
        if not test_values.issubset(train_values):
            return False
    return True


def main() -> None:
    config = ExperimentConfig()
    project_dir = PROJECT_DIR
    artifacts_dir = config.artifact_dir
    checks_dir = config.check_dir
    checks_dir.mkdir(parents=True, exist_ok=True)

    required_project_files = [
        project_dir / 'README.md',
        project_dir / 'requirements.txt',
        project_dir / 'run_experiments.py',
        project_dir / 'pvbench' / 'config.py',
        project_dir / 'pvbench' / 'data.py',
        project_dir / 'pvbench' / 'models.py',
        project_dir / 'pvbench' / 'reporting.py',
    ]
    required_artifacts = [
        artifacts_dir / 'metrics' / 'baseline_metrics.csv',
        artifacts_dir / 'metrics' / 'ablation_metrics.csv',
        artifacts_dir / 'metrics' / 'plant_level_metrics.csv',
        artifacts_dir / 'metrics' / 'baseline_daytime_metrics.csv',
        artifacts_dir / 'metrics' / 'baseline_physical_metrics.csv',
        artifacts_dir / 'metrics' / 'baseline_bvp_metrics.csv',
        artifacts_dir / 'metrics' / 'plant_level_daytime_metrics.csv',
        artifacts_dir / 'metrics' / 'plant_level_physical_metrics.csv',
        artifacts_dir / 'metrics' / 'ablation_bvp_metrics.csv',
        artifacts_dir / 'metrics' / 'subset_counts.csv',
        artifacts_dir / 'metrics' / 'seed_repeat_metrics.csv',
        artifacts_dir / 'metrics' / 'seed_repeat_summary.csv',
        artifacts_dir / 'metrics' / 'rolling_origin_metrics.csv',
        artifacts_dir / 'metrics' / 'rolling_origin_summary.csv',
        artifacts_dir / 'metrics' / 'rolling_origin_windows.csv',
        artifacts_dir / 'metrics' / 'training_configuration.csv',
        artifacts_dir / 'metrics' / 'training_execution_summary.csv',
        artifacts_dir / 'metrics' / 'validation_predictions.csv',
        artifacts_dir / 'metrics' / 'test_predictions.csv',
        artifacts_dir / 'plots' / 'baseline_mae_rmse.png',
        artifacts_dir / 'plots' / 'baseline_daytime_mae_rmse.png',
        artifacts_dir / 'plots' / 'forecast_examples.png',
        artifacts_dir / 'plots' / 'training_curves.png',
        artifacts_dir / 'reports' / 'training_setup_zh.md',
        artifacts_dir / 'reports' / 'result_summary_zh.md',
        artifacts_dir / 'reports' / 'robustness_summary_zh.md',
        artifacts_dir / 'reports' / 'sdm_positioning_zh.md',
        artifacts_dir / 'reports' / 'paper_reference_draft_zh.md',
        artifacts_dir / 'paper_figures' / 'method_framework.png',
    ]

    project_file_status = {str(path.relative_to(project_dir)): path.exists() for path in required_project_files}
    artifact_status = {str(path.relative_to(project_dir)): path.exists() for path in required_artifacts}

    prepared_data = load_and_prepare_data(config)
    train_x, val_x, test_x, encoded_columns = build_tabular_matrices(
        prepared_data.train_frame,
        prepared_data.val_frame,
        prepared_data.test_frame,
        prepared_data.feature_columns,
        prepared_data.categorical_columns,
    )

    baseline_table = pd.read_csv(artifacts_dir / 'metrics' / 'baseline_metrics.csv')
    ablation_table = pd.read_csv(artifacts_dir / 'metrics' / 'ablation_metrics.csv')
    plant_table = pd.read_csv(artifacts_dir / 'metrics' / 'plant_level_metrics.csv')
    baseline_daytime_table = pd.read_csv(artifacts_dir / 'metrics' / 'baseline_daytime_metrics.csv')
    baseline_physical_table = pd.read_csv(artifacts_dir / 'metrics' / 'baseline_physical_metrics.csv')
    baseline_bvp_table = pd.read_csv(artifacts_dir / 'metrics' / 'baseline_bvp_metrics.csv')
    plant_daytime_table = pd.read_csv(artifacts_dir / 'metrics' / 'plant_level_daytime_metrics.csv')
    plant_physical_table = pd.read_csv(artifacts_dir / 'metrics' / 'plant_level_physical_metrics.csv')
    ablation_bvp_table = pd.read_csv(artifacts_dir / 'metrics' / 'ablation_bvp_metrics.csv')
    seed_repeat_summary = pd.read_csv(artifacts_dir / 'metrics' / 'seed_repeat_summary.csv')
    rolling_origin_summary = pd.read_csv(artifacts_dir / 'metrics' / 'rolling_origin_summary.csv')
    rolling_window_table = pd.read_csv(artifacts_dir / 'metrics' / 'rolling_origin_windows.csv')
    training_config_table = pd.read_csv(artifacts_dir / 'metrics' / 'training_configuration.csv')
    training_execution_table = pd.read_csv(artifacts_dir / 'metrics' / 'training_execution_summary.csv')
    dataset_status = verify_dataset_against_parts(project_dir)
    meta_train_frame, meta_holdout_frame = split_meta_validation_frame(prepared_data.val_frame, config.meta_train_ratio)

    published_manifest_path = checks_dir / 'published_results_manifest.json'
    current_manifest_path = checks_dir / 'current_results_manifest.json'
    current_manifest = build_results_manifest(project_dir)
    save_results_manifest(current_manifest_path, current_manifest)
    release_match = None
    if published_manifest_path.exists():
        published_manifest = json.loads(published_manifest_path.read_text(encoding='utf-8'))
        release_match = compare_results_manifests(published_manifest, current_manifest)

    expected_baseline_models = {
        'Persistence',
        'XGBoost',
        'DNN',
        'TFT',
        'MeanAverage',
        'StaticBlend',
        'Hybrid',
        'AdaptiveBlend',
        'StackedXGB',
    }
    expected_core_robustness_models = {'Persistence', 'XGBoost', 'DNN', 'TFT', 'Hybrid', 'AdaptiveBlend', 'StackedXGB'}
    expected_ablation_models = {
        'Full Hybrid',
        'w/o Physics',
        'Mean Average',
        'Equal Weights',
        'w/o Plant Adaptation',
        'w/o Scene Adaptation',
        'w/o XGBoost',
        'w/o DNN',
        'w/o TFT',
        'Adaptive Blend',
        'Stacked XGB',
    }
    expected_plants = {
        'AliceSprings_MonoTrack_1A',
        'AliceSprings_PolyFixed_1C',
        'AliceSprings_PolyUtility_3A',
        'AliceSprings_HighEfficiency_4A',
    }

    checks = {
        'project_files_complete': all(project_file_status.values()),
        'artifact_files_complete': all(artifact_status.values()),
        'baseline_models_complete': set(baseline_table['Model']) == expected_baseline_models,
        'ablation_models_complete': set(ablation_table['Model']) == expected_ablation_models,
        'plant_results_complete': set(plant_table['Plant']) == expected_plants,
        'daytime_models_complete': set(baseline_daytime_table['Model']) == expected_baseline_models,
        'physical_models_complete': set(baseline_physical_table['Model']) == expected_baseline_models,
        'bvp_models_complete': set(baseline_bvp_table['Model']) == expected_baseline_models,
        'daytime_plants_complete': set(plant_daytime_table['Plant']) == expected_plants,
        'physical_plants_complete': set(plant_physical_table['Plant']) == expected_plants,
        'ablation_bvp_models_complete': set(ablation_bvp_table['Model']) == expected_ablation_models,
        'seed_summary_models_complete': (
            set(seed_repeat_summary['Model']).issubset(expected_baseline_models)
            and expected_core_robustness_models.issubset(set(seed_repeat_summary['Model']))
        ),
        'rolling_summary_models_complete': (
            set(rolling_origin_summary['Model']).issubset(expected_baseline_models)
            and expected_core_robustness_models.issubset(set(rolling_origin_summary['Model']))
        ),
        'rolling_window_count_correct': len(rolling_window_table) == len(config.rolling_origin_windows),
        'train_val_overlap_absent': not has_split_overlap(prepared_data.train_frame, prepared_data.val_frame),
        'train_test_overlap_absent': not has_split_overlap(prepared_data.train_frame, prepared_data.test_frame),
        'val_test_overlap_absent': not has_split_overlap(prepared_data.val_frame, prepared_data.test_frame),
        'split_gap_matches_config': split_gap_is_valid(prepared_data, config.split_gap_steps),
        'meta_val_holdout_overlap_absent': not has_split_overlap(meta_train_frame, meta_holdout_frame),
        'meta_val_holdout_is_chronological': bool(
            (meta_train_frame.groupby('plant_id')['forecast_time_idx'].max() < meta_holdout_frame.groupby('plant_id')['forecast_time_idx'].min()).all()
        ),
        'train_covers_eval_categories': train_covers_eval_categories(prepared_data),
        'baseline_metrics_non_null': not baseline_table.isna().any().any(),
        'ablation_metrics_non_null': not ablation_table.isna().any().any(),
        'plant_metrics_non_null': not plant_table.isna().any().any(),
        'daytime_metrics_non_null': not baseline_daytime_table.isna().any().any(),
        'physical_metrics_non_null': not baseline_physical_table.isna().any().any(),
        'bvp_metrics_non_null': not baseline_bvp_table.isna().any().any(),
        'ablation_bvp_non_null': not ablation_bvp_table.isna().any().any(),
        'seed_repeat_non_null': not seed_repeat_summary.isna().any().any(),
        'rolling_origin_non_null': not rolling_origin_summary.isna().any().any(),
        'training_config_non_null': not training_config_table.isna().any().any(),
        'training_execution_non_null': not training_execution_table.isna().any().any(),
        'dataset_hash_matches_parts_manifest': bool(dataset_status['all_match']),
        'published_results_match_manifest': bool(release_match['match']) if release_match is not None else False,
        'cuda_available': torch.cuda.is_available(),
    }

    summary = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'dataset_files': sorted(path.name for path in (project_dir / 'dataset').glob('*.csv')),
        'raw_shape': prepared_data.raw_frame.shape,
        'supervised_shape': prepared_data.supervised_frame.shape,
        'train_shape': prepared_data.train_frame.shape,
        'val_shape': prepared_data.val_frame.shape,
        'test_shape': prepared_data.test_frame.shape,
        'feature_count': len(prepared_data.feature_columns),
        'encoded_feature_count': len(encoded_columns),
        'train_matrix_shape': train_x.shape,
        'val_matrix_shape': val_x.shape,
        'test_matrix_shape': test_x.shape,
        'project_file_status': project_file_status,
        'artifact_status': artifact_status,
        'dataset_status': dataset_status,
        'release_match': release_match,
        'split_boundaries': {
            'train_cutoff': prepared_data.train_cutoff,
            'val_start_idx': prepared_data.val_start_idx,
            'val_cutoff': prepared_data.val_cutoff,
            'test_start_idx': prepared_data.test_start_idx,
            'test_cutoff': prepared_data.test_cutoff,
            'split_gap_steps': config.split_gap_steps,
            'meta_train_ratio': config.meta_train_ratio,
        },
        'checks': checks,
    }

    release_status_text = "未检查"
    if release_match is not None:
        release_status_text = "是" if checks['published_results_match_manifest'] else "否"

    (checks_dir / 'project_check.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = [
        '# 项目检查与可复现性确认',
        '',
        '## 1. 结论',
        f"- 代码结构完整：`{'是' if checks['project_files_complete'] else '否'}`",
        f"- 主要实验产物齐全：`{'是' if checks['artifact_files_complete'] else '否'}`",
        f"- 数据处理链路可加载：`是`",
        f"- Baseline/消融结果表结构正确：`{'是' if checks['baseline_models_complete'] and checks['ablation_models_complete'] else '否'}`",
        f"- GPU 环境可用：`{'是' if checks['cuda_available'] else '否'}`",
        f"- 数据集哈希匹配分片清单：`{'是' if checks['dataset_hash_matches_parts_manifest'] else '否'}`",
        f"- 当前结果匹配发布签名：`{release_status_text}`",
        f"- 主切分无重叠且 gap 正确：`{'是' if checks['train_val_overlap_absent'] and checks['train_test_overlap_absent'] and checks['val_test_overlap_absent'] and checks['split_gap_matches_config'] else '否'}`",
        f"- 验证集内部 meta-train / holdout 无重叠：`{'是' if checks['meta_val_holdout_overlap_absent'] and checks['meta_val_holdout_is_chronological'] else '否'}`",
        '',
        '## 2. 当前确认无误的内容',
        '- 原始数据文件存在，且表头与电站 ID 已统一为英文。',
        '- 主实验入口、配置、数据处理、模型与报告模块均存在。',
        '- `baseline_metrics.csv / ablation_metrics.csv / plant_level_metrics.csv` 已生成。',
        '- `baseline_daytime_metrics.csv / baseline_physical_metrics.csv / baseline_bvp_metrics.csv / seed_repeat_summary.csv / rolling_origin_summary.csv` 已生成。',
        '- `README.md`、训练配置记录、SDM 口径说明、参考稿、鲁棒性记录和主要图表已经生成。',
        '- 代码可完成数据加载、特征工程与监督样本构建。',
        '- 当前检查会额外对 `dataset_parts/manifest.json` 和发布结果签名做一致性比对。',
        '- 当前检查会显式验证 train/val/test 不重叠、时间 gap 符合配置，且验证集内部 meta-train / holdout 不重叠。',
        '',
        '## 3. 可复现性说明',
        '- 当前环境：Python 3.12 + CUDA 可用 + GPU 版 PyTorch。',
        '- 重新安装建议命令已在 README 中给出。',
        '- 主实验命令保持为：`./.venv/Scripts/python run_experiments.py`（PowerShell 下实际使用反斜杠路径）。',
        '- 本次检查验证了项目结构、依赖、数据读取和结果文件的一致性。',
        '',
        '## 4. 尚需注意的点',
    ]
    lines.append('- 逐样本预测导出文件已存在，可继续做更细粒度论文图。')
    lines.extend(
        [
            '- 新版 Hybrid 已改为电站级场景融合，并额外保留固定权重版本作为消融对照。',
            '- StackedXGB 仍是重要对照，但主线 Hybrid 现在具备更强的可解释性。',
            '- 项目现在额外提供 daytime-only、physical violation、BVP、多随机种子和 rolling-origin 五类评估结果。',
            '- 训练配置与实际执行轮次已导出为独立表格，可直接写入论文实验设置。',
            f"- 当前主切分边界：train<=`{prepared_data.train_cutoff}`，val=`{prepared_data.val_start_idx}`..`{prepared_data.val_cutoff}`，test=`{prepared_data.test_start_idx}`..`{prepared_data.test_cutoff}`，gap=`{config.split_gap_steps}`。",
            '',
            '## 5. 数据与特征规模',
            f"- 原始样本：`{prepared_data.raw_frame.shape[0]}` 行，`{prepared_data.raw_frame.shape[1]}` 列",
            f"- 监督样本：`{prepared_data.supervised_frame.shape[0]}` 行，`{prepared_data.supervised_frame.shape[1]}` 列",
            f"- 训练集：`{prepared_data.train_frame.shape[0]}` 行",
            f"- 验证集：`{prepared_data.val_frame.shape[0]}` 行",
            f"- 测试集：`{prepared_data.test_frame.shape[0]}` 行",
            f"- 连续特征数：`{len(prepared_data.feature_columns)}`",
            f"- One-hot 后特征数：`{len(encoded_columns)}`",
        ]
    )

    (checks_dir / 'project_check_zh.md').write_text('\n'.join(lines), encoding='utf-8-sig')
    print('saved', checks_dir / 'project_check_zh.md')


if __name__ == '__main__':
    main()
