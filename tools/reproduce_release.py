from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pvbench import ExperimentConfig
from tools.generate_paper_figures import main as generate_paper_figures_main
from tools.merge_dataset_parts import restore_one_file
from tools.reproducibility import (
    build_results_manifest,
    clean_release_outputs,
    compare_results_manifests,
    save_results_manifest,
    verify_dataset_against_parts,
)
from tools.staged_runner import run_main_stage, run_rolling_window_stage
from tools.verify_project import main as verify_project_main
from tools.write_reports import main as write_reports_main


def restore_dataset_if_needed(project_dir: Path, force_restore: bool) -> None:
    dataset_status = verify_dataset_against_parts(project_dir)
    if dataset_status["all_match"] and not force_restore:
        return

    manifest_path = project_dir / "dataset_parts" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset part manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for file_record in manifest["files"]:
        restore_one_file(
            parts_dir=project_dir / "dataset_parts",
            output_dir=project_dir / "dataset",
            file_record=file_record,
            overwrite=True,
        )
        print(f"restored {file_record['source_name']}", flush=True)


def write_reproduction_report(check_dir: Path, comparison: dict[str, object], dataset_status: dict[str, object]) -> None:
    lines = [
        "# 发布结果复现检查",
        "",
        f"- 数据集哈希匹配：`{'是' if dataset_status['all_match'] else '否'}`",
        f"- 发布结果签名匹配：`{'是' if comparison['match'] else '否'}`",
        "",
        "## 结果签名差异",
    ]
    if comparison["match"]:
        lines.append("- 当前输出与发布结果签名一致。")
    else:
        if comparison["missing"]:
            lines.append(f"- 缺失文件：`{', '.join(comparison['missing'])}`")
        if comparison["unexpected"]:
            lines.append(f"- 额外文件：`{', '.join(comparison['unexpected'])}`")
        if comparison["mismatched"]:
            lines.append("- 哈希不一致文件：")
            for row in comparison["mismatched"]:
                lines.append(f"  - `{row['path']}`")

    report_path = check_dir / "reproduction_check_zh.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"saved {report_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cleanly reproduce the published release artifacts and compare hashes.")
    parser.add_argument("--compare-only", action="store_true", help="Skip rerunning experiments and only compare current outputs.")
    parser.add_argument("--force-restore-dataset", action="store_true", help="Always restore dataset CSVs from split parts.")
    args = parser.parse_args()

    config = ExperimentConfig()
    published_manifest_path = config.check_dir / "published_results_manifest.json"
    current_manifest_path = config.check_dir / "current_results_manifest.json"

    restore_dataset_if_needed(PROJECT_DIR, force_restore=args.force_restore_dataset)

    if not args.compare_only:
        clean_release_outputs(
            PROJECT_DIR,
            keep_paths={
                "artifacts/checks/published_results_manifest.json",
            },
        )
        run_main_stage(config)
        for index in range(1, len(config.rolling_origin_windows) + 1):
            run_rolling_window_stage(config, index)
        write_reports_main()
        generate_paper_figures_main()
        verify_project_main()

    current_manifest = build_results_manifest(PROJECT_DIR)
    save_results_manifest(current_manifest_path, current_manifest)

    if not published_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing published results manifest: {published_manifest_path}. "
            "Create it once from the current release outputs before running compare mode."
        )

    published_manifest = json.loads(published_manifest_path.read_text(encoding="utf-8"))
    comparison = compare_results_manifests(published_manifest, current_manifest)
    dataset_status = verify_dataset_against_parts(PROJECT_DIR)
    write_reproduction_report(config.check_dir, comparison, dataset_status)

    if not comparison["match"]:
        raise SystemExit("Current outputs do not match the published release manifest.")

    print("Release reproduction check passed.", flush=True)


if __name__ == "__main__":
    main()
