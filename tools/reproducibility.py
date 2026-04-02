from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def release_signature_paths(project_dir: Path) -> list[Path]:
    return [
        project_dir / "artifacts" / "metrics" / "baseline_metrics.csv",
        project_dir / "artifacts" / "metrics" / "ablation_metrics.csv",
        project_dir / "artifacts" / "metrics" / "plant_level_metrics.csv",
        project_dir / "artifacts" / "metrics" / "baseline_daytime_metrics.csv",
        project_dir / "artifacts" / "metrics" / "baseline_physical_metrics.csv",
        project_dir / "artifacts" / "metrics" / "plant_level_daytime_metrics.csv",
        project_dir / "artifacts" / "metrics" / "plant_level_physical_metrics.csv",
        project_dir / "artifacts" / "metrics" / "subset_counts.csv",
        project_dir / "artifacts" / "metrics" / "seed_repeat_metrics.csv",
        project_dir / "artifacts" / "metrics" / "seed_repeat_summary.csv",
        project_dir / "artifacts" / "metrics" / "rolling_origin_metrics.csv",
        project_dir / "artifacts" / "metrics" / "rolling_origin_summary.csv",
        project_dir / "artifacts" / "metrics" / "rolling_origin_windows.csv",
        project_dir / "artifacts" / "metrics" / "training_configuration.csv",
        project_dir / "artifacts" / "metrics" / "training_execution_summary.csv",
        project_dir / "artifacts" / "reports" / "training_setup_zh.md",
        project_dir / "artifacts" / "reports" / "result_summary_zh.md",
        project_dir / "artifacts" / "reports" / "robustness_summary_zh.md",
        project_dir / "artifacts" / "reports" / "sdm_positioning_zh.md",
        project_dir / "artifacts" / "reports" / "paper_reference_draft_zh.md",
    ]


def build_results_manifest(project_dir: Path, paths: list[Path] | None = None) -> dict:
    manifest_paths = release_signature_paths(project_dir) if paths is None else paths
    files: list[dict[str, str | int]] = []
    for path in manifest_paths:
        files.append(
            {
                "path": str(path.relative_to(project_dir)).replace("\\", "/"),
                "exists": path.exists(),
                "size": int(path.stat().st_size) if path.exists() else 0,
                "sha256": sha256_file(path) if path.exists() else "",
            }
        )
    return {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }


def save_results_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def compare_results_manifests(published_manifest: dict, current_manifest: dict) -> dict[str, object]:
    published_map = {entry["path"]: entry for entry in published_manifest["files"]}
    current_map = {entry["path"]: entry for entry in current_manifest["files"]}

    missing = sorted(path for path in published_map if path not in current_map)
    unexpected = sorted(path for path in current_map if path not in published_map)
    mismatched: list[dict[str, str]] = []

    for path in sorted(set(published_map) & set(current_map)):
        published_entry = published_map[path]
        current_entry = current_map[path]
        if (
            published_entry["exists"] != current_entry["exists"]
            or published_entry["size"] != current_entry["size"]
            or published_entry["sha256"] != current_entry["sha256"]
        ):
            mismatched.append(
                {
                    "path": path,
                    "published_sha256": str(published_entry["sha256"]),
                    "current_sha256": str(current_entry["sha256"]),
                }
            )

    return {
        "match": not missing and not unexpected and not mismatched,
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
    }


def verify_dataset_against_parts(project_dir: Path) -> dict[str, object]:
    manifest_path = project_dir / "dataset_parts" / "manifest.json"
    dataset_dir = project_dir / "dataset"
    if not manifest_path.exists():
        return {"manifest_exists": False, "all_match": False, "files": []}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_rows: list[dict[str, object]] = []
    all_match = True
    for record in manifest["files"]:
        path = dataset_dir / record["source_name"]
        exists = path.exists()
        size_ok = exists and path.stat().st_size == record["original_size"]
        sha_ok = exists and sha256_file(path) == record["original_sha256"]
        file_rows.append(
            {
                "path": str(path.relative_to(project_dir)).replace("\\", "/"),
                "exists": exists,
                "size_ok": size_ok,
                "sha_ok": sha_ok,
            }
        )
        all_match = all_match and exists and size_ok and sha_ok
    return {
        "manifest_exists": True,
        "all_match": all_match,
        "files": file_rows,
    }


def clean_release_outputs(project_dir: Path, keep_paths: set[str] | None = None) -> None:
    keep_paths = keep_paths or set()
    targets = [
        project_dir / "artifacts" / "metrics",
        project_dir / "artifacts" / "plots",
        project_dir / "artifacts" / "paper_figures",
        project_dir / "artifacts" / "reports",
        project_dir / "artifacts" / "checks",
        project_dir / "artifacts" / "logs",
    ]

    for target in targets:
        if not target.exists():
            continue
        if target.name == "logs":
            shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)
            continue

        for item in target.iterdir():
            relative = str(item.relative_to(project_dir)).replace("\\", "/")
            if relative in keep_paths:
                continue
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink()
