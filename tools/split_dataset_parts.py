from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compress_file(source: Path, target: Path, compresslevel: int) -> None:
    with source.open("rb") as source_handle, gzip.open(target, "wb", compresslevel=compresslevel) as target_handle:
        while chunk := source_handle.read(1024 * 1024):
            target_handle.write(chunk)


def split_binary_file(source: Path, output_dir: Path, chunk_size: int) -> list[dict[str, int | str]]:
    parts: list[dict[str, int | str]] = []
    with source.open("rb") as handle:
        index = 1
        while chunk := handle.read(chunk_size):
            part_name = f"{source.name}.part{index:03d}"
            part_path = output_dir / part_name
            part_path.write_bytes(chunk)
            parts.append(
                {
                    "name": part_name,
                    "size": len(chunk),
                    "sha256": sha256_bytes(chunk),
                }
            )
            index += 1
    return parts


def build_manifest(source_dir: Path, output_dir: Path, chunk_size_mb: int, compresslevel: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in output_dir.glob("*.part*"):
        stale_file.unlink()
    compressed_temp_dir = output_dir / ".tmp"
    compressed_temp_dir.mkdir(parents=True, exist_ok=True)

    files = []
    chunk_size = chunk_size_mb * 1024 * 1024

    for csv_path in sorted(source_dir.glob("*.csv")):
        compressed_path = compressed_temp_dir / f"{csv_path.name}.gz"
        compress_file(csv_path, compressed_path, compresslevel=compresslevel)
        parts = split_binary_file(compressed_path, output_dir, chunk_size)
        files.append(
            {
                "source_name": csv_path.name,
                "original_size": csv_path.stat().st_size,
                "original_sha256": sha256_file(csv_path),
                "compressed_name": compressed_path.name,
                "compressed_size": compressed_path.stat().st_size,
                "compressed_sha256": sha256_file(compressed_path),
                "parts": parts,
            }
        )
        compressed_path.unlink()

    compressed_temp_dir.rmdir()
    return {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir).replace("\\", "/"),
        "compression": {"type": "gzip", "compresslevel": compresslevel},
        "chunk_size_bytes": chunk_size,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress and split dataset CSV files into GitHub-friendly parts.")
    parser.add_argument("--source-dir", default="dataset", help="Directory containing the raw CSV files.")
    parser.add_argument("--output-dir", default="dataset_parts", help="Directory where split files will be written.")
    parser.add_argument("--chunk-size-mb", type=int, default=8, help="Chunk size in MiB for each part.")
    parser.add_argument("--compresslevel", type=int, default=6, help="gzip compresslevel.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    manifest = build_manifest(
        source_dir=source_dir,
        output_dir=output_dir,
        chunk_size_mb=args.chunk_size_mb,
        compresslevel=args.compresslevel,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
