from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def restore_one_file(parts_dir: Path, output_dir: Path, file_record: dict, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / file_record["source_name"]
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"Target already exists: {target_path}. Use --overwrite to replace it.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".gz", dir=output_dir) as temp_handle:
        temp_compressed_path = Path(temp_handle.name)

    digest = hashlib.sha256()
    with temp_compressed_path.open("wb") as combined_handle:
        for part in file_record["parts"]:
            part_path = parts_dir / part["name"]
            if not part_path.exists():
                raise FileNotFoundError(f"Missing part file: {part_path}")
            part_bytes = part_path.read_bytes()
            if hashlib.sha256(part_bytes).hexdigest() != part["sha256"]:
                raise ValueError(f"Checksum mismatch for part: {part_path.name}")
            combined_handle.write(part_bytes)
            digest.update(part_bytes)

    if digest.hexdigest() != file_record["compressed_sha256"]:
        raise ValueError(f"Compressed file checksum mismatch for {file_record['source_name']}")

    original_digest = hashlib.sha256()
    with gzip.open(temp_compressed_path, "rb") as compressed_handle, target_path.open("wb") as output_handle:
        while chunk := compressed_handle.read(1024 * 1024):
            output_handle.write(chunk)
            original_digest.update(chunk)

    temp_compressed_path.unlink()

    if target_path.stat().st_size != file_record["original_size"]:
        raise ValueError(f"Size mismatch for restored file: {target_path.name}")
    if original_digest.hexdigest() != file_record["original_sha256"]:
        raise ValueError(f"Checksum mismatch for restored file: {target_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge split dataset parts back into CSV files.")
    parser.add_argument("--parts-dir", default="dataset_parts", help="Directory containing manifest.json and part files.")
    parser.add_argument("--output-dir", default="dataset", help="Directory where restored CSV files will be written.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in the output directory.")
    args = parser.parse_args()

    parts_dir = Path(args.parts_dir)
    manifest_path = parts_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)

    for file_record in manifest["files"]:
        restore_one_file(parts_dir, output_dir, file_record, overwrite=args.overwrite)
        print(f"Restored {file_record['source_name']}")


if __name__ == "__main__":
    main()
