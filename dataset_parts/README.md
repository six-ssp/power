# Dataset Parts

The raw CSV files are kept out of Git history, while GitHub-friendly compressed split parts are stored in this directory.

## Restore the dataset

```powershell
.\.venv\Scripts\python tools\merge_dataset_parts.py --overwrite
```

By default, the restored files will be written into `dataset/`.

## Rebuild the split files from local raw CSVs

```powershell
.\.venv\Scripts\python tools\split_dataset_parts.py
```

This command reads `dataset/*.csv`, compresses them with gzip, and writes split parts plus `manifest.json` into `dataset_parts/`.
