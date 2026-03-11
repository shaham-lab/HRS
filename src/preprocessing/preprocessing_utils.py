"""Shared utilities for the preprocessing pipeline."""

import hashlib
import json
import os
from typing import Any, cast

import pandas as pd


def _check_required_keys(config: dict, required_keys: list[str]) -> None:
    """Raise KeyError if any required config key is missing."""
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}'")


def _output_is_valid(path: str, expected_rows: int, embedding_col: str) -> bool:
    """Return True if a completed embedding parquet exists at `path` and is usable.

    Checks:
    - File exists
    - Can be read as a parquet
    - Has the expected number of rows (matches the input feature file)
    - Contains the expected embedding column
    - No null values in the embedding column (a partial write would leave nulls)
    """
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return False
    if len(df) != expected_rows:
        return False
    if embedding_col not in df.columns:
        return False
    if df[embedding_col].isnull().any():
        return False
    return True


def _gz_or_csv(base_dir: str, subdir: str, table: str) -> str:
    """Return the .csv.gz path if it exists, otherwise the .csv path.

    Does not require the file to exist; callers that need the actual
    resolved path for optional sources should check ``os.path.exists``
    on the returned value themselves.
    """
    gz = os.path.join(base_dir, subdir, f"{table}.csv.gz")
    if os.path.exists(gz):
        return gz
    return os.path.join(base_dir, subdir, f"{table}.csv")


def _load_csv(path_gz: str, path_csv: str, **kwargs: Any) -> pd.DataFrame:
    if os.path.exists(path_gz):
        return cast(pd.DataFrame, pd.read_csv(path_gz, **kwargs))
    if os.path.exists(path_csv):
        return cast(pd.DataFrame, pd.read_csv(path_csv, **kwargs))
    raise FileNotFoundError(
        f"Neither {path_gz} nor {path_csv} found."
    )


# ---------------------------------------------------------------------------
# Source-file hash utilities
# ---------------------------------------------------------------------------

def _file_hash(path: str, chunk_size: int = 1 << 20) -> str:
    """Return the MD5 hex digest of a file. Reads in chunks to handle large files.

    MD5 is used for speed on large MIMIC-IV source files (not for security).
    To use SHA-256 instead, replace ``hashlib.md5()`` with ``hashlib.sha256()``.
    """
    h = hashlib.md5()  # noqa: S324
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _load_hash_registry(registry_path: str) -> dict:
    """Load the hash registry JSON from disk. Returns empty dict if not found."""
    if not os.path.exists(registry_path):
        return {}
    with open(registry_path, "r", encoding="utf-8") as f:
        return cast(dict, json.load(f))


def _save_hash_registry(registry_path: str, registry: dict) -> None:
    """Persist the hash registry to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(registry_path)), exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def _sources_unchanged(
    module_name: str,
    source_paths: list,
    output_paths: list,
    registry_path: str,
    logger: Any,
) -> bool:
    """Return True iff all outputs exist, all sources exist, and stored hashes match.

    If True, the caller can safely skip re-running the module.
    Logs the reason for any False result.
    """
    # Check all outputs exist
    for p in output_paths:
        if not os.path.exists(p):
            logger.info("[%s] Output not found (%s) — will run.", module_name, os.path.basename(p))
            return False

    # Check all sources exist
    for p in source_paths:
        if not os.path.exists(p):
            logger.warning("[%s] Source missing: %s — will rerun.", module_name, p)
            return False

    # Load registry and compare hashes
    registry = _load_hash_registry(registry_path)
    stored = registry.get(module_name, {})

    for p in source_paths:
        current_hash = _file_hash(p)
        stored_hash = stored.get(p)
        if stored_hash != current_hash:
            logger.info(
                "[%s] Source file changed (%s) — will rerun.", module_name, os.path.basename(p)
            )
            return False

    output_names = [os.path.basename(p) for p in output_paths]
    logger.info("[%s] Skipping — outputs up to date: %s", module_name, ", ".join(output_names))
    return True


def _record_hashes(
    module_name: str,
    source_paths: list,
    registry_path: str,
) -> None:
    """After a successful module run, record current hashes of all source files."""
    registry = _load_hash_registry(registry_path)
    registry[module_name] = {
        p: _file_hash(p) for p in source_paths if os.path.exists(p)
    }
    _save_hash_registry(registry_path, registry)
