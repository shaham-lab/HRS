"""Shared utilities for the preprocessing pipeline."""

import hashlib
import json
import os
from typing import Any, cast

import pandas as pd


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
            logger.info("[%s] Output missing: %s — will rerun.", module_name, p)
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
                "[%s] Source changed (or first run): %s — will rerun.", module_name, p
            )
            return False

    logger.info("[%s] All sources unchanged and outputs exist — skipping.", module_name)
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
