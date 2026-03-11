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


def _link_hadm_for_row(
    row: pd.Series,
    admissions_df: pd.DataFrame,
    tolerance: Any,
) -> float | None:
    """Resolve hadm_id for a single null-hadm_id row via time-window linkage.

    Looks for admissions for the same subject_id where charttime falls within
    [admittime - tolerance, dischtime + tolerance].

    Returns the resolved hadm_id as float, or None if unresolvable.
    When multiple admissions match, picks the one whose admittime is closest
    to charttime.

    ``pd.to_datetime`` is called on ``row["charttime"]`` intentionally: it is
    idempotent when the value is already a Timestamp (e.g. from ``parse_dates``)
    and handles string/NaT inputs gracefully, making this helper reusable
    regardless of how the calling DataFrame was loaded.
    """
    ct = pd.to_datetime(row["charttime"])
    candidates = admissions_df[admissions_df["subject_id"] == row["subject_id"]].copy()
    if candidates.empty:
        return None
    admit_times = pd.to_datetime(candidates["admittime"])
    if "dischtime" in candidates.columns:
        disch_times = pd.to_datetime(candidates["dischtime"])
        window_mask = (admit_times - tolerance <= ct) & (ct <= disch_times + tolerance)
    else:
        window_mask = (admit_times - tolerance <= ct)
    matches = candidates[window_mask]
    if len(matches) == 0:
        return None
    if len(matches) == 1:
        return float(matches.iloc[0]["hadm_id"])
    # Multiple matches: pick the one whose admittime is closest to charttime
    matches = matches.copy()
    matches["_gap"] = (pd.to_datetime(matches["admittime"]) - ct).abs()
    best_idx = matches["_gap"].idxmin()
    return float(matches.loc[best_idx, "hadm_id"])


def _load_d_labitems(hosp_dir: str) -> pd.DataFrame:
    """Load and clean the d_labitems lookup table from *hosp_dir*.

    Returns a DataFrame with:
    - Columns: itemid, label, fluid, category
    - Whitespace stripped from fluid and category
    - Artifact rows removed (fluid in {'I', 'Q', 'fluid'})
    """
    _artifact_fluids = frozenset({"I", "Q", "fluid"})
    d_labitems = _load_csv(
        os.path.join(hosp_dir, "d_labitems.csv.gz"),
        os.path.join(hosp_dir, "d_labitems.csv"),
        usecols=["itemid", "label", "fluid", "category"],
    )
    d_labitems["fluid"] = d_labitems["fluid"].str.strip()
    d_labitems["category"] = d_labitems["category"].str.strip()
    return d_labitems[~d_labitems["fluid"].isin(_artifact_fluids)].copy()


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
