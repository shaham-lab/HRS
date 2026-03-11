"""
build_lab_text_lines.py – Helper for formatting lab event rows as text lines.

This module is not a standalone pipeline step — it is called from extract_labs.py.
It provides a vectorised function that converts a DataFrame of lab events into
per-row text line strings using the canonical format:

    [HH:MM] {label}: {value} {unit} (ref: lower-upper) [ABNORMAL]

Where:
    [HH:MM]           – elapsed time since admittime (e.g. [02:14] = 2 h 14 min)
    {value}           – valuenum formatted to 2 decimal places if available,
                        otherwise the text value field
    (ref: lower-upper)– omitted if either bound is null
    [ABNORMAL]        – appended when flag == "abnormal" OR when valuenum is not
                        null and falls outside [ref_range_lower, ref_range_upper]

Input DataFrame must have columns:
    charttime, admittime, label, value, valuenum, valueuom,
    ref_range_lower, ref_range_upper, flag
"""

import pandas as pd


def build_lab_text_line_series(df: pd.DataFrame) -> pd.Series:
    """Return a Series of lab text lines, one per row of df.

    Vectorised implementation for performance on large DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with required lab event columns (see module docstring).

    Returns
    -------
    pd.Series
        String series aligned to df's index.
    """
    # Elapsed time since admittime
    elapsed = pd.to_datetime(df["charttime"]) - pd.to_datetime(df["admittime"])
    # Clip to 0 in case charttime is slightly before admittime (e.g. data entry
    # rounding artefacts or events recorded at admission boundary).
    total_minutes = (elapsed.dt.total_seconds() // 60).clip(lower=0).astype(int)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    time_strs = hours.astype(str).str.zfill(2) + ":" + minutes.astype(str).str.zfill(2)

    # Value string: prefer numeric, fall back to text
    value_num = pd.to_numeric(df["valuenum"], errors="coerce")
    has_num = pd.notna(value_num)
    text_values = df["value"].fillna("").astype(str).str.strip()
    value_strs = pd.Series(
        [f"{v:.2f}" if h else t for h, v, t in zip(has_num, value_num, text_values)],
        index=df.index,
    )

    # Unit
    uom = df["valueuom"].fillna("").astype(str).str.strip()
    uom_strs = uom.where(uom == "", " " + uom)

    # Reference range
    ref_lower = pd.to_numeric(df["ref_range_lower"], errors="coerce")
    ref_upper = pd.to_numeric(df["ref_range_upper"], errors="coerce")
    has_ref = pd.notna(ref_lower) & pd.notna(ref_upper)
    ref_strs = pd.Series(
        [f" (ref: {lo:g}-{hi:g})" if h else ""
         for h, lo, hi in zip(has_ref, ref_lower, ref_upper)],
        index=df.index,
    )

    # Abnormal flag: flagged OR valuenum outside reference range
    flag_col = df["flag"].fillna("").astype(str).str.strip().str.lower()
    is_abnormal = flag_col == "abnormal"
    # Also check if valuenum is outside reference range
    out_of_range = has_num & has_ref & (
        (value_num < ref_lower) | (value_num > ref_upper)
    )
    is_abnormal = is_abnormal | out_of_range
    flag_strs = is_abnormal.map(lambda x: " [ABNORMAL]" if x else "")

    result = (
        "[" + time_strs + "] "
        + df["label"].astype(str) + ": "
        + value_strs + uom_strs + ref_strs + flag_strs
    )
    return result


def _compute_row_abnormal_flag(row) -> bool:
    """Return True if the lab event row should be flagged as abnormal.

    A row is abnormal if flag == "abnormal" OR if valuenum falls outside
    [ref_range_lower, ref_range_upper].
    """
    is_abnormal = str(row.get("flag", "")).strip().lower() == "abnormal"
    if not is_abnormal:
        ref_lower = row.get("ref_range_lower")
        ref_upper = row.get("ref_range_upper")
        if pd.notna(row.get("valuenum")) and pd.notna(ref_lower) and pd.notna(ref_upper):
            try:
                vn = float(row["valuenum"])
                is_abnormal = vn < float(ref_lower) or vn > float(ref_upper)
            except (TypeError, ValueError):
                pass
    return is_abnormal


def build_lab_text_line_row(row) -> str:
    """Convert a single lab event row to a text line string.

    Row-wise version for use with DataFrame.apply(). For large DataFrames,
    prefer build_lab_text_line_series() which is fully vectorised.

    Parameters
    ----------
    row : pandas Series
        A single row from a lab events DataFrame (see module docstring).

    Returns
    -------
    str
        Formatted text line.
    """
    # Elapsed time since admittime
    try:
        elapsed = pd.to_datetime(row["charttime"]) - pd.to_datetime(row["admittime"])
        total_minutes = int(max(0, elapsed.total_seconds() // 60))
        hours = total_minutes // 60
        minutes = total_minutes % 60
        time_str = f"{hours:02d}:{minutes:02d}"
    except (TypeError, ValueError, OverflowError):
        time_str = "00:00"

    # Value: prefer numeric formatted to 2dp, fall back to text value
    if pd.notna(row.get("valuenum")):
        try:
            value_str = f"{float(row['valuenum']):.2f}"
        except (TypeError, ValueError):
            value_str = str(row.get("value", "")).strip()
    else:
        value_str = str(row.get("value", "")).strip()

    # Unit
    uom = str(row.get("valueuom", "")).strip()
    uom_str = f" {uom}" if uom else ""

    # Reference range — only include when both bounds are present
    ref_lower = row.get("ref_range_lower")
    ref_upper = row.get("ref_range_upper")
    ref_str = ""
    if pd.notna(ref_lower) and pd.notna(ref_upper):
        try:
            ref_str = f" (ref: {float(ref_lower):g}-{float(ref_upper):g})"
        except (TypeError, ValueError):
            ref_str = f" (ref: {ref_lower}-{ref_upper})"

    # Abnormal flag: flagged as "abnormal" OR valuenum outside reference range
    flag_str = " [ABNORMAL]" if _compute_row_abnormal_flag(row) else ""

    return f"[{time_str}] {row['label']}: {value_str}{uom_str}{ref_str}{flag_str}"
