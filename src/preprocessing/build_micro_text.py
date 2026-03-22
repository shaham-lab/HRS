"""
build_micro_text.py – Helper module for microbiology text construction.

Provides three functions:
    clean_comment(text, cleaning_config) -> str | None
    build_event_text(row, cleaning_config) -> str
    aggregate_panel_text(panel_df, cleaning_config) -> pd.DataFrame

Imported by extract_microbiology.py.
"""

import re
import logging

import pandas as pd

logger = logging.getLogger(__name__)

_INTERP_PRIORITY = {"R": 0, "S": 1, "I": 2}


def clean_comment(text, cleaning_config: dict) -> "str | None":
    """Clean a microbiology comment string.

    Returns a cleaned string or None if the comment should be discarded.
    """
    # Step 1: discard null/blank/dash-only strings
    if text is None:
        return None
    text = str(text)
    if not text.strip():
        return None
    if re.fullmatch(r'_+|-+', text.strip()):
        return None

    max_sentences = int(cleaning_config.get("max_sentences", 2))
    max_chars = int(cleaning_config.get("max_chars", 200))
    discard_prefixes = cleaning_config.get("discard_prefixes", [])
    strip_triggers = cleaning_config.get("strip_triggers", [])

    # Step 2: discard_prefix check (after stripping leading whitespace)
    text = text.lstrip()
    for prefix in discard_prefixes:
        if text.startswith(prefix):
            return None

    # Step 3: strip_trigger truncation — first match wins
    for trigger in strip_triggers:
        idx = text.find(trigger)
        if idx > 0:
            text = text[:idx]
            break

    # Step 4: sentence splitting and truncation
    parts = re.split(r'\.\s{2,}|\.\n', text)
    parts = parts[:max_sentences]
    if len(parts) == 2:
        text = parts[0] + ".  " + parts[1]
    else:
        text = parts[0]

    # Step 5: post-strip artifact cleanup
    text = text.rstrip()
    text = re.sub(r'\.\s+\($', '', text)
    text = re.sub(r'\s+\($', '', text)
    text = text.rstrip('(')
    text = text.rstrip('.')
    text = text.strip()

    # Step 6: hard-truncate and final check
    text = text[:max_chars]
    if not text.strip():
        return None
    return text


def build_event_text(row, cleaning_config: dict) -> str:
    """Build event text string for a single microbiologyevents row.

    row may be a dict or a pandas Series.
    """
    def _get(field):
        if isinstance(row, dict):
            return row.get(field)
        return row[field] if field in row.index else None

    test_name = str(_get("test_name") or "")
    spec_type_desc = str(_get("spec_type_desc") or "")

    org_name_raw = _get("org_name")
    has_org = pd.notna(org_name_raw) and bool(str(org_name_raw).strip())

    ab_name_raw = _get("ab_name")
    has_ab = pd.notna(ab_name_raw) and bool(str(ab_name_raw).strip())

    interp_raw = _get("interpretation")
    susc_part = f"{ab_name_raw}:{interp_raw}" if has_ab else ""

    comment_raw = _get("comments")
    cleaned_comment = clean_comment(comment_raw, cleaning_config)

    if has_org:
        base = f"{test_name} [{spec_type_desc}]: {org_name_raw}"
        if susc_part:
            base += f" | {susc_part}"
        if cleaned_comment is not None:
            base += f" | {cleaned_comment}"
        return base
    if cleaned_comment is not None:
        return f"{test_name} [{spec_type_desc}]: {cleaned_comment}"
    return f"{test_name} [{spec_type_desc}]: pending"


def aggregate_panel_text(panel_df: pd.DataFrame, cleaning_config: dict) -> pd.DataFrame:
    """Aggregate microbiology events for one panel into per-admission text rows.

    Parameters
    ----------
    panel_df:
        DataFrame of microbiologyevents rows already filtered to this panel
        and the admission time window.
    cleaning_config:
        Comment cleaning configuration dict from micro_panel_config.yaml.

    Returns
    -------
    DataFrame with columns [subject_id, hadm_id, text], one row per admission.
    """
    if panel_df.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "text"])

    event_group_keys = [
        "subject_id", "hadm_id", "charttime", "test_name", "spec_type_desc", "org_name",
    ]

    records = []
    for group_keys, group in panel_df.groupby(event_group_keys, dropna=False, sort=False):
        subject_id, hadm_id, charttime, test_name, spec_type_desc, org_name = group_keys

        # Collect best interpretation per antibiotic
        ab_interps: dict[str, str] = {}
        ab_order: list[str] = []
        for _, row in group.iterrows():
            ab = row.get("ab_name") if hasattr(row, "get") else row["ab_name"]
            interp = row.get("interpretation") if hasattr(row, "get") else row["interpretation"]
            if not (pd.notna(ab) and str(ab).strip()):
                continue
            ab_str = str(ab)
            interp_str = str(interp) if pd.notna(interp) else ""
            if ab_str not in ab_interps:
                ab_interps[ab_str] = interp_str
                ab_order.append(ab_str)
            else:
                existing = ab_interps[ab_str]
                existing_pri = _INTERP_PRIORITY.get(existing, 99)
                new_pri = _INTERP_PRIORITY.get(interp_str, 99)
                if new_pri < existing_pri:
                    ab_interps[ab_str] = interp_str

        susc_string = ", ".join(f"{ab}:{ab_interps[ab]}" for ab in ab_order) if ab_order else ""

        # Take first non-None cleaned comment across all rows in the group
        cleaned_comment = None
        for _, row in group.iterrows():
            comment_raw = row.get("comments") if hasattr(row, "get") else row["comments"]
            cleaned_comment = clean_comment(comment_raw, cleaning_config)
            if cleaned_comment is not None:
                break

        has_org = pd.notna(org_name) and bool(str(org_name).strip())
        if has_org:
            event_text = f"{test_name} [{spec_type_desc}]: {org_name}"
            if susc_string:
                event_text += f" | {susc_string}"
            if cleaned_comment is not None:
                event_text += f" | {cleaned_comment}"
        elif cleaned_comment is not None:
            event_text = f"{test_name} [{spec_type_desc}]: {cleaned_comment}"
        else:
            event_text = f"{test_name} [{spec_type_desc}]: pending"

        records.append({
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "charttime": charttime,
            "text": event_text,
        })

    if not records:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "text"])

    events_df = pd.DataFrame(records)

    # Aggregate per (subject_id, hadm_id): sort by charttime, dedup, join
    admission_texts = []
    for (subject_id, hadm_id), adm_group in events_df.groupby(
        ["subject_id", "hadm_id"], sort=False
    ):
        sorted_events = adm_group.sort_values("charttime")["text"].tolist()

        # Deduplicate adjacent identical texts
        deduped: list[str] = []
        for evt in sorted_events:
            if not deduped or evt != deduped[-1]:
                deduped.append(evt)

        # Exact dedup across all events (preserve first-occurrence order)
        seen: set[str] = set()
        final: list[str] = []
        for evt in deduped:
            if evt not in seen:
                seen.add(evt)
                final.append(evt)

        admission_texts.append({
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "text": " | ".join(final),
        })

    return pd.DataFrame(admission_texts, columns=["subject_id", "hadm_id", "text"])
