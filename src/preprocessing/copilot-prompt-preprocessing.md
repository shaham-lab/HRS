# GitHub Copilot Prompt — Align preprocessing codebase with Feature Pre-Processing specification

## Context

You are working on the CDSS-ML preprocessing pipeline for MIMIC-IV. Four files are open:

- `DATA_PROCESSING.md` — current technical reference document (partially out of date)
- `preprocessing-runtime-instructions.md` — runtime/configuration documentation (partially out of date)
- `extract_demographics.py` — one implemented module (representative of the module pattern)
- `inspect_data.py` — diagnostic utility

The authoritative specification is the **Feature Pre-Processing document** (the Word document). The code and markdown docs must be updated to match it. Below are all required changes, grouped by file and topic.

---

## 1. Changes to `DATA_PROCESSING.md`

### 1a. Split ratios — correct the defaults

The document currently states the default split as **70/15/15**. Change this to **80/10/10** throughout.

Affected locations:
- Section 1 "Split strategy" prose
- Section 3 table row `Split ratios`

### 1b. Diagnosis history text format (F2) — replace pipe-concatenation with structured format

Current text says diagnoses are concatenated with `|` as separator and sorted by admission time. Replace the entire F2 section description with the following:

> All diagnoses from prior admissions are formatted as a structured text block, ordered chronologically by admission date. Each visit is rendered as a dated section header followed by one diagnosis long_title per line. The format is:
>
> ```
> Past Diagnoses:
>
> Visit (YYYY-MM-DD):
> {long_title}
> {long_title}
>
> Visit (YYYY-MM-DD):
> {long_title}
> ```
>
> First-time admissions (no prior visits) produce an empty string.

### 1c. Discharge history text format (F3) — replace separator and add date header format

Current text says notes are concatenated with `\n\n---\n\n` as separator. Replace with:

> Multiple prior discharge notes are concatenated in chronological order. Each note is prefixed with a dated header line. The format is:
>
> ```
> Prior Discharge Summary (YYYY-MM-DD):
> [cleaned note text]
>
> Prior Discharge Summary (YYYY-MM-DD):
> [cleaned note text]
> ```
>
> First-time admissions (no prior discharge notes) produce an empty string.

### 1d. Lab text line format (F6) — change to relative timestamps and remove fluid/category from inline label

Current text line format is:
```
[HH:MM] {label} ({fluid}/{category}): {value} {unit} (ref: lower-upper) [ABNORMAL] [STAT]
```

Change to:
```
[HH:MM] {label}: {value} {unit} (ref: lower-upper) [ABNORMAL]
```

Specific changes:
- `[HH:MM]` already uses relative elapsed time — confirm and keep as-is (elapsed hours and minutes since `admittime`, not absolute clock time)
- Remove `({fluid}/{category})` from the inline label — fluid and category are captured in the lab group structure, not repeated per line
- Remove `[STAT]` flag — this is not included in the final text line format
- `[ABNORMAL]` is appended when `flag = "abnormal"` OR when `valuenum` falls outside `[ref_range_lower, ref_range_upper]`; update the condition description accordingly

### 1e. Embedding method — replace CLS with mean pooling

Section 6 "BERT embeddings" currently states:

> `Embedding method | [CLS] token from the final hidden state`

Replace with:

> `Embedding method | Mean pooling over all non-padding content tokens from the final hidden layer`

Add a note below the table:

> Mean pooling is used because Clinical_ModernBERT is deployed as a frozen feature extractor, not fine-tuned end-to-end. Mean pooling ensures every content token — every lab measurement, diagnostic term, and clinical observation — contributes equally to the embedding. This is especially important for long multi-visit and multi-measurement texts (F2, F3, F6–F18, F19).

### 1f. Default BERT model — update to Clinical_ModernBERT

Current default model is `emilyalsentzer/Bio_ClinicalBERT`. Update to:
- `BERT_MODEL_NAME` default: `Simonlee711/Clinical_ModernBERT`
- `BERT_MAX_LENGTH` default: `8192` (not `512` — Clinical_ModernBERT supports an 8,192 token context window)

Affected locations:
- Section 6 table
- Any inline references to the model name or max length

### 1g. Add new Section for hadm_id and stay_id handling

Insert a new section **"Identifier handling"** between Section 3 (splits) and Section 4 (feature extraction). Content:

**Identifier hierarchy**

```
subject_id  (patient)
  └── hadm_id  (hospital admission)
        └── stay_id  (ICU stay within that admission)
```

**stay_id**

`stay_id` identifies a single ICU stay within an `hadm_id`. It appears in ICU module tables: `icustays`, `chartevents`, `inputevents`, `outputevents`, `procedureevents`, `datetimeevents`. In the current pipeline, `stay_id` is only encountered indirectly when `chartevents` is used as a height/weight fallback in `extract_demographics`; the join there is performed via `hadm_id` so `stay_id` does not need to be handled explicitly. `stay_id` becomes critical in the MDP phase for modelling clinical interventions.

**Missing hadm_id**

Several MIMIC-IV tables contain records with null `hadm_id`. This is most significant in `labevents` (10–20% of rows) and also affects `note` and `chartevents`. The handling strategy is configurable via `HADM_LINKAGE_STRATEGY` in `preprocessing.yaml`:

- `"drop"` (default): Records with null `hadm_id` are excluded. Each module logs the count and percentage of records dropped.
- `"link"`: For records with a valid `subject_id` and `charttime` but null `hadm_id`, the pipeline attempts to assign an `hadm_id` by matching `charttime` against the patient's admission windows within a tolerance of `HADM_LINKAGE_TOLERANCE_HOURS` hours (default 1). If exactly one window matches, `hadm_id` is assigned. If multiple match, the closest is used. If none match, the record is dropped. All outcomes are logged and saved to `hadm_linkage_stats.json`.

### 1h. Update lab section to reference 13 independent lab group features (F6–F18)

The current document describes a single feature "F6 — Lab events". Update:
- Change the feature label from F6 to F6–F18
- Note that `extract_labs.py` produces a single long-format parquet covering all 13 lab groups
- The 13 groups are defined by `(fluid × category)` combinations derived from `d_labitems` and stored in `lab_panel_config.yaml` (generated by `build_lab_panel_config.py`)
- At embedding time, each group is embedded independently, producing 13 embedding parquets: `lab_{group_name}_embeddings.parquet`
- Each lab group embedding is a fully independent feature, exactly parallel to triage, radiology, and other embedding features — it has its own projection layer in the neural network and is independently selectable/maskable by the MDP agent
- Update the embedding table in Section 6 to add all 13 lab embedding rows

### 1i. Update Section 6 embedding table — lab group embeddings included in final dataset

In the embedding table, add 13 rows, one per lab group:

| Input parquet | Text column | Output parquet | Embedding column |
|---|---|---|---|
| `labs_features.parquet` (filtered to group) | `lab_text_line` (concatenated per admission per group) | `lab_blood_gas_embeddings.parquet` | `lab_blood_gas_embedding` |
| … | … | `lab_blood_chemistry_embeddings.parquet` | `lab_blood_chemistry_embedding` |
| … | … | `lab_blood_hematology_embeddings.parquet` | `lab_blood_hematology_embedding` |
| … | … | `lab_urine_chemistry_embeddings.parquet` | `lab_urine_chemistry_embedding` |
| … | … | `lab_urine_hematology_embeddings.parquet` | `lab_urine_hematology_embedding` |
| … | … | `lab_other_body_fluid_chemistry_embeddings.parquet` | `lab_other_body_fluid_chemistry_embedding` |
| … | … | `lab_other_body_fluid_hematology_embeddings.parquet` | `lab_other_body_fluid_hematology_embedding` |
| … | … | `lab_ascites_embeddings.parquet` | `lab_ascites_embedding` |
| … | … | `lab_pleural_embeddings.parquet` | `lab_pleural_embedding` |
| … | … | `lab_csf_embeddings.parquet` | `lab_csf_embedding` |
| … | … | `lab_bone_marrow_embeddings.parquet` | `lab_bone_marrow_embedding` |
| … | … | `lab_joint_fluid_embeddings.parquet` | `lab_joint_fluid_embedding` |
| … | … | `lab_stool_embeddings.parquet` | `lab_stool_embedding` |

Add a note after the table:

> Lab group embedding parquets are included in `final_cdss_dataset.parquet` as 13 independent columns, discovered and joined automatically by `combine_dataset.py`. Lab group embedding columns are always a 768-float array — admissions with no events in a given group carry a zero vector, consistent with the empty-text convention used throughout the pipeline. `labs_features.parquet` (the long-format raw event data) remains excluded from the final dataset as it is superseded by the embedding parquets.

---

## 2. Changes to `preprocessing-runtime-instructions.md`

### 2a. Split ratio defaults — correct 70/15/15 to 80/10/10

Update the config table row for `SPLIT_TRAIN`, `SPLIT_DEV`, `SPLIT_TEST` example values from `0.70`/`0.15`/`0.15` to `0.80`/`0.10`/`0.10`.

### 2b. BERT model and max length defaults

Update the config table:
- `BERT_MODEL_NAME` example: `"Simonlee711/Clinical_ModernBERT"`
- `BERT_MAX_LENGTH` example: `8192`

### 2c. Add two new config keys to the configuration table

Add the following rows to the `preprocessing.yaml` configuration table in Section 3:

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `HADM_LINKAGE_STRATEGY` | `str` | How to handle records with null `hadm_id`. `"drop"` excludes them (default); `"link"` attempts time-window linkage using `charttime` and admission windows. | `"drop"` |
| `HADM_LINKAGE_TOLERANCE_HOURS` | `int` | Hours of tolerance outside `admittime`/`dischtime` used when `HADM_LINKAGE_STRATEGY` is `"link"`. Ignored when strategy is `"drop"`. | `1` |

### 2d. Add new lab-related modules to the directory layout and pipeline steps

In the directory layout (Section 2):
- Under `src/preprocessing/`, add: `build_lab_panel_config.py`, `build_lab_text_lines.py`
- Under `data/input/features/`, add: `labs_features.parquet` (already present — confirm it remains)
- Under `data/input/embeddings/`, add: `lab_{group}_embeddings.parquet` (×13, e.g. `lab_blood_chemistry_embeddings.parquet`)
- Under `data/input/classifications/`, add: `lab_panel_config.yaml`, `hadm_linkage_stats.json`

In the pipeline execution order diagram (Section 4), note that `build_lab_panel_config.py` must run before `extract_labs.py`.

### 2e. Update the output files reference table (Section 7)

Add the following rows:
- `lab_panel_config.yaml` — `data/input/classifications/` — YAML — `build_lab_panel_config` — Defines the 13 lab group names and their constituent itemids, derived from `d_labitems`
- `hadm_linkage_stats.json` — `data/input/classifications/` — JSON — all modules — Per-module counts of null hadm_id records: dropped, linked, ambiguous-resolved, unresolvable
- `lab_{group}_embeddings.parquet` (×13) — `data/input/embeddings/` — Parquet — `embed_features` — One row per admission per lab group; `lab_{group}_embedding` array column (768 floats); null for admissions with no events in that group

Update the `final_cdss_dataset.parquet` description:

> One row per admission; all features and labels joined. Includes demographics, all 5 non-lab embedding columns, and all 13 lab group embedding columns as independent columns. Lab group embedding columns are always a 768-float array — admissions with no events in a given group carry a zero vector, consistent with the empty-text convention used throughout the pipeline. `labs_features.parquet` (long-format raw event data) is excluded — it is superseded by the 13 per-group embedding parquets which are included. The 13 lab group embeddings are discovered and joined automatically by `combine_dataset.py` via dynamic parquet discovery in `EMBEDDINGS_DIR`.

---

## 3. Changes to `extract_demographics.py`

### 3a. Add HADM_LINKAGE_STRATEGY and HADM_LINKAGE_TOLERANCE_HOURS to required/optional config keys

In the `run()` function, read two new optional config values:
```python
hadm_linkage_strategy = config.get("HADM_LINKAGE_STRATEGY", "drop").lower()
hadm_linkage_tolerance_hours = int(config.get("HADM_LINKAGE_TOLERANCE_HOURS", 1))
```

Pass these to any internal function that reads `chartevents` or other tables that may have null `hadm_id`. (In practice, `chartevents` in `extract_demographics.py` joins on `hadm_id` and the primary risk is height/weight rows where `hadm_id` is null.)

### 3b. Add null hadm_id logging to `_extract_chart_vitals`

At the point where `chartevents` chunks are read, before the `itemid` filter, add:

```python
null_hadm = chunk["hadm_id"].isna().sum()
if null_hadm > 0:
    logger.info(
        "chartevents chunk %d: %d rows (%.1f%%) have null hadm_id — strategy: %s",
        i, null_hadm, 100 * null_hadm / len(chunk), hadm_linkage_strategy,
    )
```

If `hadm_linkage_strategy == "drop"`, filter out null `hadm_id` rows immediately after logging:
```python
chunk = chunk[chunk["hadm_id"].notna()]
```

If `hadm_linkage_strategy == "link"`, implement the linkage logic:
- For rows with null `hadm_id` but valid `subject_id` and `charttime`, attempt to find a matching `hadm_id` from `admissions` where `admittime - tolerance <= charttime <= dischtime + tolerance`
- If exactly one match: assign `hadm_id`
- If multiple matches: assign the one whose window most closely contains `charttime` (smallest gap to nearest boundary), log as ambiguous-resolved
- If no match: drop and log as unresolvable
- Accumulate counts per outcome and log a summary after all chunks are processed

### 3c. Add hadm_linkage_stats.json output

After processing all chunks in `_extract_chart_vitals` (or in `run()` if stats are accumulated there), write or update `hadm_linkage_stats.json` in `classifications_dir`:

```json
{
  "extract_demographics": {
    "chartevents": {
      "total_null_hadm": 12345,
      "dropped": 12345,
      "linked": 0,
      "ambiguous_resolved": 0,
      "unresolvable": 0
    }
  }
}
```

The file should be updated (merged), not overwritten, so other modules can append their own entries.

### 3d. Update module docstring

Update the module docstring at the top of the file to document the two new config keys:
```
HADM_LINKAGE_STRATEGY        – "drop" (default) or "link"; how to handle null hadm_id in chartevents
HADM_LINKAGE_TOLERANCE_HOURS – hours of tolerance for time-window linkage (default 1, only used when strategy is "link")
```

---

## 4. Changes to `inspect_data.py`

### 4a. Add null hadm_id diagnostics to `_inspect_labevents`

In `_inspect_labevents`, after loading the dataframe, add:
```python
null_hadm_count = df["hadm_id"].isna().sum()
null_hadm_pct = 100 * df["hadm_id"].isna().mean()
print(f"\nNull hadm_id: {null_hadm_count:,} ({null_hadm_pct:.1f}%) — these are outpatient/unlinked events")
print("  → HADM_LINKAGE_STRATEGY in preprocessing.yaml controls how these are handled.")
```

### 4b. Add null hadm_id diagnostics to `_inspect_chartevents`

Same pattern as 4a — report the count and percentage of null `hadm_id` rows in `chartevents`.

### 4c. Report Clinical_ModernBERT token budget context for long texts

In `_inspect_discharge` and `_inspect_radiology`, after printing the first 3 note previews, add a character/token estimate:
```python
avg_len = df["text"].dropna().str.len().mean()
print(f"\nAverage note length: {avg_len:,.0f} characters (~{avg_len / 4:.0f} tokens estimated)")
print("  Clinical_ModernBERT context window: 8,192 tokens. Notes exceeding this will be truncated.")
```

### 4d. Add `_inspect_lab_panel_config` function

Add a new inspection function that reads `lab_panel_config.yaml` from `CLASSIFICATIONS_DIR` if it exists:

```python
def _inspect_lab_panel_config(classifications_dir: str) -> None:
    path = os.path.join(classifications_dir, "lab_panel_config.yaml")
    _print_header("lab_panel_config.yaml", path)
    if not os.path.exists(path):
        print("NOT FOUND — run build_lab_panel_config.py first.")
        return
    import yaml
    with open(path) as f:
        config = yaml.safe_load(f)
    print(f"\nTotal lab groups: {len(config)}")
    for group_name, items in config.items():
        print(f"  {group_name}: {len(items)} itemids")
```

Call this from `main()` after `_inspect_d_labitems`, passing `config.get("CLASSIFICATIONS_DIR", "")`.

### 4e. Add `_inspect_hadm_linkage_stats` function

Add a new inspection function that reads `hadm_linkage_stats.json` from `CLASSIFICATIONS_DIR` if it exists:

```python
def _inspect_hadm_linkage_stats(classifications_dir: str) -> None:
    path = os.path.join(classifications_dir, "hadm_linkage_stats.json")
    _print_header("hadm_linkage_stats.json", path)
    if not os.path.exists(path):
        print("NOT FOUND — will be created after pipeline runs.")
        return
    import json
    with open(path) as f:
        stats = json.load(f)
    for module, tables in stats.items():
        print(f"\n  Module: {module}")
        for table, counts in tables.items():
            print(f"    {table}:")
            for k, v in counts.items():
                print(f"      {k}: {v:,}")
```

Call this from `main()` at the end, after `_inspect_edstays`.

---

## 5. New files to create

The following files do not yet exist and must be created. Stubs with correct structure and docstrings are sufficient — full implementation can follow.

### 5a. `build_lab_panel_config.py`

Create a new module that:
- Reads `d_labitems` from `MIMIC_DATA_DIR/hosp/`
- Groups items by `(fluid, category)` to produce 13 lab group names
- Filters out artefact rows where `fluid` is in `["I", "Q", "fluid"]`
- Writes the resulting group definitions to `CLASSIFICATIONS_DIR/lab_panel_config.yaml`
- Each group is a dict mapping a snake_case group name (e.g. `blood_chemistry`) to a list of `itemid` integers
- Follows the same module pattern as `extract_demographics.py`: `run(config: dict) -> None`, hash-based skip check, logging

The 13 expected group names (derived from fluid × category combinations present in MIMIC-IV `d_labitems`) are:
```
blood_gas, blood_chemistry, blood_hematology,
urine_chemistry, urine_hematology,
other_body_fluid_chemistry, other_body_fluid_hematology,
ascites, pleural, csf, bone_marrow, joint_fluid, stool
```

Note: some groups (e.g. `ascites`, `pleural`) span multiple categories (Chemistry + Hematology) — use the fluid name alone as the group key in those cases.

### 5b. `build_lab_text_lines.py`

Create a helper module (not a standalone pipeline step — called from `extract_labs.py`) that:
- Takes a DataFrame of lab events with columns: `charttime`, `label`, `value`, `valuenum`, `valueuom`, `ref_range_lower`, `ref_range_upper`, `flag`, `admittime`
- Returns a `lab_text_line` string per row using the format:
  ```
  [HH:MM] {label}: {value} {unit} (ref: lower-upper) [ABNORMAL]
  ```
  Where:
  - `[HH:MM]` = elapsed time since `admittime` (e.g. `[02:14]` = 2 hours 14 minutes after admission)
  - `{value}` = `valuenum` formatted to 2 decimal places if available, else the text `value` field
  - `(ref: lower-upper)` omitted if either bound is null
  - `[ABNORMAL]` appended when `flag == "abnormal"` OR when `valuenum` is not null and falls outside `[ref_range_lower, ref_range_upper]`

### 5c. Update `embed_features.py` — add 13 lab group embedding tasks

`embed_features.py` currently embeds 5 text features via the `_TEXT_FEATURES` list. Extend it to also embed all 13 lab groups. The logic differs from the other features because the input is long-format (one row per lab event) rather than one row per admission.

Add the following to `embed_features.py`:

**Load lab panel config at startup:**
```python
lab_panel_config_path = os.path.join(
    str(config["CLASSIFICATIONS_DIR"]), "lab_panel_config.yaml"
)
if not os.path.exists(lab_panel_config_path):
    raise FileNotFoundError(
        f"lab_panel_config.yaml not found at {lab_panel_config_path}. "
        "Run build_lab_panel_config.py first."
    )
import yaml
with open(lab_panel_config_path) as f:
    lab_panel_config: dict[str, list[int]] = yaml.safe_load(f)
```

**For each lab group, build per-admission text and embed:**

After embedding the 5 standard text features, iterate over `lab_panel_config`:

```python
labs_path = os.path.join(features_dir, "labs_features.parquet")
if os.path.exists(labs_path):
    labs_df = pd.read_parquet(labs_path)  # long format: subject_id, hadm_id, itemid, lab_text_line

    for group_name, itemids in lab_panel_config.items():
        output_filename = f"lab_{group_name}_embeddings.parquet"
        output_path = os.path.join(embeddings_dir, output_filename)
        embedding_col = f"lab_{group_name}_embedding"

        # Filter to this group's itemids
        group_df = labs_df[labs_df["itemid"].isin(itemids)]

        # Aggregate: concatenate all text lines per admission in chronological order
        group_text = (
            group_df.sort_values("charttime")
            .groupby(["subject_id", "hadm_id"])["lab_text_line"]
            .apply(lambda lines: "\n".join(lines))
            .reset_index()
            .rename(columns={"lab_text_line": "text"})
        )

        # Get the full admission universe from splits and left-join so
        # admissions with no events in this group get an empty string → zero vector
        splits_df = pd.read_parquet(
            os.path.join(str(config["CLASSIFICATIONS_DIR"]), "data_splits.parquet")
        )[["subject_id", "hadm_id"]].drop_duplicates()
        group_text = splits_df.merge(group_text, on=["subject_id", "hadm_id"], how="left")
        group_text["text"] = group_text["text"].fillna("")

        texts = group_text["text"].tolist()
        embeddings = _embed_texts(texts, tokenizer, model, device, max_length, batch_size)

        out_df = group_text[["subject_id", "hadm_id"]].copy()
        out_df[embedding_col] = list(embeddings)
        out_df.to_parquet(output_path, index=False)
        logger.info(
            "Saved %s embeddings to %s  (shape=%s, embedding_dim=%d)",
            group_name, output_path, out_df.shape, embeddings.shape[1],
        )
else:
    logger.warning(
        "labs_features.parquet not found at %s — lab group embeddings skipped.", labs_path
    )
```

**Update the module docstring** to document that `embed_features.py` now also reads `CLASSIFICATIONS_DIR` (for `lab_panel_config.yaml` and `data_splits.parquet`) and requires `labs_features.parquet` in `FEATURES_DIR`. Add `CLASSIFICATIONS_DIR` to the `required_keys` list in `run()`.

---

## 6. Summary of config key changes

Ensure `preprocessing.yaml` (not provided here, but referenced by all modules) has the following keys with the specified defaults. All modules that read these keys must use `config.get(key, default)` — never hardcode:

| Key | New default | Previously |
|-----|-------------|------------|
| `SPLIT_TRAIN` | `0.80` | `0.70` |
| `SPLIT_DEV` | `0.10` | `0.15` |
| `SPLIT_TEST` | `0.10` | `0.15` |
| `BERT_MODEL_NAME` | `"Simonlee711/Clinical_ModernBERT"` | `"emilyalsentzer/Bio_ClinicalBERT"` |
| `BERT_MAX_LENGTH` | `8192` | `512` |
| `HADM_LINKAGE_STRATEGY` | `"drop"` | *(new key)* |
| `HADM_LINKAGE_TOLERANCE_HOURS` | `1` | *(new key)* |

---

## 7. Conventions to preserve across all changes

- **No hardcoding** — all paths, model names, split ratios, thresholds read from `config` dict
- **Hash-based skip check** — all pipeline modules check `_sources_unchanged()` before running; write hashes only after successful completion
- **Streaming** — `labevents` and `chartevents` are processed in chunks (500,000 rows default); never load the full table into memory
- **Train-only statistics** — any statistics used for imputation or normalisation are computed from the train split only and saved to JSON; dev/test use the pre-computed values
- **Logging** — use `logger.info` / `logger.warning` (not `print`) in pipeline modules; `print` is acceptable in `inspect_data.py` only
- **Graceful degradation** — missing optional tables (omr, chartevents) cause a logged warning but do not crash the module
