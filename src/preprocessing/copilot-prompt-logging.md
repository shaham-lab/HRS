# GitHub Copilot Prompt — Improve logging and progress reporting across the pipeline

## Context

You are working on the CDSS-ML preprocessing pipeline for MIMIC-IV. The following files are open:

- `run_pipeline.py`
- `create_splits.py`
- `extract_demographics.py`
- `extract_diag_history.py`
- `extract_discharge_history.py`
- `extract_triage_and_complaint.py`
- `extract_labs.py`
- `extract_radiology.py`
- `extract_y_data.py`
- `embed_features.py`
- `combine_dataset.py`
- `build_lab_panel_config.py`

A review of the logging and progress reporting identified the following problems:

1. `run_pipeline.py` gives no indication of overall pipeline progress — which step out of how many, elapsed time per module, total wall time.
2. Most modules have no step-level progress bar. Only `create_splits.py` and `extract_y_data.py` use one, but neither updates the bar description to show the current step name — the bar just ticks forward silently.
3. `tqdm.pandas()` bars used in several modules (`extract_diag_history`, `extract_discharge_history`, `extract_radiology`, `extract_triage_and_complaint`, `extract_labs`) show a generic label with no row counts or completion percentages visible from the log.
4. `embed_features.py` has no module-level step bar. The user sees 18 separate embedding tasks fire without any framing of overall embedding progress (e.g. "Embedding feature 3/18").
5. Several modules log nothing between "Loading X" and "Saved X" — the user has no visibility into intermediate steps (joining, filtering, imputing).
6. Log messages lack counts that would let the user detect data loss — e.g. how many rows were dropped, how many admissions were matched, how many were skipped due to missing source files.
7. The `_sources_unchanged` skip message in `preprocessing_utils.py` is clear, but skipped modules produce no banner in `run_pipeline.py`, making the log hard to parse when most modules skip.
8. No wall-time reporting anywhere — the user cannot tell which module was slow.

Apply all changes described below. Preserve all existing logic exactly — only add or improve logging and progress reporting.

---

## 1. `run_pipeline.py` — pipeline-level progress and timing

### 1a. Add overall pipeline progress bar

Replace the `_run_module` function and the execution loop in `main()` with the following pattern:

```python
import time

def _run_module(name: str, config: dict, idx: int, total: int) -> float:
    """Run a single module and return elapsed seconds."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("  STEP %d/%d — %s", idx, total, name)
    logger.info("=" * 70)
    t0 = time.time()
    module = _import_module(name)
    module.run(config)
    elapsed = time.time() - t0
    logger.info("  STEP %d/%d — %s completed in %.1fs", idx, total, name, elapsed)
    return elapsed
```

Update the execution loop in `main()`:

```python
t_pipeline_start = time.time()
n = len(modules_to_run)

for idx, module_name in enumerate(modules_to_run, start=1):
    config["FORCE_RERUN"] = args.force or (module_name in (args.force_modules or []))
    elapsed = _run_module(module_name, config, idx, n)

total_elapsed = time.time() - t_pipeline_start
logger.info("")
logger.info("=" * 70)
logger.info("  PIPELINE COMPLETE")
logger.info("  Modules run : %s", modules_to_run)
logger.info("  Total time  : %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
logger.info("=" * 70)
```

### 1b. Print a plan before execution starts

After `modules_to_run` is determined and before the execution loop, add:

```python
logger.info("")
logger.info("Pipeline plan — %d module(s) to run:", len(modules_to_run))
for idx, name in enumerate(modules_to_run, start=1):
    logger.info("  %d. %s", idx, name)
logger.info("")
```

---

## 2. `create_splits.py` — update step bar to show current step name

The bar currently uses `pbar.update(1)` without updating the description. Replace every `pbar.update(1)` with a two-liner that first sets the description to the completed step name, then advances:

```python
# Pattern to apply at every pbar.update(1) call:
pbar.set_description(f"create_splits — {steps[current_step_index]}")
pbar.update(1)
```

The simplest way to implement this is to track the step index explicitly:

```python
steps = [
    "Load admissions",
    "Build patient stats",
    "Split train / devtest",
    "Split dev / test",
    "Assign split labels",
    "Save data_splits.parquet",
]
with tqdm(total=len(steps), desc="create_splits", unit="step", dynamic_ncols=True) as pbar:
    # Step 0 — Load admissions
    pbar.set_description("create_splits — loading admissions")
    # ... load logic ...
    logger.info("  Loaded %d admissions for %d unique patients",
                len(admissions), admissions["subject_id"].nunique())
    pbar.update(1)

    # Step 1 — Build patient stats
    pbar.set_description("create_splits — computing patient outcome rates")
    # ... stats logic ...
    pbar.update(1)

    # Step 2 — Train/devtest split
    pbar.set_description("create_splits — stratified train/devtest split")
    # ... split logic ...
    pbar.update(1)

    # Step 3 — Dev/test split
    pbar.set_description("create_splits — stratified dev/test split")
    # ... split logic ...
    logger.info("  Patient counts — train: %d  dev: %d  test: %d",
                len(train_patients), len(dev_patients), len(test_patients))
    pbar.update(1)

    # Step 4 — Assign labels
    pbar.set_description("create_splits — assigning split labels to admissions")
    # ... assign logic ...
    pbar.update(1)

    # Step 5 — Save
    pbar.set_description("create_splits — saving data_splits.parquet")
    # ... save logic ...
    logger.info("  Saved %d rows (%d admissions, %d patients) to %s",
                len(splits_df), splits_df["hadm_id"].nunique(),
                splits_df["subject_id"].nunique(), output_path)
    pbar.update(1)
```

---

## 3. `extract_y_data.py` — update step bar to show current step name

Apply the same pattern as `create_splits.py`. Replace bare `pbar.update(1)` calls with `set_description` then `update`:

```python
steps = [
    "Load admissions",
    "Compute Y1 — in-hospital mortality",
    "Compute Y2 — 30-day readmission",
    "Save y_labels.parquet",
]
with tqdm(total=len(steps), desc="extract_y_data", unit="step", dynamic_ncols=True) as pbar:

    pbar.set_description("extract_y_data — loading admissions")
    # ... load logic ...
    logger.info("  Loaded %d admissions for %d patients",
                len(admissions), admissions["subject_id"].nunique())
    pbar.update(1)

    pbar.set_description("extract_y_data — computing Y1 (mortality)")
    # ... Y1 logic ...
    logger.info("  Y1 positive rate: %.2f%%  (%d deceased admissions)",
                100 * labels["y1_mortality"].mean(), labels["y1_mortality"].sum())
    pbar.update(1)

    pbar.set_description("extract_y_data — computing Y2 (30-day readmission)")
    # ... Y2 logic ...
    died_mask = labels["y1_mortality"] == 1
    logger.info("  Y2 positive rate (excl. deaths): %.2f%%  (%d readmitted)",
                100 * labels.loc[~died_mask, "y2_readmission"].mean(),
                int(labels.loc[~died_mask, "y2_readmission"].sum()))
    logger.info("  Y2 excluded (deceased): %d admissions", int(died_mask.sum()))
    pbar.update(1)

    pbar.set_description("extract_y_data — saving y_labels.parquet")
    # ... save logic ...
    logger.info("  Saved %d label rows to %s", len(labels), output_path)
    pbar.update(1)
```

---

## 4. `extract_demographics.py` — add a step-level progress bar

This module currently has no step bar — only scattered `logger.info` calls and a `tqdm` chunk iterator. Wrap the entire `run()` body in a step bar:

```python
steps = [
    "Load source tables",
    "Extract age and gender",
    "Extract vitals from OMR",
    "Extract vitals from chartevents (fallback)",
    "Merge vitals",
    "Compute imputation statistics",
    "Impute missing values",
    "Assemble demographic_vec",
    "Save demographics_features.parquet",
]
with tqdm(total=len(steps), desc="extract_demographics", unit="step", dynamic_ncols=True) as pbar:

    pbar.set_description("extract_demographics — loading source tables")
    # ... load patients, admissions, omr ...
    logger.info("  Loaded %d patients, %d admissions", len(patients), len(admissions))
    pbar.update(1)

    pbar.set_description("extract_demographics — extracting age and gender")
    # ... age/gender logic ...
    pbar.update(1)

    pbar.set_description("extract_demographics — extracting vitals from OMR")
    # ... OMR logic ...
    logger.info("  OMR: %d height/weight records for %d admissions",
                omr_hits, omr_admission_count)
    pbar.update(1)

    pbar.set_description("extract_demographics — extracting vitals from chartevents (fallback)")
    # ... chartevents logic ...
    logger.info("  chartevents fallback: %d height/weight records for %d admissions",
                chart_hits, chart_admission_count)
    pbar.update(1)

    pbar.set_description("extract_demographics — merging vitals")
    # ... merge logic ...
    logger.info("  After merge: height missing %.1f%%  weight missing %.1f%%  BMI missing %.1f%%",
                100 * df["height_missing"].mean(),
                100 * df["weight_missing"].mean(),
                100 * df["bmi_missing"].mean())
    pbar.update(1)

    pbar.set_description("extract_demographics — computing imputation statistics (train split only)")
    # ... stats logic ...
    logger.info("  Imputation stats computed for %d strata", n_strata)
    pbar.update(1)

    pbar.set_description("extract_demographics — imputing missing height and weight")
    # ... imputation logic ...
    logger.info("  Imputed %d height values, %d weight values", n_imputed_h, n_imputed_w)
    pbar.update(1)

    pbar.set_description("extract_demographics — assembling demographic_vec")
    # ... assemble logic ...
    pbar.update(1)

    pbar.set_description("extract_demographics — saving demographics_features.parquet")
    # ... save logic ...
    logger.info("  Saved %d rows to %s", len(out_df), output_path)
    pbar.update(1)
```

---

## 5. `extract_diag_history.py` — add a step-level progress bar

```python
steps = [
    "Load source tables",
    "Attach ICD long_title to diagnoses",
    "Build prior-visit text per admission",
    "Save diag_history_features.parquet",
]
with tqdm(total=len(steps), desc="extract_diag_history", unit="step", dynamic_ncols=True) as pbar:

    pbar.set_description("extract_diag_history — loading source tables")
    # ... load diagnoses_icd, d_icd_diagnoses, admissions ...
    logger.info("  Loaded %d diagnosis records for %d admissions",
                len(diagnoses), diagnoses["hadm_id"].nunique())
    pbar.update(1)

    pbar.set_description("extract_diag_history — attaching ICD long_title")
    # ... merge d_icd logic ...
    n_unmapped = diagnoses["long_title"].isna().sum()
    logger.info("  ICD merge: %d unmapped codes (%.1f%%)",
                n_unmapped, 100 * n_unmapped / len(diagnoses))
    pbar.update(1)

    pbar.set_description("extract_diag_history — building prior-visit text")
    # ... prior visit logic + tqdm.pandas ...
    logger.info("  Built diagnosis history for %d admissions (%d with prior visits)",
                len(out_df), int((out_df["diag_history_text"] != "").sum()))
    pbar.update(1)

    pbar.set_description("extract_diag_history — saving diag_history_features.parquet")
    # ... save ...
    logger.info("  Saved %d rows to %s", len(out_df), output_path)
    pbar.update(1)
```

---

## 6. `extract_discharge_history.py` — add a step-level progress bar

```python
steps = [
    "Load discharge notes",
    "Clean note text",
    "Load admissions",
    "Build prior-visit text per admission",
    "Save discharge_history_features.parquet",
]
with tqdm(total=len(steps), desc="extract_discharge_history", unit="step", dynamic_ncols=True) as pbar:

    pbar.set_description("extract_discharge_history — loading discharge notes")
    # ... load notes ...
    logger.info("  Loaded %d discharge notes for %d admissions",
                len(notes), notes["hadm_id"].nunique())
    n_null_hadm = notes["hadm_id"].isna().sum()
    if n_null_hadm:
        logger.info("  Dropped %d notes with null hadm_id (%.1f%%)",
                    n_null_hadm, 100 * n_null_hadm / (len(notes) + n_null_hadm))
    pbar.update(1)

    pbar.set_description("extract_discharge_history — cleaning note text")
    # ... tqdm.pandas clean ...
    pbar.update(1)

    pbar.set_description("extract_discharge_history — loading admissions")
    # ... load admissions ...
    logger.info("  Loaded %d admissions", len(admissions))
    pbar.update(1)

    pbar.set_description("extract_discharge_history — building prior-visit text")
    # ... prior visit concat logic ...
    logger.info("  Built discharge history for %d admissions (%d with prior notes)",
                len(out_df), int((out_df["discharge_history_text"] != "").sum()))
    pbar.update(1)

    pbar.set_description("extract_discharge_history — saving discharge_history_features.parquet")
    # ... save ...
    logger.info("  Saved %d rows to %s", len(out_df), output_path)
    pbar.update(1)
```

---

## 7. `extract_triage_and_complaint.py` — add a step-level progress bar

```python
steps = [
    "Load triage table",
    "Resolve hadm_id via edstays",
    "Resolve hadm_id via intime fallback",
    "Build triage text",
    "Extract chief complaint",
    "Save triage and complaint parquets",
]
with tqdm(total=len(steps), desc="extract_triage_and_complaint", unit="step", dynamic_ncols=True) as pbar:

    pbar.set_description("extract_triage_and_complaint — loading triage table")
    # ... load triage ...
    logger.info("  Loaded %d triage rows", len(triage))
    pbar.update(1)

    pbar.set_description("extract_triage_and_complaint — resolving hadm_id via edstays")
    # ... edstays join ...
    n_linked = int(triage["hadm_id"].notna().sum())
    logger.info("  After edstays join: %d / %d rows have hadm_id (%.1f%%)",
                n_linked, len(triage), 100 * n_linked / len(triage))
    pbar.update(1)

    pbar.set_description("extract_triage_and_complaint — resolving hadm_id via intime fallback")
    # ... fallback linkage ...
    n_after = int(triage["hadm_id"].notna().sum())
    n_dropped = len(triage) - int((triage["hadm_id"].notna()).sum())
    logger.info("  After fallback: %d additional rows linked", n_after - n_linked)
    logger.info("  Dropping %d rows with no resolvable hadm_id (non-admitted ED visits)", n_dropped)
    pbar.update(1)

    pbar.set_description("extract_triage_and_complaint — building triage text")
    # ... tqdm.pandas build triage text ...
    logger.info("  Built triage text for %d admissions", len(triage_out))
    pbar.update(1)

    pbar.set_description("extract_triage_and_complaint — extracting chief complaint")
    # ... chief complaint logic ...
    n_empty_cc = int((complaint_out["chief_complaint_text"] == "").sum())
    logger.info("  Chief complaint: %d admissions with text, %d empty",
                len(complaint_out) - n_empty_cc, n_empty_cc)
    pbar.update(1)

    pbar.set_description("extract_triage_and_complaint — saving parquets")
    # ... save ...
    logger.info("  Saved triage: %d rows  |  chief complaint: %d rows",
                len(triage_out), len(complaint_out))
    pbar.update(1)
```

---

## 8. `extract_labs.py` — add a step-level progress bar and improve chunk logging

```python
steps = [
    "Load d_labitems and admissions",
    "Stream and filter labevents",
    "Apply admission window filter",
    "Build lab text lines",
    "Sort and save labs_features.parquet",
]
with tqdm(total=len(steps), desc="extract_labs", unit="step", dynamic_ncols=True) as pbar:

    pbar.set_description("extract_labs — loading d_labitems and admissions")
    # ... load d_labitems, admissions ...
    logger.info("  d_labitems: %d items across %d fluids and %d categories",
                len(d_labitems),
                d_labitems["fluid"].nunique(),
                d_labitems["category"].nunique())
    logger.info("  Admissions: %d rows loaded for window filtering", len(admissions))
    pbar.update(1)

    pbar.set_description("extract_labs — streaming labevents chunks")
    # ... chunk loop (existing tqdm chunk iterator kept as-is) ...
    # Add a summary log after the loop:
    total_rows_before_filter = sum(len(c) for c in all_chunks)
    logger.info("  labevents: %d rows retained after streaming and hadm_id handling",
                total_rows_before_filter)
    pbar.update(1)

    pbar.set_description("extract_labs — applying admission window filter")
    # ... window filter logic ...
    n_after_window = len(labs)
    logger.info("  After window filter (%s): %d rows for %d admissions",
                f"{lab_window_hours}h" if lab_window_hours else "full",
                n_after_window, labs["hadm_id"].nunique())
    pbar.update(1)

    pbar.set_description("extract_labs — building lab text lines")
    # ... tqdm.pandas build text lines ...
    pbar.update(1)

    pbar.set_description("extract_labs — saving labs_features.parquet")
    # ... sort and save ...
    logger.info("  Saved %d rows (%d unique admissions, %d unique itemids) to %s",
                len(out_df), out_df["hadm_id"].nunique(),
                out_df["itemid"].nunique(), output_path)
    pbar.update(1)
```

Also improve the per-chunk logging inside the streaming loop. Replace the existing chunk-level log with:

```python
# Inside the chunk loop — replace or supplement existing logger.info:
logger.info(
    "  Chunk %d: %d rows read  |  %d null hadm_id (%s)  |  %d retained after filters",
    i,
    raw_chunk_len,
    null_hadm_count,
    f"strategy: {hadm_linkage_strategy}",
    len(chunk),
)
```

---

## 9. `extract_radiology.py` — add a step-level progress bar

```python
steps = [
    "Load radiology notes",
    "Clean note text",
    "Load admissions",
    "Filter to admission window",
    "Select most recent note per admission",
    "Save radiology_features.parquet",
]
with tqdm(total=len(steps), desc="extract_radiology", unit="step", dynamic_ncols=True) as pbar:

    pbar.set_description("extract_radiology — loading radiology notes")
    # ... load notes ...
    logger.info("  Loaded %d radiology notes for %d admissions",
                len(notes), notes["hadm_id"].nunique())
    n_null_hadm = notes["hadm_id"].isna().sum()
    if n_null_hadm:
        logger.info("  Dropped %d notes with null hadm_id (%.1f%%)",
                    n_null_hadm, 100 * n_null_hadm / (len(notes) + n_null_hadm))
    pbar.update(1)

    pbar.set_description("extract_radiology — cleaning note text")
    # ... tqdm.pandas clean ...
    pbar.update(1)

    pbar.set_description("extract_radiology — loading admissions")
    # ... load admissions ...
    logger.info("  Loaded %d admissions", len(admissions))
    pbar.update(1)

    pbar.set_description("extract_radiology — filtering to admission window")
    # ... window filter ...
    n_in_window = int(in_window.sum())
    logger.info("  Notes in admission window: %d / %d (%.1f%% retained)",
                n_in_window, len(notes_merged), 100 * n_in_window / max(len(notes_merged), 1))
    pbar.update(1)

    pbar.set_description("extract_radiology — selecting most recent note per admission")
    # ... most_recent logic ...
    n_with_note = int(out_df["radiology_text"].notna().sum() &
                     (out_df["radiology_text"] != "").any())
    logger.info("  Admissions with radiology note: %d / %d",
                int((out_df["radiology_text"] != "").sum()), len(out_df))
    pbar.update(1)

    pbar.set_description("extract_radiology — saving radiology_features.parquet")
    # ... save ...
    logger.info("  Saved %d rows to %s", len(out_df), output_path)
    pbar.update(1)
```

---

## 10. `embed_features.py` — add a module-level step bar showing 18 total embedding tasks

The user currently has no visibility into overall embedding progress across all 18 features. Add a top-level progress bar:

```python
# Before the text feature loop, compute total work:
n_text_features = len([f for f in _TEXT_FEATURES
                        if os.path.exists(os.path.join(features_dir, f[0]))])
n_lab_groups = len(lab_panel_config) if lab_panel_config_loaded else 0
total_tasks = n_text_features + n_lab_groups

with tqdm(total=total_tasks, desc="embed_features", unit="feature", dynamic_ncols=True) as pbar:

    # --- Non-lab text features ---
    for (input_filename, text_col, output_filename, embedding_col) in _TEXT_FEATURES:
        pbar.set_description(f"embed_features — {embedding_col}")
        # ... existing embed logic ...
        logger.info("  [%d/%d] %s: %d texts embedded (dim=%d)",
                    pbar.n + 1, total_tasks, embedding_col, len(texts), embeddings.shape[1])
        pbar.update(1)

    # --- Lab group features ---
    for group_name, itemids in lab_panel_config.items():
        pbar.set_description(f"embed_features — lab_{group_name}_embedding")
        # ... existing lab group embed logic ...
        n_non_empty = int((group_text["text"] != "").sum())
        logger.info("  [%d/%d] lab_%s: %d admissions (%d with events, %d zero vectors)",
                    pbar.n + 1, total_tasks, group_name,
                    len(texts_group), n_non_empty, len(texts_group) - n_non_empty)
        pbar.update(1)
```

Also improve the BERT model loading log — this is the slowest single step and currently produces only one line:

```python
# Before
logger.info("Loading BERT model '%s' on device '%s'…", model_name, device)

# After
logger.info("Loading BERT model: %s", model_name)
logger.info("  Device: %s", device)
logger.info("  Max token length: %d", max_length)
logger.info("  Batch size: %d", batch_size)
logger.info("  Total embedding tasks: %d (%d text features + %d lab groups)",
            total_tasks, n_text_features, n_lab_groups)
```

---

## 11. `combine_dataset.py` — add a step-level progress bar

```python
steps = [
    "Load data_splits.parquet",
    "Merge y_labels",
    "Merge feature parquets",
    "Merge embedding parquets",
    "Validate and save final_cdss_dataset.parquet",
]
with tqdm(total=len(steps), desc="combine_dataset", unit="step", dynamic_ncols=True) as pbar:

    pbar.set_description("combine_dataset — loading data_splits.parquet")
    # ... load splits ...
    logger.info("  Splits: %d admissions (%d train  %d dev  %d test)",
                len(base),
                int((base["split"] == "train").sum()),
                int((base["split"] == "dev").sum()),
                int((base["split"] == "test").sum()))
    pbar.update(1)

    pbar.set_description("combine_dataset — merging y_labels")
    # ... merge labels ...
    n_missing_y1 = int(base["y1_mortality"].isna().sum())
    if n_missing_y1:
        logger.warning("  %d admissions missing Y1 after label merge", n_missing_y1)
    pbar.update(1)

    pbar.set_description("combine_dataset — merging feature parquets")
    # ... existing features loop (keep inner tqdm or remove it) ...
    pbar.update(1)

    pbar.set_description("combine_dataset — merging embedding parquets")
    # ... existing embeddings loop ...
    logger.info("  Merged %d embedding files", len(embedding_files))
    pbar.update(1)

    pbar.set_description("combine_dataset — saving final_cdss_dataset.parquet")
    # ... validate split col, save ...
    logger.info("  Final dataset: %d rows × %d columns", base.shape[0], base.shape[1])
    logger.info("  Columns: %s", list(base.columns))
    pbar.update(1)
```

---

## 12. `build_lab_panel_config.py` — improve log messages

The per-group detail logging (one `logger.info` per group × 13 groups) clutters the log. Replace the per-group loop with a summary table:

```python
# Before
logger.info("Lab panel config: %d groups, total %d itemids", ...)
for group_name, items in sorted(groups.items()):
    logger.info("  %s: %d itemids", group_name, len(items))

# After
logger.info("Lab panel config: %d groups, %d total itemids",
            len(groups), sum(len(v) for v in groups.values()))
logger.info("  %-45s  %s", "Group", "Item count")
logger.info("  " + "-" * 55)
for group_name in sorted(groups):
    logger.info("  %-45s  %d", group_name, len(groups[group_name]))
logger.info("Saved lab panel config → %s", output_path)
```

Also add a start log at the top of `run()`:

```python
logger.info("Building lab panel config from d_labitems…")
```

---

## 13. `preprocessing_utils.py` — improve skip and hash messages

The skip message currently says "All sources unchanged and outputs exist — skipping." with no indication of what was skipped. Improve it:

```python
# Before
logger.info("[%s] All sources unchanged and outputs exist — skipping.", module_name)

# After
logger.info("[%s] Sources unchanged — skipping (outputs already exist).", module_name)
```

Also add output file names to the skip message so the user can confirm which files were skipped:

```python
output_names = [os.path.basename(p) for p in output_paths]
logger.info("[%s] Skipping — outputs up to date: %s", module_name, ", ".join(output_names))
```

And improve the "will rerun" messages to indicate whether it's a first run or a change:

```python
# For output missing:
logger.info("[%s] Output not found (%s) — will run.", module_name, os.path.basename(p))

# For source changed:
logger.info("[%s] Source file changed (%s) — will rerun.", module_name, os.path.basename(p))
```

---

## Conventions to preserve across all changes

- **Never use `print()`** in pipeline modules (`print` is only acceptable in `inspect_data.py`)
- **All new `logger` calls use `logger.info` or `logger.warning`** — never `logger.debug` (the default log level is INFO throughout the pipeline)
- **All `tqdm` bars use `dynamic_ncols=True`** so they resize correctly in narrow terminals
- **`set_description()` must be called before the work it labels**, not after, so the user sees what is happening before it happens
- **Counts in log messages must be formatted with commas** for readability (`%d` → `f"{n:,}"` or `logger.info("  %d rows", n)` — `%d` is fine for moderate numbers, but use `f"{n:,}"` for anything that could exceed 100,000)
- **Do not remove any existing `logger.info` or `logger.warning` calls** — only add to them or improve their message text
- **Do not remove any existing `tqdm` bars** — only add step bars above them or improve their `desc`
