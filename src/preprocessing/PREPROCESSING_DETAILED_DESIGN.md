# CDSS-ML Preprocessing — Detailed Design

## Table of Contents

1. [Identifier Hierarchy](#1-identifier-hierarchy)
2. [Data Splits — Implementation](#2-data-splits--implementation)
3. [Module Implementation](#3-module-implementation)
   - [build_lab_panel_config.py](#build_lab_panel_configpy)
   - [create_splits.py](#create_splitspy)
   - [extract_demographics.py](#extract_demographicspy)
   - [extract_diag_history.py](#extract_diag_historypy)
   - [extract_discharge_history.py](#extract_discharge_historypy)
   - [extract_triage_and_complaint.py](#extract_triage_and_complaintpy)
   - [extract_labs.py](#extract_labspy)
   - [extract_microbiology.py](#extract_microbiologypy)
   - [extract_radiology.py](#extract_radiologypy)
   - [extract_y_data.py](#extract_y_datapy)
   - [embed_features.py](#embed_featurespy)
   - [combine_dataset.py](#combine_datasetpy)
4. [Embedding Implementation Detail](#4-embedding-implementation-detail)
   - [Model](#model)
   - [Mean Pooling](#mean-pooling)
   - [Per-Feature Token Length Caps](#per-feature-token-length-caps)
   - [Admission-Slice Batching](#admission-slice-batching)
   - [Multi-GPU Execution Within a Slice](#multi-gpu-execution-within-a-slice)
   - [GPU Load Balancing](#gpu-load-balancing)
   - [Feature-Level Resume](#feature-level-resume)
   - [Record-Level Resume](#record-level-resume)
5. [Final Dataset Assembly](#5-final-dataset-assembly)
6. [Configuration Reference](#6-configuration-reference)
7. [Memory Requirements](#7-memory-requirements)

---

## 1. Identifier Hierarchy

MIMIC-IV uses three nested identifiers:

```
subject_id   (patient — persistent across all visits)
  └── hadm_id    (hospital admission — the unit of prediction)
        └── stay_id    (ICU stay within an admission)
```

**`hadm_id` is the primary join key** throughout the pipeline. All feature parquets carry `(subject_id, hadm_id)` as their key columns.

**`stay_id`** appears only indirectly in Phase 1 — via `chartevents` in the demographics fallback (joined on `hadm_id`). It becomes critical in the MDP phase for modelling clinical interventions.

### Missing `hadm_id`

Several tables contain records with null `hadm_id` (~10–20% of `labevents`, lower rates in `note` and `chartevents`). Handling is configurable via `HADM_LINKAGE_STRATEGY`:

| Strategy             | Behaviour                                                                              |
|----------------------|----------------------------------------------------------------------------------------|
| `"drop"` *(default)* | Exclude records with null `hadm_id`. Count and percentage logged per module.           |
| `"link"`             | Time-window `hadm_id` linkage; assign closest match; log to `hadm_linkage_stats.json`. |

**`"link"` detail:** match `charttime` against `[admittime − tolerance, dischtime + tolerance]` per
`subject_id` (tolerance = `HADM_LINKAGE_TOLERANCE_HOURS` h). Assign if exactly one match; assign
closest if multiple; drop if none.

---

## 2. Data Splits — Implementation

`create_splits.py` performs a patient-level stratified 3-way split:

```
1. Load admissions → compute hospital_expire_flag per subject_id
2. Build patient table: subject_id, has_death (1 if any admission is Y1=1)
3. sklearn.model_selection.train_test_split with stratify=has_death, seed=42
4. Output: data_splits.parquet — one row per hadm_id with split column
```

All admissions of a patient go to the same split. Stratification on Y1 only — Y2 would create degenerate strata because deceased patients always have Y2=NaN.

---

## 3. Module Implementation

### `build_lab_panel_config.py`

Derives lab groups from `d_labitems` and writes `classifications/lab_panel_config.yaml`.

**Algorithm:**
1. Load `d_labitems` — retain `itemid`, `fluid`, `category`
2. Drop artefact rows where `fluid` in `["I", "Q", "fluid"]`
3. For each row, assign to a group using `_FLUID_CATEGORY_MAP` if the `(fluid, category)` pair has an explicit mapping
4. For fluids in `_SINGLE_GROUP_FLUIDS` (Ascites, Pleural, Cerebrospinal Fluid, Bone Marrow, Joint Fluid, Stool), all itemids regardless of category are merged under a single group key; `Cerebrospinal Fluid` is mapped to the key `csf` via `_SINGLE_GROUP_NAME_OVERRIDES`
5. Blood gas panels for non-blood fluids (`Urine`, `Other Body Fluid`, `Fluid`) are folded into their closest existing group rather than creating new groups
6. Write YAML: `{group_name: [itemid, ...], ...}`

**Output — 13 groups:**

| # | Group name | Fluid | Category | Items | Emb. dim |
|---|-----------|-------|----------|-------|---------|
| 1 | `blood_gas` | Blood | Blood Gas | 34 | 768 |
| 2 | `blood_chemistry` | Blood | Chemistry | 267 | 768 |
| 3 | `blood_hematology` | Blood | Hematology | 198 | 768 |
| 4 | `urine_chemistry` | Urine | Chemistry | 72 | 768 |
| 5 | `urine_hematology` | Urine | Hematology + Blood Gas | 65 | 768 |
| 6 | `other_body_fluid_chemistry` | Other Body Fluid | Chemistry | 65 | 768 |
| 7 | `other_body_fluid_hematology` | Other Body Fluid | Hematology + Blood Gas | 62 | 768 |
| 8 | `ascites` | Ascites | Chemistry + Hematology | 40 | 768 |
| 9 | `pleural` | Pleural | Chemistry + Hematology | 39 | 768 |
| 10 | `csf` | Cerebrospinal Fluid | Chemistry + Hematology | 34 | 768 |
| 11 | `bone_marrow` | Bone Marrow | Hematology | 44 | 768 |
| 12 | `joint_fluid` | Joint Fluid | Blood Gas + Chemistry + Hematology | 38 | 768 |
| 13 | `stool` | Stool | Chemistry + Hematology | 18 | 768 |

---

### `extract_demographics.py`

Produces `demographic_vec = [age, gender, height_cm, weight_kg, bmi, height_missing, weight_missing, bmi_missing]`.

**Age:**
```
age = anchor_age + (year(admittime) - anchor_year)
```
Corrects for MIMIC-IV's year-shift anonymisation.

**Gender:** `M = 1.0`, `F = 0.0`.

**Height / Weight source priority:**

| Priority | Source | Notes |
|----------|--------|-------|
| 1 | `omr.result_name` contains "Height" / "Weight" | `chartdate ≤ admittime`. Inches → cm (×2.54). Lbs → kg (×0.453592). |
| 2–5 | `chartevents` itemids: height 226707, 226730; weight 226512, 224639, 226531, 226846 | First value within admission window. Unit conversion per itemid. |

Plausibility filters: height 50–250 cm, weight 20–400 kg. Values outside range discarded before imputation.

After merge (pre-imputation): height missing 41.5%, weight missing 29.7%, BMI missing 43.2%.

**Imputation:**
- Missingness flags set **before** any imputation
- Method: sample from `N(mean, std)` per `(age_bin × gender)` stratum
- Age bins: `[18–29, 30–44, 45–64, 65–74, 75+]` — 11 strata (5 age × 2 gender, minus one absent stratum)
- Statistics computed on **train split only**, saved to `imputation_stats.json`
- Applied identically to dev/test; fallback to global stats if stratum absent
- Result: 226,873 height values imputed, 162,179 weight values imputed

**BMI:** Use OMR `bmi` value if present. Derive as `weight_kg / (height_cm / 100)²` if absent. **Never imputed independently.**

**Memory:** `chartevents` is streamed in 500k-row chunks — never loaded fully into memory.

---

### `extract_diag_history.py`

Builds a structured text block of prior-visit ICD diagnoses.

**Algorithm:**
1. Load `diagnoses_icd` + join `d_icd_diagnoses` for `long_title`
2. Load current admission `admittime` for each `hadm_id`
3. For each target admission, collect all prior admissions with `admittime < current admittime` for the same `subject_id`
4. Sort prior admissions chronologically; format as text block

**Text format:**
```
Past Diagnoses:

Visit (2018-03-12):
Chronic kidney disease, stage 3
Hypertension, unspecified

Visit (2019-07-24):
Acute kidney injury
```

First-time admissions (no prior visits) → empty string → zero vector at embedding time.

---

### `extract_discharge_history.py`

Concatenates prior-visit discharge notes with dated headers.

**Algorithm:**
1. Load `note/discharge`
2. For each target admission, collect prior discharge notes (`charttime < current admittime`)
3. Clean each note: strip everything before the first `"Allergies:"` marker (removes boilerplate header)
4. Concatenate chronologically with dated section headers

**Text format:**
```
Prior Discharge Summary (2018-03-12):
Allergies: Penicillin
[clinical note body...]

Prior Discharge Summary (2019-07-24):
Allergies: None known
[clinical note body...]
```

Admissions with no prior discharge notes → empty string → zero vector at embedding time.

---

### `extract_triage_and_complaint.py`

Extracts two features from the ED visit: a structured triage text (F4) and raw chief complaint (F5).

**`hadm_id` resolution for ED visits:**
1. Primary: join `edstays` on `stay_id → hadm_id` (direct linkage)
2. Fallback: for `subject_id` with no direct link, find the inpatient admission where `admittime` is closest to and ≥ `intime`
3. Non-admitted ED visits (no matching `hadm_id`) are excluded

**F4 — Triage text template:**
```
Triage assessment: temperature {T}°C, heart rate {HR} bpm,
respiratory rate {RR} breaths/min, O2 saturation {O2}%,
blood pressure {SBP}/{DBP} mmHg, pain score {pain}/10, acuity level {acuity}.
```
Missing fields rendered as `"N/A"`.

**F5 — Chief complaint:**
- Primary source: `triage.chiefcomplaint`
- Fallback: `chartevents` itemid 223112
- No cleaning or templating — raw text only

---

### `extract_labs.py`

Produces a long-format parquet of all lab events within the admission window.

**Algorithm:**
1. Stream `labevents` in 500k-row chunks
2. For each chunk: join `admissions` on `hadm_id` to get `admittime`
3. Filter: `(admittime − ED_LOOKBACK_HOURS) ≤ charttime ≤ (admittime + LAB_ADMISSION_WINDOW)`. The lookback window (default 24h) captures pre-admission ED labs drawn before formal inpatient `admittime`. Null `hadm_id` events in the lookback window are resolved via `pd.merge_asof()` with `direction="forward"` when `HADM_LINKAGE_STRATEGY` is `"link"`, replacing the previous `iterrows()` row-by-row approach.
4. Format each retained event as a text line
5. Concatenate all chunks → write `labs_features.parquet`

**Text line format:**
```
[HH:MM] {label}: {value} {unit} (ref: lower-upper) [ABNORMAL]
```

- `[HH:MM]` = elapsed hours:minutes since `admittime` (not wall clock time)
- `valuenum` formatted with Python's `{:g}` specifier when available (e.g. Troponin 0.004 → `"0.004"`, Glucose 142.0 → `"142"`); preserves clinically significant precision and drops trailing zeros; `value` text field otherwise
- `(ref: lower-upper)` omitted when either bound is null
- `[ABNORMAL]` appended when `flag == "abnormal"` **or** `valuenum` outside `[ref_range_lower, ref_range_upper]`

**Example (blood_chemistry group):**
```
[00:14] Glucose: 8.2 mmol/L [ABNORMAL]
[00:14] Sodium: 138 mEq/L
[00:14] Potassium: 6.1 mEq/L [ABNORMAL]
[08:32] Creatinine: 1.8 mg/dL [ABNORMAL]
```

**Admission window:** `LAB_ADMISSION_WINDOW` — integer hours or `"full"`. Default: 24.

**Lab groups:** 13 groups derived from `d_labitems` — see `build_lab_panel_config.py` for full group table.

---

### `extract_microbiology.py`

Produces 37 text parquets — one per microbiology panel — from `microbiologyevents`. Each parquet contains one row per admission with a `text` column holding the aggregated panel text for that admission.

**Algorithm:**
1. Load `micro_panel_config.yaml` from `MICRO_PANEL_CONFIG_PATH` — panel combo dict, excluded tests, excluded spec_types, comment cleaning rules
2. Stream `microbiologyevents` (or load fully if memory permits)
3. Drop excluded tests and excluded spec_types
4. Apply null `hadm_id` strategy (`MICRO_NULL_HADM_STRATEGY`):
   - `"drop"`: exclude all rows with null `hadm_id`, log count
   - `"link"`: time-window linkage against admissions; classify as linked / ambiguous / unresolvable; save audit to `micro_linkage_stats.json`
5. Join `admissions` on `hadm_id` to get `admittime` (and `dischtime` if `MICRO_WINDOW_HOURS = "full_admission"`)
6. Apply time window filter based on `MICRO_WINDOW_HOURS`:
   - Integer N: `(admittime − ED_LOOKBACK_HOURS) ≤ charttime ≤ (admittime + N hours)`. Default lookback: 24h (`ED_LOOKBACK_HOURS` config key).
   - `"full_admission"`: `(admittime − ED_LOOKBACK_HOURS) ≤ charttime ≤ dischtime`
7. Assign panel via `(test_name.strip(), spec_type_desc.strip())` lookup; log unassigned combos
8. Clean `comments` column via `clean_comment()` (see comment cleaning spec)
9. Build per-event text string (Cases A/B/C — see below)
10. Group by `(subject_id, hadm_id, panel)` — deduplicate, sort by `charttime`, concatenate with ` | `, tokenise and truncate to `BERT_MAX_LENGTH`
11. Write one parquet per panel to `FEATURES_DIR/micro_<panel_name>.parquet`

**Per-event text construction:**

| Case | Condition | Format |
|------|-----------|--------|
| A | `org_name` present | `{test_name} [{spec_type_desc}]: {org_name} \| {susc_string} \| {cleaned_comment}` |
| B | `org_name` null, comment present | `{test_name} [{spec_type_desc}]: {cleaned_comment}` |
| C | `org_name` null, comment null | `{test_name} [{spec_type_desc}]: pending` |

**Susceptibility string:** For each `(org_name, ab_name)` pair within an admission+panel, select interpretation by priority R > S > I. I appears only when no R or S exists. Antibiotics listed in source data order. Example: `OXACILLIN:R, VANCOMYCIN:S, CLINDAMYCIN:I`

**Comment cleaning:** Six-step pipeline — null check, discard-entirely prefixes, trigger-word truncation, sentence splitting (first 2 sentences, not on `:`), artifact cleanup, hard truncation to `MICRO_COMMENT_MAX_CHARS`. Full specification in `microbiology_comments_cleaning_spec.md`. Rules configurable in `micro_panel_config.yaml` under `comment_cleaning`.

**Empty panels:** Admissions with no events in a panel within the time window receive an empty string — produces a zero vector at embedding time.

**Helper module:** `build_micro_text.py` — contains `clean_comment()`, `build_event_text()`, and `aggregate_panel_text()` functions, imported by `extract_microbiology.py`.

---

Selects the most recent radiology note within the current admission window.

**Algorithm:**
1. Load `note/radiology` — drop null `hadm_id`
2. Filter to notes within admission window (`admittime ≤ charttime ≤ dischtime`)
3. Per `hadm_id`, select the note with the latest `charttime`
4. Clean: strip everything before the first `"EXAMINATION:"` marker

Admissions with no radiology note → empty string → zero vector at embedding time.

---

### `extract_y_data.py`

Produces `y_labels.parquet` with Y1 and Y2 per admission.

**Y1:** Direct from `admissions.hospital_expire_flag`. Values: 0 (survived) or 1 (died in hospital).

**Y2:** For each admission where Y1 = 0, check whether any subsequent admission for the same `subject_id` has `admittime ≤ dischtime + 30 days`. Y2 = 1 if yes, 0 if no. **Y2 = NaN for all admissions where Y1 = 1** (deceased patients cannot be readmitted).

**Null `dischtime` handling:** Surviving admissions (Y1=0) with null `dischtime` cannot have a valid Y2 computed. These are identified, logged with count and percentage per split, and excluded from the output entirely — they are not assigned NaN. In MIMIC-IV v3.1 no such records exist (verified by EDA), but the exclusion step is retained as a defensive check against future data versions.

**Required assertions** — both must pass before writing output:

```python
assert df.loc[df['y1_mortality'] == 1, 'y2_readmission'].isna().all(), \
    "Deceased patients must have Y2=NaN"

assert df.loc[df['y1_mortality'] == 0, 'y2_readmission'].notna().all(), \
    "Surviving patients must have a valid Y2 — check dischtime upstream"
```

The second assertion enforces that no surviving patient has a missing Y2. A surviving patient with missing Y2 is a data quality error, not a deceased patient, and must not be silently absorbed into the NaN masking.

---

### `embed_features.py`

Embeds all 55 text features using Clinical_ModernBERT. Full implementation details in [Section 4](#4-embedding-implementation-detail).

**CLI entry point:**
```bash
python embed_features.py --config config/preprocessing.yaml \
                         --slice-index 0
```
Also callable as `run(config, slice_index)` from `run_pipeline.py`.

`--slice-index` specifies which admission slice this job processes (0-based). The total number of slices is derived at runtime: `n_slices = ceil(total_admissions / (BERT_SLICE_SIZE_PER_GPU × n_gpus))`. Omitting `--slice-index` defaults to `0`. To run the full dataset in a single job, set `BERT_SLICE_SIZE_PER_GPU` large enough to cover all admissions.

**Feature inputs:**

| Feature | Input parquet | Text column |
|---------|---------------|-------------|
| `diag_history` | `diag_history_features.parquet` | `text` |
| `discharge_history` | `discharge_history_features.parquet` | `text` |
| `triage` | `triage_features.parquet` | `text` |
| `chief_complaint` | `chief_complaint_features.parquet` | `text` |
| `radiology` | `radiology_features.parquet` | `text` |
| `lab_{group}` ×13 | `labs_features.parquet` (filtered by group's itemid list) | `lab_text_line` (concatenated per hadm_id) |
| `micro_{panel}` ×37 | `micro_{panel_name}.parquet` | `text` |

**Labs processing:** `labs_df` is filtered to the current slice's `hadm_id` set, then scanned **once** before the 13-group embedding loop to build per-group text maps.

**Microbiology processing:** Each of the 37 micro panel parquets is loaded and filtered to the current slice's `hadm_id` set independently. Text is already aggregated per admission — no further grouping needed.

---

### `combine_dataset.py`

Builds `final_cdss_dataset.parquet` from `data_splits.parquet` as the admission universe.

**Algorithm:**
1. Start with `data_splits.parquet`
2. Left-join `y_labels.parquet` on `hadm_id`
3. Left-join `demographics_features.parquet` on `hadm_id`
4. Scan `EMBEDDINGS_DIR` for all `*.parquet` files dynamically
5. Left-join each embedding parquet on `hadm_id`
6. Build canonical column order from config files via `_build_canonical_columns(config)`: reads lab panel names from `LAB_PANEL_CONFIG_PATH` and microbiology panel names from `MICRO_PANEL_CONFIG_PATH` in insertion order. Column order: `[metadata] + [labels] + [demographic_vec] + [F2-F5 fixed embeddings] + [13 lab group embeddings] + [radiology] + [37 micro panel embeddings]`.
7. Assert all expected columns are present — raises `ValueError` listing any missing columns if extraction or embedding is incomplete.
8. Reorder `df` to canonical order and write `final_cdss_dataset.parquet`.

All joins are **left joins** — admissions missing a non-lab/micro feature receive null for that column. Lab and microbiology embedding columns are always a 768-float array (zero vector for admissions with no events — never null).

**Intentionally excluded:** `labs_features.parquet` (superseded by per-group embedding parquets), all raw text parquets including `micro_<panel>.parquet` (superseded by embedding parquets).

---

## 4. Embedding Implementation Detail

### Model

| Property | Value |
|----------|-------|
| Model identifier | `Simonlee711/Clinical_ModernBERT` |
| Pre-training corpus | PubMed abstracts, MIMIC-IV clinical notes, medical ontologies |
| Context window | 8,192 tokens (RoPE positional encoding + Flash Attention) |
| Hidden size | 768 |
| Load call | `BertModel.from_pretrained(model_name, add_pooling_layer=False)` |
| HF logging | `hf_logging.set_verbosity_error()` — suppresses non-error output |

The model is used as a **frozen feature extractor** — weights are never updated.

---

### Mean Pooling

```
token hidden states (final layer):
  t₁    t₂    t₃   ...   tₙ   [PAD] [PAD]
   │     │     │           │
   └─────┴─────┴─────...───┘
              mean
               │
               ▼
         768-d embedding vector
```

Mean pooling is chosen over `[CLS]` because Clinical_ModernBERT is not fine-tuned — the `[CLS]` token does not develop task-specific semantics in this regime. Mean pooling ensures every content token contributes equally, which is important for long texts (600-token discharge notes, 40-measurement lab timelines).

**Implementation:**
```
attention_mask = batch["attention_mask"]           # (B, L)
hidden = model(**batch).last_hidden_state          # (B, L, 768)
mask = attention_mask.unsqueeze(-1).float()        # (B, L, 1)
embedding = (hidden * mask).sum(1) / mask.sum(1)  # (B, 768)
```

**Empty / missing features:** Zero vector of dimension 768 — never null in output parquet.

---

### Per-Feature Token Length Caps

Per-feature caps are applied as `min(BERT_MAX_LENGTH, cap)` at tokenisation time, using `padding="longest"` per batch (not global padding). This prevents short texts from being padded to the global 8,192 maximum.

| Feature | Cap (tokens) | Rationale |
|---------|-------------|-----------|
| `chief_complaint` | 64 | Single phrase, typically 5–15 words |
| `triage` | 256 | Structured template, bounded length |
| `diag_history` | 512 | ICD label list — verbose but bounded |
| `radiology` | 1,024 | Single report |
| All lab groups ×13 | 2,048 | Lab timeline — longer for complex admissions |
| All microbiology panels ×37 | 512 | Dense structured text — median event text ~25 chars; 512 tokens comfortably covers even high-volume panels |
| `discharge_history` | 4,096 | Concatenated multi-visit notes |

Attention is O(n²) in sequence length. Padding chief complaint to 64 vs 8,192 tokens is ~16,000× less GPU compute per token.

**Dynamic batch size:** `_effective_batch_size(base_batch_size, max_length_cap)` scales the batch size inversely with `max_length_cap` to maintain a constant token budget per GPU forward pass.

---

### Admission-Slice Batching

#### Motivation

The full admission corpus cannot be embedded in a single SLURM job due to partition time limits. The safe per-GPU throughput is approximately **20,000 admissions per 12-hour window**, determined empirically from cluster runs.

The `BERT_SLICE_SIZE_PER_GPU` config key sets this limit. With 2 GPUs, each SLURM job covers `BERT_SLICE_SIZE_PER_GPU × n_gpus` admissions. The total number of slices is computed at runtime — no manual counting required. Reducing `BERT_SLICE_SIZE_PER_GPU` fits shorter partition windows; increasing it reduces the number of jobs needed on faster hardware.

#### Slice Layout (default: `BERT_SLICE_SIZE_PER_GPU = 20000`, 2 GPUs)

Each job covers 40,000 admissions (20,000 per GPU). The number of slices is `ceil(total_admissions / 40000)`.

| Slice | Admission rows | Per-GPU rows |
|-------|---------------|-------------|
| 0 | rows 0 – 39,999 | 20,000 |
| 1 | rows 40,000 – 79,999 | 20,000 |
| … | … | 20,000 |
| N−1 | rows (N−1)×40,000 – end | ≤ 20,000 |

#### CLI Interface

```bash
python embed_features.py --config config/preprocessing.yaml \
                         --slice-index 5
```

- `BERT_SLICE_SIZE_PER_GPU` (config key): admissions per GPU per job. Default: `20000`. This is the single knob for tuning time-window fit.
- `--slice-index` (CLI arg, also `BERT_SLICE_INDEX` in config): 0-based index of the slice this job processes. `src/preprocessing/submit_all.sh` sets this per job automatically.
- `n_slices` is never passed manually — it is always computed as `ceil(total_admissions / (BERT_SLICE_SIZE_PER_GPU × n_gpus))`.

#### Slice Computation

```
per_job = config["BERT_SLICE_SIZE_PER_GPU"] * n_gpus
n_slices = math.ceil(len(all_hadm_ids) / per_job)
slice_start = slice_index * per_job
slice_end   = min(slice_start + per_job, len(all_hadm_ids))
slice_hadm_ids = set(all_hadm_ids[slice_start:slice_end])
```

`all_hadm_ids` is the sorted list of all `hadm_id` values from `data_splits.parquet` — deterministic and reproducible across runs.

#### Output Appending

Each slice reads the existing output parquet (if present), concatenates its new rows with the existing data using `pa.concat_tables()`, and atomically overwrites the output file via a `.tmp` intermediate and `os.replace()`. PyArrow is used throughout — `fastparquet` cannot serialise `fixed_size_list` (`float32`, 768) embedding columns. The resulting file is a valid parquet readable by `pandas`/`pyarrow`. Concurrent writes are prevented by running slices sequentially (chained via `--dependency=afterok`).

#### Slice-Level Completeness Check

`check_embed_status.py` determines whether a slice is complete by checking whether all `hadm_id` values for that slice's range are present in the output parquets. Row count comparison against expected cumulative counts is no longer used — `hadm_id` presence is the authoritative completeness signal.

`check_embed_status.py` also validates extraction completeness before embedding: it checks that all expected feature parquets exist in `FEATURES_DIR` (including all 37 `micro_` panel parquets) before embedding proceeds.

Slices always run sequentially (chained via `--dependency=afterok`) — this ensures each slice appends cleanly to the previous slice's output without concurrent write conflicts.

---

### Multi-GPU Execution Within a Slice

Each embed SLURM job processes one slice using 2 GPUs. The slice's `hadm_ids` are split evenly between the two workers. Both workers run **in parallel** — each writes to its own per-worker temporary parquets, which the main process merges after both complete. This avoids concurrent writes to shared files while achieving true GPU parallelism (~2× throughput vs sequential execution).

```
Main process (one slice: ~40k admissions)
├── Load all input parquets, filter to slice_hadm_ids
├── Split slice into GPU-0 half and GPU-1 half (~20k each)
├── LPT-order 55 features within each worker (most expensive first)
│
├── spawn Worker 0 (cuda:0) ─────────────────────────────────┐
│     embeds 55 features for hadm_ids[0:20k]                 │ parallel
└── spawn Worker 1 (cuda:1) ─────────────────────────────────┘
      embeds 55 features for hadm_ids[20k:40k]
              │
              │  both write to per-worker temp parquets:
              │    discharge_history_embeddings.worker0.parquet
              │    discharge_history_embeddings.worker1.parquet
              │    ...
              ▼
        main process joins after both workers complete
              │
              ▼
        merge worker parquets → discharge_history_embeddings.parquet
                               → lab_blood_chemistry_embeddings.parquet
                               → micro_blood_culture_routine_embeddings.parquet
                               → ... (55 output parquets)
```

**Per-worker temporary parquets:** Each worker writes to
`{output_path}.worker{rank}` (e.g. `discharge_history_embeddings.parquet.worker0`).
After both workers finish, the main process reads each pair, concatenates them,
and writes the merged result to the final output path atomically via
`{output_path}.tmp` → `os.replace()`. Per-worker temp files are deleted after
successful merge.

**Resume behaviour with parallel workers:** Feature-level and record-level
resume operate on the final merged output parquet, not the per-worker temps.
If a job is killed mid-merge, the per-worker temps are still present on restart
and the merge is re-run. If a job is killed mid-embedding, the incomplete
per-worker temps are detected (row count < expected) and that worker
re-embeds from scratch for that feature.

`torch.multiprocessing.get_context("spawn")` is required for CUDA.
Single-GPU path runs in-process with no temporary files.

---

### GPU Load Balancing

**Purpose:** Since both workers process the same 55 features (on different admission halves), LPT is used to order features **within each worker** so the most expensive features start first. This gives better progress visibility in logs and ensures the GPU is never idle waiting for a trivially short feature at the end.

**Cost estimate per feature per worker:** `cost = len(worker_texts) × max_length_cap`

**Ordering within each worker:**
```
worker_tasks.sort(key=lambda t: len(t["texts"]) * t["max_length"], reverse=True)
```

Example ordering for a worker half:
```
discharge_history     (×admissions × 4096) → processed first
lab_blood_chemistry   (×admissions × 2048)
lab_blood_hematology  (×admissions × 2048)
...
micro_{panel}         (×admissions × 512)
...
chief_complaint       (×admissions × 64)  → processed last
```

Both workers process features in the same LPT order, so their logs are directly comparable.

---

### Feature-Level Resume

Before embedding each feature within a slice, the code checks whether this slice's rows are already present in the output parquet. If the feature parquet exists and contains all expected `hadm_id` values for this slice, the feature is skipped entirely for this slice.

| Check | Catches |
|-------|---------|
| Output parquet exists | Slice never started for this feature |
| Slice's hadm_ids all present | Partial slice write |
| No null values for slice rows | Partial embedding within slice |

Features passing all checks are logged as `[SKIP slice=N feature=X]` and not re-embedded. Override with `BERT_FORCE_REEMBED: true` in config.

**Atomic slice writes:** within a slice, each feature's batch is written to a `.tmp` file then `os.replace()`-d onto the append target, preventing a partially written batch from being mistaken for a complete one.

---

### Record-Level Resume

Feature-level resume (above) handles the case where a complete slice was written for a feature. Record-level resume handles the case where a slice was interrupted mid-feature — some rows were appended, but not all.

**Design:**

```
For each feature in each slice:
  1. Compute slice_hadm_ids (the hadm_ids this slice/feature should produce)
  2. If output parquet exists: load its hadm_ids → build already_done set
  3. pending = slice_hadm_ids − already_done
  4. If pending is empty: SKIP (feature-level resume handles this)
  5. Embed pending texts in batches of BERT_CHECKPOINT_INTERVAL rows
  6. After each interval: append batch to output parquet via fastparquet
  7. On slice completion: log row count
```

**Append mechanics (fastparquet):**
```
import fastparquet as fp

worker_path = output_path + f".worker{rank}"

# First write for this worker/feature
fp.write(worker_path, df_batch, compression="snappy")

# Subsequent checkpoint appends within the same worker
fp.write(worker_path, df_batch, compression="snappy", append=True)
```

Each worker appends only to its own `{output_path}.worker{rank}` file.
After both workers finish, the main process merges the two per-worker files:
```
df = pd.concat([
    pd.read_parquet(output_path + ".worker0"),
    pd.read_parquet(output_path + ".worker1"),
])
# atomic write to final path
df.to_parquet(tmp_path)
os.replace(tmp_path, output_path)
```
`fastparquet` append mode adds row groups without reading the full file.
The merged result is a valid parquet readable by `pandas`/`pyarrow`.

**Checkpoint interval:** `BERT_CHECKPOINT_INTERVAL` — rows between appends. Default: 10,000. Too small → high I/O overhead. Too large → large re-work on kill (at most `BERT_CHECKPOINT_INTERVAL` rows lost per interruption).

**Configuration keys:**

| Key | Default | Description |
|-----|---------|-------------|
| `BERT_SLICE_SIZE_PER_GPU` | `20000` | Admissions per GPU per SLURM job; determines total slice count |
| `BERT_SLICE_INDEX` | `0` | Slice index for this job (overridden by `--slice-index` CLI arg) |
| `BERT_CHECKPOINT_INTERVAL` | `10000` | Rows between within-feature incremental appends |
| `BERT_FORCE_REEMBED` | `false` | If true, ignores all slice/feature/record-level state |

**Resume hierarchy summary:**

| Level | Granularity | Trigger | Action |
|-------|-------------|---------|--------|
| Slice-level | Per SLURM job | `check_embed_status.py` at submit time | Job not submitted if slice complete |
| Feature-level | Per feature per slice | All slice `hadm_ids` present in output | Feature skipped for this slice |
| Record-level | Per checkpoint interval | Some (not all) slice `hadm_ids` present | Only missing rows embedded |

---

## 5. Final Dataset Assembly

### Join diagram

```
data_splits.parquet
        │
        ├── LEFT JOIN y_labels.parquet              on hadm_id
        │       y1_mortality, y2_readmission
        │
        ├── LEFT JOIN demographics_features.parquet  on hadm_id
        │       demographic_vec  [8 floats]
        │
        ├── LEFT JOIN diag_history_embeddings.parquet
        ├── LEFT JOIN discharge_history_embeddings.parquet
        ├── LEFT JOIN triage_embeddings.parquet
        ├── LEFT JOIN chief_complaint_embeddings.parquet
        ├── LEFT JOIN radiology_embeddings.parquet
        │       {feature}_embedding  [768 floats each]
        │
        ├── LEFT JOIN lab_{group}_embeddings.parquet  ×13
        │       lab_{group}_embedding  [768 floats, zero vector if no events]
        │
        └── LEFT JOIN micro_{panel}_embeddings.parquet  ×37
                micro_{panel}_embedding  [768 floats, zero vector if no events]
```

### Output schema

| Column | Type | Source |
|--------|------|--------|
| `subject_id` | int64 | `data_splits.parquet` |
| `hadm_id` | int64 | `data_splits.parquet` |
| `split` | str | `data_splits.parquet` |
| `y1_mortality` | int8 | `y_labels.parquet` |
| `y2_readmission` | float32 (NaN for deceased) | `y_labels.parquet` |
| `demographic_vec` | float32[8] | `demographics_features.parquet` |
| `diag_history_embedding` | float32[768] | embeddings/ |
| `discharge_history_embedding` | float32[768] | embeddings/ |
| `triage_embedding` | float32[768] | embeddings/ |
| `chief_complaint_embedding` | float32[768] | embeddings/ |
| `radiology_embedding` | float32[768] | embeddings/ |
| `lab_blood_gas_embedding` | float32[768] | embeddings/ |
| `lab_blood_chemistry_embedding` | float32[768] | embeddings/ |
| `lab_blood_hematology_embedding` | float32[768] | embeddings/ |
| `lab_urine_chemistry_embedding` | float32[768] | embeddings/ |
| `lab_urine_hematology_embedding` | float32[768] | embeddings/ |
| `lab_other_body_fluid_chemistry_embedding` | float32[768] | embeddings/ |
| `lab_other_body_fluid_hematology_embedding` | float32[768] | embeddings/ |
| `lab_ascites_embedding` | float32[768] | embeddings/ |
| `lab_pleural_embedding` | float32[768] | embeddings/ |
| `lab_csf_embedding` | float32[768] | embeddings/ |
| `lab_bone_marrow_embedding` | float32[768] | embeddings/ |
| `lab_joint_fluid_embedding` | float32[768] | embeddings/ |
| `lab_stool_embedding` | float32[768] | embeddings/ |
| `micro_blood_culture_routine_embedding` | float32[768] | embeddings/ |
| `micro_blood_bottle_gram_stain_embedding` | float32[768] | embeddings/ |
| `micro_urine_culture_embedding` | float32[768] | embeddings/ |
| `micro_urine_viral_embedding` | float32[768] | embeddings/ |
| `micro_urinary_antigens_embedding` | float32[768] | embeddings/ |
| `micro_respiratory_non_invasive_embedding` | float32[768] | embeddings/ |
| `micro_respiratory_invasive_embedding` | float32[768] | embeddings/ |
| `micro_respiratory_afb_embedding` | float32[768] | embeddings/ |
| `micro_respiratory_viral_embedding` | float32[768] | embeddings/ |
| `micro_respiratory_pcp_legionella_embedding` | float32[768] | embeddings/ |
| `micro_gram_stain_respiratory_embedding` | float32[768] | embeddings/ |
| `micro_gram_stain_wound_tissue_embedding` | float32[768] | embeddings/ |
| `micro_gram_stain_csf_embedding` | float32[768] | embeddings/ |
| `micro_wound_culture_embedding` | float32[768] | embeddings/ |
| `micro_hardware_and_lines_culture_embedding` | float32[768] | embeddings/ |
| `micro_pleural_culture_embedding` | float32[768] | embeddings/ |
| `micro_peritoneal_culture_embedding` | float32[768] | embeddings/ |
| `micro_joint_fluid_culture_embedding` | float32[768] | embeddings/ |
| `micro_fluid_culture_embedding` | float32[768] | embeddings/ |
| `micro_bone_marrow_culture_embedding` | float32[768] | embeddings/ |
| `micro_csf_culture_embedding` | float32[768] | embeddings/ |
| `micro_fungal_tissue_wound_embedding` | float32[768] | embeddings/ |
| `micro_fungal_respiratory_embedding` | float32[768] | embeddings/ |
| `micro_fungal_fluid_embedding` | float32[768] | embeddings/ |
| `micro_mrsa_staph_screen_embedding` | float32[768] | embeddings/ |
| `micro_resistance_screen_embedding` | float32[768] | embeddings/ |
| `micro_cdiff_embedding` | float32[768] | embeddings/ |
| `micro_stool_bacterial_embedding` | float32[768] | embeddings/ |
| `micro_stool_parasitology_embedding` | float32[768] | embeddings/ |
| `micro_herpesvirus_serology_embedding` | float32[768] | embeddings/ |
| `micro_hepatitis_hiv_embedding` | float32[768] | embeddings/ |
| `micro_syphilis_serology_embedding` | float32[768] | embeddings/ |
| `micro_misc_serology_embedding` | float32[768] | embeddings/ |
| `micro_herpesvirus_culture_antigen_embedding` | float32[768] | embeddings/ |
| `micro_gc_chlamydia_sti_embedding` | float32[768] | embeddings/ |
| `micro_vaginal_genital_flora_embedding` | float32[768] | embeddings/ |
| `micro_throat_strep_embedding` | float32[768] | embeddings/ |

**Total: 61 columns** (3 metadata + 2 labels + 1 structured vector + 55 embeddings: 5 text + 13 lab + 37 microbiology)

---

## 6. Configuration Reference

All configuration in `config/preprocessing.yaml`. No module reads this file directly — `run_pipeline.py` loads it and passes the dict to each module's `run(config)`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `MIMIC_DATA_DIR` | str | — | Root of MIMIC-IV download (`hosp/`, `icu/` subdirs) |
| `MIMIC_NOTE_DIR` | str | `MIMIC_DATA_DIR` | Root of `mimic-iv-note` (`note/` subdir) |
| `MIMIC_ED_DIR` | str | `MIMIC_DATA_DIR` | Root of `mimic-iv-ed` (`ed/` subdir) |
| `SPLIT_TRAIN` | float | `0.80` | Patient fraction for train |
| `SPLIT_DEV` | float | `0.10` | Patient fraction for dev |
| `SPLIT_TEST` | float | `0.10` | Patient fraction for test |
| `BERT_MODEL_NAME` | str | `Simonlee711/Clinical_ModernBERT` | HuggingFace model identifier |
| `BERT_MAX_LENGTH` | int | `8192` | Global token limit; per-feature caps applied on top |
| `BERT_BATCH_SIZE` | int | `32` | Base batch size; auto-scaled per feature |
| `BERT_DEVICE` | str | `cuda` | Falls back to CPU if CUDA unavailable |
| `BERT_MAX_GPUS` | int\|null | `null` | Max GPUs to use; null = all available |
| `BERT_SLICE_SIZE_PER_GPU` | int | `20000` | Admissions per GPU per SLURM job; controls number of slices |
| `BERT_SLICE_INDEX` | int | `0` | Slice index for this job; overridden by `--slice-index` CLI arg |
| `BERT_FORCE_REEMBED` | bool | `false` | Bypass all slice/feature/record-level resume |
| `BERT_CHECKPOINT_INTERVAL` | int | `10000` | Rows between within-feature checkpoint appends |
| `LAB_ADMISSION_WINDOW` | int\|`"full"` | `24` | Hours of lab events from `admittime`; `"full"` = entire admission |
| `HADM_LINKAGE_STRATEGY` | str | `"drop"` | `"drop"` or `"link"` for null `hadm_id` records in lab/note/chartevents |
| `HADM_LINKAGE_TOLERANCE_HOURS` | int | `2` | Tolerance in hours for time-window linkage (lab/note/chartevents) |
| `MICRO_WINDOW_HOURS` | int\|`"full_admission"` | `72` | Hours of microbiology events from `admittime`; `"full_admission"` = entire admission |
| `MICRO_NULL_HADM_STRATEGY` | str | `"drop"` | `"drop"` or `"link"` for null `hadm_id` in microbiologyevents |
| `MICRO_LINK_TOLERANCE_HOURS` | int | `2` | Tolerance in hours for microbiology hadm_id linkage |
| `MICRO_INCLUDE_COMMENTS` | bool | `true` | Include cleaned comments field in microbiology text representation |
| `MICRO_COMMENT_MAX_SENTENCES` | int | `2` | Max sentences retained after comment cleaning |
| `MICRO_COMMENT_MAX_CHARS` | int | `200` | Hard character limit applied after sentence extraction |
| `PREPROCESSING_DIR` | str | `data/preprocessing` | Root output directory |
| `FEATURES_DIR` | str | `data/preprocessing/features` | Raw feature parquets |
| `EMBEDDINGS_DIR` | str | `data/preprocessing/features/embeddings` | Embedding parquets |
| `CLASSIFICATIONS_DIR` | str | `data/preprocessing/classifications` | Labels, config artefacts |
| `HASH_REGISTRY_PATH` | str | `data/preprocessing/source_hashes.json` | MD5 registry for incremental runs |

---

## 7. Memory Requirements

Memory is expressed as formulas based on config parameters so estimates remain valid as the corpus size changes. Let **A** = total admissions, **S** = `BERT_SLICE_SIZE_PER_GPU`, **G** = number of GPUs per job, **M** = model hidden size (768).

### pipeline_job (CPU only)

| Process | Formula | Dominant term |
|---------|---------|---------------|
| `extract_discharge_history` | O(total discharge notes × avg note length) | All discharge notes loaded into memory before concatenation |
| All other extract modules | O(A) | Admission-indexed joins; well within available RAM |
| **Recommended allocation** | **64 GB** | Discharge history is the bottleneck |

### embed_job (2 GPUs, one slice)

Let **W** = S × G = admissions per job (default: 40,000).

| Component | Formula | Notes |
|-----------|---------|-------|
| Main process — feature parquets | O(W × 55 features × avg text length) | Filtered to slice hadm_ids before worker spawn |
| Main process — labs parquet | O(W × avg lab events per admission) | Long-format, filtered to slice |
| Per GPU worker — model weights | ~570 MB fixed | Clinical_ModernBERT frozen weights |
| Per GPU worker — embedding buffer | O(S × M × 4 bytes) = O(S × 3072 bytes) | float32[768] per admission per feature batch |
| Per GPU worker — batch buffer | O(batch_size × max_length_cap × 4 bytes) | Scales with `BERT_BATCH_SIZE` and per-feature cap |
| Per-worker temp parquets | O(S × 55 × M × 4 bytes) per worker | Written incrementally; peak = one full feature written |
| **Recommended allocation** | **64 GB per job** | Headroom for both workers + main process simultaneously |

### combine_job (CPU only)

| Component | Formula | Notes |
|-----------|---------|-------|
| All embedding parquets in memory | O(A × 55 × M × 4 bytes) | All 55 parquets joined simultaneously |
| At default A=546k, M=768 | ≈ 546,000 × 55 × 768 × 4 ≈ **92 GB** | Exceeds 64 GB — load and join incrementally per parquet |
| **Recommended allocation** | **32 GB** | Incremental join keeps peak well below full load |

**Note on combine_job:** `combine_dataset.py` joins embedding parquets one at a time rather than loading all 55 simultaneously, keeping peak memory proportional to a single parquet (O(A × M × 4 bytes) ≈ 1.7 GB at default settings) plus the growing output dataframe.
