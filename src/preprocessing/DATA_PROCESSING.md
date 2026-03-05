# CDSS Preprocessing — Data Processing Reference

This document describes the clinical and technical choices made in the CDSS
preprocessing pipeline. It is aimed at researchers and reviewers who want to
understand what data is used and how it is transformed.

For instructions on how to run the pipeline, see
[preprocessing-runtime-instructions.md](preprocessing-runtime-instructions.md).

---

## 1. Overview

### Prediction targets

| Target | Name | Definition |
|--------|------|------------|
| Y1 | In-hospital mortality | `admissions.hospital_expire_flag` — binary flag set to 1 if the patient died during the admission |
| Y2 | 30-day readmission | Derived — 1 if the patient has a subsequent admission with `admittime` within 30 days after `dischtime` of the current admission |

### Split strategy

Patients (identified by `subject_id`) are split into **train / dev / test**
(70 / 15 / 15) using a stratified split with a **fixed random seed of 42**.

**Patient-level splitting** is used — not admission-level — to prevent data
leakage: a single patient may have multiple admissions, and if these were
assigned to different splits, information from one admission could bleed into
the model's evaluation on another.

Stratification is performed on a **per-patient binary flag**: whether any of
the patient's admissions resulted in in-hospital death (`hospital_expire_flag > 0`).
Y2 is not included in stratification because patients who died are excluded from
Y2 (see [Section 5](#5-labels-extract_y_datapy)), which makes joint
stratification on both targets unstable.

Every `hadm_id` inherits the split label of its `subject_id`.

---

## 2. Source tables

| Table | Location | Used by | Columns extracted |
|-------|----------|---------|-------------------|
| `admissions` | `hosp/` | all modules | `subject_id`, `hadm_id`, `admittime`, `dischtime`, `hospital_expire_flag` |
| `patients` | `hosp/` | `extract_demographics` | `subject_id`, `gender`, `anchor_age`, `anchor_year` |
| `omr` | `hosp/` | `extract_demographics` | `subject_id`, `chartdate`, `result_name`, `result_value` |
| `diagnoses_icd` | `hosp/` | `extract_diag_history` | `subject_id`, `hadm_id`, `icd_code`, `icd_version` |
| `d_icd_diagnoses` | `hosp/` | `extract_diag_history` | `icd_code`, `icd_version`, `long_title` |
| `d_labitems` | `hosp/` | `extract_labs` | `itemid`, `label`, `fluid`, `category` |
| `labevents` | `hosp/` | `extract_labs` | `subject_id`, `hadm_id`, `itemid`, `charttime`, `value`, `valuenum`, `valueuom`, `ref_range_lower`, `ref_range_upper`, `flag`, `priority` |
| `chartevents` | `icu/` | `extract_demographics` | `subject_id`, `hadm_id`, `itemid`, `charttime`, `valuenum` |
| `discharge` | `note/` (mimic-iv-note) | `extract_discharge_history` | `subject_id`, `hadm_id`, `charttime`, `text` |
| `radiology` | `note/` (mimic-iv-note) | `extract_radiology` | `subject_id`, `hadm_id`, `charttime`, `text` |
| `triage` | `ed/` (mimic-iv-ed) | `extract_triage_and_complaint` | `subject_id`, `stay_id`, `temperature`, `heartrate`, `resprate`, `o2sat`, `sbp`, `dbp`, `pain`, `acuity`, `chiefcomplaint` |
| `edstays` | `ed/` (mimic-iv-ed) | `extract_triage_and_complaint` | `subject_id`, `stay_id`, `hadm_id`, `intime` |

---

## 3. Train/Dev/Test splitting (`create_splits.py`)

| Property | Value |
|----------|-------|
| Unit of splitting | `subject_id` (patient level, not admission level) |
| Stratification variable | Binary — `1` if any of the patient's admissions has `hospital_expire_flag > 0`, else `0` |
| Split ratios | Configurable via `SPLIT_TRAIN`, `SPLIT_DEV`, `SPLIT_TEST` (default 70/15/15) |
| Random seed | 42 (fixed for reproducibility) |
| Output | `data_splits.parquet` — one row per `hadm_id` with a `split` column (`train`/`dev`/`test`) |

**Why patient-level splitting?** If multiple admissions of the same patient
were assigned to different splits, features from earlier admissions (e.g.
diagnosis history, discharge history) would contain information from the held-
out split, creating leakage.

**Why Y2 is excluded from stratification:** Patients who died (`hospital_expire_flag = 1`)
receive `NaN` for Y2 (they cannot be readmitted). Including them in a joint
stratification on Y1 and Y2 would create degenerate strata, so only Y1 is
used for stratification.

---

## 4. Feature extraction

### F1 — Demographics (`extract_demographics.py`)

**Output column:** `demographic_vec` — array of 8 floats:
`[age, gender, height_cm, weight_kg, bmi, height_missing, weight_missing, bmi_missing]`

#### Age

Derived as `anchor_age + (admit_year - anchor_year)` from the `patients` table,
where `admit_year` is extracted from `admittime`. This corrects for the fact
that `anchor_age` corresponds to `anchor_year`, not the admission year.

#### Gender

Encoded as `M = 1.0`, `F = 0.0`. Unknown values → `NaN`.

#### Height

| Priority | Source | Condition | Transformation |
|----------|--------|-----------|----------------|
| 1 | `omr.result_name` contains `"Height"` | `chartdate ≤ admittime` (leakage control) | If `result_name` contains `"Inches"`, multiply by 2.54; otherwise assume cm |
| 2 | `chartevents` `itemid = 226707` | First recorded value within admission window | Inches to cm (× 2.54) |
| 3 | `chartevents` `itemid = 226730` | First recorded value within admission window | Use as-is (already cm) |

Plausibility filter: 50–250 cm. Values outside this range are discarded.

#### Weight

| Priority | Source | Condition | Transformation |
|----------|--------|-----------|----------------|
| 1 | `omr.result_name` contains `"Weight"` | `chartdate ≤ admittime` (leakage control) | If `result_name` contains `"Lbs"`, multiply by 0.453592; otherwise assume kg |
| 2 | `chartevents` `itemid = 226512` | First recorded value within admission window | Use as-is (kg) — Admission Weight |
| 3 | `chartevents` `itemid = 224639` | First recorded value within admission window | Use as-is (kg) — Daily Weight |
| 4 | `chartevents` `itemid = 226531` | First recorded value within admission window | lbs to kg (× 0.453592) — Admission Weight lbs |
| 5 | `chartevents` `itemid = 226846` | First recorded value within admission window | Use as-is (kg) — Feeding Weight |

Plausibility filter: 20–400 kg. Values outside this range are discarded.

#### BMI

- **Primary source:** `omr.result_name` contains `"BMI"`, used as-is.
- **Derived fallback:** `weight_kg / (height_cm / 100)²` when BMI is absent but
  both height and weight are available.
- BMI is **never independently imputed** — it is always derived from imputed
  height/weight when the measured value is missing.

#### Missingness handling and imputation

Binary flags `height_missing`, `weight_missing`, and `bmi_missing` are set
**before** any imputation step, preserving information about which values were
originally absent.

Imputation statistics (mean and standard deviation) are computed on the
**train split only** and saved to `imputation_stats.json` to prevent leakage
into dev/test sets. Statistics are stratified by **(age-bin × gender)**, where
age bins are: `18–29`, `30–44`, `45–64`, `65–74`, `75+`.

Missing values are filled by sampling from N(mean, std) of the patient's
stratum. If the stratum is absent in the training data, global statistics are
used as a fallback.

**Execution order:** missingness flags → height/weight imputation → BMI
derivation.

---

### F2 — Diagnosis history (`extract_diag_history.py`)

**Output column:** `diag_history_text` — a single string per admission.

- **Source:** `diagnoses_icd` joined to `d_icd_diagnoses` on `(icd_code, icd_version)`
  to obtain `long_title`.
- **Leakage control:** Only diagnoses from admissions where
  `admittime < current admittime` (strictly prior visits) are included.
- **Transformation:** `long_title` values from all prior admissions are
  concatenated with `|` as separator, sorted by admission time.
- First-time admissions (no prior visits) produce an empty string.

---

### F3 — Discharge history (`extract_discharge_history.py`)

**Output column:** `discharge_history_text` — a single string per admission.

- **Source:** `note/discharge` (mimic-iv-note module), looked up under
  `MIMIC_NOTE_DIR/note/` first, then `MIMIC_DATA_DIR/note/`, then
  `MIMIC_DATA_DIR/hosp/` as a last resort.
- **Leakage control:** Only notes from admissions where
  `admittime < current admittime` (strictly prior visits) are included.
- **Text cleaning:** Everything before the first `"Allergies:"` marker is
  removed from each note.
- Multiple prior discharge notes are concatenated in chronological order with
  `\n\n---\n\n` as separator.
- First-time admissions (no prior discharge notes) produce an empty string.

---

### F4 — Triage (`extract_triage_and_complaint.py`)

**Output column:** `triage_text` — a single string per admission.

- **Source:** `ed/triage` joined to `ed/edstays` on `stay_id` to resolve `hadm_id`
  (mimic-iv-ed module, resolved via `MIMIC_ED_DIR/ed/` first).
  - **Primary linkage:** `stay_id → hadm_id` directly from `edstays`.
  - **Fallback linkage:** for ED visits with null `hadm_id` in `edstays`, the
    closest hospital admission with `admittime ≥ intime` for the same `subject_id`
    is used as an approximate link.
  - ED visits with no resolvable `hadm_id` (non-admitted visits) are excluded.
- **Transformation:** Structured triage fields are rendered as a natural-language
  sentence using the template:

  ```
  Triage assessment: temperature {temperature}°C, heart rate {heartrate} bpm,
  respiratory rate {resprate} breaths/min, O2 saturation {o2sat}%,
  blood pressure {sbp}/{dbp} mmHg, pain score {pain}/10, acuity level {acuity}.
  ```

  Missing fields are rendered as `"N/A"`.

---

### F5 — Chief complaint (`extract_triage_and_complaint.py`)

**Output column:** `chief_complaint_text` — a raw text string per admission.

- **Primary source:** `triage.chiefcomplaint` column (if present). `hadm_id` is
  resolved via `edstays` exactly as in F4 above (same join logic).
- **Fallback:** `chartevents` `itemid = 223112`.
- No cleaning or templating is applied.

---

### F6 — Lab events (`extract_labs.py`)

**Output:** Long-format parquet — one row per lab event.

**Output columns:** `subject_id`, `hadm_id`, `charttime`, `itemid`, `label`,
`fluid`, `category`, `lab_text_line`.

- **Source:** `labevents` joined to `d_labitems` on `itemid` for `label`,
  `fluid`, and `category`.
- **Filtering applied:**
  - Rows with null `hadm_id` are removed (~70 % of `labevents` are outpatient).
  - Rows where both `value` and `valuenum` are null are removed.
  - Unmapped `itemid` values (not present in `d_labitems`) are removed.
  - `d_labitems` artefact rows with `fluid` in `["I", "Q", "fluid"]` are
    removed.
  - Admission window filter: `admittime ≤ charttime ≤ dischtime`.
- **Streaming:** `labevents` is read in chunks of 500,000 rows to manage memory.
- **Text line format per event:**
  ```
  [HH:MM] {label} ({fluid}/{category}): {value} {unit} (ref: lower-upper) [ABNORMAL] [STAT]
  ```
  - `valuenum` is used when available (formatted to 2 decimal places); otherwise
    the text `value` field is used.
  - Reference range is omitted when either bound is null.
  - `[ABNORMAL]` is appended only when `flag = "abnormal"`.
  - `[STAT]` is appended only when `priority = "STAT"`.

**Note:** Lab embedding is intentionally deferred to training time. The MDP
agent selects a subset of `itemid` values, their text lines are concatenated
chronologically, and passed to the language model for encoding. `labs_features.parquet`
is excluded from `final_cdss_dataset.parquet` and joined dynamically at
training/inference time.

---

### F7 — Radiology notes (`extract_radiology.py`)

**Output column:** `radiology_text` — a single string per admission.

- **Source:** `note/radiology` (mimic-iv-note module), resolved with the same
  search order as F3 (`MIMIC_NOTE_DIR/note/` → `MIMIC_DATA_DIR/note/` →
  `MIMIC_DATA_DIR/hosp/`).
- **Filter:** Only notes within the admission window
  (`admittime ≤ charttime ≤ dischtime`) are considered.
- **Selection:** The **most recent** note per admission is used.
- **Text cleaning:** Everything before the first `"EXAMINATION:"` marker is
  removed.

---

## 5. Labels (`extract_y_data.py`)

### Y1 — In-hospital mortality

| Property | Value |
|----------|-------|
| Source | `admissions.hospital_expire_flag` |
| Type | Binary integer: `1 = died`, `0 = survived` |
| Output column | `y1_mortality` |
| Transformation | None |

### Y2 — 30-day readmission

| Property | Value |
|----------|-------|
| Definition | `1` if the patient has a subsequent admission with `admittime` within 30 days after `dischtime` of the current admission |
| Subsequent admission condition | `next_admittime > dischtime` (strictly after discharge) AND `next_admittime ≤ dischtime + 30 days` |
| Deceased patients | Excluded — patients with `hospital_expire_flag = 1` receive `NaN` for Y2 (they cannot be readmitted) |
| Output column | `y2_readmission` |

**Output file:** `y_labels.parquet` — one row per `hadm_id` with
`y1_mortality` and `y2_readmission` columns.

---

## 6. BERT embeddings (`embed_features.py`)

| Property | Value |
|----------|-------|
| Model | Configurable via `BERT_MODEL_NAME` (default: `emilyalsentzer/Bio_ClinicalBERT`) |
| Embedding method | `[CLS]` token from the final hidden state |
| Null/empty text | Zero vector of the same dimensionality as the model's hidden size |
| Truncation | Inputs are truncated to `BERT_MAX_LENGTH` tokens (default: 512) |
| Batch size | `BERT_BATCH_SIZE` samples per inference call (default: 32) |
| Device | `BERT_DEVICE` (`"cuda"` or `"cpu"`); falls back to CPU with a warning if CUDA is unavailable |

**Features embedded:**

| Input parquet | Text column | Output parquet | Embedding column |
|---------------|-------------|----------------|------------------|
| `diag_history_features.parquet` | `diag_history_text` | `diag_history_embeddings.parquet` | `diag_history_embedding` |
| `discharge_history_features.parquet` | `discharge_history_text` | `discharge_history_embeddings.parquet` | `discharge_history_embedding` |
| `triage_features.parquet` | `triage_text` | `triage_embeddings.parquet` | `triage_embedding` |
| `chief_complaint_features.parquet` | `chief_complaint_text` | `chief_complaint_embeddings.parquet` | `chief_complaint_embedding` |
| `radiology_features.parquet` | `radiology_text` | `radiology_embeddings.parquet` | `radiology_embedding` |

Lab features (F6) are **not embedded here** — they are in long format and
embedded dynamically at training time.

---

## 7. Final dataset assembly (`combine_dataset.py`)

`combine_dataset.py` assembles the final flat dataset by starting from the
admission universe defined by `data_splits.parquet` and left-joining features
in the following order:

1. `y_labels.parquet` (labels)
2. `demographics_features.parquet` (structured features)
3. All parquets in `EMBEDDINGS_DIR/` (BERT embeddings)

**Intentionally excluded:**
- `labs_features.parquet` — stored in long format; joined dynamically at
  training time when the MDP agent selects a subset of `itemid` values.
- Raw text parquets — superseded by embedding parquets.

**Output:** `data/input/classifications/final_cdss_dataset.parquet` — one row per `hadm_id`; missing
feature values appear as nulls.

---

## 8. Design principles

### No leakage

- Imputation statistics (mean, std for height/weight) are computed on the
  **train split only** and saved to `imputation_stats.json`; dev/test
  imputation uses these pre-computed statistics.
- Prior-visit features (diagnosis history, discharge history) use a **strict
  temporal filter** (`prior admittime < current admittime`).
- OMR vital signs use `chartdate ≤ admittime` to exclude measurements taken
  during the current admission.

### No hardcoding

All file paths, model names, split ratios, batch sizes, and output directories
are read from `preprocessing.yaml`. No script contains hardcoded paths or
tunable parameters.

### Reproducibility

- Random seed **42** is fixed for all train/dev/test splits.
- Imputation statistics are persisted to `imputation_stats.json` so that
  re-running `extract_demographics` on a new machine produces identical output.
- Source file MD5 hashes are persisted to `source_hashes.json`, making
  incremental runs deterministic.

### Memory safety

- `labevents` and `chartevents` are streamed in chunks of **500,000 rows**.
  Unit conversion, filtering, and text formatting are applied per chunk.
- The post-chunk merge with `admissions` is performed in memory; a machine
  with at least 32 GB RAM is recommended.

### Graceful degradation

- `BERT_DEVICE: cuda` → CPU fallback with a logged warning if CUDA is
  unavailable.
- Per-stratum imputation → global statistics fallback if the stratum is absent
  from the training data.
- Missing `omr` or `chartevents` tables → logged warning; module still
  completes with higher missingness rates.

### Incremental runs

MD5 hashes of source files are recorded to `source_hashes.json` **after** each
module completes successfully. On subsequent runs, a module is skipped if all
source hashes match and all output files exist. Crashes mid-run leave no stale
hashes, ensuring the module re-runs on the next invocation. The `--force` and
`--force-module` CLI flags bypass hash checking when needed.
