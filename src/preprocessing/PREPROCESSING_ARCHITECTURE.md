# CDSS-ML Preprocessing — Design & Architecture

## Table of Contents

1. [Overview](#1-overview)
2. [Identifier Hierarchy](#2-identifier-hierarchy)
3. [Data Splits](#3-data-splits)
4. [Feature Set](#4-feature-set)
5. [Pipeline Architecture](#5-pipeline-architecture)
6. [Module Reference](#6-module-reference)
7. [Feature Extraction Detail](#7-feature-extraction-detail)
8. [Embedding Architecture](#8-embedding-architecture)
9. [Final Dataset Assembly](#9-final-dataset-assembly)
10. [Configuration Reference](#10-configuration-reference)
11. [Design Principles](#11-design-principles)
12. [Directory Structure](#12-directory-structure)

---

## 1. Overview

The CDSS-ML preprocessing pipeline transforms raw MIMIC-IV clinical data into a fixed-schema feature dataset ready for supervised classification and reinforcement learning. Two prediction targets are produced:

| Target | Name | Definition | Output column |
|--------|------|------------|---------------|
| Y1 | In-hospital mortality | `admissions.hospital_expire_flag` | `y1_mortality` |
| Y2 | 30-day readmission | Subsequent admission within 30 days of `dischtime`; NaN for deceased patients | `y2_readmission` |

The output is a single parquet file (`final_cdss_dataset.parquet`) with one row per hospital admission, containing 19 feature representations (1 structured vector + 18 embeddings), 2 labels, and a split assignment column.

---

## 2. Identifier Hierarchy

MIMIC-IV uses three nested identifiers:

```
subject_id   (patient — persistent across all visits)
  └── hadm_id    (hospital admission — the unit of prediction)
        └── stay_id    (ICU stay within an admission)
```

**`hadm_id` is the primary join key** throughout the pipeline. All feature parquets carry `(subject_id, hadm_id)` as their key columns.

**`stay_id`** is encountered in ICU module tables (`chartevents`, `inputevents`, `procedureevents`). In the current Phase 1 pipeline it appears only indirectly via `chartevents` in the demographics fallback — the join is performed on `hadm_id`, so `stay_id` does not need explicit handling. It becomes critical in the MDP phase for modelling clinical interventions.

### Missing `hadm_id`

Several tables contain records with null `hadm_id` (~10–20% of `labevents`, lower rates in `note` and `chartevents`). Handling is configurable:

| Strategy | Behaviour |
|----------|-----------|
| `"drop"` *(default)* | Exclude records with null `hadm_id`. Count and percentage logged per module. |
| `"link"` | Attempt time-window linkage: match `charttime` against patient admission windows within `HADM_LINKAGE_TOLERANCE_HOURS` tolerance. Assign if exactly one match; assign closest if multiple; drop if none. All outcomes logged to `hadm_linkage_stats.json`. |

---

## 3. Data Splits

Splitting is performed **at the patient level** (`subject_id`), not at the admission level. This prevents features derived from prior admissions (F2, F3) from creating leakage between splits.

```
All admissions
      │
      ▼
Group by subject_id
      │
      ▼
Stratify by: binary flag — does any admission have hospital_expire_flag = 1?
      │
      ├──► Train  (default 80%)
      ├──► Dev    (default 10%)
      └──► Test   (default 10%)
```

All admissions of a given patient are assigned to the same split. Split ratios are configurable via `SPLIT_TRAIN`, `SPLIT_DEV`, `SPLIT_TEST`. Random seed is fixed at 42.

**Why Y1 only for stratification:** Deceased patients receive NaN for Y2 (they cannot be readmitted). Including Y2 in stratification would create degenerate strata, so only Y1 is used.

Output: `data_splits.parquet` — columns: `subject_id`, `hadm_id`, `split`.

---

## 4. Feature Set

The input feature vector X covers 19 feature slots across two representation types:

| ID | Feature | Source | Type | Representation | Visible at episode start |
|----|---------|--------|------|----------------|--------------------------|
| F1 | Demographics | `patients`, `admissions`, `omr`, `chartevents` | Numeric | 8-float vector | ✓ Always |
| F2 | Diagnosis History (prior visits) | `diagnoses_icd`, `d_icd_diagnoses` | Coded text | 768-d mean-pool embedding | ✓ Always |
| F3 | Discharge History (prior visits) | `note/discharge` | Free text | 768-d mean-pool embedding | ✓ Always |
| F4 | Triage (current visit) | `triage`, `edstays` | Structured→text | 768-d mean-pool embedding | ✓ Always |
| F5 | Chief Complaint (current visit) | `triage.chiefcomplaint` | Free text | 768-d mean-pool embedding | ✓ Always |
| F6 | Labs — Blood Gas | `labevents` | Structured→text | 768-d mean-pool embedding | ✗ Maskable |
| F7 | Labs — Blood Chemistry | `labevents` | Structured→text | 768-d mean-pool embedding | ✗ Maskable |
| F8 | Labs — Blood Hematology | `labevents` | Structured→text | 768-d mean-pool embedding | ✗ Maskable |
| F9 | Labs — Urine Chemistry | `labevents` | Structured→text | 768-d mean-pool embedding | ✗ Maskable |
| F10 | Labs — Urine Hematology | `labevents` | Structured→text | 768-d mean-pool embedding | ✗ Maskable |
| F11 | Labs — Other Body Fluid Chemistry | `labevents` | Structured→text | 768-d mean-pool embedding | ✗ Maskable |
| F12 | Labs — Other Body Fluid Hematology | `labevents` | Structured→text | 768-d mean-pool embedding | ✗ Maskable |
| F13 | Labs — Ascites | `labevents` | Structured→text | 768-d mean-pool embedding | ✗ Maskable |
| F14 | Labs — Pleural | `labevents` | Structured→text | 768-d mean-pool embedding | ✗ Maskable |
| F15 | Labs — CSF | `labevents` | Structured→text | 768-d mean-pool embedding | ✗ Maskable |
| F16 | Labs — Bone Marrow | `labevents` | Structured→text | 768-d mean-pool embedding | ✗ Maskable |
| F17 | Labs — Joint Fluid | `labevents` | Structured→text | 768-d mean-pool embedding | ✗ Maskable |
| F18 | Labs — Stool | `labevents` | Structured→text | 768-d mean-pool embedding | ✗ Maskable |
| F19 | Radiology Note (current visit) | `note/radiology` | Free text | 768-d mean-pool embedding | ✗ Maskable |

**F1–F5** are always visible to both the classifier and the MDP agent. **F6–F19** are maskable — the MDP agent selects which to unlock during an episode; each is an independent feature slot with its own projection layer in the neural network.

---

## 5. Pipeline Architecture

### Execution Order

```
                    ┌─────────────────────┐
                    │   create_splits.py   │
                    │  data_splits.parquet │
                    └──────────┬──────────┘
                               │  (all extract_* depend on this)
              ┌────────────────┼────────────────────┐
              │                │                    │
              ▼                ▼                    ▼
   extract_demographics   extract_labs       extract_y_data
   extract_diag_history   (requires          y_labels.parquet
   extract_discharge_     build_lab_panel_
   history                config first)
   extract_triage_and_
   complaint
   extract_radiology
              │                │
              └────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ embed_features │
              │  (18 parquets) │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────────┐
              │  combine_dataset   │
              │ final_cdss_dataset │
              └───────────────────┘
```

**Rules:**
- `create_splits.py` must run first
- `build_lab_panel_config.py` must run before `extract_labs.py`
- All `extract_*` modules are independent of each other and can run in parallel
- `embed_features.py` requires all `extract_*` modules to have completed
- `combine_dataset.py` requires `embed_features.py` and `extract_y_data.py`

### Module Summary

| Step | Module | Output |
|------|--------|--------|
| 0 | `build_lab_panel_config.py` | `lab_panel_config.yaml` |
| 1 | `create_splits.py` | `data_splits.parquet` |
| 2 | `extract_demographics.py` | `demographics_features.parquet` |
| 3 | `extract_diag_history.py` | `diag_history_features.parquet` |
| 4 | `extract_discharge_history.py` | `discharge_history_features.parquet` |
| 5 | `extract_triage_and_complaint.py` | `triage_features.parquet`, `chief_complaint_features.parquet` |
| 6 | `extract_labs.py` | `labs_features.parquet` (long format) |
| 7 | `extract_radiology.py` | `radiology_features.parquet` |
| 8 | `extract_y_data.py` | `y_labels.parquet` |
| 9 | `embed_features.py` | 18 embedding parquets |
| 10 | `combine_dataset.py` | `final_cdss_dataset.parquet` |

---

## 6. Module Reference

### `build_lab_panel_config.py`

Reads `d_labitems`, groups items by `(fluid × category)`, and writes `lab_panel_config.yaml` mapping each of the 13 group names to a list of `itemid` integers. Artefact rows where `fluid` is in `["I", "Q", "fluid"]` are removed before grouping. Groups where a single fluid spans multiple categories (Ascites, Pleural, CSF, Bone Marrow, Joint Fluid, Stool) are merged into a single group keyed by fluid name.

### `create_splits.py`

Patient-level stratified 3-way split. Stratification variable: binary flag — does any admission for the patient have `hospital_expire_flag = 1`? Random seed 42. Output is one row per `hadm_id` with a `split` column.

### `extract_demographics.py`

Produces an 8-float vector per admission. Sources: `patients` (age, gender), `omr` (preferred height/weight/BMI), `chartevents` (fallback height/weight). Missingness flags created before imputation. Imputation statistics computed on train split only and saved to `imputation_stats.json`. Missing height/weight imputed by sampling from `N(mean, std)` per `(age_bin × gender)` stratum. BMI derived from imputed height/weight when absent, never imputed independently.

### `extract_diag_history.py`

Builds a structured text block of ICD diagnoses from all prior admissions (strictly before current `admittime`). Formatted with dated section headers and one `long_title` per line per visit. Empty string for first-time admissions.

### `extract_discharge_history.py`

Concatenates discharge notes from all prior admissions with dated headers. Text cleaning removes everything before the first `"Allergies:"` marker in each note. Empty string for first-time admissions.

### `extract_triage_and_complaint.py`

Extracts triage structured fields and chief complaint from the ED visit corresponding to the current admission. `hadm_id` is resolved via `edstays` (primary: `stay_id → hadm_id`) with a fallback to the closest admission by `intime` for ED visits not linked to an inpatient admission. Non-admitted ED visits are excluded.

### `extract_labs.py`

Streams `labevents` in chunks, filters to current admission window, formats each event as a timestamped text line. Output is long-format (one row per event) with a `lab_text_line` column. Used as input to `embed_features.py` for group-level embedding.

### `extract_radiology.py`

Selects the most recent radiology note within the current admission window. Text cleaning removes everything before the first `"EXAMINATION:"` marker. Empty string if no radiology notes exist for the admission.

### `extract_y_data.py`

Y1: `hospital_expire_flag` directly from `admissions`. Y2: 1 if any subsequent admission has `admittime` within 30 days of `dischtime`; NaN for patients with `hospital_expire_flag = 1`.

### `embed_features.py`

Embeds all 18 text features using Clinical_ModernBERT with mean pooling. For the 5 non-lab features, reads the corresponding text parquet directly. For each of the 13 lab groups, filters `labs_features.parquet` to the group's `itemid` list, concatenates text lines per admission chronologically, and embeds the resulting text. Admissions with no events in a given group receive a zero vector.

### `combine_dataset.py`

Starts from `data_splits.parquet` as the admission universe. Left-joins `y_labels.parquet`, `demographics_features.parquet`, and all `*.parquet` files discovered in `EMBEDDINGS_DIR`. The 13 lab group embedding parquets are joined as independent nullable columns alongside the 5 non-lab embeddings.

---

## 7. Feature Extraction Detail

### F1 — Demographics

```
Output: demographic_vec — [age, gender, height_cm, weight_kg, bmi,
                           height_missing, weight_missing, bmi_missing]
```

**Age:** `anchor_age + (admit_year − anchor_year)` — corrects for the year shift in MIMIC-IV anonymisation.

**Gender:** `M = 1.0`, `F = 0.0`.

**Height / Weight source priority:**

| Priority | Source | Notes |
|----------|--------|-------|
| 1 | `omr` — `result_name` contains "Height"/"Weight" | `chartdate ≤ admittime` (leakage control). Inches → cm (×2.54). Lbs → kg (×0.453592). |
| 2–5 | `chartevents` itemids (height: 226707, 226730; weight: 226512, 224639, 226531, 226846) | First value within admission window. Unit conversion per itemid. |

Plausibility filters: height 50–250 cm, weight 20–400 kg.

**Imputation** (height and weight only):
- Missingness flags set **before** imputation
- Statistics: `N(mean, std)` per `(age_bin × gender)` stratum, computed on **train split only**
- Saved to `imputation_stats.json`; applied identically to dev/test
- Age bins: 18–29, 30–44, 45–64, 65–74, 75+
- Fallback to global statistics if stratum absent from training data

**BMI:** Use OMR value if present. Derive as `weight_kg / (height_cm / 100)²` if absent. Never imputed independently.

---

### F2 — Diagnosis History

```
Output: diag_history_text — single string per admission
Leakage control: prior admissions only (admittime < current admittime)
```

Text format:
```
Past Diagnoses:

Visit (2018-03-12):
Chronic kidney disease, stage 3
Hypertension

Visit (2019-07-24):
Acute kidney injury
```

---

### F3 — Discharge History

```
Output: discharge_history_text — single string per admission
Leakage control: prior admissions only (admittime < current admittime)
Cleaning: remove everything before first "Allergies:" marker in each note
```

Text format:
```
Prior Discharge Summary (2018-03-12):
Allergies: Penicillin
[clinical note body...]

Prior Discharge Summary (2019-07-24):
Allergies: None known
[clinical note body...]
```

---

### F4 — Triage

```
Output: triage_text — single string per admission
```

Template:
```
Triage assessment: temperature {T}°C, heart rate {HR} bpm,
respiratory rate {RR} breaths/min, O2 saturation {O2}%,
blood pressure {SBP}/{DBP} mmHg, pain score {pain}/10, acuity level {acuity}.
```

Missing fields rendered as `"N/A"`.

**`hadm_id` resolution:** Primary via `edstays.stay_id → hadm_id`. Fallback via closest `admittime ≥ intime` for same `subject_id`. Non-admitted ED visits excluded.

---

### F5 — Chief Complaint

```
Output: chief_complaint_text — raw text string per admission
Primary source: triage.chiefcomplaint
Fallback: chartevents itemid = 223112
No cleaning or templating applied.
```

---

### F6–F18 — Laboratory Results (13 independent groups)

```
Output: labs_features.parquet — long format (one row per lab event)
Leakage control: current admission only, within LAB_ADMISSION_WINDOW
```

**Lab groups** (derived from `d_labitems` fluid × category, stored in `lab_panel_config.yaml`):

| Group name | Fluid | Category |
|------------|-------|----------|
| `blood_gas` | Blood | Blood Gas |
| `blood_chemistry` | Blood | Chemistry |
| `blood_hematology` | Blood | Hematology |
| `urine_chemistry` | Urine | Chemistry |
| `urine_hematology` | Urine | Hematology |
| `other_body_fluid_chemistry` | Other Body Fluid | Chemistry |
| `other_body_fluid_hematology` | Other Body Fluid | Hematology |
| `ascites` | Ascites | Chemistry + Hematology |
| `pleural` | Pleural | Chemistry + Hematology |
| `csf` | Cerebrospinal Fluid | Chemistry + Hematology |
| `bone_marrow` | Bone Marrow | Hematology |
| `joint_fluid` | Joint Fluid | Blood Gas + Chemistry + Hematology |
| `stool` | Stool | Chemistry + Hematology |

**Admission window:** Configurable via `LAB_ADMISSION_WINDOW` — integer hours (e.g. `24`) or `"full"` for entire admission. Default: 24 hours.

**Text line format per event:**
```
[HH:MM] {label}: {value} {unit} (ref: lower-upper) [ABNORMAL]
```

- `[HH:MM]` = elapsed time since `admittime` (not absolute clock time)
- `valuenum` formatted to 2 dp when available; text `value` field otherwise
- `(ref: lower-upper)` omitted when either bound is null
- `[ABNORMAL]` appended when `flag == "abnormal"` OR when `valuenum` falls outside `[ref_range_lower, ref_range_upper]`

Example (`blood_chemistry` group):
```
[00:14] Glucose: 8.20 mmol/L [ABNORMAL]
[00:14] Sodium: 138.00 mEq/L
[00:14] Potassium: 6.10 mEq/L [ABNORMAL]
[08:32] Creatinine: 1.80 mg/dL [ABNORMAL]
```

---

### F19 — Radiology

```
Output: radiology_text — single string per admission
Selection: most recent note within admission window
Cleaning: remove everything before first "EXAMINATION:" marker
```

---

## 8. Embedding Architecture

### Model

| Property | Value |
|----------|-------|
| Model | `Simonlee711/Clinical_ModernBERT` |
| Pre-training | PubMed abstracts, MIMIC-IV clinical notes, medical ontologies (ICD codes) |
| Context window | 8,192 tokens (RoPE positional encoding + Flash Attention) |
| Hidden size | 768 |
| Config key | `BERT_MODEL_NAME` |

### Pooling method: Mean pooling

All embeddings use **mean pooling** over the final hidden states of all non-padding content tokens:

```
token hidden states (final layer):
  t₁    t₂    t₃   ...   tₙ   [PAD] [PAD]
   │     │     │           │
   └─────┴─────┴─────...───┘
              mean
               │
               ▼
         768-d embedding
```

Mean pooling is used because Clinical_ModernBERT is deployed as a **frozen feature extractor** — not fine-tuned end-to-end on the classification task. In this regime, the `[CLS]` token does not reliably encode full-sequence semantics; mean pooling ensures every content token contributes equally to the final vector. This is especially important for long clinical texts: a multi-visit diagnosis history, a discharge note with 600+ tokens, or a lab timeline with 40+ measurements.

**Empty / missing features:** Zero vector of dimension 768.

### Embedding inputs per feature

| Feature | Input to embedding model |
|---------|--------------------------|
| F2 Diagnosis history | Full structured text block (all prior visits concatenated) |
| F3 Discharge history | Full concatenated prior discharge notes |
| F4 Triage | Natural-language triage template string |
| F5 Chief complaint | Raw chief complaint text |
| F6–F18 Lab groups | Text lines for the group's events, concatenated chronologically, one text per admission per group |
| F19 Radiology | Cleaned radiology note text |

### Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `BERT_MODEL_NAME` | `Simonlee711/Clinical_ModernBERT` | HuggingFace model identifier |
| `BERT_MAX_LENGTH` | `8192` | Maximum token length; inputs truncated if exceeded |
| `BERT_BATCH_SIZE` | `32` | Samples per inference call |
| `BERT_DEVICE` | `cuda` | Falls back to CPU automatically if CUDA unavailable |

---

## 9. Final Dataset Assembly

`combine_dataset.py` builds the final flat dataset by left-joining all features onto the admission universe from `data_splits.parquet`:

```
data_splits.parquet          (admission universe — subject_id, hadm_id, split)
        │
        ├── LEFT JOIN y_labels.parquet
        │       y1_mortality, y2_readmission
        │
        ├── LEFT JOIN demographics_features.parquet
        │       demographic_vec  (8 floats)
        │
        ├── LEFT JOIN diag_history_embeddings.parquet
        │       diag_history_embedding  (768 floats)
        │
        ├── LEFT JOIN discharge_history_embeddings.parquet
        │       discharge_history_embedding  (768 floats)
        │
        ├── LEFT JOIN triage_embeddings.parquet
        │       triage_embedding  (768 floats)
        │
        ├── LEFT JOIN chief_complaint_embeddings.parquet
        │       chief_complaint_embedding  (768 floats)
        │
        ├── LEFT JOIN lab_blood_gas_embeddings.parquet
        ├── LEFT JOIN lab_blood_chemistry_embeddings.parquet
        ├── LEFT JOIN lab_blood_hematology_embeddings.parquet
        ├── LEFT JOIN lab_urine_chemistry_embeddings.parquet
        ├── LEFT JOIN lab_urine_hematology_embeddings.parquet
        ├── LEFT JOIN lab_other_body_fluid_chemistry_embeddings.parquet
        ├── LEFT JOIN lab_other_body_fluid_hematology_embeddings.parquet
        ├── LEFT JOIN lab_ascites_embeddings.parquet
        ├── LEFT JOIN lab_pleural_embeddings.parquet
        ├── LEFT JOIN lab_csf_embeddings.parquet
        ├── LEFT JOIN lab_bone_marrow_embeddings.parquet
        ├── LEFT JOIN lab_joint_fluid_embeddings.parquet
        ├── LEFT JOIN lab_stool_embeddings.parquet
        │       lab_{group}_embedding  (768 floats each, zero vector if no events)
        │
        └── LEFT JOIN radiology_embeddings.parquet
                radiology_embedding  (768 floats)
```

The 13 lab group embeddings are discovered dynamically by scanning `EMBEDDINGS_DIR` for `*.parquet` files — no hardcoded list. All left joins mean admissions missing a non-lab feature receive null for that column. Lab group embedding columns are always a 768-float array — admissions with no events in a given group receive a zero vector, consistent with the empty-text convention used throughout the pipeline.

**Intentionally excluded from final dataset:**
- `labs_features.parquet` — superseded by the 13 per-group embedding parquets
- Raw text parquets — superseded by embedding parquets

**Final schema summary:**

| Column | Type | Source |
|--------|------|--------|
| `subject_id` | int | `data_splits.parquet` |
| `hadm_id` | int | `data_splits.parquet` |
| `split` | str | `data_splits.parquet` |
| `y1_mortality` | int | `y_labels.parquet` |
| `y2_readmission` | float (NaN for deceased) | `y_labels.parquet` |
| `demographic_vec` | float[8] | `demographics_features.parquet` |
| `diag_history_embedding` | float[768] | `diag_history_embeddings.parquet` |
| `discharge_history_embedding` | float[768] | `discharge_history_embeddings.parquet` |
| `triage_embedding` | float[768] | `triage_embeddings.parquet` |
| `chief_complaint_embedding` | float[768] | `chief_complaint_embeddings.parquet` |
| `lab_{group}_embedding` ×13 | float[768] | `lab_{group}_embeddings.parquet` |
| `radiology_embedding` | float[768] | `radiology_embeddings.parquet` |

---

## 10. Configuration Reference

All configuration is centralised in `config/preprocessing.yaml` (repository root). No module reads this file directly — `run_pipeline.py` loads it and passes the resulting dict to each module's `run()` function.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `MIMIC_DATA_DIR` | str | — | Root of the MIMIC-IV download (`hosp/`, `icu/` subdirs) |
| `MIMIC_NOTE_DIR` | str | `MIMIC_DATA_DIR` | Root of `mimic-iv-note` module (`note/` subdir) |
| `MIMIC_ED_DIR` | str | `MIMIC_DATA_DIR` | Root of `mimic-iv-ed` module (`ed/` subdir) |
| `SPLIT_TRAIN` | float | `0.80` | Fraction of patients for training |
| `SPLIT_DEV` | float | `0.10` | Fraction of patients for development |
| `SPLIT_TEST` | float | `0.10` | Fraction of patients for test |
| `BERT_MODEL_NAME` | str | `Simonlee711/Clinical_ModernBERT` | HuggingFace model identifier |
| `BERT_MAX_LENGTH` | int | `8192` | Maximum tokeniser length |
| `BERT_BATCH_SIZE` | int | `32` | Embedding batch size |
| `BERT_DEVICE` | str | `cuda` | Inference device; falls back to CPU if unavailable |
| `LAB_ADMISSION_WINDOW` | int or `"full"` | `24` | Hours of lab events to include from `admittime`; `"full"` = entire admission |
| `HADM_LINKAGE_STRATEGY` | str | `"drop"` | How to handle null `hadm_id` records: `"drop"` or `"link"` |
| `HADM_LINKAGE_TOLERANCE_HOURS` | int | `1` | Tolerance in hours for time-window linkage (only used when strategy is `"link"`) |
| `PREPROCESSING_DIR` | str | `data/preprocessing` | Root output directory; `data_splits.parquet` and `source_hashes.json` are written here |
| `FEATURES_DIR` | str | `data/preprocessing/features` | Output directory for raw feature parquets |
| `EMBEDDINGS_DIR` | str | `data/preprocessing/features/embeddings` | Output directory for embedding parquets |
| `CLASSIFICATIONS_DIR` | str | `data/preprocessing/classifications` | Output directory for labels and JSON artefacts |
| `HASH_REGISTRY_PATH` | str | `data/preprocessing/source_hashes.json` | Path to MD5 hash registry for incremental run detection |

---

## 11. Design Principles

### No target leakage

| Rule | Enforcement |
|------|-------------|
| Prior-visit features (F2, F3) use only admissions strictly before current `admittime` | Temporal filter in `extract_diag_history.py`, `extract_discharge_history.py` |
| Lab events restricted to current admission window | `admittime ≤ charttime ≤ admittime + LAB_ADMISSION_WINDOW` |
| OMR vitals restricted to `chartdate ≤ admittime` | Prevents current-admission measurements entering F1 |
| Imputation statistics computed on train split only | Saved to `imputation_stats.json`; applied identically to dev/test |
| No normalisation before split | Normalisation deferred to model training |

### No hardcoding

All file paths, model names, split ratios, batch sizes, window sizes, and thresholds are read from `config/preprocessing.yaml`. No pipeline module contains a hardcoded path or tunable parameter.

### Reproducibility

- Random seed 42 fixed for all train/dev/test splits
- Imputation statistics persisted to `imputation_stats.json`
- Source file MD5 hashes persisted to `source_hashes.json` after each successful module run
- Incremental runs: a module is skipped if all source hashes match and all output files exist; hashes are written only after successful completion, so crashes leave no stale state

### Memory safety

- `labevents` and `chartevents` are streamed in chunks (500,000 rows default)
- Post-chunk merges are performed in memory; minimum 32 GB RAM recommended
- Chunk size is configurable per module (not a global config key — set in module source)

### Graceful degradation

| Missing resource | Behaviour |
|-----------------|-----------|
| `omr` table | Warning logged; falls back to `chartevents` only for height/weight |
| `chartevents` table | Warning logged; demographics module completes with higher missingness |
| CUDA unavailable | Warning logged; automatic fallback to CPU in `embed_features.py` |
| Imputation stratum absent from train data | Falls back to global training statistics |
| No lab events for an admission in a given group | Empty string → zero vector |

---

## 12. Directory Structure

```
HRS/
├── environment.yml
├── config/
│   └── preprocessing.yaml              # Central preprocessing configuration
├── src/
│   └── preprocessing/
│       ├── run_pipeline.py                     # Orchestrator CLI
│       ├── inspect_data.py                     # Read-only diagnostic utility
│       ├── preprocessing_utils.py              # Shared utilities (hashing, CSV loading)
│       ├── build_lab_panel_config.py           # Step 0 — generates lab_panel_config.yaml
│       ├── create_splits.py                    # Step 1
│       ├── extract_demographics.py             # Step 2
│       ├── extract_diag_history.py             # Step 3
│       ├── extract_discharge_history.py        # Step 4
│       ├── extract_triage_and_complaint.py     # Step 5
│       ├── extract_labs.py                     # Step 6
│       ├── extract_radiology.py                # Step 7
│       ├── extract_y_data.py                   # Step 8
│       ├── embed_features.py                   # Step 9
│       ├── combine_dataset.py                  # Step 10
│       └── build_lab_text_lines.py             # Helper — called by extract_labs.py
│
└── data/
    └── preprocessing/                          # Generated artefacts (git-ignored)
        ├── data_splits.parquet
        ├── source_hashes.json
        ├── features/
        │   ├── demographics_features.parquet
        │   ├── diag_history_features.parquet
        │   ├── discharge_history_features.parquet
        │   ├── triage_features.parquet
        │   ├── chief_complaint_features.parquet
        │   ├── labs_features.parquet               # Long format — input to embed_features
        │   ├── radiology_features.parquet
        │   └── embeddings/
        │       ├── diag_history_embeddings.parquet
        │       ├── discharge_history_embeddings.parquet
        │       ├── triage_embeddings.parquet
        │       ├── chief_complaint_embeddings.parquet
        │       ├── radiology_embeddings.parquet
        │       ├── lab_blood_gas_embeddings.parquet
        │       ├── lab_blood_chemistry_embeddings.parquet
        │       ├── lab_blood_hematology_embeddings.parquet
        │       ├── lab_urine_chemistry_embeddings.parquet
        │       ├── lab_urine_hematology_embeddings.parquet
        │       ├── lab_other_body_fluid_chemistry_embeddings.parquet
        │       ├── lab_other_body_fluid_hematology_embeddings.parquet
        │       ├── lab_ascites_embeddings.parquet
        │       ├── lab_pleural_embeddings.parquet
        │       ├── lab_csf_embeddings.parquet
        │       ├── lab_bone_marrow_embeddings.parquet
        │       ├── lab_joint_fluid_embeddings.parquet
        │       └── lab_stool_embeddings.parquet
        └── classifications/
            ├── y_labels.parquet
            ├── final_cdss_dataset.parquet
            ├── lab_panel_config.yaml
            ├── imputation_stats.json
            └── hadm_linkage_stats.json
```
