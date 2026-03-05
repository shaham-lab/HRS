# Preprocessing Pipeline for CDSS-ML (MIMIC-IV)

This document describes the modular preprocessing pipeline for the Clinical Decision Support System (CDSS) built on MIMIC-IV data.

---

## Directory Structure

```
preprocessing/
├── preprocessing.yaml              # Central configuration file
├── preprocessing.md                # This documentation
├── create_splits.py                # Step 1 – patient-level stratified splits
├── extract_demographics.py         # Feature: age, gender, height, weight, BMI
├── extract_diag_history.py         # Feature: prior-visit ICD diagnosis text
├── extract_discharge_history.py    # Feature: prior-visit discharge summary text
├── extract_triage_and_complaint.py # Feature: triage structured data + chief complaint
├── extract_labs.py                 # Feature: lab results (current admission)
├── extract_radiology.py            # Feature: most-recent radiology note (current admission)
├── extract_y_data.py               # Labels: Y1 (mortality) and Y2 (30-day readmission)
├── embed_features.py               # BERT embeddings for all text features
├── combine_dataset.py              # Merges all features into final dataset
└── run_pipeline.py                 # Orchestrator with argparse CLI
```

### Generated Artefacts

| Directory                     | Contents                                                                                                     |
|-------------------------------|--------------------------------------------------------------------------------------------------------------|
| `data/input/features/`             | Raw feature parquet files (demographics, labs, text)                                                         |
| `data/input/embeddings/`           | BERT embedding parquet files                                                                                 |
| `data/input/classifications/`      | Labels (`y_labels.parquet`), splits (`data_splits.parquet`), imputation statistics (`imputation_stats.json`) |

---

## Configuration (`preprocessing.yaml`)

| Key                  | Description                                           | Example value                         |
|----------------------|-------------------------------------------------------|---------------------------------------|
| `MIMIC_DATA_DIR`     | Path to raw MIMIC-IV CSV/parquet files                | `~/data/physionet.org/files/mimiciv/3.1` |
| `SPLIT_TRAIN`        | Fraction of patients assigned to train set            | `0.70`                                |
| `SPLIT_DEV`          | Fraction of patients assigned to dev set              | `0.15`                                |
| `SPLIT_TEST`         | Fraction of patients assigned to test set             | `0.15`                                |
| `BERT_MODEL_NAME`    | HuggingFace model identifier                          | `emilyalsentzer/Bio_ClinicalBERT`     |
| `BERT_MAX_LENGTH`    | Maximum token length for BERT tokenizer               | `512`                                 |
| `BERT_BATCH_SIZE`    | Batch size used when computing embeddings             | `32`                                  |
| `BERT_DEVICE`        | Compute device (`cuda` or `cpu`)                      | `cuda`                                |
| `FEATURES_DIR`       | Output directory for raw feature parquets             | `data/input/features`                 |
| `EMBEDDINGS_DIR`     | Output directory for embedding parquets               | `data/input/embeddings`               |
| `CLASSIFICATIONS_DIR`| Output directory for labels and splits                | `data/input/classifications`          |

> **No hardcoded paths, split ratios, or model names appear in any Python script.** All values are read exclusively from `preprocessing.yaml` at runtime, passed in via `run_pipeline.py`.

---

## Module Descriptions

### 1. `create_splits.py` – Patient-level stratified splits

* **Sources**: `admissions` table, split ratios from config.
* **Logic**: Groups admissions by `subject_id` to prevent patient-level data leakage. Computes a per-patient outcome rate (`hospital_expire_flag`) and uses it to stratify patients into Train / Dev / Test sets in the configured proportions.
* **Output**: `data/input/classifications/data_splits.parquet` — columns: `subject_id`, `hadm_id`, `split`.

---

### 2. `extract_demographics.py` – Age, gender, vitals

* **Sources**: `patients`, `admissions`, `omr`, `chartevents`, `data_splits.parquet`.
* **Logic**:
  * Extracts age, gender, height, weight, BMI.
  * OMR is preferred; `chartevents` is used as fallback.
  * Creates binary missingness indicators (`height_missing`, `weight_missing`, `bmi_missing`).
  * Computes per-stratum (age-bin × gender) mean/std **on train split only** and saves them to `imputation_stats.json`.
  * Imputes missing height/weight by sampling from `N(mean, std)` using train-derived statistics.
  * BMI is derived from height/weight when missing, never independently imputed.
  * No normalisation is applied.
* **Output**: `data/input/features/demographics_features.parquet` — column `demographic_vec` (array of 8 floats: `[Age, Gender, Height, Weight, BMI, height_missing, weight_missing, bmi_missing]`).

---

### 3. `extract_diag_history.py` – Prior-visit diagnosis text

* **Sources**: `diagnoses_icd`, `d_icd_diagnoses`, `admissions`.
* **Logic**: Concatenates ICD `long_title` values from all admissions that precede the current admission (strictly before `admittime`).
* **Output**: `data/input/features/diag_history_features.parquet` — columns: `subject_id`, `hadm_id`, `diag_history_text`.

---

### 4. `extract_discharge_history.py` – Prior-visit discharge summaries

* **Sources**: `note` table (discharge type), `admissions`.
* **Logic**: Retrieves discharge notes from prior admissions only. Removes everything before the first occurrence of `"Allergies:"`.
* **Output**: `data/input/features/discharge_history_features.parquet` — columns: `subject_id`, `hadm_id`, `discharge_history_text`.

---

### 5. `extract_triage_and_complaint.py` – Triage data and chief complaint

* **Sources**: `triage` table, early `chartevents`.
* **Logic**: Converts triage structured fields to a natural-language template. Extracts chief complaint as raw text.
* **Output**:
  * `data/input/features/triage_features.parquet` — columns: `subject_id`, `hadm_id`, `triage_text`.
  * `data/input/features/chief_complaint_features.parquet` — columns: `subject_id`, `hadm_id`, `chief_complaint_text`.

---

### 6. `extract_labs.py` – Lab events (current admission, long format)

* **Sources**: `labevents`, `d_labitems`, `admissions`.
* **Logic**:
  * Reads `labevents` in chunks of 500,000 rows for memory efficiency.
  * Filters out rows with no `hadm_id` (~70% of lab events in MIMIC-IV are outpatient and not linked to an admission).
  * Filters out rows where both `value` and `valuenum` are null.
  * Joins `label`, `fluid`, `category` from `d_labitems`. Strips whitespace and removes artifact rows (`fluid` in `["I", "Q", "fluid"]`).
  * Filters to events within the current admission window (`admittime` ≤ `charttime` ≤ `dischtime`).
  * Converts each lab event to a chronological natural-language text line:
    `[HH:MM] {label} ({fluid}/{category}): {value} {unit} (ref: lower-upper) [ABNORMAL] [STAT]`
    Reference range, abnormal flag, and STAT priority are omitted when not present.
  * Sorts events chronologically within each admission.
  * **No aggregation, no pivoting, no wide format.**
  * Lab embedding is intentionally deferred to training/inference time. The MDP agent selects a subset of `itemid`s, their text lines are concatenated in chronological order, and passed to the language model for encoding.
* **Output**: `data/input/features/labs_features.parquet` — long format, one row per lab event. Columns: `subject_id`, `hadm_id`, `charttime`, `itemid`, `label`, `fluid`, `category`, `lab_text_line`.
* **Note**: `labs_features.parquet` is excluded from `final_cdss_dataset.parquet` and joined dynamically at training time.

---

### 7. `extract_radiology.py` – Radiology notes (current admission)

* **Sources**: `note` table (radiology type).
* **Logic**: Selects the most recent radiology note during the current admission. Removes everything before the first occurrence of `"EXAMINATION:"`.
* **Output**: `data/input/features/radiology_features.parquet` — columns: `subject_id`, `hadm_id`, `radiology_text`.

---

### 8. `extract_y_data.py` – Labels

* **Targets**: Y1 (in-hospital mortality), Y2 (30-day readmission).
* **Logic**:
  * Y1: `admissions.hospital_expire_flag`.
  * Y2: 1 if the patient has a subsequent admission within 30 days of `dischtime`. Patients with `hospital_expire_flag = 1` are excluded from Y2 (set to `NaN`).
* **Output**: `data/input/classifications/y_labels.parquet`.

---

### 9. `embed_features.py` – BERT sentence embeddings

* **Sources**: text parquets from `data/input/features/`, BERT config from `preprocessing.yaml`.
* **Logic**:
  * Loads the model/tokenizer specified by `BERT_MODEL_NAME`.
  * Falls back to CPU with a warning if `BERT_DEVICE: cuda` is set but CUDA is unavailable.
  * Embeds text using the `[CLS]` token representation.
  * Outputs a zero vector for null/empty text.
  * Processes in batches of `BERT_BATCH_SIZE`.
* **Output**: one parquet per text feature saved to `data/input/embeddings/`.

---

### 10. `combine_dataset.py` – Final dataset assembly

* **Logic**: Left-joins all embedding parquets, demographics, and classifications on (`subject_id`, `hadm_id`). Excludes raw text parquets and `labs_features.parquet` (which is in long format and joined dynamically at training time). Ensures the `split` column is present.
* **Output**: `final_cdss_dataset.parquet`.

---

### 11. `run_pipeline.py` – Orchestrator

* Loads all configuration from `preprocessing.yaml`.
* Exposes a CLI via `argparse`; pass `--all` to run the full pipeline or individual module names to run specific steps.
* Enforces execution order: `create_splits` → `extract_*` → `embed_features` → `combine_dataset`.
* Passes the loaded config dict to every module — no module reads `preprocessing.yaml` directly.

---

## Running the Pipeline

```bash
# Run the full pipeline
python preprocessing/run_pipeline.py --all

# Run only the split creation step
python preprocessing/run_pipeline.py --create_splits

# Run only specific extract modules
python preprocessing/run_pipeline.py --extract_demographics --extract_labs

# Run embedding step
python preprocessing/run_pipeline.py --embed_features

# Run final combine step
python preprocessing/run_pipeline.py --combine_dataset
```

---

## Design Principles

1. **No leakage**: imputation statistics are derived exclusively from the train split.
2. **No hardcoding**: all paths, model names, and hyperparameters live in `preprocessing.yaml`.
3. **Reproducibility**: imputation statistics are persisted to `imputation_stats.json`.
4. **Memory safety**: large tables (e.g., `labevents`) are read in configurable chunks.
5. **Graceful degradation**: CUDA fall-back to CPU; per-stratum fall-back to global statistics.
