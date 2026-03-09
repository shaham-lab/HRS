# Preprocessing Pipeline — CDSS-ML (MIMIC-IV)

This document is a developer reference for the preprocessing pipeline. For runtime instructions and configuration, see `preprocessing-runtime-instructions.md`. For the full architecture specification, see `PREPROCESSING_ARCHITECTURE.md`.

---

## Directory Structure

```
HRS/
├── config/
│   └── preprocessing.yaml              # Central configuration
├── src/
│   └── preprocessing/
│       ├── run_pipeline.py                     # Orchestrator CLI
│       ├── preprocessing_utils.py              # Shared utilities
│       ├── build_lab_panel_config.py           # Step 0
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
└── data/
    └── preprocessing/
        ├── data_splits.parquet
        ├── source_hashes.json
        ├── features/
        │   ├── demographics_features.parquet
        │   ├── diag_history_features.parquet
        │   ├── discharge_history_features.parquet
        │   ├── triage_features.parquet
        │   ├── chief_complaint_features.parquet
        │   ├── labs_features.parquet
        │   ├── radiology_features.parquet
        │   └── embeddings/
        │       ├── diag_history_embeddings.parquet
        │       ├── discharge_history_embeddings.parquet
        │       ├── triage_embeddings.parquet
        │       ├── chief_complaint_embeddings.parquet
        │       ├── radiology_embeddings.parquet
        │       └── lab_{group}_embeddings.parquet  (×13)
        └── classifications/
            ├── y_labels.parquet
            ├── final_cdss_dataset.parquet
            ├── lab_panel_config.yaml
            ├── imputation_stats.json
            └── hadm_linkage_stats.json
```

---

## Configuration (`config/preprocessing.yaml`)

All configuration is centralised in `config/preprocessing.yaml`. No module reads this file directly — `run_pipeline.py` loads it and passes the resulting dict to each module's `run()` function.

| Key | Description | Default |
|-----|-------------|---------|
| `MIMIC_DATA_DIR` | Root of MIMIC-IV download | — |
| `MIMIC_NOTE_DIR` | Root of mimic-iv-note module | `MIMIC_DATA_DIR` |
| `MIMIC_ED_DIR` | Root of mimic-iv-ed module | `MIMIC_DATA_DIR` |
| `SPLIT_TRAIN` | Train fraction | `0.80` |
| `SPLIT_DEV` | Dev fraction | `0.10` |
| `SPLIT_TEST` | Test fraction | `0.10` |
| `BERT_MODEL_NAME` | HuggingFace model identifier | `Simonlee711/Clinical_ModernBERT` |
| `BERT_MAX_LENGTH` | Maximum token length | `8192` |
| `BERT_BATCH_SIZE` | Embedding batch size | `32` |
| `BERT_DEVICE` | Inference device | `cuda` |
| `LAB_ADMISSION_WINDOW` | Hours from admittime for lab extraction; `"full"` = entire admission | `24` |
| `HADM_LINKAGE_STRATEGY` | How to handle null `hadm_id`: `"drop"` or `"link"` | `"drop"` |
| `HADM_LINKAGE_TOLERANCE_HOURS` | Tolerance in hours for time-window linkage | `1` |
| `PREPROCESSING_DIR` | Root output directory | `data/preprocessing` |
| `FEATURES_DIR` | Output directory for raw feature parquets | `data/preprocessing/features` |
| `EMBEDDINGS_DIR` | Output directory for embedding parquets | `data/preprocessing/features/embeddings` |
| `CLASSIFICATIONS_DIR` | Output directory for labels and final dataset | `data/preprocessing/classifications` |
| `HASH_REGISTRY_PATH` | Path to MD5 hash registry | `data/preprocessing/source_hashes.json` |

---

## Module Descriptions

### Step 0 — `build_lab_panel_config.py`

Reads `d_labitems`, groups `itemid`s by `(fluid × category)` into 13 named lab groups, and writes `lab_panel_config.yaml` to `CLASSIFICATIONS_DIR`. Must run before `extract_labs.py`.

Groups: `blood_gas`, `blood_chemistry`, `blood_hematology`, `urine_chemistry`, `urine_hematology`, `other_body_fluid_chemistry`, `other_body_fluid_hematology`, `ascites`, `pleural`, `csf`, `bone_marrow`, `joint_fluid`, `stool`.

### Step 1 — `create_splits.py`

Patient-level stratified 3-way split (train/dev/test) by `subject_id`. Stratified by patient-level `hospital_expire_flag` rate. Output: `data/preprocessing/data_splits.parquet`.

### Step 2 — `extract_demographics.py`

Extracts age, gender, height, weight, BMI. Sources: `patients`, `admissions`, `omr` (preferred), `chartevents` (fallback). Creates missingness flags before imputation. Computes stratum statistics (age-bin × gender) on train split only; saves to `imputation_stats.json`. Implements `HADM_LINKAGE_STRATEGY` for null `hadm_id` in `chartevents`. Output: `demographics_features.parquet` — `demographic_vec` array of 8 floats.

### Step 3 — `extract_diag_history.py`

Prior-visit ICD diagnosis text per admission. Only admissions strictly before current `admittime` are included. Format: dated section headers with one `long_title` per line per visit. Empty string if no prior visits.

### Step 4 — `extract_discharge_history.py`

Prior-visit discharge summary text. Text cleaning removes everything before `"Allergies:"`. Notes concatenated with dated headers in chronological order.

### Step 5 — `extract_triage_and_complaint.py`

Triage structured fields formatted as a natural-language template. Chief complaint as raw text. `hadm_id` resolved via `edstays` (primary) and intime-based fallback.

### Step 6 — `extract_labs.py`

Lab events from the current admission in long format (one row per event). Controlled by `LAB_ADMISSION_WINDOW`. Respects `HADM_LINKAGE_STRATEGY` for null `hadm_id` in `labevents`. Text line format:

```
[HH:MM] {label}: {value} {unit} (ref: lower-upper) [ABNORMAL]
```

`[HH:MM]` is elapsed time since `admittime`. `[ABNORMAL]` is appended when `flag == "abnormal"` OR `valuenum` is outside reference range.

### Step 7 — `extract_radiology.py`

Most recent radiology note during the current admission. Text cleaning removes everything before `"EXAMINATION:"`.

### Step 8 — `extract_y_data.py`

Y1: `hospital_expire_flag`. Y2: readmission within 30 days of `dischtime`; NaN for deceased patients.

### Step 9 — `embed_features.py`

Embeds all text features using `Clinical_ModernBERT` (`Simonlee711/Clinical_ModernBERT`). **Mean pooling** over all non-padding content tokens from the final hidden layer. Empty/null text produces a zero vector. Produces 5 non-lab embedding parquets and 13 lab group embedding parquets (one per group defined in `lab_panel_config.yaml`). All 13 lab group parquets always contain a valid embedding — admissions with no events in a given group receive a zero vector.

### Step 10 — `combine_dataset.py`

Left-joins all embedding parquets (discovered dynamically from `EMBEDDINGS_DIR`), `demographics_features.parquet`, `y_labels.parquet`, and `data_splits.parquet` on `(subject_id, hadm_id)`. Starts from the admission universe in `data_splits.parquet`. Output: `final_cdss_dataset.parquet` — one row per admission.

---

## Running the Pipeline

```bash
# Full pipeline
python src/preprocessing/run_pipeline.py --all

# Individual steps
python src/preprocessing/run_pipeline.py --create_splits
python src/preprocessing/run_pipeline.py --extract_demographics --extract_labs
python src/preprocessing/run_pipeline.py --embed_features
python src/preprocessing/run_pipeline.py --combine_dataset

# Force rerun of a specific module
python src/preprocessing/run_pipeline.py --extract_labs --force-module extract_labs
```

---

## Design Principles

1. **No leakage** — imputation statistics derived from train split only; prior-visit features use strictly-before-admittime filter.
2. **No hardcoding** — all paths, model names, and hyperparameters in `config/preprocessing.yaml`.
3. **Reproducibility** — imputation statistics persisted to `imputation_stats.json`; source file hashes to `source_hashes.json`.
4. **Memory safety** — large tables (`labevents`, `chartevents`) read in configurable chunks.
5. **Graceful degradation** — CUDA falls back to CPU; missing optional sources are logged and skipped.

