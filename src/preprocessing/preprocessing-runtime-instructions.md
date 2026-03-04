# Preprocessing Pipeline — Runtime Instructions

This document explains how to configure and run the CDSS preprocessing pipeline
end-to-end on MIMIC-IV data. For a technical description of the data-processing
choices, see [DATA_PROCESSING.md](DATA_PROCESSING.md).

---

## 1. Prerequisites

### Python version

Python **3.10 or later** is required.

### Required packages

| Package | Purpose |
|---------|---------|
| `pandas` | DataFrame operations |
| `numpy` | Numerical processing |
| `scikit-learn` | Stratified train/dev/test splitting |
| `transformers` | HuggingFace BERT tokenizer and model |
| `torch` | BERT inference backend |
| `pyyaml` | Configuration file parsing |
| `pyarrow` | Parquet read/write |

Install all dependencies:

```bash
pip install -r requirements.txt
```

### MIMIC-IV data access

MIMIC-IV requires **credentialed access** via [PhysioNet](https://physionet.org/content/mimiciv/).
After obtaining access, download the following modules and place them at the
paths configured in `preprocessing.yaml` (see [Section 3](#3-configuration-preprocessingyaml)):

**`hosp/` subdirectory** (under `MIMIC_DATA_DIR`):

| Table | Used by |
|-------|---------|
| `admissions` | all modules |
| `patients` | `extract_demographics` |
| `omr` | `extract_demographics` |
| `diagnoses_icd` | `extract_diag_history` |
| `d_icd_diagnoses` | `extract_diag_history` |
| `d_labitems` | `extract_labs` |
| `labevents` | `extract_labs` |

**`icu/` subdirectory** (under `MIMIC_DATA_DIR`):

| Table | Used by |
|-------|---------|
| `chartevents` | `extract_demographics` (height/weight fallback) |

**`note/` subdirectory** (under `MIMIC_NOTE_DIR`, i.e. the `mimic-iv-note` module):

| Table | Used by |
|-------|---------|
| `discharge` | `extract_discharge_history` |
| `radiology` | `extract_radiology` |

**`ed/` subdirectory** (under `MIMIC_DATA_DIR`):

| Table | Used by |
|-------|---------|
| `triage` | `extract_triage_and_complaint` |

All tables are expected as **`.csv.gz`** (gzip-compressed CSV). Uncompressed
**`.csv`** is supported as a fallback where `.csv.gz` is absent.

---

## 2. Directory layout

```
HRS/
├── requirements.txt
├── src/
│   └── preprocessing/
│       ├── preprocessing.yaml                  # Central configuration
│       ├── run_pipeline.py                     # Orchestrator CLI
│       ├── inspect_data.py                     # Read-only diagnostic utility
│       ├── create_splits.py
│       ├── extract_demographics.py
│       ├── extract_diag_history.py
│       ├── extract_discharge_history.py
│       ├── extract_triage_and_complaint.py
│       ├── extract_labs.py
│       ├── extract_radiology.py
│       ├── extract_y_data.py
│       ├── embed_features.py
│       ├── combine_dataset.py
│       └── preprocessing_utils.py
│
└── input/                                      # Generated artefacts (git-ignored)
    ├── features/
    │   ├── demographics_features.parquet
    │   ├── diag_history_features.parquet
    │   ├── discharge_history_features.parquet
    │   ├── triage_features.parquet
    │   ├── chief_complaint_features.parquet
    │   ├── labs_features.parquet
    │   └── radiology_features.parquet
    ├── embeddings/
    │   ├── diag_history_embeddings.parquet
    │   ├── discharge_history_embeddings.parquet
    │   ├── triage_embeddings.parquet
    │   ├── chief_complaint_embeddings.parquet
    │   └── radiology_embeddings.parquet
    └── classifications/
        ├── data_splits.parquet
        ├── y_labels.parquet
        ├── imputation_stats.json
        ├── source_hashes.json
        └── final_cdss_dataset.parquet
```

---

## 3. Configuration (`preprocessing.yaml`)

All configuration is centralised in `src/preprocessing/preprocessing.yaml`.
No paths or parameters are hardcoded in any pipeline script — all values flow
from this file.

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `MIMIC_DATA_DIR` | `str` | Root of the MIMIC-IV data download (contains `hosp/`, `icu/`, `ed/`). Supports `~` expansion. | `"~/data/physionet.org/files/mimiciv/3.1"` |
| `MIMIC_NOTE_DIR` | `str` | Root of the `mimic-iv-note` module (contains `note/`). Falls back to `MIMIC_DATA_DIR/note` if omitted. Supports `~` expansion. | `"~/data/physionet.org/files/mimic-iv-note/2.2"` |
| `SPLIT_TRAIN` | `float` | Fraction of patients for training. Must sum to 1.0 with `SPLIT_DEV` and `SPLIT_TEST`. | `0.70` |
| `SPLIT_DEV` | `float` | Fraction of patients for dev/validation. | `0.15` |
| `SPLIT_TEST` | `float` | Fraction of patients for testing. | `0.15` |
| `BERT_MODEL_NAME` | `str` | HuggingFace model identifier used for embedding text features. | `"emilyalsentzer/Bio_ClinicalBERT"` |
| `BERT_MAX_LENGTH` | `int` | Maximum token length passed to the BERT tokenizer. | `512` |
| `BERT_BATCH_SIZE` | `int` | Number of text samples per inference batch. | `32` |
| `BERT_DEVICE` | `str` | Inference device: `"cuda"` or `"cpu"`. Falls back to CPU automatically if CUDA is unavailable. | `"cuda"` |
| `FEATURES_DIR` | `str` | Output directory for raw feature parquets. | `"input/features"` |
| `EMBEDDINGS_DIR` | `str` | Output directory for BERT embedding parquets. | `"input/embeddings"` |
| `CLASSIFICATIONS_DIR` | `str` | Output directory for label parquets, split files, and JSON artefacts. | `"input/classifications"` |
| `HASH_REGISTRY_PATH` | `str` | Path to the JSON file that stores MD5 hashes of source files for incremental-run detection. | `"input/classifications/source_hashes.json"` |

---

## 4. Running the pipeline

All commands are run from the **repository root**.

### Full pipeline

```bash
python src/preprocessing/run_pipeline.py --all
```

### Individual steps

```bash
python src/preprocessing/run_pipeline.py --create_splits
python src/preprocessing/run_pipeline.py --extract_demographics
python src/preprocessing/run_pipeline.py --extract_diag_history
python src/preprocessing/run_pipeline.py --extract_discharge_history
python src/preprocessing/run_pipeline.py --extract_triage_and_complaint
python src/preprocessing/run_pipeline.py --extract_labs
python src/preprocessing/run_pipeline.py --extract_radiology
python src/preprocessing/run_pipeline.py --extract_y_data
python src/preprocessing/run_pipeline.py --embed_features
python src/preprocessing/run_pipeline.py --combine_dataset
```

Multiple steps can be combined in a single invocation:

```bash
python src/preprocessing/run_pipeline.py --extract_demographics --extract_labs
```

### Force flags

| Flag | Effect |
|------|--------|
| `--force` | Force rerun of **all** selected modules even if source files are unchanged. |
| `--force-module MODULE [MODULE ...]` | Force rerun of **specific named modules** only. |

```bash
# Rerun everything regardless of hashes
python src/preprocessing/run_pipeline.py --all --force

# Rerun only extract_labs and downstream steps
python src/preprocessing/run_pipeline.py --extract_labs --embed_features --combine_dataset \
    --force-module extract_labs embed_features combine_dataset

# --force and --force-module can be combined (--force takes precedence for its modules)
python src/preprocessing/run_pipeline.py --all --force-module extract_labs
```

### Custom config file

```bash
python src/preprocessing/run_pipeline.py --all \
    --config /path/to/my_preprocessing.yaml
```

### Recommended execution order

The pipeline has the following dependency structure:

```
create_splits
  └─► extract_demographics       ─┐
  └─► extract_diag_history        │
  └─► extract_discharge_history   │  (these can run in parallel)
  └─► extract_triage_and_complaint│
  └─► extract_labs                │
  └─► extract_radiology           │
  └─► extract_y_data             ─┘
        └─► embed_features
              └─► combine_dataset
```

**`create_splits` must complete first.** All seven `extract_*` modules depend
on `data_splits.parquet` and can be run in parallel once splits exist.
`embed_features` requires all text feature parquets. `combine_dataset` requires
all embedding parquets and labels.

---

## 5. Hash-based incremental runs

### How it works

Each module records the MD5 hash of every source file it reads in
`source_hashes.json` (configured via `HASH_REGISTRY_PATH`). On a subsequent
run, the module compares current file hashes against the stored values. If all
source hashes match **and** all output files exist, the module logs a skip
message and returns immediately without re-processing.

The hash registry is written **only after** a module completes successfully.
If a run crashes mid-way, no stale hashes are recorded and the module will
reprocess on the next run.

### When to use `--force` vs `--force-module`

| Scenario | Recommended flag |
|----------|-----------------|
| Initial run (no hashes yet) | Neither — hash check is a no-op when outputs are missing |
| Source files completely unchanged, want to re-embed with a new BERT model | `--force-module embed_features combine_dataset` |
| One source table updated | `--force-module <affected_module> embed_features combine_dataset` |
| Want to wipe and redo everything | `--force` |

### Example: only `labevents.csv.gz` was updated

```bash
python src/preprocessing/run_pipeline.py \
    --extract_labs --embed_features --combine_dataset \
    --force-module extract_labs embed_features combine_dataset
```

`extract_demographics`, `extract_diag_history`, etc. are not re-run because
their source files have not changed.

---

## 6. Inspecting the data

`inspect_data.py` is a **read-only** diagnostic utility that loads each
MIMIC-IV source file and prints shape, dtypes, null counts, and a short
numeric summary. It writes no files and is safe to run at any time.

```bash
python src/preprocessing/inspect_data.py
python src/preprocessing/inspect_data.py --config /path/to/preprocessing.yaml
```

---

## 7. Output files reference

| File | Location | Format | Produced by | Description |
|------|----------|--------|-------------|-------------|
| `data_splits.parquet` | `input/classifications/` | Parquet | `create_splits` | One row per admission with `split` column (`train`/`dev`/`test`) |
| `demographics_features.parquet` | `input/features/` | Parquet | `extract_demographics` | One row per admission; `demographic_vec` array column (8 floats) |
| `diag_history_features.parquet` | `input/features/` | Parquet | `extract_diag_history` | One row per admission; `diag_history_text` string column |
| `discharge_history_features.parquet` | `input/features/` | Parquet | `extract_discharge_history` | One row per admission; `discharge_history_text` string column |
| `triage_features.parquet` | `input/features/` | Parquet | `extract_triage_and_complaint` | One row per admission; `triage_text` string column |
| `chief_complaint_features.parquet` | `input/features/` | Parquet | `extract_triage_and_complaint` | One row per admission; `chief_complaint_text` string column |
| `labs_features.parquet` | `input/features/` | Parquet | `extract_labs` | Long format — one row per lab event; columns: `subject_id`, `hadm_id`, `charttime`, `itemid`, `label`, `fluid`, `category`, `lab_text_line` |
| `radiology_features.parquet` | `input/features/` | Parquet | `extract_radiology` | One row per admission; `radiology_text` string column |
| `y_labels.parquet` | `input/classifications/` | Parquet | `extract_y_data` | One row per admission; `y1_mortality` and `y2_readmission` columns |
| `imputation_stats.json` | `input/classifications/` | JSON | `extract_demographics` | Per-stratum (age-bin × gender) mean/std used for height/weight imputation, computed on train split only |
| `source_hashes.json` | `input/classifications/` | JSON | all modules | MD5 hashes of source files per module; drives incremental-run skipping |
| `diag_history_embeddings.parquet` | `input/embeddings/` | Parquet | `embed_features` | One row per admission; `diag_history_embedding` array column |
| `discharge_history_embeddings.parquet` | `input/embeddings/` | Parquet | `embed_features` | One row per admission; `discharge_history_embedding` array column |
| `triage_embeddings.parquet` | `input/embeddings/` | Parquet | `embed_features` | One row per admission; `triage_embedding` array column |
| `chief_complaint_embeddings.parquet` | `input/embeddings/` | Parquet | `embed_features` | One row per admission; `chief_complaint_embedding` array column |
| `radiology_embeddings.parquet` | `input/embeddings/` | Parquet | `embed_features` | One row per admission; `radiology_embedding` array column |
| `final_cdss_dataset.parquet` | `input/classifications/` | Parquet | `combine_dataset` | One row per admission; all features and labels joined; labs excluded (long-format, joined at training time) |

---

## 8. Troubleshooting

### `data_splits.parquet not found`

**Cause:** `combine_dataset` (or another downstream module) ran before
`create_splits`.

**Fix:** Run `create_splits` first, then re-run the failing module.

```bash
python src/preprocessing/run_pipeline.py --create_splits
```

---

### CUDA requested but not available

**Symptom:** `embed_features` raises a CUDA-related error or falls back with a
warning.

**Fix:** Set `BERT_DEVICE: cpu` in `preprocessing.yaml`. The pipeline
automatically falls back to CPU if CUDA is requested but unavailable, but
setting it explicitly avoids the warning.

---

### Module skipped unexpectedly

**Cause:** The hash check found that all source files are unchanged and all
output files exist, so the module was skipped.

**Fix:** Use `--force-module` to force a specific module to rerun regardless of
hashes:

```bash
python src/preprocessing/run_pipeline.py --extract_labs \
    --force-module extract_labs
```

---

### OMR or `chartevents` not found

**Cause:** These tables are optional. If `omr.csv.gz` is missing,
`extract_demographics` logs a warning and falls back to `chartevents` only.
If `chartevents` is also missing, the module still succeeds but height and
weight will have higher missingness.

**Effect:** The `height_missing`, `weight_missing`, and `bmi_missing` flags in
`demographic_vec` will reflect higher rates of imputation, but the pipeline
does not crash.

---

### Memory errors on `labevents` or `chartevents`

**Cause:** These are the largest MIMIC-IV tables. The pipeline streams them in
chunks of 500,000 rows to limit peak memory usage. However, the post-chunk
merge with admissions is performed in memory.

**Fix:** Run on a machine with at least **32 GB RAM**. If memory is still
exhausted, reduce `_CHUNK_SIZE` in `extract_labs.py` and
`extract_demographics.py` (this increases I/O but reduces peak memory).
