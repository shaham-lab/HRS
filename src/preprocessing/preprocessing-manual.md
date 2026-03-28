# Preprocessing Pipeline — Manual

This document explains how to configure and run the CDSS preprocessing pipeline
end-to-end on MIMIC-IV data. For a technical description of the data-processing
choices, see [DATA_PROCESSING.md](DATA_PROCESSING.md).

---

## 1. Prerequisites

### Python version

Python **3.12** is required (managed via the project's conda environment).

### Environment setup

The project uses **conda** for environment management. Install
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
if you do not already have it, then create and activate the environment from
the repository root:

```bash
conda env create -f environment.yml
conda activate hrs
```

The environment installs all required packages, including:

| Package        | Purpose                              |
| -------------- | ------------------------------------ |
| `pandas`       | DataFrame operations                 |
| `numpy`        | Numerical processing                 |
| `scikit-learn` | Stratified train/dev/test splitting  |
| `transformers` | HuggingFace BERT tokenizer and model |
| `torch`        | BERT inference backend               |
| `pyyaml`       | Configuration file parsing           |
| `pyarrow`      | Parquet read/write                   |

To update an existing environment after changes to `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

### MIMIC-IV data access

MIMIC-IV requires **credentialed access** via [PhysioNet](https://physionet.org/content/mimiciv/).
After obtaining access, download the following modules and place them at the
paths configured in `preprocessing.yaml` (see [Section 3](#3-configuration-configpreprocessingyaml)):

**`hosp/` subdirectory** (under `MIMIC_DATA_DIR`):

| Table             | Used by                |
| ----------------- | ---------------------- |
| `admissions`      | all modules            |
| `patients`        | `extract_demographics` |
| `omr`             | `extract_demographics` |
| `diagnoses_icd`   | `extract_diag_history` |
| `d_icd_diagnoses` | `extract_diag_history` |
| `d_labitems`      | `extract_labs`         |
| `labevents`       | `extract_labs`         |

**`icu/` subdirectory** (under `MIMIC_DATA_DIR`):

| Table         | Used by                                         |
| ------------- | ----------------------------------------------- |
| `chartevents` | `extract_demographics` (height/weight fallback) |

**`note/` subdirectory** (under `MIMIC_NOTE_DIR`, i.e. the `mimic-iv-note` module):

| Table       | Used by                     |
| ----------- | --------------------------- |
| `discharge` | `extract_discharge_history` |
| `radiology` | `extract_radiology`         |

**`ed/` subdirectory** (under `MIMIC_ED_DIR`, i.e. the `mimic-iv-ed` module):

| Table     | Used by                                                                       |
| --------- | ----------------------------------------------------------------------------- |
| `triage`  | `extract_triage_and_complaint`                                                |
| `edstays` | `extract_triage_and_complaint` (bridge table to resolve `stay_id → hadm_id`) |

All tables are expected as **`.csv.gz`** (gzip-compressed CSV). Uncompressed
**`.csv`** is supported as a fallback where `.csv.gz` is absent.

---

## 2. Directory layout

```
HRS/
├── environment.yml
├── config/
│   ├── preprocessing.yaml              # Central preprocessing configuration
│   └── micro_panel_config.yaml         # Microbiology panel definitions (version-controlled)
├── src/
│   └── preprocessing/
│       ├── run_pipeline.py                     # Orchestrator CLI
│       ├── inspect_data.py                     # Read-only diagnostic utility
│       ├── create_splits.py
│       ├── extract_demographics.py
│       ├── extract_diag_history.py
│       ├── extract_discharge_history.py
│       ├── extract_triage_and_complaint.py
│       ├── build_lab_text_lines.py             # Helper called by extract_labs
│       ├── extract_labs.py
│       ├── build_micro_text.py                 # Helper called by extract_microbiology
│       ├── extract_microbiology.py
│       ├── extract_radiology.py
│       ├── extract_y_data.py
│       ├── embed_features.py
│       ├── combine_dataset.py
│       ├── preprocessing_utils.py
│       ├── pipeline_job.sh             # Slurm: all steps except embed_features
│       ├── labs_extract_job.sh         # Slurm: extract_labs only
│       ├── micro_extract_job.sh        # Slurm: extract_microbiology only
│       ├── embed_job.sh                # Slurm: embed one GPU slice
│       ├── combine_job.sh              # Slurm: combine_dataset
│       └── submit_all.sh              # Slurm: auto-submit entrypoint
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
        │   ├── labs_features.parquet
        │   ├── micro_<panel>.parquet
        │   ├── radiology_features.parquet
        │   └── embeddings/
        │       ├── diag_history_embeddings.parquet
        │       ├── discharge_history_embeddings.parquet
        │       ├── triage_embeddings.parquet
        │       ├── chief_complaint_embeddings.parquet
        │       ├── radiology_embeddings.parquet
        │       └── lab_{group}_embeddings.parquet  # ×13, e.g. lab_blood_chemistry_embeddings.parquet
        └── classifications/
            ├── y_labels.parquet
            ├── imputation_stats.json
            ├── lab_panel_config.yaml
            ├── hadm_linkage_stats.json
            └── final_cdss_dataset.parquet
```

---

## 3. Configuration (`config/preprocessing.yaml`)

All configuration is centralised in `config/preprocessing.yaml` (repository root).
No paths or parameters are hardcoded in any pipeline script — all values flow
from this file.

| Key                   | Type    | Description                                                                                                                                                                                      | Example                                          |
| --------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
| `MIMIC_DATA_DIR`      | `str`   | Root of the MIMIC-IV data download (contains `hosp/`, `icu/`). Supports `~` expansion.                                                                                                           | `"~/data/physionet.org/files/mimiciv/3.1"`       |
| `MIMIC_NOTE_DIR`      | `str`   | Root of the `mimic-iv-note` module (contains `note/`). Falls back to `MIMIC_DATA_DIR/note` if omitted. Supports `~` expansion.                                                                   | `"~/data/physionet.org/files/mimic-iv-note/2.2"` |
| `MIMIC_ED_DIR`        | `str`   | Root of the `mimic-iv-ed` module (contains `ed/`). The triage table is looked up here first; falls back to `MIMIC_DATA_DIR/ed/` then `MIMIC_DATA_DIR/hosp/` if omitted. Supports `~` expansion.  | `"~/data/physionet.org/files/mimic-iv-ed/2.2"`   |
| `SPLIT_TRAIN`         | `float` | Fraction of patients for training. Must sum to 1.0 with `SPLIT_DEV` and `SPLIT_TEST`.                                                                                                            | `0.80`                                           |
| `SPLIT_DEV`           | `float` | Fraction of patients for dev/validation.                                                                                                                                                         | `0.10`                                           |
| `SPLIT_TEST`          | `float` | Fraction of patients for testing.                                                                                                                                                                | `0.10`                                           |
| `BERT_MODEL_NAME`     | `str`   | HuggingFace model identifier used for embedding text features.                                                                                                                                   | `"Simonlee711/Clinical_ModernBERT"`              |
| `BERT_MAX_LENGTH`     | `int`   | Maximum token length passed to the BERT tokenizer.                                                                                                                                               | `8192`                                           |
| `BERT_BATCH_SIZE`     | `int`   | Number of text samples per inference batch.                                                                                                                                                      | `32`                                             |
| `BERT_DEVICE`         | `str`   | Inference device: `"cuda"` or `"cpu"`. Falls back to CPU automatically if CUDA is unavailable.                                                                                                   | `"cuda"`                                         |
| `PREPROCESSING_DIR`   | `str`   | Root output directory; `data_splits.parquet` and `source_hashes.json` are written here.                                                                                                         | `"data/preprocessing"`                           |
| `FEATURES_DIR`        | `str`   | Output directory for raw feature parquets.                                                                                                                                                       | `"data/preprocessing/features"`                  |
| `EMBEDDINGS_DIR`      | `str`   | Output directory for BERT embedding parquets.                                                                                                                                                    | `"data/preprocessing/features/embeddings"`       |
| `CLASSIFICATIONS_DIR` | `str`   | Output directory for label parquets and JSON artefacts.                                                                                                                                          | `"data/preprocessing/classifications"`           |
| `HASH_REGISTRY_PATH`  | `str`   | Path to the JSON file that stores MD5 hashes of source files for incremental-run detection.                                                                                                      | `"data/preprocessing/source_hashes.json"`        |
| `HADM_LINKAGE_STRATEGY` | `str` | How to handle records with null `hadm_id`. `"drop"` excludes them (default); `"link"` attempts time-window linkage using `charttime` and admission windows.                                    | `"drop"`                                         |
| `HADM_LINKAGE_TOLERANCE_HOURS` | `int` | Hours of tolerance outside `admittime`/`dischtime` used when `HADM_LINKAGE_STRATEGY` is `"link"`. Ignored when strategy is `"drop"`.                                                   | `2`                                              |
| `LAB_ADMISSION_WINDOW` | `int` or `"full"` | Hours from `admittime` to include in `labs_features.parquet`. Integer: include events within this many hours of `admittime`. `"full"`: include all events within the full admission. | `24`                                             |
| `BERT_MAX_GPUS`        | `int` or `null`   | Number of GPUs to use per embed slice job. `null` defaults to 2. Controls how many GPU processes are spawned inside each `embed_job.sh` Slurm job.                                   | `2`                                              |
| `BERT_SLICE_SIZE_PER_GPU` | `int`          | Number of admissions each GPU processes per embed slice. Together with `BERT_MAX_GPUS` and the total admission count, determines how many Slurm embed slice jobs are created: `ceil(total_admissions / (BERT_SLICE_SIZE_PER_GPU × BERT_MAX_GPUS))`. | `20000`                                          |

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
python src/preprocessing/run_pipeline.py --build_lab_panel_config
python src/preprocessing/run_pipeline.py --extract_labs
python src/preprocessing/run_pipeline.py --extract_microbiology
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

| Flag                                 | Effect                                                                      |
| ------------------------------------ | --------------------------------------------------------------------------- |
| `--force`                            | Force rerun of **all** selected modules even if source files are unchanged. |
| `--force-module MODULE [MODULE ...]` | Force rerun of **specific named modules** only.                             |

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
  └─► build_lab_panel_config ─► extract_labs
  └─► extract_microbiology
  └─► extract_radiology           │
  └─► extract_y_data             ─┘
        └─► embed_features
              └─► combine_dataset
```

**`create_splits` must complete first.** `build_lab_panel_config` must run
before `extract_labs`. `micro_panel_config.yaml` is version-controlled and
requires no build step — `extract_microbiology` can run as soon as splits exist.
All other `extract_*` modules can run in parallel once splits exist.

---

### Running on a Slurm cluster

For HPC environments, four Slurm shell scripts are provided in
`src/preprocessing/`. The intended workflow splits the pipeline into three
separate job types because embedding is GPU-intensive and must be
checkpointed across many slices:

| Script             | Slurm resources          | What it does                                                                                        |
| ------------------ | ------------------------ | --------------------------------------------------------------------------------------------------- |
| `pipeline_job.sh`  | 4 CPUs, 64 GB RAM        | Runs the full `run_pipeline.py --all --skip-modules embed_features` (all steps except embedding).   |
| `embed_job.sh`     | 2 GPUs, 8 CPUs, 64 GB RAM | Runs `embed_features.py --slice-index <i>` for one slice of admissions.                            |
| `combine_job.sh`   | 4 CPUs, 32 GB RAM        | Runs `run_pipeline.py --modules combine_dataset` once all embedding slices are complete.            |
| `submit_all.sh`    | —                        | Auto-detect state and submit the correct subset of jobs with `afterok` dependency chaining.         |

#### Recommended workflow: `submit_all.sh`

`submit_all.sh` is the only script you normally need to call. Run it from
the repository root:

```bash
bash src/preprocessing/submit_all.sh
```

It checks the current pipeline state, then submits only the jobs that still
need to run:

| State detected                              | Jobs submitted                                             |
| ------------------------------------------- | ---------------------------------------------------------- |
| Nothing done yet                            | `pipeline_job` → N embed slices (chained) → `combine_job` |
| Preprocessing done, embedding incomplete    | Remaining embed slices (chained) → `combine_job`           |
| All embeddings complete                     | `combine_job` only                                         |
| Everything complete                         | Nothing — prints status and exits                          |

Re-run `submit_all.sh` at any time; it always resumes from where the pipeline
left off without duplicating work.

#### How embed slices are created

The number of Slurm embed slice jobs is computed automatically from
`BERT_SLICE_SIZE_PER_GPU` and `BERT_MAX_GPUS` in `config/preprocessing.yaml`:

```
N_slices = ceil(total_admissions / (BERT_SLICE_SIZE_PER_GPU × BERT_MAX_GPUS))
```

With the default settings (`BERT_SLICE_SIZE_PER_GPU: 20000`, `BERT_MAX_GPUS: 2`)
and ~546 k admissions this produces **14 slices**. Each slice is submitted as a
separate `embed_job.sh` job. Slices are chained sequentially with
`--dependency=afterok` to prevent concurrent write conflicts on the parquet
checkpoint files.

To submit an individual embed slice manually:

```bash
sbatch src/preprocessing/embed_job.sh <slice_index>
# e.g.
sbatch src/preprocessing/embed_job.sh 3
```

#### Monitoring and logs

```bash
# Check queued / running jobs
squeue -u $USER

# Tail preprocessing log
tail -f logs/hrs_preprocessing_<job_id>.out

# Tail an embedding slice log
tail -f logs/hrs_embed_<job_id>.out

# Tail combine log
tail -f logs/hrs_combine_<job_id>.out
```

All logs are written to the `logs/` directory at the repository root.

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

| Scenario                                                                   | Recommended flag                                                  |
| -------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Initial run (no hashes yet)                                                | Neither — hash check is a no-op when outputs are missing          |
| Source files completely unchanged, want to re-embed with a new BERT model  | `--force-module embed_features combine_dataset`                   |
| One source table updated                                                   | `--force-module <affected_module> embed_features combine_dataset` |
| Want to wipe and redo everything                                           | `--force`                                                         |

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

| File                                   | Location                          | Format  | Produced by                    | Description                                                                                                                                 |
| -------------------------------------- | --------------------------------- | ------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `data_splits.parquet`                  | `data/preprocessing/`             | Parquet | `create_splits`                | One row per admission with `split` column (`train`/`dev`/`test`)                                                                            |
| `source_hashes.json`                   | `data/preprocessing/`             | JSON    | all modules                    | MD5 hashes of source files per module; drives incremental-run skipping                                                                      |
| `demographics_features.parquet`        | `data/preprocessing/features/`    | Parquet | `extract_demographics`         | One row per admission; `demographic_vec` array column (8 floats)                                                                            |
| `diag_history_features.parquet`        | `data/preprocessing/features/`    | Parquet | `extract_diag_history`         | One row per admission; `diag_history_text` string column                                                                                    |
| `discharge_history_features.parquet`   | `data/preprocessing/features/`    | Parquet | `extract_discharge_history`    | One row per admission; `discharge_history_text` string column                                                                               |
| `triage_features.parquet`              | `data/preprocessing/features/`    | Parquet | `extract_triage_and_complaint` | One row per admission; `triage_text` string column                                                                                          |
| `chief_complaint_features.parquet`     | `data/preprocessing/features/`    | Parquet | `extract_triage_and_complaint` | One row per admission; `chief_complaint_text` string column                                                                                 |
| `labs_features.parquet`                | `data/preprocessing/features/`    | Parquet | `extract_labs`                 | Long format — one row per lab event; columns: `subject_id`, `hadm_id`, `charttime`, `itemid`, `label`, `fluid`, `category`, `lab_text_line` |
| `radiology_features.parquet`           | `data/preprocessing/features/`    | Parquet | `extract_radiology`            | One row per admission; `radiology_text` string column                                                                                       |
| `diag_history_embeddings.parquet`      | `data/preprocessing/features/embeddings/` | Parquet | `embed_features`      | One row per admission; `diag_history_embedding` array column                                                                                |
| `discharge_history_embeddings.parquet` | `data/preprocessing/features/embeddings/` | Parquet | `embed_features`      | One row per admission; `discharge_history_embedding` array column                                                                           |
| `triage_embeddings.parquet`            | `data/preprocessing/features/embeddings/` | Parquet | `embed_features`      | One row per admission; `triage_embedding` array column                                                                                      |
| `chief_complaint_embeddings.parquet`   | `data/preprocessing/features/embeddings/` | Parquet | `embed_features`      | One row per admission; `chief_complaint_embedding` array column                                                                             |
| `radiology_embeddings.parquet`         | `data/preprocessing/features/embeddings/` | Parquet | `embed_features`      | One row per admission; `radiology_embedding` array column                                                                                   |
| `lab_{group}_embeddings.parquet` (×13) | `data/preprocessing/features/embeddings/` | Parquet | `embed_features`      | One row per admission per lab group; `lab_{group}_embedding` array column (768 floats); zero vector for admissions with no events in that group |
| `y_labels.parquet`                     | `data/preprocessing/classifications/` | Parquet | `extract_y_data`          | One row per admission; `y1_mortality` and `y2_readmission` columns                                                                          |
| `imputation_stats.json`                | `data/preprocessing/classifications/` | JSON    | `extract_demographics`    | Per-stratum (age-bin × gender) mean/std used for height/weight imputation, computed on train split only                                     |
| `lab_panel_config.yaml`                | `data/preprocessing/classifications/` | YAML    | `build_lab_panel_config`  | Defines the 13 lab group names and their constituent itemids, derived from `d_labitems`                                                     |
| `micro_panel_config.yaml`              | `config/`                             | YAML    | version-controlled         | Defines the 37 microbiology panel names. Each panel entry includes a human-readable `description` field (e.g. `"Microbiology panel: blood culture"`) and a `combos` list of `[test_name, spec_type_desc]` pairs. Path declared via `MICRO_PANEL_CONFIG_PATH` in `config/preprocessing.yaml`. |
| `hadm_linkage_stats.json`              | `data/preprocessing/classifications/` | JSON    | all modules               | Per-module counts of null hadm_id records: dropped, linked, ambiguous-resolved, unresolvable                                               |
| `final_cdss_dataset.parquet`           | `data/preprocessing/classifications/` | Parquet | `combine_dataset`         | One row per admission; all features and labels joined. Includes demographics, all 5 non-lab embedding columns, and all 13 lab group embedding columns as independent columns. `labs_features.parquet` (long-format raw event data) is excluded — it is superseded by the 13 per-group embedding parquets. The 13 lab group embeddings are discovered and joined automatically by `combine_dataset.py` via dynamic parquet discovery in `EMBEDDINGS_DIR`. |

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

**Fix:** Set `BERT_DEVICE: cpu` in `config/preprocessing.yaml`. The pipeline
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
