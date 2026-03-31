# CDSS-ML Preprocessing — Architecture

## Table of Contents

1. [Overview](#1-overview)
2. [Prediction Targets](#2-prediction-targets)
3. [Data](#3-data)
4. [Feature Set](#4-feature-set)
5. [Pipeline Overview](#5-pipeline-overview)
6. [Module Summary](#6-module-summary)
7. [Embedding Strategy](#7-embedding-strategy)
8. [Final Dataset](#8-final-dataset)
9. [Design Principles](#9-design-principles)
10. [Directory Structure](#10-directory-structure)
11. [SLURM Execution](#11-slurm-execution)

---

## 1. Overview

The CDSS-ML preprocessing pipeline transforms raw MIMIC-IV clinical data into a fixed-schema parquet dataset ready for supervised classification and reinforcement learning.

**Input:** MIMIC-IV v3.1 raw CSV tables  
**Output:** `final_cdss_dataset.parquet` — one row per hospital admission, 61 columns

The pipeline is fully configuration-driven, resumable, and runs on a SLURM cluster with multi-GPU embedding support. Features span five clinical domains: demographics, clinical history (prior visits), current-visit structured data, laboratory results (13 groups), and microbiology results (37 panels).

---

## 2. Prediction Targets

| Target | Column | Definition |
|--------|--------|------------|
| Y1 — In-hospital mortality | `y1_mortality` | `admissions.hospital_expire_flag` |
| Y2 — 30-day readmission | `y2_readmission` | Subsequent admission within 30 days of `dischtime`; **NaN for deceased patients** |

---

## 3. Data

**Source:** MIMIC-IV v3.1

**Splits** (patient-level stratified, seed 42):

| Split | Fraction |
|-------|----------|
| Train | 80% |
| Dev   | 10% |
| Test  | 10% |

Splitting is at the **patient level** (`subject_id`) — all admissions of a patient are in the same split, preventing leakage from prior-visit features (F2, F3). Stratification uses Y1 only (Y2 is NaN for deceased patients and would create degenerate strata).

---

## 4. Feature Set

The design specifies **56 feature slots**: 1 structured vector + 55 embeddings (5 text + 13 lab groups + 37 microbiology panels), each embedding 768-dimensional.

| ID | Feature | Representation | MDP Visibility |
|----|---------|----------------|----------------|
| F1 | Demographics | 8-float vector | ✓ Always visible |
| F2 | Diagnosis History (prior visits) | 768-d embedding | ✓ Always visible |
| F3 | Discharge History (prior visits) | 768-d embedding | ✓ Always visible |
| F4 | Triage (current visit) | 768-d embedding | ✓ Always visible |
| F5 | Chief Complaint (current visit) | 768-d embedding | ✓ Always visible |
| F6–F18 | Lab Results — **13 groups** (current visit) | 768-d embedding per group | ✗ Maskable |
| F19 | Radiology Note (current visit) | 768-d embedding | ✗ Maskable |
| F20–F56 | Microbiology Results — **37 panels** (current visit) | 768-d embedding per panel | ✗ Maskable |

F1–F5 are always available to both the classifier and MDP agent. F6–F56 are maskable — the MDP agent selects which to unlock per episode; each is an independent feature slot with its own projection layer.

**Lab group design:** Groups are derived from unique `(fluid, category)` combinations in `d_labitems`. Fluids spanning multiple categories (Ascites, Pleural, Cerebrospinal Fluid, Joint Fluid, Bone Marrow, Stool) are merged into a single group per fluid, yielding 13 canonical groups. See `PREPROCESSING_DETAILED_DESIGN.md` for the full group table.

**Microbiology panel design:** Panels are keyed on `(test_name, spec_type_desc)` pairs — the same test on different specimen types is a clinically distinct event. 37 panels cover blood cultures, respiratory specimens, urine, sterile fluids, gram stains, fungal, serology, resistance screening, GI, STI, and genital workup. Post-mortem specimens are excluded entirely to prevent target leakage (Y1). Panel membership is defined in `micro_panel_config.yaml`. See `PREPROCESSING_DETAILED_DESIGN.md` for the full panel table.

---

## 5. Pipeline Overview

### Execution DAG

```
                    ┌───────────────────────┐
                    │   create_splits.py     │
                    │   data_splits.parquet  │
                    └──────────┬────────────┘
                               │
               ┌───────────────┴───────────────┐
               │                               │
    ┌──────────▼────────────┐     ┌────────────────────────┐
    │ build_lab_panel_       │     │ micro_panel_config.yaml │
    │ config.py              │     │ (version-controlled     │
    │ lab_panel_config.yaml  │     │  config, no build step) │
    └──────────┬────────────┘     └────────────┬───────────┘
               │                               │
               └───────────────┬───────────────┘
                               │
          ┌────────────────────┼──────────────────────┐
          │                    │                      │
          ▼                    ▼                      ▼
 extract_demographics    extract_labs          extract_y_data
 extract_diag_history    extract_triage_       y_labels.parquet
 extract_discharge_      and_complaint
 history                 extract_radiology
                         extract_microbiology
          │                    │
          └────────────────────┘
                     │
                     ▼
       ┌─────────────────────────────┐
       │    embed_features.py         │
       │    admission-slice batching  │
       │                             │
       │  Slice 0  (hadm 0–40k)      │
       │  Slice 1  (hadm 40k–80k)    │
       │  ...                         │
       │  Slice 13 (hadm 520k–546k)  │
       │                             │
       │  Each slice: 2 GPUs         │
       │  Each GPU: 20k admissions   │
       └──────────────┬──────────────┘
                      │
                      ▼
              ┌───────────────────┐
              │  combine_dataset   │
              │ final_cdss_dataset │
              └───────────┬───────┘
                          │
                          ▼
              ┌───────────────────────────────┐
              │  reduce_dataset (optional)     │
              │ reduced_cdss_dataset.parquet   │
              └───────────────────────────────┘
```

### Dependency Rules

- `create_splits` → must run first
- `build_lab_panel_config` → must run before `extract_labs`; `micro_panel_config.yaml` is version-controlled and requires no build step
- All `extract_*` → independent of each other, can run in parallel
- `embed_features` → requires all `extract_*` complete; runs as **14 sequential SLURM jobs**
- `combine_dataset` → requires all embed slices complete
- `reduce_dataset` (optional) → runs after `combine_dataset` and consumes `final_cdss_dataset.parquet`

### Runtime (SLURM, GPU cluster)

| Phase | Jobs | Time per job | Total |
|-------|------|-------------|-------|
| Preprocessing (steps 0–9) | 1 | ~25 min | ~25 min |
| Embedding (step 10, 14 slices) | 14 | ≤6 hrs | ≤84 hrs wall, runs sequentially (2 GPUs parallel within each job) |
| Combine (step 11) | 1 | ~1 min | ~1 min |
| Dimensionality reduction (step 12, optional) | 1 | ~20–40 min (CPU) | ~20–40 min |

Optional step 12 (`reduce_dataset.py`) runs after combine, streaming one embedding column at a time. Typical runtime is CPU-bound (~20–40 minutes when reducing all embedding columns — 55 by default — to 128 dims with SVD on a 64 GB CPU node) and stays within the existing ≤64 GB RAM envelope because only one column is loaded at a time.

---

## 6. Module Summary

| Step | Module | Output | Notes |
|------|--------|--------|-------|
| 0a | `build_lab_panel_config.py` | `lab_panel_config.yaml` | Must run before extract_labs |
| 1 | `create_splits.py` | `data_splits.parquet` | Patient-level, stratified, seed 42 |
| 2 | `extract_demographics.py` | `demographics_features.parquet` | 8-float vector per admission |
| 3 | `extract_diag_history.py` | `diag_history_features.parquet` | Prior visits only |
| 4 | `extract_discharge_history.py` | `discharge_history_features.parquet` | Prior visits only |
| 5 | `extract_triage_and_complaint.py` | `triage_features.parquet`, `chief_complaint_features.parquet` | ED visit linkage |
| 6 | `extract_labs.py` | `labs_features.parquet` | Long format, 16.8M rows |
| 7 | `extract_microbiology.py` | `micro_<panel>.parquet` × 37 | One parquet per panel; 72h window default |
| 8 | `extract_radiology.py` | `radiology_features.parquet` | Most recent note per admission |
| 9 | `extract_y_data.py` | `y_labels.parquet` | Y1 + Y2 labels |
| 10 | `embed_features.py` | 55 embedding parquets | 14 SLURM jobs × 2 GPUs × 20k admissions |
| 11 | `combine_dataset.py` | `final_cdss_dataset.parquet` | Left-join all features; 61 columns |
| 12 | `reduce_dataset.py` | `reduced_cdss_dataset.parquet` | Column-wise embedding dimensionality reduction; fit on train split only (no leakage) |

Supporting scripts: `check_embed_status.py` (state detection for `submit_all.sh`), `preprocessing_utils.py` (hashing/IO utilities), `build_lab_text_lines.py` (helper for `extract_labs`), `build_micro_text.py` (helper for `extract_microbiology` — comment cleaning and text construction). `micro_panel_config.yaml` is a version-controlled config file in `config/` — no build step required.

---

## 7. Embedding Strategy

**Model:** `Simonlee711/Clinical_ModernBERT` — trained on PubMed, MIMIC-IV notes, and medical ontologies. 8,192-token context window, 768-d hidden size. Used as a **frozen feature extractor** (not fine-tuned).

**Pooling:** Mean pooling over all non-padding content tokens from the final hidden layer. Preferred over `[CLS]` because the model is not fine-tuned — mean pooling ensures every token contributes equally, which is critical for long clinical texts and for the dense structured-text format used in microbiology panels.

**Missing features:** Zero vector (768 floats). Lab groups and microbiology panels with no events for an admission receive a zero vector, never null.

**Performance:** Per-feature token length caps (64–8,192) prevent padding short texts to the full context window. Batch size auto-scales inversely with sequence length.

**Multi-GPU within a job:** The slice's admissions are split evenly between 2 GPU workers (~20k each). Both workers run **in parallel** — each embeds all 55 features for its own admission half, writing to per-worker temporary parquets. The main process merges the per-worker parquets into the shared output parquets after both workers complete. LPT ordering within each worker (features sorted by estimated compute cost descending) ensures the most expensive features start first for better progress visibility. Each GPU worker loads its own model copy.

**Admission-slice batching:** The full admission corpus is divided into slices based on `BERT_SLICE_SIZE_PER_GPU` (default: 20,000 admissions per GPU). With 2 GPUs, each slice covers 40,000 admissions. The number of slices is computed automatically at runtime. Each slice runs as a separate SLURM job (≤12h). Slices run sequentially. Each slice reads the existing output parquet, concatenates its new rows with the existing rows using PyArrow, and atomically overwrites the output file. PyArrow is used throughout because other parquet writers cannot serialise `fixed_size_list(float32, 768)` embedding columns. Adjusting `BERT_SLICE_SIZE_PER_GPU` is the only knob needed to fit different partition time limits.

**Resume:** Three levels — (1) slice-level: a slice is detected as complete when all its admission IDs are already present in the output parquet and skipped; (2) feature-level: within a slice, completed feature-parquet segments are skipped; (3) record-level: within a feature in a slice, already-embedded rows are skipped via incremental checkpointing.

**Microbiology-specific configuration:** Microbiology panels use separate config keys from lab features — `MICRO_WINDOW_HOURS` (default 72h vs lab's 24h), `MICRO_NULL_HADM_STRATEGY`, and `MICRO_LINK_TOLERANCE_HOURS`. These are resolved at extract time; `embed_features.py` consumes the pre-extracted text parquets and is agnostic to the extraction configuration.

---

## 8. Final Dataset

`final_cdss_dataset.parquet` — one row per hospital admission, 61 columns:

| Group | Count | Type |
|-------|-------|------|
| Metadata (`subject_id`, `hadm_id`, `split`) | 3 | int / str |
| Labels (`y1_mortality`, `y2_readmission`) | 2 | int / float |
| Demographics (`demographic_vec`) | 1 | float[8] |
| Text embeddings (F2–F5, F19) | 5 | float[768] each |
| Lab group embeddings (F6–F18, 13 groups) | 13 | float[768] each |
| Microbiology panel embeddings (F20–F56, 37 panels) | 37 | float[768] each |

Embedding columns are discovered dynamically from `EMBEDDINGS_DIR` — no hardcoded list.

Two artefacts are available for downstream training: (1) the full-dimensionality dataset (≈50 GB) and (2) the reduced-dimensionality dataset (e.g., ≈17 GB when reduced to 128 dimensions) produced by `reduce_dataset.py`, allowing the reward model to choose based on hardware constraints.

---

## 9. Design Principles

**No target leakage** — prior-visit features use only admissions strictly before current `admittime`. Lab events restricted to current admission window. Post-mortem specimens excluded from all microbiology panels — their presence perfectly predicts Y1 (in-hospital mortality) and would introduce a direct causal shortcut. Imputation statistics computed on train split only, persisted and applied identically to dev/test.

**No hardcoding** — all paths, model names, split ratios, batch sizes, and thresholds in `config/preprocessing.yaml`. Lab groups derived dynamically from `d_labitems`. Microbiology panel membership, comment cleaning rules, excluded tests, and excluded specimen types defined in `config/micro_panel_config.yaml` (version-controlled; path declared via `MICRO_PANEL_CONFIG_PATH`).

**Reproducibility** — random seed 42. Imputation stats and source MD5 hashes persisted. Incremental runs skip modules whose source hashes match and outputs exist.

**Memory safety** — `labevents`, `microbiologyevents`, and `chartevents` streamed in 500k-row chunks. 64 GB RAM required for both pipeline and embed SLURM jobs.

**Time-window safety** — embedding is split into admission slices of ≤80k records (≤40k per GPU) to fit within the 12-hour SLURM partition limit. Each slice is a self-contained job that appends to the shared output parquets.

**Graceful degradation** — missing OMR/chartevents falls back gracefully. Missing CUDA falls back to CPU. Missing lab events or microbiology events → zero vector. Missing `lab_panel_config.yaml` or `micro_panel_config.yaml` (at path from `MICRO_PANEL_CONFIG_PATH`) → respective embeddings skipped with warning.

---

## 10. Directory Structure

```
HRS/
├── config/
│   ├── preprocessing.yaml              # All configuration — single source of truth
│   └── micro_panel_config.yaml         # Microbiology panel definitions (version-controlled)
├── src/preprocessing/
│   ├── pipeline_job.sh                 # SLURM: preprocessing (no GPU, 64G)
│   ├── embed_job.sh                    # SLURM: one embed slice (2× L4 GPU, 64G)
│   ├── combine_job.sh                  # SLURM: combine (no GPU, 32G)
│   ├── submit_all.sh                   # Auto-submit with state detection
│   ├── micro_extract_job.sh            # SLURM: extract_microbiology only
│   ├── labs_extract_job.sh             # SLURM: extract_labs only
│   ├── run_pipeline.py                 # Orchestrator CLI
│   ├── check_embed_status.py           # State detection for submit_all.sh
│   ├── preprocessing_utils.py          # Shared utilities
│   ├── build_lab_panel_config.py       # Step 0a
│   ├── create_splits.py                # Step 1
│   ├── extract_demographics.py         # Step 2
│   ├── extract_diag_history.py         # Step 3
│   ├── extract_discharge_history.py    # Step 4
│   ├── extract_triage_and_complaint.py # Step 5
│   ├── extract_labs.py                 # Step 6
│   ├── extract_microbiology.py         # Step 7
│   ├── extract_radiology.py            # Step 8
│   ├── extract_y_data.py               # Step 9
│   ├── embed_features.py               # Step 10 — accepts --slice-index
│   ├── combine_dataset.py              # Step 11
│   ├── build_lab_text_lines.py         # Helper for extract_labs
│   └── build_micro_text.py             # Helper for extract_microbiology
└── data/preprocessing/                 # Generated artefacts (git-ignored)
    ├── data_splits.parquet
    ├── source_hashes.json
    ├── features/
    │   ├── [feature parquets ×8]
    │   └── embeddings/
    │       └── [embedding parquets ×55]
    └── classifications/
        ├── y_labels.parquet
        ├── final_cdss_dataset.parquet
        ├── lab_panel_config.yaml
        ├── imputation_stats.json
        ├── hadm_linkage_stats.json
        └── micro_linkage_stats.json
```

---

## 11. SLURM Execution

**Cluster:** University HPC cluster — login node and partition names are defined in `config/preprocessing.yaml`.  
**Partitions:** Two GPU partitions are supported — a standard time-limit partition and a shorter queue-wait partition. Partition names, GPU type, time limits, and GPU count per job are all configurable.

### Capacity Sizing

The safe per-GPU admission limit is ~20,000 admissions per 2-GPU job, giving ~40,000 per SLURM job. This was determined empirically from cluster runs and is controlled by `BERT_SLICE_SIZE_PER_GPU` in config — the number of slices is computed automatically at runtime. The addition of 37 microbiology panels increases per-admission compute within each slice but does not change the number of slices — the admission count per slice is unchanged.

| Total admissions | Per-GPU limit (`BERT_SLICE_SIZE_PER_GPU`) | Per-job (2 GPUs) | Jobs required |
|-----------------|------------------------------------------|-----------------|--------------|
| Computed at runtime | 20,000 | 40,000 | **14** (based on current corpus) |

### Scripts

All SLURM scripts live in `src/preprocessing/` alongside the Python modules they invoke.

| Script | GPUs | Purpose |
|--------|------|---------|
| `pipeline_job.sh` | 0 | Steps 0a–9 (CPU only) |
| `micro_extract_job.sh` | 0 | `extract_microbiology` only (CPU) — run before `submit_all.sh` |
| `labs_extract_job.sh` | 0 | `extract_labs` only (CPU) — run before `submit_all.sh` |
| `embed_job.sh` | 2 | One admission slice — takes `--slice-index N` (passed by `submit_all.sh`) |
| `combine_job.sh` | 0 | Step 11 — combine only (CPU) |
| `submit_all.sh` | — | Detects state, submits all pending slices chained via `--dependency=afterok` |

`extract_labs` and `extract_microbiology` must be submitted as dedicated SLURM jobs before running `submit_all.sh`. `submit_all.sh` handles embedding and combine only — it will exit with an error if extraction parquets are missing, printing the exact command to run. Use `check_embed_status.py` to verify extraction completeness before submitting embed jobs.

### Auto-submit State Detection and Job Chaining

```bash
cd ~/Python/HRS
bash src/preprocessing/submit_all.sh
```

`check_embed_status.py` scans embedding parquets for total row count, determines which slices are complete, and exits with:
- **2** → preprocessing incomplete → submits pipeline → 14 embed slices → combine
- **1** → embedding incomplete or extraction incomplete →  
  if extraction incomplete: prints error with command to run extraction job and exits;  
  if extraction complete: submits all embed slices → combine
- **0** → all embeddings complete → submits combine only

The 14 embed slice jobs are submitted as a dependency chain:

```
embed_slice_0
    └──(afterok)── embed_slice_1
                       └──(afterok)── embed_slice_2
                                          └──(afterok)── ... embed_slice_13
                                                                   └──(afterok)── combine
```

Re-running `submit_all.sh` is always safe. Each slice job detects its already-completed rows via record-level resume and skips them, so a re-submitted slice only processes what remains.

> See `PREPROCESSING_DETAILED_DESIGN.md` for full per-module implementation details, embedding internals, admission-slice batching design, and configuration reference.
