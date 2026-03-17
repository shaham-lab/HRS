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
**Output:** `final_cdss_dataset.parquet` — one row per hospital admission, 24 columns

The pipeline is fully configuration-driven, resumable, and runs on a SLURM cluster with multi-GPU embedding support.

---

## 2. Prediction Targets

| Target | Column | Definition |
|--------|--------|------------|
| Y1 — In-hospital mortality | `y1_mortality` | `admissions.hospital_expire_flag` |
| Y2 — 30-day readmission | `y2_readmission` | Subsequent admission within 30 days of `dischtime`; **NaN for deceased patients** |

---

## 3. Data

**Source:** MIMIC-IV v3.1 — 546,028 admissions, 223,452 patients

**Splits** (patient-level stratified, seed 42):

| Split | Patients | Admissions |
|-------|----------|------------|
| Train (80%) | 178,761 | 435,160 |
| Dev (10%) | 22,345 | 54,934 |
| Test (10%) | 22,346 | 55,934 |

Splitting is at the **patient level** (`subject_id`) — all admissions of a patient are in the same split, preventing leakage from prior-visit features (F2, F3). Stratification uses Y1 only (Y2 is NaN for deceased patients and would create degenerate strata).

**Label statistics:**
- Y1 positive rate: 2.16% (11,801 deceased admissions)
- Y2 positive rate (excl. deaths): 20.14% (107,617 readmitted)

---

## 4. Feature Set

The design specifies **19 feature slots**: 1 structured vector + 18 embeddings (5 text + 13 lab groups), each embedding 768-dimensional.

| ID | Feature | Representation | MDP Visibility |
|----|---------|----------------|----------------|
| F1 | Demographics | 8-float vector | ✓ Always visible |
| F2 | Diagnosis History (prior visits) | 768-d embedding | ✓ Always visible |
| F3 | Discharge History (prior visits) | 768-d embedding | ✓ Always visible |
| F4 | Triage (current visit) | 768-d embedding | ✓ Always visible |
| F5 | Chief Complaint (current visit) | 768-d embedding | ✓ Always visible |
| F6–F18 | Lab Results — **13 groups** (current visit) | 768-d embedding per group | ✗ Maskable |
| F19 | Radiology Note (current visit) | 768-d embedding | ✗ Maskable |

F1–F5 are always available to both the classifier and MDP agent. F6–F19 are maskable — the MDP agent selects which to unlock per episode; each is an independent feature slot with its own projection layer.

**Lab group design:** Groups are derived from unique `(fluid, category)` combinations in `d_labitems`. Fluids spanning multiple categories (Ascites, Pleural, Cerebrospinal Fluid, Joint Fluid, Bone Marrow, Stool) are merged into a single group per fluid, yielding 13 canonical groups. See `PREPROCESSING_DETAILED_DESIGN.md` for the full group table.

---

## 5. Pipeline Overview

### Execution DAG

```
                    ┌───────────────────────┐
                    │   create_splits.py     │
                    │   data_splits.parquet  │
                    └──────────┬────────────┘
                               │
                    ┌──────────▼────────────┐
                    │ build_lab_panel_       │
                    │ config.py              │
                    │ lab_panel_config.yaml  │
                    └──────────┬────────────┘
                               │
          ┌────────────────────┼──────────────────────┐
          │                    │                      │
          ▼                    ▼                      ▼
 extract_demographics    extract_labs          extract_y_data
 extract_diag_history    extract_triage_       y_labels.parquet
 extract_discharge_      and_complaint
 history                 extract_radiology
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
              └───────────────────┘
```

### Dependency Rules

- `create_splits` → must run first
- `build_lab_panel_config` → must run before any `extract_*`
- All `extract_*` → independent of each other, can run in parallel
- `embed_features` → requires all `extract_*` complete; runs as **7 sequential SLURM jobs**
- `combine_dataset` → requires all embed slices complete

### Runtime (SLURM, L4 GPU, 64 GB RAM)

| Phase | Jobs | Time per job | Total |
|-------|------|-------------|-------|
| Preprocessing (steps 0–8) | 1 | ~18 min | ~18 min |
| Embedding (step 9, 14 slices) | 14 | ≤6 hrs | ≤84 hrs wall, runs sequentially (2 GPUs parallel within each job) |
| Combine (step 10) | 1 | ~1 min | ~1 min |

---

## 6. Module Summary

| Step | Module | Output | Notes |
|------|--------|--------|-------|
| 0 | `build_lab_panel_config.py` | `lab_panel_config.yaml` | Must run before extract_labs |
| 1 | `create_splits.py` | `data_splits.parquet` | Patient-level, stratified, seed 42 |
| 2 | `extract_demographics.py` | `demographics_features.parquet` | 8-float vector per admission |
| 3 | `extract_diag_history.py` | `diag_history_features.parquet` | Prior visits only |
| 4 | `extract_discharge_history.py` | `discharge_history_features.parquet` | Prior visits only |
| 5 | `extract_triage_and_complaint.py` | `triage_features.parquet`, `chief_complaint_features.parquet` | ED visit linkage |
| 6 | `extract_labs.py` | `labs_features.parquet` | Long format, 16.8M rows |
| 7 | `extract_radiology.py` | `radiology_features.parquet` | Most recent note per admission |
| 8 | `extract_y_data.py` | `y_labels.parquet` | Y1 + Y2 labels |
| 9 | `embed_features.py` | 18 embedding parquets | 14 SLURM jobs × 2 GPUs × 20k admissions |
| 10 | `combine_dataset.py` | `final_cdss_dataset.parquet` | Left-join all features |

Supporting scripts: `check_embed_status.py` (state detection for `submit_all.sh`), `preprocessing_utils.py` (hashing/IO utilities), `build_lab_text_lines.py` (helper for `extract_labs`).

---

## 7. Embedding Strategy

**Model:** `Simonlee711/Clinical_ModernBERT` — trained on PubMed, MIMIC-IV notes, and medical ontologies. 8,192-token context window, 768-d hidden size. Used as a **frozen feature extractor** (not fine-tuned).

**Pooling:** Mean pooling over all non-padding content tokens from the final hidden layer. Preferred over `[CLS]` because the model is not fine-tuned — mean pooling ensures every token contributes equally, which is critical for long clinical texts.

**Missing features:** Zero vector (768 floats). Lab groups with no events for an admission receive a zero vector, never null.

**Performance:** Per-feature token length caps (64–4,096) prevent padding short texts to 8,192 tokens. Batch size auto-scales inversely with sequence length.

**Multi-GPU within a job:** The slice's admissions are split evenly between 2 GPU workers (~20k each). Both workers run **in parallel** — each embeds all 18 features for its own admission half, writing to per-worker temporary parquets. The main process merges the per-worker parquets into the shared output parquets after both workers complete. LPT ordering within each worker (features sorted by estimated compute cost descending) ensures the most expensive features start first for better progress visibility. Each GPU worker loads its own model copy.

**Admission-slice batching:** The full admission corpus is divided into slices based on `BERT_SLICE_SIZE_PER_GPU` (default: 20,000 admissions per GPU). With 2 GPUs, each slice covers 40,000 admissions, giving **14 slices** for 546,028 admissions. Each slice runs as a separate SLURM job (≤12h). Slices run sequentially and append their results into the same output parquets via `fastparquet` append mode. Adjusting `BERT_SLICE_SIZE_PER_GPU` is the only knob needed to fit different partition time limits.

**Resume:** Three levels — (1) slice-level: a completed slice is detected by row count and skipped; (2) feature-level: within a slice, completed feature-parquet segments are skipped; (3) record-level: within a feature in a slice, already-embedded rows are skipped via incremental checkpointing.

---

## 8. Final Dataset

`final_cdss_dataset.parquet` — 546,028 rows × 24 columns:

| Group | Count | Type |
|-------|-------|------|
| Metadata (`subject_id`, `hadm_id`, `split`) | 3 | int / str |
| Labels (`y1_mortality`, `y2_readmission`) | 2 | int / float |
| Demographics (`demographic_vec`) | 1 | float[8] |
| Text embeddings (F2–F5, F19) | 5 | float[768] each |
| Lab group embeddings (F6–F18, 13 groups) | 13 | float[768] each |

Embedding columns are discovered dynamically from `EMBEDDINGS_DIR` — no hardcoded list.

---

## 9. Design Principles

**No target leakage** — prior-visit features use only admissions strictly before current `admittime`. Lab events restricted to current admission window. Imputation statistics computed on train split only, persisted and applied identically to dev/test.

**No hardcoding** — all paths, model names, split ratios, batch sizes, and thresholds in `config/preprocessing.yaml`. Lab groups derived dynamically from `d_labitems`.

**Reproducibility** — random seed 42. Imputation stats and source MD5 hashes persisted. Incremental runs skip modules whose source hashes match and outputs exist.

**Memory safety** — `labevents` and `chartevents` streamed in 500k-row chunks. 64 GB RAM required for both pipeline and embed SLURM jobs.

**Time-window safety** — embedding is split into admission slices of ≤80k records (≤40k per GPU) to fit within the 12-hour SLURM partition limit. Each slice is a self-contained job that appends to the shared output parquets.

**Graceful degradation** — missing OMR/chartevents falls back gracefully. Missing CUDA falls back to CPU. Missing lab events → zero vector. Missing `lab_panel_config.yaml` → lab embeddings skipped with warning.

---

## 10. Directory Structure

```
HRS/
├── config/
│   └── preprocessing.yaml              # All configuration — single source of truth
├── src/preprocessing/
│   ├── pipeline_job.sh                 # SLURM: preprocessing (no GPU, 64G)
│   ├── embed_job.sh                    # SLURM: one embed slice (2× L4 GPU, 64G)
│   ├── combine_job.sh                  # SLURM: combine (no GPU, 32G)
│   ├── submit_all.sh                   # Auto-submit with state detection
│   ├── run_pipeline.py                 # Orchestrator CLI
│   ├── check_embed_status.py           # State detection for submit_all.sh
│   ├── preprocessing_utils.py          # Shared utilities
│   ├── build_lab_panel_config.py       # Step 0
│   ├── create_splits.py                # Step 1
│   ├── extract_demographics.py         # Step 2
│   ├── extract_diag_history.py         # Step 3
│   ├── extract_discharge_history.py    # Step 4
│   ├── extract_triage_and_complaint.py # Step 5
│   ├── extract_labs.py                 # Step 6
│   ├── extract_radiology.py            # Step 7
│   ├── extract_y_data.py               # Step 8
│   ├── embed_features.py               # Step 9 — accepts --slice-index
│   ├── combine_dataset.py              # Step 10
│   └── build_lab_text_lines.py         # Helper for extract_labs
└── data/preprocessing/                 # Generated artefacts (git-ignored)
    ├── data_splits.parquet
    ├── source_hashes.json
    ├── features/
    │   ├── [feature parquets ×7]
    │   └── embeddings/
    │       └── [embedding parquets ×18]
    └── classifications/
        ├── y_labels.parquet
        ├── final_cdss_dataset.parquet
        ├── lab_panel_config.yaml
        ├── imputation_stats.json
        └── hadm_linkage_stats.json
```

---

## 11. SLURM Execution

**Cluster:** BIU SLURM — `slurm-login1/2/3.lnx.biu.ac.il`  
**Partitions:**
- `L4-12h` — NVIDIA L4 (sm_89), **12-hour** limit, max 2 GPUs per user
- `L4-4h` — same hardware, **4-hour** limit (shorter queue wait)

### Capacity Sizing

136,507 admissions failed to complete in 12 hours on 2 GPUs. The safe per-GPU limit is therefore ~20,000 admissions, giving ~40,000 per 2-GPU SLURM job. This is controlled by `BERT_SLICE_SIZE_PER_GPU` in config — the number of slices is computed automatically at runtime.

| Total admissions | Per-GPU limit (`BERT_SLICE_SIZE_PER_GPU`) | Per-job (2 GPUs) | Jobs required |
|-----------------|------------------------------------------|-----------------|--------------|
| 546,028 | 20,000 | 40,000 | **14** |

### Scripts

All four scripts live in `src/preprocessing/` alongside the Python modules they invoke.

| Script | GPUs | RAM | Purpose |
|--------|------|-----|---------|
| `pipeline_job.sh` | 0 | 64G | Steps 0–8 (CPU only) |
| `embed_job.sh` | 2 | 64G | One admission slice — takes `--slice-index N` (passed by `submit_all.sh`) |
| `combine_job.sh` | 0 | 32G | Step 10 — combine only (CPU) |
| `submit_all.sh` | — | — | Detects state, submits all pending slices chained via `--dependency=afterok` |

### Auto-submit State Detection and Job Chaining

```bash
cd ~/Python/HRS
bash src/preprocessing/submit_all.sh
```

`check_embed_status.py` scans embedding parquets for total row count, determines which slices are complete, and exits with:
- **2** → preprocessing incomplete → submits pipeline → 7 embed slices → combine
- **1** → embedding incomplete → submits remaining slice jobs → combine
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
