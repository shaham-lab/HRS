# CDSS-ML Reward Model — Architecture

> **Location:** `HRS/src/reward_model/reward_model_architecture.md`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prediction Targets](#2-prediction-targets)
3. [Data and Inputs](#3-data-and-inputs)
4. [Feature Set](#4-feature-set)
5. [Pipeline Overview](#5-pipeline-overview)
6. [Module Summary](#6-module-summary)
7. [Technology Stack](#7-technology-stack)
8. [Key Subsystem Detail](#8-key-subsystem-detail)
9. [Final Output](#9-final-output)
10. [Security](#10-security)
11. [Performance](#11-performance)
12. [Design Principles](#12-design-principles)
13. [Directory Structure](#13-directory-structure)
14. [Infrastructure and Execution](#14-infrastructure-and-execution)

---

## 1. Overview

The CDSS-ML Reward Model is a supervised feedforward neural network that consumes a fixed-length patient feature vector derived from `final_cdss_dataset.parquet` — produced by `HRS/src/preprocessing` — and outputs two calibrated probability scores: in-hospital mortality (Y1) and 30-day readmission conditional on survival (Y2). Once trained, the model is frozen and used exclusively as an inference module by the RL agent, which computes a potential-based reward from the delta in output probabilities between consecutive episode states. The three most important architectural properties are: masking-aware training (the network trains under random and adversarial feature zeroing to simulate partial information availability at RL inference time), multi-GPU distributed training via PyTorch DDP (training runs across 2 GPUs by default, configurable via `NUM_GPUS`), and full configurability (all hyperparameters, schedule parameters, and architectural dimensions are defined in `config/reward_model.yaml` with no hardcoded values).

See `reward_model_design.docx` for the full design rationale. See `PREPROCESSING_ARCHITECTURE.md` for the upstream pipeline that produces the input dataset.

---

## 2. Prediction Targets

| Target | Column | Definition | Population | Positive rate |
|--------|--------|------------|------------|---------------|
| Y1 — In-hospital mortality | `y1_mortality` | `admissions.hospital_expire_flag = 1` | All admissions | ~8–10% |
| Y2 — 30-day readmission | `y2_readmission` | Unplanned readmission within 30 days of `dischtime` | Survivors only (`y1_mortality = 0`) | ~20% |

**NaN rule:** `y2_readmission` is `NaN` (float32) for all admissions where `y1_mortality = 1`. This is guaranteed by `extract_y_data.py` upstream and validated by `Mimic4DataLoader` at load time. The readmission head learns `P(readmitted | survived)` — deceased patients contribute zero gradient to Y2. See Section 8.1 for runtime contract enforcement.

---

## 3. Data and Inputs

**Source:** `HRS/data/preprocessing/classifications/final_cdss_dataset.parquet` — produced by `combine_dataset.py` (Step 11 of the preprocessing pipeline). The reward model reads this file directly and does not query MIMIC-IV or any intermediate preprocessing artefacts.

**Schema reference:** `PREPROCESSING_DATA_MODEL.md` Section 3.12. The reward model depends on the following columns being present and correctly typed:

- `hadm_id` (int64, non-null) — primary key
- `split` (varchar: `train` / `dev` / `test`) — pre-assigned by the preprocessing pipeline
- `y1_mortality` (int8, non-null)
- `y2_readmission` (float32, nullable — NaN for deceased)
- `demographic_vec` (float32[8], non-null)
- All 55 `*_embedding` columns (float32[768], non-null — zero vector for missing features)

**Split strategy:** Patient-level splitting (`subject_id`) is applied upstream by the preprocessing pipeline. The reward model reads the `split` column directly — it does not re-split the data. All statistics derived from data (class weights, temperature scaling parameters) are computed from `split = 'train'` rows only.

| Split | Fraction | Stratification |
|-------|----------|----------------|
| Train | 80% | Y1 (patient-level mortality rate), seed 42 |
| Dev | 10% | Y1 |
| Test | 10% | Y1 |

---

## 4. Feature Set

The input vector X is constructed by concatenating all feature columns from `final_cdss_dataset.parquet` in the canonical column order defined in `PREPROCESSING_DATA_MODEL.md` Section 3.12. **No separate feature index map config file exists** — feature slot boundaries are derived at load time from the dataset column names and their declared dimensions (8 for `demographic_vec`, 768 for all `*_embedding` columns). The derived index map is held in memory by the `Mimic4DataLoader` (a dataset-specific subclass of the generic `DataLoader` base) and passed to `masking.py` and `train.py`.

**Total input dimensionality:** 8 + (55 × 768) = **42,248 dimensions**

| ID | Column name | Dim | RL visibility |
|----|------------|-----|---------------|
| F1 | `demographic_vec` | 8 | Always visible |
| F2 | `diag_history_embedding` | 768 | Always visible |
| F3 | `discharge_history_embedding` | 768 | Always visible |
| F4 | `triage_embedding` | 768 | Always visible |
| F5 | `chief_complaint_embedding` | 768 | Always visible |
| F6 | `lab_blood_gas_embedding` | 768 | Maskable |
| F7 | `lab_blood_chemistry_embedding` | 768 | Maskable |
| F8 | `lab_blood_hematology_embedding` | 768 | Maskable |
| F9 | `lab_urine_chemistry_embedding` | 768 | Maskable |
| F10 | `lab_urine_hematology_embedding` | 768 | Maskable |
| F11 | `lab_other_body_fluid_chemistry_embedding` | 768 | Maskable |
| F12 | `lab_other_body_fluid_hematology_embedding` | 768 | Maskable |
| F13 | `lab_ascites_embedding` | 768 | Maskable |
| F14 | `lab_pleural_embedding` | 768 | Maskable |
| F15 | `lab_csf_embedding` | 768 | Maskable |
| F16 | `lab_bone_marrow_embedding` | 768 | Maskable |
| F17 | `lab_joint_fluid_embedding` | 768 | Maskable |
| F18 | `lab_stool_embedding` | 768 | Maskable |
| F19 | `radiology_embedding` | 768 | Maskable |
| F20 | `micro_blood_culture_routine_embedding` | 768 | Maskable |
| F21 | `micro_blood_bottle_gram_stain_embedding` | 768 | Maskable |
| F22 | `micro_urine_culture_embedding` | 768 | Maskable |
| F23 | `micro_urine_viral_embedding` | 768 | Maskable |
| F24 | `micro_urinary_antigens_embedding` | 768 | Maskable |
| F25 | `micro_respiratory_non_invasive_embedding` | 768 | Maskable |
| F26 | `micro_respiratory_invasive_embedding` | 768 | Maskable |
| F27 | `micro_respiratory_afb_embedding` | 768 | Maskable |
| F28 | `micro_respiratory_viral_embedding` | 768 | Maskable |
| F29 | `micro_respiratory_pcp_legionella_embedding` | 768 | Maskable |
| F30 | `micro_gram_stain_respiratory_embedding` | 768 | Maskable |
| F31 | `micro_gram_stain_wound_tissue_embedding` | 768 | Maskable |
| F32 | `micro_gram_stain_csf_embedding` | 768 | Maskable |
| F33 | `micro_wound_culture_embedding` | 768 | Maskable |
| F34 | `micro_hardware_and_lines_culture_embedding` | 768 | Maskable |
| F35 | `micro_pleural_culture_embedding` | 768 | Maskable |
| F36 | `micro_peritoneal_culture_embedding` | 768 | Maskable |
| F37 | `micro_joint_fluid_culture_embedding` | 768 | Maskable |
| F38 | `micro_fluid_culture_embedding` | 768 | Maskable |
| F39 | `micro_bone_marrow_culture_embedding` | 768 | Maskable |
| F40 | `micro_csf_culture_embedding` | 768 | Maskable |
| F41 | `micro_fungal_tissue_wound_embedding` | 768 | Maskable |
| F42 | `micro_fungal_respiratory_embedding` | 768 | Maskable |
| F43 | `micro_fungal_fluid_embedding` | 768 | Maskable |
| F44 | `micro_mrsa_staph_screen_embedding` | 768 | Maskable |
| F45 | `micro_resistance_screen_embedding` | 768 | Maskable |
| F46 | `micro_cdiff_embedding` | 768 | Maskable |
| F47 | `micro_stool_bacterial_embedding` | 768 | Maskable |
| F48 | `micro_stool_parasitology_embedding` | 768 | Maskable |
| F49 | `micro_herpesvirus_serology_embedding` | 768 | Maskable |
| F50 | `micro_hepatitis_hiv_embedding` | 768 | Maskable |
| F51 | `micro_syphilis_serology_embedding` | 768 | Maskable |
| F52 | `micro_misc_serology_embedding` | 768 | Maskable |
| F53 | `micro_herpesvirus_culture_antigen_embedding` | 768 | Maskable |
| F54 | `micro_gc_chlamydia_sti_embedding` | 768 | Maskable |
| F55 | `micro_vaginal_genital_flora_embedding` | 768 | Maskable |
| F56 | `micro_throat_strep_embedding` | 768 | Maskable |

**RL visibility:** F1–F5 are always unmasked at episode start. F6–F56 are maskable — the RL agent reveals them by transitioning their slots from zero to their pre-computed embedding values. All embeddings are static and pre-computed per admission by the preprocessing pipeline; no re-embedding occurs during RL episodes. State evolution is driven entirely by which feature slots the agent unmasks.

---

## 5. Pipeline Overview

The reward model sits downstream of `HRS/src/preprocessing` and upstream of the RL agent.

```
HRS/src/preprocessing
    └── data/preprocessing/classifications/final_cdss_dataset.parquet
                    │
                    ▼
        ┌──────────────────────────────────────┐
        │   mimic4_data_loader.py              │  read parquet, validate schema,
        │                                       │  derive feature index map,
        │                                       │  build train/dev/test tensors
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │   train.py  (torchrun, NUM_GPUS=2)    │  DDP across GPUs, masking curriculum,
        │                                       │  forward/backward, early stopping,
        │   GPU 0 ──── mini-batch shard 0       │  checkpointing (rank 0 only)
        │   GPU 1 ──── mini-batch shard 1       │
        │       └── all-reduce gradients        │
        └──────────────┬───────────────────────┘
                       │  best_model.pt (rank 0)
                       ▼
        ┌──────────────────────────────────────┐
        │   calibrate.py                        │  temperature scaling on dev split
        └──────────────┬───────────────────────┘
                       │
                       ▼
          frozen model artefact
    data/reward_model/checkpoints/best_model.pt
                       │
                       ▼
        HRS/src/rl_agent  (inference only — read-only)
```

**Dependency rules:**
- `HRS/src/preprocessing` must have produced `final_cdss_dataset.parquet` before any reward model job runs.
- `Mimic4DataLoader` must pass all schema assertions before `train.py` starts.
- `calibrate.py` requires `best_model.pt` from a completed training run.
- The reward model never writes to `HRS/data/preprocessing/` — that directory is read-only from this module's perspective.

**Runtime estimate (preliminary):**

| Phase | Estimated time | Notes |
|-------|---------------|-------|
| Data loading and validation | 2–5 min | Parquet read + schema assertions |
| Training | Hours–days | 2-GPU DDP; adversarial batches cost 2× per batch |
| Calibration | < 5 min | Single forward pass on dev split, single GPU |

---

## 6. Module Summary

| # | Module | Output | Notes |
|---|--------|--------|-------|
| 1 | `data_loader.py` / `mimic4_data_loader.py` | `DatasetBundle` + feature index map | `DataLoader` base with `Mimic4DataLoader` implementation; enforces upstream data contract and raises on failure with reference to `PREPROCESSING_DATA_MODEL.md` |
| 2 | `model.py` | `RewardModel` class | MLP definition only — no training logic; wrapped in `DistributedDataParallel` by `train.py` |
| 3 | `masking.py` | Masked input tensors | Reads feature index map from `mimic4_data_loader.py`; implements random, adversarial, and no-mask modes |
| 4 | `loss.py` | Scalar loss tensor | Dynamic NaN masking for `y2_readmission`; weighted BCE per head |
| 5 | `train.py` | Checkpoint files | DDP entry point via `torchrun`; masking curriculum; AdamW + cosine LR; early stopping; metric logging on rank 0 |
| 6 | `calibrate.py` | `calibration_params.json` | Per-head temperature scaling on dev split; single GPU |
| 7 | `inference.py` | Probability tensors | Frozen forward pass; consumed by RL agent; single GPU |

Supporting scripts (not in the training pipeline): `validate_contract.py` (standalone schema assertion runner without training), `export_model.py` (serialise frozen model for RL consumption).

---

## 7. Technology Stack

### Language and Runtime

Python 3.11+. CUDA 12.x required for GPU training. `torchrun` is used as the DDP launcher — it manages process spawning, rank assignment, and the `MASTER_ADDR`/`MASTER_PORT` environment variables. Consistent with `HRS/src/preprocessing` in language, CUDA version, and cluster environment.

### Core Frameworks and Libraries

| Library | Version | Role | Why chosen |
|---------|---------|------|------------|
| `torch` | ≥ 2.2 | Neural network, DDP, training loop, autograd | Standard; `BCEWithLogitsLoss` supports `pos_weight` natively; `DistributedDataParallel` for multi-GPU |
| `torch.distributed` | — | All-reduce gradient synchronisation across GPUs | Built into PyTorch; nccl backend for GPU-to-GPU communication |
| `pandas` | ≥ 2.0 | Parquet loading, label validation, NaN operations | Consistent with preprocessing; `df.loc` NaN operations are the canonical contract step |
| `pyarrow` | ≥ 14.0 | Parquet reads | Read-only access — no append needed, unlike preprocessing which uses `fastparquet` for append-mode writes |
| `pydantic` | ≥ 2.0 | `config/reward_model.yaml` validation at startup | Catches misconfigured schedules before training starts; consistent with preprocessing |
| `pyyaml` | ≥ 6.0 | Config loading | Human-editable; loaded once at startup; same convention as `config/preprocessing.yaml` |
| `scikit-learn` | ≥ 1.4 | AUROC, AUPRC, ECE metrics; calibration utilities | Mature, well-tested implementations |
| `numpy` | ≥ 1.26 | Tensor construction, NaN masking | Required by pandas and torch interop |

### Data Structures and Storage Formats

- **Parquet (`final_cdss_dataset.parquet`, input, read-only):** Columnar; predicate pushdown used for `split` filtering. Produced upstream via `fastparquet` append mode — this module reads with `pyarrow`.
- **YAML (`config/reward_model.yaml`):** All hyperparameters and schedule parameters. Validated by Pydantic. Follows the same convention as `config/preprocessing.yaml`.
- **PyTorch checkpoint (`best_model.pt`):** Weights (unwrapped from DDP), optimizer state, epoch, curriculum state, config snapshot. Written by rank 0 only. Enables full SLURM resume.
- **JSON (`calibration_params.json`):** Per-head temperature values `T_y1` and `T_y2`. Human-readable audit trail.
- **Parquet (`training_metrics.parquet`):** Per-epoch metrics written by rank 0. Columnar for downstream analysis.

### What Was Explicitly Rejected

- **`BCELoss` (without logits):** Rejected — numerically unstable; no native `pos_weight` support.
- **3-class softmax output:** Rejected. `y2_readmission` is NaN for deceased patients — a softmax implicitly assumes Y2 is defined for all patients. Two independent sigmoid heads with dynamic NaN masking correctly represent the label structure.
- **Internal projection / fusion layers:** Rejected. Dimensionality reduction of BERT embeddings belongs in `HRS/src/preprocessing`, not in the network.
- **Separate `feature_index_map.yaml` config:** Rejected. Feature slot boundaries are derived at load time from the canonical column order in `PREPROCESSING_DATA_MODEL.md` Section 3.12. A separate file would duplicate that information and create a consistency risk.
- **Fixed global importance ranking for adversarial masking:** Rejected. A ranking computed once does not adapt as the network evolves. Per-batch per-sample gradient-based selection (Shaham et al., 2016) is used instead.
- **`torch.multiprocessing` spawn workers (preprocessing pattern):** Not applicable here. Preprocessing uses worker processes because the bottleneck is independent embedding jobs per feature. Training uses DDP where all processes must synchronise gradients every step — `torchrun` with `nccl` backend is the correct multi-GPU pattern.
- **`fastparquet` for reading:** Preprocessing uses `fastparquet` for append-mode writes; read-only access here uses `pyarrow` to avoid the fastparquet/pyarrow serialisation incompatibility encountered in the preprocessing pipeline.

---

## 8. Key Subsystem Detail

### 8.1 Upstream Data Contract

`Mimic4DataLoader` enforces the following assertions at startup. All failures raise with a descriptive error message referencing `PREPROCESSING_DATA_MODEL.md` and the producing module.

**`y2_readmission` must be float32 with mathematical NaN for all deceased patients (`y1_mortality = 1`).** If a deceased patient carries `0.0` instead of NaN, the survivor mask `~torch.isnan(y2)` silently includes that patient in the readmission loss, corrupting `P(readmitted | survived)` without any visible error.

**All `*_embedding` columns must be float32 non-null.** Missing features are zero vectors — never NaN. NaN inside an embedding column propagates silently through all MLP layers and corrupts loss, gradients, and adversarial importance scores.

**`y1_mortality` must be int8 non-null (0 or 1) for every row.**

### 8.2 Feature Index Map Derivation

At load time, `Mimic4DataLoader` reads the ordered column list from `final_cdss_dataset.parquet` and constructs a feature index map in memory: `{'demographic_vec': (0, 8), 'diag_history_embedding': (8, 776), ...}`. This map is passed to `masking.py` and `train.py` and never written to disk. The canonical column order is defined in `PREPROCESSING_DATA_MODEL.md` Section 3.12 — any upstream change to feature count or order is automatically reflected at reward model load time.

### 8.3 Multi-GPU Distributed Training

Training uses PyTorch `DistributedDataParallel` (DDP) launched via `torchrun`. The number of GPUs is controlled by `NUM_GPUS` in `config/reward_model.yaml` (default: 2). `torchrun` spawns one process per GPU, each with a unique `rank`. Each process loads a full copy of the dataset and model. A `DistributedSampler` shards the training data across ranks so each GPU processes a non-overlapping subset of the mini-batch. Gradients are all-reduced across GPUs after each backward pass. Checkpointing, metric logging, and masking curriculum state are managed by rank 0 only to avoid duplicate writes.

The `nccl` backend is used for GPU-to-GPU communication, consistent with best practice for PyTorch DDP on CUDA hardware. If only 1 GPU is available, training runs in single-process mode without DDP wrapping.

Adversarial masking under DDP requires attention: the first forward/backward pass (to compute gradient norms) must be performed with `model.no_sync()` to suppress premature all-reduce, since only the second pass should trigger gradient synchronisation across GPUs.

### 8.4 Masking Strategy and Curriculum

Three masking modes are applied externally to the network before each forward pass. The network receives a fixed-length 42,248-dim float32 tensor with no awareness of which slots are masked.

**Random masking** zeroes one or more feature slots selected uniformly at random per sample (default k=1, configurable). **Adversarial masking** implements Shaham et al. (2016) robust optimisation adapted for discrete feature-slot masking: a first forward pass with `model.no_sync()` computes per-feature gradient L2 norms by aggregating `∂L/∂x` over each slot's index range; the slot with the highest norm per sample is zeroed; a second forward/backward pass (with DDP all-reduce) updates weights. This doubles the cost of adversarial batches. **No masking** passes the full vector unchanged.

The probability of each mode evolves via a configurable sigmoid crossover schedule. Default: 100% random at epoch 0, transitioning to 33%/33%/33% by the final epoch. All schedule parameters (`masking_start_ratios`, `masking_end_ratios`, `transition_shape`, `transition_midpoint_epoch`) are in `config/reward_model.yaml`.

### 8.5 Loss Function and Class Imbalance

Total loss: `L = w1 * L_Y1 + w2 * L_Y2`. `L_Y1` uses `BCEWithLogitsLoss` with `pos_weight_y1 ≈ 9.0` (computed from training rows). `L_Y2` applies a dynamic per-batch NaN mask before `BCEWithLogitsLoss` with `pos_weight_y2 ≈ 4.0` (computed from training survivors only). The all-deceased-batch edge case sets `L_Y2 = 0.0` explicitly. Both `pos_weight` values are computed once from `split = 'train'` rows on rank 0 and broadcast to all ranks before training begins.

### 8.6 Neural Network Architecture

The network is a feedforward MLP with a gradual funnel. Under DDP, each GPU holds a full model copy (~1.32 GB for Hidden 1 alone at float32). With 2 GPUs and AdamW optimizer state, total GPU memory per device is approximately 14–18 GB at batch size 256 per GPU (512 effective). The recommended mitigation if memory is exceeded is PCA reduction of BERT embeddings (768 → 256) applied in `HRS/src/preprocessing` — input reduces to ~14,088 dims with no architectural change to this module.

| Layer | In | Out | Activation | Regularisation |
|-------|----|-----|------------|----------------|
| Hidden 1 | 42,248 | 8,192 | ReLU | BatchNorm + Dropout |
| Hidden 2 | 8,192 | 2,048 | ReLU | BatchNorm + Dropout |
| Hidden 3 | 2,048 | 512 | ReLU | BatchNorm + Dropout |
| Hidden 4 | 512 | 128 | ReLU | BatchNorm + Dropout |
| Head Y1 | 128 | 1 | Sigmoid | — |
| Head Y2 | 128 | 1 | Sigmoid | — |

All widths and dropout rates are configurable in `config/reward_model.yaml`.

### 8.7 Post-Training Calibration

Temperature scaling is applied on the dev split after training converges, on a single GPU. A scalar `T` is learned per head via NLL minimisation — weights are not modified. Well-calibrated probabilities are critical because the RL reward is `ΔP` between consecutive states. `T_y1` and `T_y2` are written to `data/reward_model/calibration_params.json` and applied at inference time.

---

## 9. Final Output

**Frozen model:** `HRS/data/reward_model/checkpoints/best_model.pt` — PyTorch state dict (unwrapped from DDP), calibration parameters, feature index map snapshot, and config. Loaded by `inference.py` for the RL agent.

**Training metrics:** `HRS/data/reward_model/training_metrics.parquet` — written by rank 0.

| Column group | Columns | Type |
|---|---|---|
| Epoch metadata | `epoch`, `wall_time_s`, `masking_random_pct`, `masking_adversarial_pct`, `masking_none_pct` | int / float32 |
| Loss | `loss_total`, `loss_y1`, `loss_y2` | float32 |
| Y1 performance | `auroc_y1`, `auprc_y1`, `ece_y1` | float32 |
| Y2 performance | `auroc_y2`, `auprc_y2`, `ece_y2` | float32 |

---

## 10. Security

This module processes de-identified MIMIC-IV data under the PhysioNet data use agreement on the university HPC cluster. Log statements reference row counts and aggregate statistics only — no individual patient record contents. Model checkpoints contain learned weights only, not training data. The PhysioNet data use agreement prohibits sharing raw MIMIC-IV data or derivatives that could re-identify patients outside the credentialed research group. No API keys or credentials are used by this module — MIMIC-IV access is managed by `HRS/src/preprocessing` only.

---

## 11. Performance

**Dominant bottleneck — Hidden 1 weight matrix.** The 42,248 × 8,192 matrix occupies ~1.32 GB at float32 per GPU. Under DDP with `nccl` all-reduce, each GPU holds a full model copy — memory savings from 2-GPU DDP come from halving the per-GPU effective batch size, not from splitting model weights. With AdamW optimizer state (~3× parameter size), Hidden 1 alone requires ~5 GB per GPU before activations.

**Effective batch size with DDP.** With `batch_size = 256` per GPU and 2 GPUs, the effective batch size is 512. Each GPU processes 256 samples per forward/backward pass. At batch size 256 per GPU, estimated total GPU memory per device is 14–18 GB.

**Adversarial masking cost.** Each adversarial batch requires two forward/backward passes (first with `no_sync()`, second with all-reduce). At the default end-state curriculum (33% adversarial), ~33% of batches cost 2×. Wall time per epoch is logged in `training_metrics.parquet` by rank 0.

**Scaling knobs:**

| Knob | Config key | Effect |
|------|-----------|--------|
| GPU count | `NUM_GPUS` | Linear throughput scaling up to available GPUs |
| Reduce Hidden 1 width | `layer_widths[0]` | Cuts first-layer parameters quadratically |
| Reduce batch size per GPU | `batch_size` | Reduces per-GPU activation memory |
| PCA in preprocessing | Applied in `HRS/src/preprocessing` | Reduces input to ~14,088 dims; no network change |
| Reduce adversarial ratio | `masking_end_ratios` | Reduces proportion of double-pass batches |

**Throughput:** Not yet characterised on target hardware. Anticipated bottleneck is Hidden 1 forward pass (dense matrix multiply), not I/O — `final_cdss_dataset.parquet` fits in RAM for the full MIMIC-IV cohort.

See `REWARD_MODEL_DETAILED_DESIGN.md` for per-module memory requirements and full configuration reference.

---

## 12. Design Principles

**No leakage across splits.** All statistics derived from data — `pos_weight_y1`, `pos_weight_y2`, temperature scaling parameters — are computed from `split = 'train'` rows only, computed on rank 0, broadcast to all ranks, and frozen for the training run. The split assignment is read from the upstream dataset; this module never re-splits data.

**Hard contracts, hard failures.** Upstream schema violations raise immediately in `Mimic4DataLoader` with descriptive error messages referencing `PREPROCESSING_DATA_MODEL.md` by section and the producing module by name.

**No hardcoded values.** Every hyperparameter, architectural dimension, schedule parameter, file path, and GPU count is defined in `config/reward_model.yaml` and validated by Pydantic on startup. Input dimensionality is derived at runtime from the dataset column schema.

**Preprocessing owns dimensionality.** BERT embedding dimensions, PCA reduction choices, and feature count are decisions made in `HRS/src/preprocessing`. The reward model accepts whatever `final_cdss_dataset.parquet` provides.

**Masking is external to the network.** The network receives a flat float32 tensor and has no awareness of masked slots. All masking logic lives in `masking.py`, entirely decoupled from `model.py`.

**Feature boundaries are derived, not declared.** The feature index map is constructed at load time from the canonical column order in `PREPROCESSING_DATA_MODEL.md` Section 3.12. No separate index map config file exists.

**One class per file.** Each class definition lives in its own `*.py` module (for example, `RewardModelConfig` in `reward_model_config.py`, `ParquetDataset` in `parquet_dataset.py`). Shared helpers remain in `reward_model_utils.py`, which now only re-exports these class modules.

**Rank 0 owns all I/O.** Under DDP, only rank 0 writes checkpoints, metrics, and logs. All ranks participate in forward/backward passes and gradient all-reduce. This prevents duplicate writes and ensures a consistent checkpoint state.

**Resumability.** Every checkpoint saves model weights (unwrapped from DDP), optimizer state, current epoch, curriculum schedule state, and a full config snapshot. Re-running `train.py --resume` via `torchrun` continues from the last checkpoint. Schema validation runs on every start regardless of resume status.

---

## 13. Directory Structure

```
HRS/
├── config/
│   ├── preprocessing.yaml                   # owned by HRS/src/preprocessing
│   └── reward_model.yaml                    # all reward model hyperparameters and schedule params
│
├── src/
│   ├── preprocessing/                       # upstream pipeline
│   │   └── ...
│   │
│   └── reward_model/
│       ├── reward_model_architecture.md     # this document
│       ├── REWARD_MODEL_DETAILED_DESIGN.md  # per-module implementation details
│       │
│       ├── mimic4_data_loader.py           # step 1 — load, validate schema, derive feature index map
│       ├── model.py                         # step 2 — RewardModel MLP definition
│       ├── masking.py                       # step 3 — random / adversarial / no-mask modes
│       ├── loss.py                          # step 4 — weighted BCE + dynamic NaN masking for Y2
│       ├── train.py                         # step 5 — DDP training loop, curriculum, checkpointing
│       ├── calibrate.py                     # step 6 — temperature scaling on dev split
│       ├── inference.py                     # step 7 — frozen forward pass for RL agent
│       │
│       ├── schema_error.py                  # shared SchemaError exception (class-only file)
│       ├── reward_model_config.py           # Pydantic RewardModelConfig + loader (class-only file)
│       ├── parquet_dataset.py               # ParquetDataset class (lazy Parquet reader)
│       ├── row_group_block_sampler.py       # RowGroupBlockSampler class (row-group-aware sampler)
│       ├── dataset_bundle.py                # DatasetBundle NamedTuple (train/dev/test bundle)
│       ├── reward_model_utils.py            # shared helpers + re-exports (no class definitions)
│       │
│       ├── reward_job.sh                    # SLURM: training job (2× GPU, 64G)
│       ├── calibrate_job.sh                 # SLURM: calibration job (1× GPU, 32G)
│       ├── submit_reward.sh                 # submit training then calibration with dependency chain
│       │
│       ├── validate_contract.py             # standalone: schema assertions without training
│       └── export_model.py                  # standalone: serialise frozen model for RL agent
│
└── data/
    ├── preprocessing/                       # [git-ignored] owned by HRS/src/preprocessing
    │   └── classifications/
    │       └── final_cdss_dataset.parquet   # primary input to reward model (read-only)
    │
    └── reward_model/                        # [git-ignored] generated by this module
        ├── checkpoints/
        │   ├── best_model.pt
        │   └── epoch_<N>.pt
        ├── training_metrics.parquet
        └── calibration_params.json
```

---

## 14. Infrastructure and Execution

### Cluster and Environment

University HPC cluster running SLURM — same cluster as `HRS/src/preprocessing`. Partition names, GPU type, time limits, and RAM are defined in `config/reward_model.yaml`. Two GPUs per training job (default; controlled by `NUM_GPUS`). GPU with ≥24 GB VRAM recommended per device. CUDA 12.x, Python 3.11+.

### Capacity Sizing

Under DDP with 2 GPUs, each GPU holds a full model copy. Hidden 1 weight matrix (42,248 × 8,192, ~1.32 GB float32) plus AdamW optimizer state requires ~5 GB per GPU before activations. At batch size 256 per GPU, total estimated GPU memory per device is 14–18 GB. A 24 GB GPU (A100 or equivalent) provides sufficient headroom. If memory is exceeded, apply PCA (768 → 256) in `HRS/src/preprocessing` — input reduces to ~14,088 dims, Hidden 1 to ~231 MB, with no changes to this module.

### Scripts

All SLURM scripts live alongside the Python modules in `HRS/src/reward_model/`.

| Script | GPUs | RAM | Purpose |
|--------|------|-----|---------|
| `reward_job.sh` | 2 | 64G | DDP training run via `torchrun` |
| `calibrate_job.sh` | 1 | 32G | Temperature scaling after training |
| `validate_contract.py` | 0 | 16G | Schema assertion check only |
| `export_model.py` | 0 | 8G | Serialise frozen model for RL agent |

### How to Run

```bash
# Validate upstream data contract before submitting training
cd ~/Python/HRS
python src/reward_model/validate_contract.py --config config/reward_model.yaml

# Submit training then calibration as a chained SLURM dependency
bash src/reward_model/submit_reward.sh

# Resume after preemption (torchrun re-launches all DDP workers)
torchrun --nproc_per_node=2 src/reward_model/train.py \
  --config config/reward_model.yaml \
  --resume
```

### Job Chain

```
[validate_contract.py]
          │  assertions pass
          ▼
[reward_job.sh]  (torchrun, 2 GPUs)
   rank 0 + rank 1 ── SLURM preemption ──► [reward_job.sh --resume]
          │  best_model.pt written by rank 0
          └──(afterok)──► [calibrate_job.sh]  (single GPU)
                                   │  calibration_params.json written
                                   ▼
                            [export_model.py]
                                   │  frozen model artefact
                                   ▼
                            HRS/src/rl_agent
```

### Resume Guarantee

Re-running `reward_job.sh --resume` relaunches `torchrun` with 2 workers. Each worker loads the latest checkpoint from `data/reward_model/checkpoints/`, restores optimizer state and curriculum schedule, and continues from the saved epoch. Schema validation via `Mimic4DataLoader` runs on every start regardless of resume status. Only rank 0 reads and writes the checkpoint — rank 1 receives the loaded state via DDP process group initialisation.

---

> See `REWARD_MODEL_DETAILED_DESIGN.md` for per-module implementation details, full `config/reward_model.yaml` reference, and per-layer memory requirements.
>
> See `PREPROCESSING_ARCHITECTURE.md` and `PREPROCESSING_DATA_MODEL.md` for the upstream pipeline that produces `final_cdss_dataset.parquet`.
