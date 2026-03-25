# CDSS-ML Reward Model — Detailed Design

> **Location:** `HRS/src/reward_model/REWARD_MODEL_DETAILED_DESIGN.md`
>
> This document is the authoritative implementation reference for the Reward Model module. For high-level structure, design rationale, and system context see `reward_model_architecture.md` in the same directory.

---

## Table of Contents

1. [Primary Identifier](#1-primary-identifier)
2. [Module Decomposition](#2-module-decomposition)
3. [Cross-Cutting Concerns](#3-cross-cutting-concerns)
   - [3.1 Configuration Loading](#31-configuration-loading)
   - [3.2 Logging](#32-logging)
   - [3.3 DDP Process Group and Rank Conventions](#33-ddp-process-group-and-rank-conventions)
   - [3.4 Shared Utilities (`reward_model_utils.py`)](#34-shared-utilities-reward_model_utilspy)
4. [Feature Index Map](#4-feature-index-map)
5. [Module Implementation](#5-module-implementation)
   - [load_dataset.py](#load_datasetpy)
   - [model.py](#modelpy)
   - [masking.py](#maskingpy)
   - [loss.py](#losspy)
   - [train.py](#trainpy)
   - [calibrate.py](#calibratepy)
   - [inference.py](#inferencepy)
   - [validate_contract.py](#validate_contractpy)
   - [export_model.py](#export_modelpy)
6. [DDP Training Implementation](#6-ddp-training-implementation)
   - [6.1 Process Launch and Initialisation](#61-process-launch-and-initialisation)
   - [6.2 Data Sharding](#62-data-sharding)
   - [6.3 Adversarial Masking Under DDP](#63-adversarial-masking-under-ddp)
   - [6.4 Rank 0 I/O Discipline](#64-rank-0-io-discipline)
7. [Adversarial Masking Gradient Computation](#7-adversarial-masking-gradient-computation)
8. [Checkpoint and Resume](#8-checkpoint-and-resume)
9. [Configuration Reference](#9-configuration-reference)
10. [Memory Requirements](#10-memory-requirements)

---

## 1. Primary Identifier

The reward model operates at the admission level. `hadm_id` (int64) is the primary key throughout — one row in `final_cdss_dataset.parquet` corresponds to one hospital admission. `subject_id` is present in the dataset but is not used by any reward model module; it is carried through for potential downstream use by the RL agent.

The reward model does not perform any identifier linkage. All identifier resolution and null-`hadm_id` handling is completed upstream by `HRS/src/preprocessing`. By the time `final_cdss_dataset.parquet` reaches this module, every row has a valid `hadm_id`.

---

## 2. Module Decomposition

### Module boundaries

Each Python file corresponds to one pipeline concern. Class definitions live in class-only modules (`reward_model_config.py`, `parquet_dataset.py`, `row_group_block_sampler.py`, `dataset_bundle.py`, `schema_error.py`) and are re-exported by `reward_model_utils.py` for backward compatibility. Inter-module state exchange happens via tensors passed as function arguments within a single process, or via checkpoint files on disk between separate SLURM jobs. No module reads `final_cdss_dataset.parquet` except `load_dataset.py`.

### Class vs plain script

| Module | Pattern | Reason |
|--------|---------|--------|
| `load_dataset.py` | Plain script, `run(config)` | Single top-to-bottom load; no shared state needed after return |
| `model.py` | **Class** (`RewardModel`) | Stateful network; instantiated once, called many times via `forward()`; must be wrappable by DDP |
| `masking.py` | **Class** (`MaskingSchedule`) | Maintains curriculum state across the training loop; epoch advancement is a stateful operation |
| `loss.py` | Plain functions | Stateless transformations; `compute_loss(logits_y1, logits_y2, y1, y2, weights)` |
| `train.py` | Plain script, DDP entry point | Top-to-bottom training loop; all state lives in checkpoint |
| `calibrate.py` | Plain script, `run(config)` | Single optimisation pass; no persistent state |
| `inference.py` | **Class** (`RewardModelInference`) | Loaded once, called repeatedly per RL step; holds frozen weights and calibration params in memory |
| `validate_contract.py` | Plain script, CLI tool | One-shot assertion run; exits with code 0 (pass) or 1 (fail) |
| `export_model.py` | Plain script, CLI tool | One-shot serialisation; no shared state |

### Encapsulation rules

All helper functions are module-private (prefixed `_`). The public interface of each pipeline module is its `run(config)` function or class constructor. `train.py` is the sole exception — it is launched directly by `torchrun` and its entry point is the module-level `main()` function. Config is always passed as a validated Pydantic model object, not a raw dict.

### File naming conventions

`*.py` for pipeline step modules, class-only modules for each class, `reward_model_utils.py` for shared helpers/re-exports, `*_job.sh` for SLURM scripts, `submit_*.sh` for submission orchestrators. All names use `snake_case`.

---

## 3. Cross-Cutting Concerns

---

### 3.1 Configuration Loading

`config/reward_model.yaml` is the single source of truth for all parameters. It is loaded and validated by any CLI entry point using the Pydantic model defined in `reward_model_config.py` (re-exported by `reward_model_utils.py`). The Pydantic model enforces types, required vs optional fields, and value constraints at startup — a misconfigured schedule or invalid path raises before any computation begins.

All path values are expanded with `os.path.expanduser` and resolved to absolute paths before being stored in the config object. No module calls `yaml.safe_load` directly. Config keys use `SCREAMING_SNAKE_CASE`. Boolean flags use YAML native `true`/`false`. Path keys end in `_DIR`, `_PATH`, or `_FILE`.

Under DDP, the config object is loaded by rank 0 and broadcast to all worker ranks via the process group before any training begins. All ranks therefore operate from an identical config snapshot for the duration of the run.

---

### 3.2 Logging

Every module obtains its logger as `logger = logging.getLogger(__name__)`, placing loggers in the hierarchy `reward_model.<module_name>`. No module calls `logging.basicConfig()` — the entry point configures the root handler once.

Under DDP, only rank 0 emits `INFO` and above to stdout. All ranks emit `ERROR` and `CRITICAL` regardless of rank, ensuring errors from any GPU are visible without duplicating progress output.

| Level | When to use |
|-------|-------------|
| `DEBUG` | Per-batch detail; gradient norm values; mask selection per sample |
| `INFO` | Epoch start/end; loss and metric values; checkpoint written; curriculum state |
| `WARNING` | Fallback used (e.g. single-GPU mode when `NUM_GPUS > 1` but only 1 available) |
| `ERROR` | A validation assertion failed but the process can report it before exiting |
| `CRITICAL` | Unrecoverable — let the exception propagate |

Log format set by the entry point: `%(asctime)s  %(levelname)-8s  %(name)s  %(message)s`

---

### 3.3 DDP Process Group and Rank Conventions

`torchrun` spawns one process per GPU. Each process has a `rank` (0 to `NUM_GPUS - 1`) and a `local_rank` (GPU index on the current node). For the default 2-GPU single-node configuration these are identical.

The process group is initialised with the `nccl` backend at the start of `train.py` using environment variables set by `torchrun` (`MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, `LOCAL_RANK`). If `NUM_GPUS = 1` or only one GPU is available at runtime, training proceeds in single-process mode without initialising a process group — no DDP wrapping, no `DistributedSampler`. This fallback is detected automatically and logged at `WARNING`.

**Rank 0 is the designated I/O rank.** Only rank 0 writes checkpoints, `training_metrics.parquet`, and `calibration_params.json`. Only rank 0 computes `pos_weight` values and broadcasts them to other ranks. All ranks participate in forward passes, backward passes, and all-reduce operations. Barrier synchronisation (`dist.barrier()`) is used at three points: after checkpoint load, after `pos_weight` broadcast, and before the final checkpoint write at training completion.

---

### 3.4 Shared Utilities (`reward_model_utils.py`)

| Function / Class | Purpose |
|------------------|---------|
| `load_and_validate_config(path)` | Load `reward_model.yaml`, validate with Pydantic, return config object (defined in `reward_model_config.py`, re-exported) |
| `build_feature_index_map(columns)` | Construct `{col_name: (start, end)}` from ordered column list; see Section 4 |
| `compute_pos_weights(df_train)` | Compute `pos_weight_y1` and `pos_weight_y2` from training split; excludes deceased rows for Y2 |
| `sigmoid_crossover(epoch, total_epochs, start_ratios, end_ratios, midpoint)` | Compute current masking mode probabilities for a given epoch |
| `get_device(local_rank)` | Return `torch.device('cuda', local_rank)` or `cpu` with CUDA availability check |
| `unwrap_ddp(model)` | Return `model.module` if wrapped in DDP, else `model` directly |
| `broadcast_tensor(tensor, src_rank)` | Broadcast a scalar tensor from `src_rank` to all ranks via process group |
| **`ParquetDataset(Dataset)`** | Class-only module `parquet_dataset.py`. Lazy row-group reads from `final_cdss_dataset.parquet`; constructor accepts open PyArrow file handle, split row indices, the feature index map, and `DATASET_ROW_GROUP_CACHE_SIZE`. Holds an LRU cache of at most `DATASET_ROW_GROUP_CACHE_SIZE` decompressed row groups in memory at any time. `__getitem__(i)` resolves the row group containing row `i`, reads it from the LRU cache or from disk, slices the requested row, concatenates feature columns in index map order into a float32 tensor, and returns `(X, y1, y2)`. `__len__` returns the number of rows in the split. Re-exported by `reward_model_utils.py`. |
| **`RowGroupBlockSampler(Sampler)`** | Class-only module `row_group_block_sampler.py`. Row-group-aware sampler to preserve Parquet row-group locality and partition row groups round-robin across DDP ranks. Re-exported by `reward_model_utils.py`. |
| **`DatasetBundle(NamedTuple)`** | Class-only module `dataset_bundle.py`. Bundles `train/dev/test` `ParquetDataset` instances plus metadata. Re-exported by `reward_model_utils.py`. |

A helper function belongs in `reward_model_utils.py` if and only if it is used by two or more modules and has no module-specific state. Shared classes sit in class-only modules and are re-exported by `reward_model_utils.py`. Functions used only once remain private to their module with a `_` prefix.

---

## 4. Feature Index Map

The feature index map defines the start and end index within the flat 42,248-dim input tensor for each feature slot. It is derived at load time by `load_dataset.py` from the ordered list of feature columns in `final_cdss_dataset.parquet`, following the canonical column order defined in `PREPROCESSING_DATA_MODEL.md` Section 3.12.

The derivation algorithm iterates the ordered column list, skips the three metadata columns (`subject_id`, `hadm_id`, `split`) and the two label columns (`y1_mortality`, `y2_readmission`), and for each remaining column assigns a start index equal to the running offset and an end index equal to start plus the column's declared dimension — 8 for `demographic_vec` and 768 for all `*_embedding` columns. The result is a dict mapping column name to a `(start, end)` tuple.

The algorithm expects to find exactly **56 feature columns**: 1 structured vector (`demographic_vec`) and **55 embedding columns** broken down as follows:

- **4 history and triage embeddings** — `diag_history_embedding`, `discharge_history_embedding`, `triage_embedding`, `chief_complaint_embedding` (F2–F5)
- **13 lab group embeddings** — `lab_blood_gas_embedding` through `lab_stool_embedding` (F6–F18)
- **1 radiology embedding** — `radiology_embedding` (F19)
- **37 microbiology panel embeddings** — `micro_blood_culture_routine_embedding` through `micro_throat_strep_embedding` (F20–F56)

If the count of `*_embedding` columns in the dataset does not equal exactly 55, or if the count within any of the four groups above does not match (4 + 13 + 1 + 37), `build_feature_index_map()` raises a `SchemaError` referencing `PREPROCESSING_DATA_MODEL.md` Section 3.12. This guards against silent miscounting if the upstream dataset schema changes.

The map is constructed once by `load_dataset.py`, stored in the returned `DatasetBundle`, and passed explicitly to `masking.py` and `train.py`. It is also saved as a snapshot inside every checkpoint file so that `inference.py` can reconstruct the same boundaries when loading a frozen model, even if the upstream dataset schema were to change between runs.

The map is never written to a standalone config or YAML file — a separate file would duplicate `PREPROCESSING_DATA_MODEL.md` Section 3.12 and create a consistency risk.

---

## 5. Module Implementation

---

### `load_dataset.py`

Loads `final_cdss_dataset.parquet`, validates the upstream data contract, constructs the feature index map, and returns lazy `ParquetDataset` objects per split plus metadata. Plain script with `run(config)` entry point. The dataset is never fully materialised into a float32 tensor — batches are read lazily from disk by `ParquetDataset.__getitem__` at training time.

**Algorithm:**
1. Read `final_cdss_dataset.parquet` from `DATASET_PATH` using `pyarrow`. Do not use `fastparquet` — the fastparquet/pyarrow serialisation incompatibility observed in the preprocessing pipeline applies here.
2. Validate column presence: assert all 61 expected columns are present in canonical order from `PREPROCESSING_DATA_MODEL.md` Section 3.12. Raise `SchemaError` referencing the data model doc if any column is missing or out of order.
3. Validate `y1_mortality`: assert dtype is int8 or float32, assert no null values. Error message references `extract_y_data.py` as the producing module.
4. Validate `y2_readmission`: assert dtype is float32. Assert all rows where `y1_mortality = 1` carry NaN. Assert all rows where `y1_mortality = 0` are non-null. Error message references `PREPROCESSING_DATA_MODEL.md` Section 3.12 and `extract_y_data.py`.
5. Validate all `*_embedding` columns: assert dtype is float32 and no null values exist in any embedding column. Error message references `combine_dataset.py`.
6. Build the feature index map via `build_feature_index_map()` from `reward_model_utils.py`. This step also validates the 55-embedding count and per-group breakdown — see Section 4.
7. Read the `split` column only (lightweight — metadata column) to determine row indices for each split. Produce three lists of row indices: `train_indices`, `dev_indices`, `test_indices`.
8. Compute `pos_weight_y1` and `pos_weight_y2` from the training split rows only via `compute_pos_weights()`. This reads only `y1_mortality` and `y2_readmission` columns for the training rows — not the full feature data. These values will be broadcast to non-rank-0 processes by `train.py`.
9. Instantiate three `ParquetDataset` objects (class-only module `parquet_dataset.py`, re-exported by `reward_model_utils.py`) — one per split — passing the open PyArrow file handle, the split's row index list, the feature index map, and `DATASET_ROW_GROUP_CACHE_SIZE`. Each `ParquetDataset` holds only the file handle and index metadata in memory. No feature data is read at this step.
10. Return a `DatasetBundle` named tuple (class-only module `dataset_bundle.py`, re-exported by `reward_model_utils.py`): `train_dataset`, `dev_dataset`, `test_dataset` (each a `ParquetDataset`), `feature_index_map`, `pos_weight_y1`, `pos_weight_y2`.

**Config keys used:** `DATASET_PATH`, `FEATURES_DIM` (assertion only — must equal D as derived from the column schema), `DATASET_ROW_GROUP_CACHE_SIZE`.

**Memory note:** The dataset is never fully resident in RAM. At any given time, only the LRU-cached row groups are held in memory — at most `DATASET_ROW_GROUP_CACHE_SIZE × row_group_size × D × 4` bytes. At the default cache size of 2, a typical row group size of ~400 rows, and D=42,248, this is approximately **135 MB**. Combined with DataLoader prefetch workers (each holding one batch buffer of ~42 MB), total dataset-related RAM is well under 1 GB — comfortably within the 64 GB SLURM allocation alongside the model, optimizer state, and OS overhead.

---

### `model.py`

Defines the `RewardModel` class — a feedforward MLP with a gradual funnel and two independent sigmoid output heads.

**Class: `RewardModel(nn.Module)`**

The constructor accepts `input_dim` (derived from the feature index map at runtime — never hardcoded), `layer_widths` (list of hidden layer output sizes), `dropout_rate`, and `activation`. It builds a sequence of `Linear` → `BatchNorm1d` → `Activation` → `Dropout` blocks for each consecutive pair in `[input_dim] + layer_widths`. After the final hidden block, two independent `Linear(layer_widths[-1], 1)` heads produce raw logits for Y1 and Y2.

`forward(x)` returns a tuple `(logits_y1, logits_y2)`, each of shape `(batch_size, 1)`. **Sigmoid is not applied inside `forward`** — raw logits are returned so that `BCEWithLogitsLoss` can be used for numerical stability in training. Sigmoid is applied only at inference time in `inference.py`.

The model is not DDP-aware — DDP wrapping is the responsibility of `train.py`. This keeps `model.py` independently testable without a process group.

**Config keys used:** `LAYER_WIDTHS`, `DROPOUT_RATE`, `ACTIVATION`.

---

### `masking.py`

Implements the `MaskingSchedule` class, which maintains the curriculum state and applies the correct masking mode to each mini-batch.

**Class: `MaskingSchedule`**

The constructor accepts the `feature_index_map`, curriculum schedule parameters (`start_ratios`, `end_ratios`, `transition_shape`, `transition_midpoint_epoch`, `total_epochs`), and `k` (features zeroed per sample in random mode, default 1).

`get_mode_probabilities(epoch)` delegates to `sigmoid_crossover()` in `reward_model_utils.py` and returns the current `(p_random, p_adversarial, p_none)` tuple for the given epoch.

`sample_mode(epoch)` draws a masking mode string — `'random'`, `'adversarial'`, or `'none'` — according to the current probabilities.

`apply_random_mask(X)` selects `k` feature slots uniformly at random per sample without replacement, then zeros the corresponding index ranges to 0.0. Returns the masked tensor.

`apply_adversarial_mask(X, grad_X)` receives the input tensor and the gradient `∂L/∂X` from the first forward/backward pass. For each sample it computes the L2 norm of the gradient over each feature slot's index range, identifies the highest-norm slot, and zeros that slot. Returns the adversarially masked tensor. The gradient computation is the responsibility of `train.py` — see Section 7.

`apply_no_mask(X)` returns `X` unchanged.

**Config keys used:** `MASKING_START_RATIOS`, `MASKING_END_RATIOS`, `MASKING_TRANSITION_SHAPE`, `MASKING_TRANSITION_MIDPOINT_EPOCH`, `MASKING_K`.

---

### `loss.py`

Plain module-level functions. No class, no state.

`compute_loss(logits_y1, logits_y2, y1, y2, pos_weight_y1, pos_weight_y2, w1, w2)` computes the total weighted loss.

For `L_Y1`: apply `BCEWithLogitsLoss` with `pos_weight = pos_weight_y1` over the full batch. `y1` is always fully populated — no masking required.

For `L_Y2`: construct the survivor mask `~torch.isnan(y2)`. If the mask is all-false (entire batch is deceased), set `L_Y2 = torch.tensor(0.0)` on the correct device to prevent NaN propagation. Otherwise apply `BCEWithLogitsLoss` with `pos_weight = pos_weight_y2` over the masked subset `logits_y2[mask]` against `y2[mask]`.

Total loss: `L = w1 * L_Y1 + w2 * L_Y2`. Returns the scalar total loss plus the two component losses separately for epoch logging.

`compute_metrics(logits_y1, logits_y2, y1, y2)` applies sigmoid to both logit tensors, constructs the survivor mask, and computes AUROC, AUPRC, and ECE for Y1 (full batch) and Y2 (survivors only) using scikit-learn. Called on the dev split by rank 0 at epoch end.

**Config keys used:** `LOSS_WEIGHT_Y1`, `LOSS_WEIGHT_Y2`.

---

### `train.py`

DDP entry point. Launched by `torchrun`. Plain script with `main()` entry point. Orchestrates the full training loop including curriculum scheduling, masking, loss, optimisation, evaluation, and checkpointing.

**Algorithm:**
1. Parse CLI arguments: `--config`, `--resume`.
2. Load and validate config via `load_and_validate_config()`.
3. Initialise the DDP process group with `nccl` backend using environment variables set by `torchrun`. If only one GPU is available, skip DDP initialisation and proceed in single-process mode (logged at `WARNING`).
4. On rank 0 only: call `load_dataset.run(config)` to load data and compute `pos_weight` values. Broadcast `pos_weight_y1` and `pos_weight_y2` to all ranks via `broadcast_tensor()`. Non-rank-0 processes wait at the broadcast and receive the scalar values without loading the dataset.
5. Instantiate `RewardModel` with `input_dim` from the feature index map. Move to `get_device(local_rank)`. Wrap in `DistributedDataParallel` if multi-GPU.
6. Instantiate `MaskingSchedule` with config parameters.
7. Instantiate `AdamW` with `LEARNING_RATE` and `WEIGHT_DECAY`. Instantiate cosine annealing scheduler with linear warmup over `LR_WARMUP_EPOCHS`.
8. If `--resume`: load latest checkpoint via the mechanism in Section 8. Restore model weights, optimiser state, scheduler state, epoch, and masking schedule state. Broadcast model weights from rank 0 to all ranks. Barrier after broadcast.
9. Construct `DistributedSampler` over the training dataset (rank 0 only holds data — see Section 6.2). Wrap in `DataLoader` with `batch_size = BATCH_SIZE_PER_GPU`.
10. For each epoch from the current epoch to `MAX_EPOCHS`:
    - Call `sampler.set_epoch(epoch)` to reshuffle per epoch.
    - For each mini-batch: sample masking mode from `MaskingSchedule`; apply the appropriate mask — adversarial mode requires two forward/backward passes using `model.no_sync()` for the first pass (see Section 6.3 and 7); compute loss via `loss.compute_loss()`; call `optimizer.step()`; `scheduler.step()`.
    - On rank 0 at epoch end: run dev evaluation via `loss.compute_metrics()` on the full dev split (no DDP); log metrics to `INFO`; append epoch row to `training_metrics.parquet`; check early stopping criterion.
    - If dev loss improved: write checkpoint (rank 0 only, see Section 8); update `best_model.pt`.
    - Broadcast early stopping signal from rank 0 to all ranks via `broadcast_tensor()`. All ranks break the training loop simultaneously on a `True` stop signal.
11. On rank 0: write final checkpoint. Call `dist.destroy_process_group()`.

**Config keys used:** `LEARNING_RATE`, `WEIGHT_DECAY`, `LR_WARMUP_EPOCHS`, `LR_MIN`, `ADAM_BETA1`, `ADAM_BETA2`, `MAX_EPOCHS`, `BATCH_SIZE_PER_GPU`, `NUM_GPUS`, `EARLY_STOPPING_PATIENCE`, `CHECKPOINT_DIR`, `CHECKPOINT_KEEP_N`, `METRICS_PATH`, all masking keys, all loss keys, all model architecture keys.

---

### `calibrate.py`

Applies temperature scaling to the best trained model. Plain script with `run(config)` entry point. Single GPU — no DDP.

**Algorithm:**
1. Load `best_model.pt` from `CHECKPOINT_DIR`. Extract model state dict and the config snapshot from the checkpoint (not the current `config/reward_model.yaml` — the checkpoint config is authoritative for architecture).
2. Instantiate `RewardModel` from the checkpoint config snapshot. Load state dict. Move to device. Call `model.eval()`.
3. Load the dev split from `final_cdss_dataset.parquet`.
4. Run a full forward pass on the dev split with `torch.no_grad()` to collect raw logits for Y1 and Y2.
5. For Y1: optimise scalar temperature `T_y1` by minimising negative log-likelihood on the full dev split using L-BFGS. The calibrated probability is `sigmoid(logit / T_y1)`.
6. For Y2: apply the survivor mask, then optimise `T_y2` on the survivor subset using L-BFGS.
7. Log pre- and post-calibration ECE for both heads for audit.
8. Write `{'T_y1': float(T_y1), 'T_y2': float(T_y2)}` to `CALIBRATION_PARAMS_PATH` as JSON.

**Config keys used:** `CHECKPOINT_DIR`, `DATASET_PATH`, `CALIBRATION_PARAMS_PATH`.

---

### `inference.py`

Provides the `RewardModelInference` class consumed by the RL agent. Loaded once per RL session, called once per episode step.

**Class: `RewardModelInference`**

The constructor accepts `checkpoint_path` and `calibration_params_path` (or the exported artefact path from `export_model.py`). It loads the frozen model weights, feature index map snapshot, and calibration parameters `T_y1` and `T_y2`. It moves the model to the target device, calls `model.eval()`, and freezes all parameters. No gradient computation ever occurs in this class.

`predict(X)` accepts a float32 tensor of shape `(N, input_dim)`. Under `torch.no_grad()`, it runs a forward pass and returns `(p_mortality, p_readmission)` — two tensors of shape `(N, 1)` containing calibrated probabilities from `sigmoid(logit_y1 / T_y1)` and `sigmoid(logit_y2 / T_y2)` respectively.

`get_feature_index_map()` returns the feature index map snapshot so the RL agent can construct correctly masked input tensors for each episode step without needing access to `final_cdss_dataset.parquet`.

**Config keys used:** None — all parameters come from the checkpoint and calibration files passed to the constructor.

---

### `validate_contract.py`

Standalone CLI tool. Runs only the schema assertions from `load_dataset.py` steps 2–5 without constructing tensors or loading data into memory. Intended to be run before submitting a training job.

**Algorithm:**
1. Load config from `--config` argument.
2. Read `final_cdss_dataset.parquet` column names and dtypes only (schema metadata, no data materialisation).
3. Run all column-presence, dtype, and NaN assertions using column statistics from the Parquet footer.
4. Print a per-assertion pass/fail summary to stdout with the first failing assertion described in full.
5. Exit code 0 if all assertions pass, code 1 if any fail.

**Config keys used:** `DATASET_PATH`.

---

### `export_model.py`

Standalone CLI tool. Produces a self-contained artefact that `inference.py` can load without access to `config/reward_model.yaml`.

**Algorithm:**
1. Load `best_model.pt` from `CHECKPOINT_DIR`.
2. Load `calibration_params.json` from `CALIBRATION_PARAMS_PATH`.
3. Construct an export dict: model state dict (unwrapped from DDP if needed via `unwrap_ddp()`), feature index map snapshot, calibration parameters `T_y1` and `T_y2`, model architecture config (layer widths, dropout), and input dimensionality.
4. Write to `EXPORT_PATH` as a PyTorch `.pt` file.
5. Log the export path and total model parameter count.

**Config keys used:** `CHECKPOINT_DIR`, `CALIBRATION_PARAMS_PATH`, `EXPORT_PATH`.

---

## 6. DDP Training Implementation

---

### 6.1 Process Launch and Initialisation

`torchrun` is the DDP launcher, invoked from `reward_job.sh` with `--nproc_per_node = NUM_GPUS` (default 2). It sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` before spawning each worker process. `train.py` reads these at startup and calls `dist.init_process_group(backend='nccl')`. Each process calls `torch.cuda.set_device(LOCAL_RANK)` to bind to its assigned GPU.

If `NUM_GPUS = 1` or `torch.cuda.device_count() < 2` at runtime, `init_process_group` is skipped. The model is not wrapped in DDP and `DistributedSampler` is replaced by `RandomSampler`. This single-GPU fallback is logged at `WARNING`.

The `nccl` backend is used for all GPU-to-GPU communication. `nccl` is the only supported backend for CUDA DDP — `gloo` is not suitable for large all-reduce operations at the gradient sizes involved here (~1.4 GB per step at full model dimensionality).

---

### 6.2 Data Sharding

Only rank 0 loads the dataset from `final_cdss_dataset.parquet`. Rank 0 calls `load_dataset.run(config)`, which returns three `ParquetDataset` objects — one per split. These are lightweight wrappers holding a PyArrow file handle and row index lists; no feature data is materialised at load time. Non-rank-0 processes do not call `load_dataset` at all — they receive only the broadcast `pos_weight` scalars from rank 0 and wait at a barrier while rank 0 loads.

Training data is sharded across GPUs via `DistributedSampler`. The sampler operates on row indices from the `train_dataset` `ParquetDataset`. For each epoch it assigns a non-overlapping subset of indices to each rank. Each rank's `DataLoader` calls `ParquetDataset.__getitem__` for its assigned indices, which reads the corresponding row groups lazily from disk via the LRU cache and materialises individual batch tensors on demand. With 2 GPUs and `BATCH_SIZE_PER_GPU = 256`, each GPU materialises 256 samples per step and the effective batch size is 512.

Dev evaluation runs on rank 0 only, iterating the full `dev_dataset` without a `DistributedSampler`, ensuring metrics reflect the complete dev set rather than a shard. The dev split is also accessed lazily via `ParquetDataset` — the full dev tensor is never held in RAM simultaneously.

---

### 6.3 Adversarial Masking Under DDP

Adversarial masking requires two forward/backward passes per batch. The first pass must not trigger a DDP all-reduce — only the second pass should synchronise gradients across GPUs.

The first pass is executed inside a `model.no_sync()` context. This suppresses the DDP backward hook so gradients from the first pass accumulate locally on each GPU without synchronisation. `masking.py apply_adversarial_mask()` uses these local gradients to compute per-sample importance scores and identify the worst-case feature slot per sample.

The second pass runs normally, triggering the all-reduce, and the synchronised gradients update model weights via `optimizer.step()`.

Each GPU independently selects its adversarial mask based on its own local gradient from its own data shard during the first pass. Masks are not synchronised across GPUs. This is intentional — it increases the diversity of adversarial patterns seen per effective batch and is consistent with the curriculum's robustness goal. The design treats each GPU's independent mask selection as a feature, not a deficiency.

---

### 6.4 Rank 0 I/O Discipline

All file writes are gated behind a rank-0 check. Non-rank-0 processes never write to disk during training.

Checkpoints are written after rank 0 evaluates early stopping on the dev split. Before writing, rank 0 calls `dist.barrier()` to ensure all ranks have completed the current epoch's passes. After writing, rank 0 calls `dist.barrier()` again so all ranks resume the next epoch from a consistent state. The early stopping decision is broadcast from rank 0 to all ranks as a scalar boolean tensor. All ranks break the loop simultaneously on receiving a stop signal.

`training_metrics.parquet` is appended per epoch by rank 0 only, using atomic writes (write to temp file, rename) to prevent partial rows if the job is preempted mid-write.

---

## 7. Adversarial Masking Gradient Computation

The gradient computation step in `train.py` feeds `masking.py apply_adversarial_mask()` with `∂L/∂X`.

**The problem:** Identifying the worst-case feature slot per sample requires the gradient of the loss with respect to the input tensor. PyTorch does not compute this by default because `X` (the data tensor loaded from the dataset) is not a leaf tensor with `requires_grad=True`.

**The mechanism:**

Before the first forward pass in an adversarial batch, `train.py` clones the batch input tensor and calls `requires_grad_(True)` on the clone. This ensures the original data tensor in the `DataLoader` is not modified. The first forward pass runs on this clone. The resulting loss is computed via `loss.compute_loss()`. `loss.backward()` is called inside `model.no_sync()` to suppress DDP all-reduce. After the backward pass, the clone's `.grad` attribute holds `∂L/∂X` of shape `(batch_size, D)`.

**Importance scoring per feature slot:**

For each sample and each of the 56 feature slots, the importance score is the L2 norm of the gradient over the slot's index range: `importance[i, f] = ||grad[i, start_f : end_f]||_2`. This is computed by iterating the feature index map and slicing `grad` accordingly. The slot with the highest importance score per sample is passed to `apply_adversarial_mask()`, which zeros the corresponding index range.

**Why L2 norm rather than raw gradient maximum:** The L2 norm aggregates all dimensions within a slot into a single scalar, making slots of different sizes comparable — `demographic_vec` has 8 dimensions while all embedding slots have 768. A raw per-dimension maximum would be dominated by the single largest gradient element regardless of which slot it belongs to, which could bias mask selection toward embedding slots even when their overall contribution is small.

**Gradient cleanup:** After the first pass, the clone tensor is discarded. `optimizer.zero_grad()` is called before the second forward pass to clear any accumulated gradients. The second pass uses the original (non-clone) batch input tensor, which has `requires_grad = False`.

---

## 8. Checkpoint and Resume

**Checkpoint contents:** Each checkpoint file at `CHECKPOINT_DIR/epoch_<N>.pt` contains:

- Model state dict, unwrapped from DDP via `unwrap_ddp()`
- Optimiser state dict
- LR scheduler state dict
- Current epoch number
- Current masking schedule state (probability ratios at this epoch)
- Best dev loss seen so far
- Feature index map snapshot
- Full config snapshot serialised from the Pydantic model

The config snapshot inside the checkpoint is authoritative for architecture reconstruction. If `config/reward_model.yaml` is modified between a run and a resume, the resumed run uses the checkpoint's config to ensure no architecture mismatch. The current YAML is still loaded for non-architecture settings (paths, logging) but architecture keys are ignored in favour of the checkpoint snapshot.

**Best model tracking:** `best_model.pt` is overwritten (via temp-file rename for atomicity) whenever dev loss improves. It always reflects the best epoch seen so far, not the most recent epoch.

**Resume mechanism:**

On `--resume`, the latest checkpoint in `CHECKPOINT_DIR` is identified by epoch number (not filesystem modification time). It is loaded on rank 0. Model, optimiser, scheduler, epoch, and masking schedule are restored. The feature index map snapshot from the checkpoint is compared against the freshly derived map from the current `final_cdss_dataset.parquet`. If they differ, training refuses to resume and raises a descriptive error — this protects against resuming after an upstream dataset change that would make the checkpoint inconsistent with the current data.

After loading, rank 0 broadcasts the full model state dict to non-rank-0 processes via `dist.broadcast_object_list()`. A `dist.barrier()` ensures all ranks have received the state before training proceeds.

**Checkpoint retention:** Only the `CHECKPOINT_KEEP_N` most recent epoch checkpoints are retained (default 3). Older checkpoints are deleted by rank 0 after each new checkpoint is written. `best_model.pt` is always retained regardless of this limit.

---

## 9. Configuration Reference

All keys defined in `config/reward_model.yaml`. Loaded and validated by `load_and_validate_config()`. ★ marks the primary tuning knob for each performance or capacity constraint.

### Model architecture

| Key | Default | Used by | Description |
|-----|---------|---------|-------------|
| `INPUT_DIM` | (derived) | `load_dataset.py`, `model.py` | Expected input dimensionality; validated against feature index map at startup |
| `LAYER_WIDTHS` | `[8192, 2048, 512, 128]` | `model.py` | Hidden layer output sizes; length determines depth |
| `★ DROPOUT_RATE` | `0.3` | `model.py` | Dropout probability per hidden layer |
| `ACTIVATION` | `relu` | `model.py` | Activation function; supports `relu`, `leaky_relu` |

### Training

| Key | Default | Used by | Description |
|-----|---------|---------|-------------|
| `MAX_EPOCHS` | `100` | `train.py` | Maximum training epochs |
| `★ BATCH_SIZE_PER_GPU` | `256` | `train.py` | Samples per GPU per step; effective batch = this × `NUM_GPUS` |
| `★ NUM_GPUS` | `2` | `train.py`, `reward_job.sh` | GPUs for DDP; `1` disables DDP |
| `LEARNING_RATE` | `1e-4` | `train.py` | Initial AdamW learning rate |
| `WEIGHT_DECAY` | `1e-5` | `train.py` | AdamW weight decay |
| `ADAM_BETA1` | `0.9` | `train.py` | AdamW first moment decay |
| `ADAM_BETA2` | `0.999` | `train.py` | AdamW second moment decay |
| `LR_WARMUP_EPOCHS` | `5` | `train.py` | Linear warmup before cosine decay |
| `LR_MIN` | `1e-6` | `train.py` | Minimum LR at end of cosine decay |
| `EARLY_STOPPING_PATIENCE` | `10` | `train.py` | Epochs without dev loss improvement before stopping |
| `CHECKPOINT_KEEP_N` | `3` | `train.py` | Most recent epoch checkpoints to retain on disk |

### Loss

| Key | Default | Used by | Description |
|-----|---------|---------|-------------|
| `LOSS_WEIGHT_Y1` | `1.0` | `loss.py` | Head weight `w1` |
| `LOSS_WEIGHT_Y2` | `1.0` | `loss.py` | Head weight `w2` |
| `POS_WEIGHT_Y1` | (computed) | `loss.py` | Positive class weight for Y1; computed from training split if omitted |
| `POS_WEIGHT_Y2` | (computed) | `loss.py` | Positive class weight for Y2; computed from survivors in training split if omitted |

### Masking curriculum

| Key | Default | Used by | Description |
|-----|---------|---------|-------------|
| `MASKING_START_RATIOS` | `{random: 1.0, adversarial: 0.0, none: 0.0}` | `masking.py` | Mode probabilities at epoch 0 |
| `MASKING_END_RATIOS` | `{random: 0.33, adversarial: 0.33, none: 0.34}` | `masking.py` | Mode probabilities at final epoch |
| `★ MASKING_TRANSITION_MIDPOINT_EPOCH` | `50` | `masking.py` | Epoch at sigmoid crossover inflection point |
| `MASKING_TRANSITION_SHAPE` | `sigmoid` | `masking.py` | Crossover curve shape |
| `MASKING_K` | `1` | `masking.py` | Feature slots zeroed per sample in random mode |

### Dataset loading

| Key | Default | Used by | Description |
|-----|---------|---------|-------------|
| `DATASET_PATH` | (required) | `load_dataset.py`, `calibrate.py`, `validate_contract.py` | Absolute path to `final_cdss_dataset.parquet` |
| `★ DATASET_ROW_GROUP_CACHE_SIZE` | `2` | `parquet_dataset.ParquetDataset` | Number of decompressed Parquet row groups held in the LRU cache per `ParquetDataset` instance; higher values increase RAM usage but reduce re-reads when the DataLoader accesses rows from the same row group across consecutive batches |

### Paths

| Key | Default | Used by | Description |
|-----|---------|---------|-------------|
| `DATASET_PATH` | (required) | `load_dataset.py`, `calibrate.py`, `validate_contract.py` | Absolute path to `final_cdss_dataset.parquet` |
| `CHECKPOINT_DIR` | `data/reward_model/checkpoints` | `train.py`, `calibrate.py`, `export_model.py` | Checkpoint directory |
| `METRICS_PATH` | `data/reward_model/training_metrics.parquet` | `train.py` | Per-epoch metrics output |
| `CALIBRATION_PARAMS_PATH` | `data/reward_model/calibration_params.json` | `calibrate.py`, `inference.py` | Temperature scaling output |
| `EXPORT_PATH` | `data/reward_model/frozen_model.pt` | `export_model.py` | Self-contained frozen model artefact |

---

## 10. Memory Requirements

Variables: **A** = total admissions (~546k). **B** = `BATCH_SIZE_PER_GPU`. **D** = input dimensionality (42,248 at full BERT embeddings). **P** = total model parameters (~364M at default widths). **C** = `DATASET_ROW_GROUP_CACHE_SIZE`. **R** = Parquet row group size (rows per group, typically ~400 at default PyArrow settings).

### Dataset RAM (rank 0 only)

The dataset is never fully materialised into a float32 tensor. `ParquetDataset` reads lazily from disk via an LRU row group cache. At any given time, the dataset-related RAM on rank 0 is bounded by the cache contents plus DataLoader prefetch buffers.

| Component | Formula | Estimated size |
|-----------|---------|----------------|
| LRU row group cache | C × R × D × 4 bytes | C=2, R=400, D=42,248 → **~135 MB** |
| DataLoader prefetch buffers | `num_workers` × B × D × 4 bytes | 4 workers, B=256, D=42,248 → **~170 MB** |
| PyArrow file handle + metadata | Fixed | ~50 MB |
| **Total dataset RAM** | | **~355 MB** |

This is well within the 64 GB SLURM allocation alongside the model, optimizer state, and OS overhead. No eager loading, no memmap file, and no PCA reduction is required.

If `DATASET_ROW_GROUP_CACHE_SIZE` is increased to improve I/O performance (reducing re-reads when shuffled batches cluster within a row group), RAM scales proportionally: each additional cached row group adds approximately C × R × D × 4 bytes (~67 MB per additional row group at default settings).

### Per-GPU memory during training

Each GPU holds a full model copy under DDP — memory savings come from halving per-GPU effective batch size, not from splitting model weights.

| Component | Formula | A100 (D=42,248) | L4 (D=42,248) |
|-----------|---------|-----------------|---------------|
| Model weights | P × 4 B | ~1.39 GB | ~1.39 GB |
| AdamW state | P × 8 B | ~2.78 GB | ~2.78 GB |
| Gradients | P × 4 B | ~1.39 GB | ~1.39 GB |
| DDP all-reduce buffer | ~P × 4 B | ~1.39 GB | ~1.39 GB |
| Activations (fwd+bkwd) | ~B × layers × 4 B | ~0.5 GB | ~0.5 GB |
| Input batch | B × D × 4 B | ~42 MB | ~42 MB |
| CUDA context + fragmentation | Fixed | ~1.5 GB | ~1.5 GB |
| **Total estimated at B=256** | | **~9.0 GB** | **~9.0 GB** |
| **Total estimated at B=512** | | **~9.5 GB** | **~9.5 GB** |

At B=256 or B=512, training fits comfortably within the A100's 40 GB and within the L4's 24 GB. GPU VRAM is not the binding constraint for this module. The binding constraint is dataset RAM on rank 0, which is resolved by `ParquetDataset` lazy loading as described above.

Adversarial masking adds one extra clone tensor of size B × D × 4 bytes per adversarial batch (~42 MB at B=256, D=42,248). This is negligible.

### Calibration and inference

| Phase | GPU VRAM | Notes |
|-------|---------|-------|
| `calibrate.py` | ~1.4 GB | Weights only; no optimiser state; no DDP buffer |
| `inference.py` | ~1.4 GB | Frozen model; `torch.no_grad()` eliminates gradient tensors |

### Recommended allocation summary

| Job | RAM | GPU config | Notes |
|-----|-----|-----------|-------|
| `reward_job.sh` (A100 default) | 64 GB | 2 × A100 40 GB | Dataset loaded lazily via `ParquetDataset`; ~355 MB dataset RAM |
| `reward_job.sh` (L4 fallback) | 64 GB | 2 × L4 24 GB | Same dataset RAM; GPU VRAM fits at B=256 |
| `calibrate_job.sh` | 32 GB | 1 × any GPU | Model fits on any available GPU |
| `validate_contract.py` | 16 GB | None | CPU only; reads Parquet schema metadata only |
| `inference.py` (RL agent) | 16 GB | 1 × any GPU | Frozen forward pass; minimal footprint |

---

> See `reward_model_architecture.md` for system context, design rationale, pipeline overview, and the upstream pipeline that produces `final_cdss_dataset.parquet`.
