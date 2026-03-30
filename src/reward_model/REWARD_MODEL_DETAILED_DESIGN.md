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
   - [data_loader.py](#data_loaderpy)
   - [checkpoint_manager.py](#checkpoint_managerpy)
   - [mimic4_data_loader.py](#mimic4_data_loaderpy)
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
11. [MIMIC-IV Reference Configuration](#11-mimic-iv-reference-configuration)

---

## 1. Primary Identifier

The reward model operates at the admission level. `hadm_id` (int64) is the primary key throughout — one row in `final_cdss_dataset.parquet` corresponds to one hospital admission. `subject_id` is present in the dataset but is not used by any reward model module; it is carried through for potential downstream use by the RL agent.

The reward model does not perform any identifier linkage. All identifier resolution and null-`hadm_id` handling is completed upstream by `HRS/src/preprocessing`. By the time `final_cdss_dataset.parquet` reaches this module, every row has a valid `hadm_id`.

---

## 2. Module Decomposition

### Module boundaries

Each Python file corresponds to one pipeline concern. Class definitions live in class-only modules (`reward_model_config.py`, `parquet_dataset.py`, `row_group_block_sampler.py`, `dataset_bundle.py`, `schema_error.py`) and are re-exported by `reward_model_utils.py` for backward compatibility. Inter-module state exchange happens via tensors passed as function arguments within a single process, or via checkpoint files on disk between separate SLURM jobs. No module reads `final_cdss_dataset.parquet` except `mimic4_data_loader.py` (via `Mimic4DataLoader`, a subclass of the generic `DataLoader` base).

### Class inventory (one class per file)

| File | Class | Role |
|------|-------|------|
| `data_loader.py` | `DataLoader` | Abstract base for dataset loaders. |
| `mimic4_data_loader.py` | `Mimic4DataLoader` | MIMIC-IV implementation of `DataLoader`. |
| `parquet_dataset.py` | `ParquetDataset` | `torch.utils.data.Dataset` backed by a Parquet file and feature index map. |
| `row_group_block_sampler.py` | `RowGroupBlockSampler` | Sampler that shards row groups round-robin across DDP ranks. |
| `dataset_bundle.py` | `DatasetBundle` | Named tuple bundling datasets, feature index map, and pos-weights. |
| `schema_error.py` | `SchemaError` | Custom exception for schema validation failures. |
| `model.py` | `RewardModel` | Feedforward MLP with T output heads (one per classification target). |
| `masking.py` | `MaskingSchedule` | Masking curriculum with random/adversarial/none modes. |
| `reward_model_config.py` | `RewardModelConfig` | Pydantic config model. |
| `checkpoint_manager.py` | `CheckpointManager` | Manages saving/loading/pruning checkpoints and validates feature index maps on resume. |
| `inference.py` | `RewardModelInference` | Frozen inference wrapper with calibration parameters. |
| `train.py` | `TrainManager` | Encapsulates training state, masking curriculum, epoch loop, optimizer/scheduler, dev eval, checkpointing. |

### Class vs plain script

| Module | Pattern | Reason |
|--------|---------|--------|
| `data_loader.py` | **Class** (`DataLoader`) | Generic base (no dataset-specific logic); single instantiation per run |
| `mimic4_data_loader.py` | **Class** (`Mimic4DataLoader`) | MIMIC-IV implementation; validates schema, constructs `DatasetBundle`; single instantiation per run |
| `model.py` | **Class** (`RewardModel`) | Stateful network; instantiated once, called many times via `forward()`; must be wrappable by DDP |
| `masking.py` | **Class** (`MaskingSchedule`) | Maintains curriculum state across the training loop; epoch advancement is a stateful operation |
| `loss.py` | Plain functions | Stateless transformations; `compute_loss(logits_list, labels_list, pos_weights, loss_weights)` for T targets |
| `train.py` | **Class** (`TrainManager`) | Holds training state (datasets, model, optimizer, scheduler, masking schedule), epoch loop, checkpointing |
| `train_main.py` | Plain script, DDP entry point | CLI + runtime init; instantiates `TrainManager` and delegates training |
| `calibrate.py` | Plain script, `run(config)` | Single optimisation pass; no persistent state |
| `inference.py` | **Class** (`RewardModelInference`) | Loaded once, called repeatedly per RL step; holds frozen weights and calibration params in memory |
| `validate_contract.py` | Plain script, CLI tool | One-shot assertion run; exits with code 0 (pass) or 1 (fail) |
| `export_model.py` | Plain script, CLI tool | One-shot serialisation; no shared state |

### Encapsulation rules

All helper functions are module-private (prefixed `_`). The public interface of each pipeline module is its `run(config)` function or class constructor. `train_main.py` is the torchrun entry point and delegates to the `TrainManager` class in `train.py`. Config is always passed as a validated Pydantic model object, not a raw dict.

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

The process group is initialised with the `nccl` backend at the start of `train_main.py` using environment variables set by `torchrun` (`MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, `LOCAL_RANK`). If `NUM_GPUS = 1` or only one GPU is available at runtime, training proceeds in single-process mode without initialising a process group — no DDP wrapping, no `DistributedSampler`. This fallback is detected automatically and logged at `WARNING`.

**Rank 0 is the designated I/O rank.** Only rank 0 writes checkpoints, `training_metrics.parquet`, and `calibration_params.json`. Only rank 0 computes `pos_weight` values (inside `TrainManager.__init__`) and broadcasts them to other ranks. All ranks participate in forward passes, backward passes, and all-reduce operations. Barrier synchronisation (`dist.barrier()`) is used at three points: after checkpoint load, after `pos_weight` broadcast, and before the final checkpoint write at training completion.

---

### 3.4 Shared Utilities (`reward_model_utils.py`)

| Function / Class | Purpose |
|------------------|---------|
| `load_and_validate_config(path)` | Load `reward_model.yaml`, validate with Pydantic, return config object (defined in `reward_model_config.py`, re-exported) |
| `get_device(local_rank)` | Return `torch.device('cuda', local_rank)` or `cpu` with CUDA availability check |
| `sigmoid_crossover(epoch, total_epochs, start_ratios, end_ratios, midpoint)` | Compute current masking mode probabilities for a given epoch (implemented in `masking.py`) |
| `unwrap_ddp(model)` | Return `model.module` if wrapped in DDP, else `model` directly (implemented in `train.py`) |
| `broadcast_tensor(tensor, src_rank)` | Broadcast a scalar tensor from `src_rank` to all ranks via process group (implemented in `train.py`) |
| **`ParquetDataset(Dataset)`** | Class-only module `parquet_dataset.py`. Lazy row-group reads from `final_cdss_dataset.parquet`; constructor accepts open PyArrow file handle, split row indices, the feature index map, and `DATASET_ROW_GROUP_CACHE_SIZE`. Holds an LRU cache of at most `DATASET_ROW_GROUP_CACHE_SIZE` decompressed row groups in memory at any time. `__getitem__(i)` resolves the row group containing row `i`, reads it from the LRU cache or from disk, slices the requested row, concatenates feature columns in index map order into a float32 tensor, and returns `(X, y1, y2)`. `__len__` returns the number of rows in the split. Re-exported by `reward_model_utils.py`. |
| **`RowGroupBlockSampler(Sampler)`** | Class-only module `row_group_block_sampler.py`. Row-group-aware sampler to preserve Parquet row-group locality and partition row groups round-robin across DDP ranks. Re-exported by `reward_model_utils.py`. |
| **`DatasetBundle(NamedTuple)`** | Class-only module `dataset_bundle.py`. Bundles `train/dev/test` `ParquetDataset` instances plus metadata. Re-exported by `reward_model_utils.py`. |

A helper function belongs in `reward_model_utils.py` if and only if it is used by two or more modules and has no module-specific state. Shared classes sit in class-only modules and are re-exported by `reward_model_utils.py`. Functions used only once remain private to their module with a `_` prefix.

---

## 4. Feature Index Map

The feature index map defines the start and end index within the flat D-dimensional input tensor for each feature slot. It is derived at load time by the dataset-specific data loader from the ordered list of feature columns in the dataset file. The derivation algorithm iterates the ordered column list, skips metadata columns (e.g. primary key, split column) and label columns, and for each remaining feature column assigns a start index equal to the running offset and an end index equal to start plus the column's declared dimension. The result is a dict mapping column name to a `(start, end)` tuple.

The number and breakdown of feature slots, their column names, and their declared dimensions are dataset-specific. For the MIMIC-IV deployment (56 slots, 1 structured vector at 8 dims, 55 BERT embeddings at 768 dims each, D=42,248) see `mimic4_feature_set.md` and Section 11.2 of this document.

The dataset-specific data loader validates the expected slot count and per-group breakdown and raises `SchemaError` referencing the upstream data model document if any count does not match. This guards against silent miscounting if the upstream dataset schema changes.

The map is constructed once by the data loader, stored in the returned `DatasetBundle`, and passed explicitly to `masking.py` and `train.py`. It is also saved as a snapshot inside every checkpoint file so that `inference.py` can reconstruct the same boundaries when loading a frozen model, even if the upstream dataset schema were to change between runs.

The map is never written to a standalone config or YAML file — a separate file would duplicate the upstream data model document and create a consistency risk.

---

## 5. Module Implementation

---

### `data_loader.py`

Defines the `DataLoader` abstract base class that establishes the contract for all dataset-specific loaders. Provides a `load(config)` template method that orchestrates the open → validate → build index map → split → bundle sequence via abstract hooks. Concrete subclasses implement the dataset-specific hooks; the template method ensures consistent ordering and error handling.

Public interface: `load(config) -> DatasetBundle`. All other methods are abstract or private.

---

### `checkpoint_manager.py`

Centralises all checkpoint read/write operations. `CheckpointManager` is the sole class permitted to write `epoch_<N>.pt` and `best_model.pt`, and the sole class that performs schema safety checks before a resume proceeds.

**Key responsibilities:**
- `save(epoch, model, optimizer, scheduler, masking_state, feature_index_map, config, dev_loss)` — writes the checkpoint dict to `CHECKPOINT_DIR/epoch_<N>.pt` and atomically updates `best_model.pt` if dev_loss improved. Prunes old epoch checkpoints keeping only `CHECKPOINT_KEEP_N` most recent.
- `load_latest() -> dict` — identifies the most recent checkpoint by epoch number (not mtime) and returns the checkpoint dict.
- `validate_feature_index_map(old_map, new_map)` — compares the feature index map stored in the checkpoint against the freshly derived map from `Mimic4DataLoader`. If keys or index ranges differ, raises `SchemaError` with a descriptive message to abort the resume and prevent training with shifted feature boundaries after an upstream dataset change.

The config snapshot inside the checkpoint is authoritative for architecture reconstruction — see Section 8.

---

### `mimic4_data_loader.py`

Houses the MIMIC-IV `Mimic4DataLoader` implementation built on the generic `DataLoader` template. `DataLoader.load()` orchestrates open/validate/read/bundle steps via template hooks; dataset-specific logic lives in subclasses. The dataset is never fully materialised into a float32 tensor — batches are read lazily from disk by `ParquetDataset.__getitem__` at training time.

**Algorithm (Mimic4DataLoader implementation):**
1. Read `final_cdss_dataset.parquet` from `DATASET_PATH` using `pyarrow`. Do not use `fastparquet` — the fastparquet/pyarrow serialisation incompatibility observed in the preprocessing pipeline applies here.
2. Validate column presence and order via `Mimic4DataLoader._validate_schema()`; assert all 61 expected columns are present in canonical order from `PREPROCESSING_DATA_MODEL.md` Section 3.12.
3. Validate labels: `y1_mortality` dtype int8/float32 and non-null; `y2_readmission` dtype float32 with NaN for deceased rows and non-null for survivors (checked inside `_validate_labels`).
4. Build the feature index map via `Mimic4DataLoader._build_feature_index_map()`. This also validates the 55-embedding count and per-group breakdown — see Section 4.
5. Read the `split` column only (lightweight — metadata column) to determine row indices for each split. Produce three lists of row indices: `train_indices`, `dev_indices`, `test_indices`.
6. Compute `pos_weight_y1` and `pos_weight_y2` from the training split rows via `Mimic4DataLoader._compute_pos_weights()` unless overridden by `POS_WEIGHT_Y1`/`POS_WEIGHT_Y2` in config.
7. Instantiate three `ParquetDataset` objects (class-only module `parquet_dataset.py`, re-exported by `reward_model_utils.py`) — one per split — passing the open PyArrow file handle, the split's row index list, the feature index map, and `DATASET_ROW_GROUP_CACHE_SIZE`. Each `ParquetDataset` holds only the file handle and index metadata in memory. No feature data is read at this step.
8. Return a `DatasetBundle` named tuple (class-only module `dataset_bundle.py`, re-exported by `reward_model_utils.py`): `train_dataset`, `dev_dataset`, `test_dataset` (each a `ParquetDataset`), `feature_index_map`, `pos_weight_y1`, `pos_weight_y2`.

**Config keys used:** `DATASET_PATH`, `FEATURES_DIM` (assertion only — must equal D as derived from the column schema), `DATASET_ROW_GROUP_CACHE_SIZE`.

**Memory note:** The dataset is never fully resident in RAM. At any given time, only the LRU-cached row groups are held in memory — at most `DATASET_ROW_GROUP_CACHE_SIZE × row_group_size × D × 4` bytes. At the default cache size of 2, a typical row group size of ~400 rows, and D=42,248, this is approximately **135 MB**. Combined with DataLoader prefetch workers (each holding one batch buffer of ~42 MB), total dataset-related RAM is well under 1 GB — comfortably within the 64 GB SLURM allocation alongside the model, optimizer state, and OS overhead.

---

### `model.py`

Defines the `RewardModel` class — a feedforward MLP with a gradual funnel and T independent sigmoid output heads, one per classification target.

**Class: `RewardModel(nn.Module)`**

The constructor accepts `input_dim` (derived from the feature index map at runtime — never hardcoded), `layer_widths` (list of hidden layer output sizes), `dropout_rates` (list of per-layer dropout probabilities — one value per hidden layer; a single float is broadcast to all layers for backward compatibility), `num_targets` (number of output heads T, default 2), and `activation`. It builds a sequence of `Linear` → `BatchNorm1d` → `Activation` → `Dropout` blocks for each consecutive pair in `[input_dim] + layer_widths`, applying `dropout_rates[i]` at layer i. After the final hidden block, T independent `Linear(layer_widths[-1], 1)` heads produce raw logits, one per target.

`forward(x)` returns a tuple of T tensors `(logits_y1, ..., logits_yT)`, each of shape `(batch_size, 1)`. **Sigmoid is not applied inside `forward`** — raw logits are returned so that `BCEWithLogitsLoss` can be used for numerical stability in training. Sigmoid is applied only at inference time in `inference.py`.

The model is not DDP-aware — DDP wrapping is performed by `TrainManager` (`train.py`) during construction. This keeps `model.py` independently testable without a process group.

**Config keys used:** `LAYER_WIDTHS`, `DROPOUT_RATES`, `ACTIVATION`, `NUM_TARGETS`.

---

### `masking.py`

Implements the `MaskingSchedule` class, which maintains the curriculum state and applies the correct masking mode to each mini-batch.

**Class: `MaskingSchedule`**

The constructor accepts the `feature_index_map`, `num_always_visible` (integer — the first `num_always_visible` slots in the index map are never candidates for masking), curriculum schedule parameters (`start_ratios`, `end_ratios`, `transition_shape`, `transition_midpoint_epoch`, `total_epochs`), and the four k-range fraction parameters (`random_k_min_fraction`, `random_k_max_fraction`, `adversarial_k_min_fraction`, `adversarial_k_max_fraction`). The constructor partitions the feature index map into `_always_visible_slots` (first `num_always_visible` entries) and `_maskable_slots` (all remaining entries). M denotes the count of maskable slots.

`get_mode_probabilities(epoch)` delegates to the module-local `sigmoid_crossover()` helper and returns the current `(p_random, p_adversarial, p_none)` tuple for the given epoch.

`sample_mode(epoch)` draws a masking mode string — `'random'`, `'adversarial'`, or `'none'` — according to the current probabilities, with explicit normalisation to guard against floating-point drift.

`_sample_k(min_fraction, max_fraction)` draws an integer k independently per sample from `Uniform(floor(min_fraction × M), ceil(max_fraction × M))`. After rounding, the lower bound is clamped to at least 1 and the upper bound to at most M−1. If lower still exceeds upper after clamping (possible only when M is very small and the fraction range is tight), lower is set equal to upper as a final safety guard.

`apply_random_mask(X)` draws k independently per sample via `_sample_k(random_k_min_fraction, random_k_max_fraction)`, selects k maskable slots uniformly at random per sample without replacement, then zeros the corresponding index ranges to 0.0. Returns a cloned masked tensor — the original is not modified.

`apply_adversarial_mask(X, grad_X)` draws k independently per sample via `_sample_k(adversarial_k_min_fraction, adversarial_k_max_fraction)`. For each sample it computes the RMS gradient magnitude per maskable slot: `importance[i, f] = ||grad_X[i, start_f:end_f]||_2 / sqrt(slot_dim_f)`. Dividing by the square root of slot dimension normalises for slot size, making `demographic_vec` (8 dims) and embedding slots (768 dims) directly comparable. Sorts all maskable slots by importance score descending and zeros the top k slots per sample. Returns a cloned masked tensor. The gradient computation is the responsibility of `TrainManager` — see Section 7.

`apply_no_mask(X)` returns `X` unchanged (no clone needed).

**Config keys used:** `MASKING_START_RATIOS`, `MASKING_END_RATIOS`, `MASKING_TRANSITION_SHAPE`, `MASKING_TRANSITION_MIDPOINT_EPOCH`, `MASKING_RANDOM_K_MIN_FRACTION`, `MASKING_RANDOM_K_MAX_FRACTION`, `MASKING_ADVERSARIAL_K_MIN_FRACTION`, `MASKING_ADVERSARIAL_K_MAX_FRACTION`, `NUM_ALWAYS_VISIBLE_FEATURES`.

---

### `loss.py`

Plain module-level functions. No class, no state.

`compute_loss(logits_list, labels_list, pos_weights, loss_weights)` computes the total weighted loss over T targets. `logits_list` and `labels_list` are lists of T tensors, one per target. `pos_weights` and `loss_weights` are lists of T floats. `loss_weights` are normalised to sum to 1.0 — normalisation is applied by `RewardModelConfig` at config load time so that callers always receive weights that already sum to 1.0.

For each target i: construct the valid-sample mask `~torch.isnan(labels_list[i])`. If the mask is all-false (no valid samples for this target in the batch), set `L_i = torch.tensor(0.0)` on the correct device to prevent NaN propagation. Otherwise apply `BCEWithLogitsLoss` with `pos_weight = pos_weights[i]` over the masked subset. Accumulate `L = Σ loss_weights[i] * L_i`. Returns the scalar total loss plus the per-target component losses separately for epoch logging.

`compute_metrics(logits_list, labels_list, masked=False)` applies sigmoid to each logits tensor, constructs per-target valid-sample masks, and computes AUROC, AUPRC, and ECE for each target using scikit-learn. Returns a dict keyed by target index: `{0: {'auroc': ..., 'auprc': ..., 'ece': ...}, 1: {...}, ...}`. The `masked` flag is included in the returned dict as a tag so callers can distinguish masked-input metrics from unmasked-input metrics in the epoch log. ECE uses equal-mass binning (adaptive bin edges via `np.percentile`) rather than equal-width binning, to produce reliable calibration estimates under class imbalance. Called on the dev split by rank 0 at epoch end using unmasked inputs. Also called on training batches for diagnostic AUROC tracking — see Section 6.4.

**Config keys used:** `LOSS_WEIGHTS` (list of T normalised weights), `POS_WEIGHTS` (list of T positive class weights).

---

### `train.py`

Defines the `TrainManager` class, which owns all training state and encapsulates the epoch loop.

- `__init__(config, rank, local_rank, world_size, is_ddp, device)` loads the dataset via `_load_datasets_and_weights()` (rank 0 reads from disk; broadcast to others), stores the feature index map and pos-weights, builds `RewardModel` (DDP-wrapped when applicable), and constructs the optimizer. It also initialises checkpoint/metrics paths and helper handles.
- `setup_training_state(ckpt_state, start_epoch)` builds the `MaskingSchedule`, `DataLoader` (with `RowGroupBlockSampler`), and LR scheduler (warmup + cosine), then restores model/optimizer/scheduler state when resuming.
- `_run_train_batch(X, labels, epoch)` executes the masking-aware mini-batch (random/adversarial/none). Adversarial mode clones `X`, sets `requires_grad_(True)`, does the first backward under `no_sync()` (when DDP) to obtain gradients, reapplies adversarial masks, then performs the second backward/step.
- `train_epochs(start_epoch, best_dev_loss)` drives the epoch loop: sets sampler epoch, iterates batches (calling `_run_train_batch` and stepping the scheduler), runs dev evaluation on rank 0, logs metrics, performs checkpoint/early-stop decisions, and broadcasts stop signals across ranks.

**Config keys used:** `LEARNING_RATE`, `WEIGHT_DECAY`, `LR_WARMUP_EPOCHS`, `LR_MIN`, `ADAM_BETA1`, `ADAM_BETA2`, `MAX_EPOCHS`, `BATCH_SIZE_PER_GPU`, `NUM_GPUS`, `EARLY_STOPPING_PATIENCE`, `CHECKPOINT_DIR`, `CHECKPOINT_KEEP_N`, `METRICS_PATH`, all masking keys, all loss keys, all model architecture keys.

---

### `train_main.py`

Plain script and DDP entry point launched by `torchrun`. It wires CLI/runtime concerns and delegates training to `TrainManager`.

**Algorithm:**
1. Parse CLI arguments: `--config`, `--resume`.
2. Load and validate config via `load_and_validate_config()`.
3. Call `_init_runtime()` to set up ranks, process group (`nccl`), and device selection (single-process fallback when only one GPU is available).
4. Instantiate `TrainManager(config, rank, local_rank, world_size, is_ddp, device)`.
5. If `--resume`, locate latest checkpoint via `CheckpointManager.find_latest()`, load on rank 0, broadcast to all ranks, and recover `start_epoch`/`best_dev_loss`.
6. Call `manager.setup_training_state(ckpt_state, start_epoch)`.
7. Execute `manager.train_epochs(start_epoch, best_dev_loss)`.
8. Destroy the process group if DDP is active.

---

### `calibrate.py`

Applies temperature scaling to the best trained model. Plain script with `run(config)` entry point. Single GPU — no DDP.

**Algorithm:**
1. Load `best_model.pt` from `CHECKPOINT_DIR`. Extract model state dict and the config snapshot from the checkpoint (not the current `config/reward_model.yaml` — the checkpoint config is authoritative for architecture).
2. Instantiate `RewardModel` from the checkpoint config snapshot. Load state dict. Move to device. Call `model.eval()`.
3. Load the dev split from `final_cdss_dataset.parquet`.
4. Run a full forward pass on the dev split with `torch.no_grad()` to collect raw logits for all T targets.
5. For each target i: construct the valid-sample mask `~torch.isnan(labels_i)`. Optimise scalar temperature `T_i` in log-space — define `log_T = torch.nn.Parameter(torch.zeros(1))` and recover `T = exp(log_T)` inside the closure. This guarantees T > 0 throughout optimisation, avoiding erratic gradients at the clamp boundary that occur with direct optimisation of T. Minimise negative log-likelihood on the valid-sample subset using L-BFGS. Return `max(exp(log_T).item(), 1e-8)` as the fitted temperature for target i.
6. Log pre- and post-calibration ECE for all T heads for audit.
7. Write `{'T_0': float, 'T_1': float, ..., 'T_{T-1}': float}` to `CALIBRATION_PARAMS_PATH` as JSON.
7. Log pre- and post-calibration ECE for both heads for audit.
8. Log the calibration parameters and export path.

**Config keys used:** `CHECKPOINT_DIR`, `DATASET_PATH`, `CALIBRATION_PARAMS_PATH`.

---

### `inference.py`

Provides the `RewardModelInference` class consumed by the RL agent. Loaded once per RL session, called once per episode step.

**Class: `RewardModelInference`**

The constructor accepts `checkpoint_path` and `calibration_params_path` (or the exported artefact path from `export_model.py`). It loads the frozen model weights, feature index map snapshot, and T calibration temperature values (one per target). It moves the model to the target device, calls `model.eval()`, and freezes all parameters. No gradient computation ever occurs in this class.

`predict(X)` accepts a float32 tensor of shape `(N, input_dim)`. Under `torch.no_grad()`, it runs a forward pass and returns a tuple of T tensors, each of shape `(N, 1)`, containing calibrated probabilities from `sigmoid(logit_i / T_i)` for each target i. Temperature values are clamped to a minimum of 1e-8 at load time to prevent division by zero.

`get_feature_index_map()` returns the feature index map snapshot so the RL agent can construct correctly masked input tensors for each episode step without needing access to `final_cdss_dataset.parquet`.

**Config keys used:** None — all parameters come from the checkpoint and calibration files passed to the constructor.

---

### `validate_contract.py`

Standalone CLI tool. Runs only the schema assertions from `Mimic4DataLoader.load()` steps 2–5 without constructing tensors or loading data into memory. Intended to be run before submitting a training job.

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
3. Construct an export dict: model state dict (unwrapped from DDP if needed via `unwrap_ddp()`), feature index map snapshot, T calibration temperature values (one per target), model architecture config (layer widths, dropout), and input dimensionality.
4. Write to `EXPORT_PATH` as a PyTorch `.pt` file.
5. Log the export path and total model parameter count.

**Config keys used:** `CHECKPOINT_DIR`, `CALIBRATION_PARAMS_PATH`, `EXPORT_PATH`.

---

## 6. DDP Training Implementation

---

### 6.1 Process Launch and Initialisation

`torchrun` is the DDP launcher, invoked from `reward_job.sh` with `--nproc_per_node = NUM_GPUS` (default 2). It sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` before spawning each worker process. `train_main.py` reads these at startup and calls `dist.init_process_group(backend='nccl')`. Each process calls `torch.cuda.set_device(LOCAL_RANK)` to bind to its assigned GPU.

If `NUM_GPUS = 1` or `torch.cuda.device_count() < 2` at runtime, `init_process_group` is skipped. The model is not wrapped in DDP and `DistributedSampler` is replaced by `RandomSampler`. This single-GPU fallback is logged at `WARNING`.

The `nccl` backend is used for all GPU-to-GPU communication. `nccl` is the only supported backend for CUDA DDP — `gloo` is not suitable for large all-reduce operations at the gradient sizes involved here (~1.4 GB per step at full model dimensionality).

---

### 6.2 Data Sharding

Only rank 0 loads the dataset from `final_cdss_dataset.parquet` inside `TrainManager.__init__`. Rank 0 calls `Mimic4DataLoader(config).load()`, which returns three `ParquetDataset` objects — one per split. These are lightweight wrappers holding a PyArrow file handle and row index lists; no feature data is materialised at load time. Non-rank-0 processes do not touch disk — they receive the `DatasetBundle` object and broadcast `pos_weight` scalars from rank 0 and wait at a barrier while rank 0 loads.

Training data is sharded across GPUs via `RowGroupBlockSampler`. The sampler shuffles at the row group level rather than globally across individual row indices, preserving Parquet row-group locality and preventing I/O thrashing on the `ParquetDataset` LRU cache. Row groups are partitioned round-robin across DDP ranks, ensuring each rank processes a non-overlapping subset of admissions per epoch. Each rank's `DataLoader` calls `ParquetDataset.__getitem__` for its assigned indices, which reads the corresponding row groups lazily from disk via the LRU cache and materialises individual batch tensors on demand. With 2 GPUs and `BATCH_SIZE_PER_GPU = 256`, each GPU materialises 256 samples per step and the effective batch size is 512.

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

The gradient computation step in `TrainManager` feeds `masking.py apply_adversarial_mask()` with `∂L/∂X`.

**The problem:** Identifying the worst-case feature slot per sample requires the gradient of the loss with respect to the input tensor. PyTorch does not compute this by default because `X` (the data tensor loaded from the dataset) is not a leaf tensor with `requires_grad=True`.

**The mechanism:**

Before the first forward pass in an adversarial batch, `TrainManager` clones the batch input tensor and calls `requires_grad_(True)` on the clone. This ensures the original data tensor in the `DataLoader` is not modified. The first forward pass runs on this clone. The resulting loss is computed via `loss.compute_loss()`. `loss.backward()` is called inside `model.no_sync()` to suppress DDP all-reduce. After the backward pass, the clone's `.grad` attribute holds `∂L/∂X` of shape `(batch_size, D)`.

**Importance scoring per feature slot (RMS normalisation):**

For each sample and each maskable feature slot, the importance score is the RMS gradient magnitude: `importance[i, f] = ||grad[i, start_f : end_f]||_2 / sqrt(slot_dim_f)`. Dividing by the square root of slot dimension normalises for slot size — `demographic_vec` has 8 dimensions while all embedding slots have 768. Without this normalisation, the L2 norm of a 768-dim gradient vector would almost always exceed an 8-dim vector regardless of actual feature importance, biasing adversarial selection toward embedding slots.

**Top-k slot selection:**

After computing importance scores for all maskable slots, sort them descending. Zero the top k slots per sample, where k is drawn independently per sample via `_sample_k(adversarial_k_min_fraction, adversarial_k_max_fraction)`. This replaces the earlier single-slot argmax with a sorted selection over the configured fraction range.

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

Checkpoint ownership is centralised in a dedicated `CheckpointManager` class (own file: `checkpoint_manager.py`). It is the sole writer/loader of `epoch_<N>.pt` and `best_model.pt`, and it performs schema safety checks before a resume proceeds. `CheckpointManager.validate_feature_index_map(old_map, new_map)` compares the feature-index map stored inside the checkpoint against the freshly derived map from `Mimic4DataLoader`; if keys or index ranges differ, it raises and aborts the resume to prevent continuing with shifted feature boundaries after an upstream dataset change.

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
| `INPUT_DIM` | `42248` (MIMIC-IV) | `mimic4_data_loader.py`, `model.py` | Expected input dimensionality; validated against feature index map at startup; update when deploying on a different dataset |
| `LAYER_WIDTHS` | `[8192, 2048, 512, 128]` | `model.py` | Hidden layer output sizes; length determines depth |
| `★ DROPOUT_RATES` | `[0.4, 0.3, 0.3, 0.2]` | `model.py` | Per-layer dropout probabilities; must match length of `LAYER_WIDTHS`. A single float is also accepted and broadcast to all layers for backward compatibility |
| `ACTIVATION` | `relu` | `model.py` | Activation function; supports `relu`, `leaky_relu` |
| `NUM_TARGETS` | `2` | `model.py`, `loss.py`, `calibrate.py`, `inference.py` | Number of classification targets T; determines number of output heads |

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
| `LOSS_WEIGHTS` | `[0.75, 0.25]` (MIMIC-IV) | `loss.py` | List of T normalised weights, one per target; `RewardModelConfig` validates they sum to 1.0 at startup |
| `POS_WEIGHTS` | (computed) | `loss.py` | List of T positive class weights; each computed from the applicable (non-NaN) training rows for that target if omitted |

### Masking curriculum

| Key | Default | Used by | Description |
|-----|---------|---------|-------------|
| `MASKING_START_RATIOS` | `{random: 1.0, adversarial: 0.0, none: 0.0}` | `masking.py` | Mode probabilities at epoch 0 |
| `MASKING_END_RATIOS` | `{random: 0.33, adversarial: 0.33, none: 0.34}` | `masking.py` | Mode probabilities at final epoch |
| `★ MASKING_TRANSITION_MIDPOINT_EPOCH` | `50` | `masking.py` | Epoch at sigmoid crossover inflection point |
| `MASKING_TRANSITION_SHAPE` | `sigmoid` | `masking.py` | Crossover curve shape |
| `MASKING_RANDOM_K_MIN_FRACTION` | `0.5` | `masking.py` | Minimum fraction of maskable slots zeroed per sample in random mode |
| `MASKING_RANDOM_K_MAX_FRACTION` | `1.0` | `masking.py` | Maximum fraction of maskable slots zeroed per sample in random mode (upper bound is M−1) |
| `MASKING_ADVERSARIAL_K_MIN_FRACTION` | `0.3` | `masking.py` | Minimum fraction of maskable slots zeroed per sample in adversarial mode |
| `MASKING_ADVERSARIAL_K_MAX_FRACTION` | `0.7` | `masking.py` | Maximum fraction of maskable slots zeroed per sample in adversarial mode |
| `NUM_ALWAYS_VISIBLE_FEATURES` | `5` | `masking.py` | Number of leading feature slots that are never masked (positional — must be the first N slots in the feature index map) |

### Dataset loading

| Key | Default | Used by | Description |
|-----|---------|---------|-------------|
| `DATASET_PATH` | (required) | `mimic4_data_loader.py`, `calibrate.py`, `validate_contract.py` | Absolute path to `final_cdss_dataset.parquet` |
| `★ DATASET_ROW_GROUP_CACHE_SIZE` | `2` | `parquet_dataset.ParquetDataset` | Number of decompressed Parquet row groups held in the LRU cache per `ParquetDataset` instance; higher values increase RAM usage but reduce re-reads when the DataLoader accesses rows from the same row group across consecutive batches |

### Paths

| Key | Default | Used by | Description |
|-----|---------|---------|-------------|
| `DATASET_PATH` | (required) | `mimic4_data_loader.py`, `calibrate.py`, `validate_contract.py` | Absolute path to `final_cdss_dataset.parquet` |
| `CHECKPOINT_DIR` | `data/reward_model/checkpoints` | `train.py`, `calibrate.py`, `export_model.py` | Checkpoint directory |
| `METRICS_PATH` | `data/reward_model/training_metrics.parquet` | `train.py` | Per-epoch metrics output |
| `CALIBRATION_PARAMS_PATH` | `data/reward_model/calibration_params.json` | `calibrate.py`, `inference.py` | Temperature scaling output |
| `EXPORT_PATH` | `data/reward_model/frozen_model.pt` | `export_model.py` | Self-contained frozen model artefact |

---

## 10. Memory Requirements

Variables: **A** = total samples (e.g. ~546k admissions for MIMIC-IV). **B** = `BATCH_SIZE_PER_GPU`. **D** = input dimensionality (42,248 at full BERT embeddings for MIMIC-IV; varies by dataset). **P** = total model parameters (~364M at default widths with D=42,248). **C** = `DATASET_ROW_GROUP_CACHE_SIZE`. **R** = Parquet row group size (rows per group, typically ~400 at default PyArrow settings).

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

## 11. MIMIC-IV Reference Configuration

This section documents the specific implementation of the generic reward model framework for MIMIC-IV clinical data. All config values in this section are the defaults in `config/reward_model.yaml`. When deploying on a different dataset, update these values — no framework code changes are required.

### 11.1 Target Definitions (T=2)

| Index | Column | Definition | Population | NaN convention |
|-------|--------|------------|------------|----------------|
| 0 — Y1 mortality | `y1_mortality` | `admissions.hospital_expire_flag = 1` | All admissions | Never NaN |
| 1 — Y2 readmission | `y2_readmission` | Unplanned readmission within 30 days of `dischtime` | Survivors (Y1=0) | NaN when Y1=1 |

Y2 = NaN for deceased patients is enforced upstream by `extract_y_data.py` and validated by `Mimic4DataLoader._validate_labels()`. The dynamic NaN mask in `loss.py` handles this per batch without removing rows from the dataset.

### 11.2 Feature Slot Summary

| Slots | Count | Always visible | Dim each | Total dims |
|-------|-------|----------------|----------|-----------|
| F1 (`demographic_vec`) | 1 | Yes | 8 | 8 |
| F2–F5 (history/triage embeddings) | 4 | Yes | 768 | 3,072 |
| F6–F18 (lab group embeddings) | 13 | No (maskable) | 768 | 9,984 |
| F19 (radiology embedding) | 1 | No (maskable) | 768 | 768 |
| F20–F56 (microbiology panel embeddings) | 37 | No (maskable) | 768 | 28,416 |
| **Total** | **56** | **5 always visible** | | **42,248** |

`NUM_ALWAYS_VISIBLE_FEATURES = 5`. Maskable slot count M = 51.

### 11.3 Default Config Values

| Key | MIMIC-IV Default | Notes |
|-----|-----------------|-------|
| `NUM_TARGETS` | `2` | Y1 mortality + Y2 readmission |
| `INPUT_DIM` | `42248` | Derived at runtime; validation only |
| `NUM_ALWAYS_VISIBLE_FEATURES` | `5` | F1–F5 positional |
| `LOSS_WEIGHT_Y1` | `0.75` | 3:1 ratio favouring mortality |
| `LOSS_WEIGHT_Y2` | `0.25` | |
| `LAYER_WIDTHS` | `[8192, 2048, 512, 128]` | Gradual funnel from D=42,248 |
| `DROPOUT_RATES` | `[0.4, 0.3, 0.3, 0.2]` | Heavier on wider early layers |

---

> See `reward_model_architecture.md` for system context, design rationale, pipeline overview, and the upstream pipeline that produces `final_cdss_dataset.parquet`.
