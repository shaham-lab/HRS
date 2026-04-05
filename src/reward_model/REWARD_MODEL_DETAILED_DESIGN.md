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
   - [reward_model.py](#reward_modelpy)
   - [masking.py](#maskingpy)
   - [metrics.py](#metricspy)
   - [reward_model_manager.py](#reward_model_managerpy)
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

The reward model operates at the admission level. `hadm_id` (int64) is the primary key throughout — one row in `full_cdss_dataset.parquet` corresponds to one hospital admission. `subject_id` is present in the dataset but is not used by any reward model module; it is carried through for potential downstream use by the RL agent.

The reward model does not perform any identifier linkage. All identifier resolution and null-`hadm_id` handling is completed upstream by `HRS/src/preprocessing`. By the time `full_cdss_dataset.parquet` reaches this module, every row has a valid `hadm_id`.

---

## 2. Module Decomposition

### Module boundaries

Each Python file corresponds to one pipeline concern. Inter-module state exchange happens via tensors passed as function arguments within a single process, or via checkpoint files on disk between separate SLURM jobs. No module reads `full_cdss_dataset.parquet` except `mimic4_data_loader.py` (via `Mimic4DataLoader`, a subclass of the generic `DataLoader` base).

### Class inventory (one class per file)

| File | Class | Role |
|------|-------|------|
| `data_loader.py` | `DataLoader` | Abstract base for dataset loaders. |
| `mimic4_data_loader.py` | `Mimic4DataLoader` | MIMIC-IV implementation of `DataLoader`. |
| `parquet_dataset.py` | `ParquetDataset` | `torch.utils.data.Dataset` backed by a Parquet file and feature index map. |
| `row_group_block_sampler.py` | `RowGroupBlockSampler` | Sampler that shards row groups round-robin across DDP ranks. |
| `dataset_bundle.py` | `DatasetBundle` | Named tuple bundling datasets, feature index map, pos-weights, and label names. |
| `reward_model.py` | `RewardModel` | Feedforward MLP with T output heads (one per classification target). |
| `masking.py` | `MaskingSchedule` | Masking curriculum with random/adversarial/none modes. |
| `reward_model_config.py` | `RewardModelConfig` | Pydantic config model. |
| `checkpoint_manager.py` | `CheckpointManager` | Manages saving and loading of `best_model_train.pt` and `best_model.pt`. |
| `inference.py` | `RewardModelInference` | Frozen inference wrapper with calibration parameters. |
| `metrics.py` | `MetricsLogger` | Appends per-epoch metrics rows to a CSV file (also houses metric helpers). |
| `reward_model_manager.py` | `RewardModelManager` | Encapsulates dataset load/broadcast, model + optimizer + scheduler construction, training state, masking curriculum, epoch loop, dev eval, checkpointing. |
| `calibrate.py` | `TemperatureCalibrator` | Manages dev-split inference and per-head temperature scaling state. |

### Class vs plain script

| Module | Pattern | Reason |
|--------|---------|--------|
| `data_loader.py` | **Class** (`DataLoader`) | Generic base (no dataset-specific logic); single instantiation per run |
| `mimic4_data_loader.py` | **Class** (`Mimic4DataLoader`) | MIMIC-IV implementation; validates schema, constructs `DatasetBundle`; single instantiation per run |
| `reward_model.py` | **Class** (`RewardModel`) | Stateful network; instantiated once, called many times via `forward()`; must be wrappable by DDP |
| `masking.py` | **Class** (`MaskingSchedule`) | Maintains curriculum state across the training loop; epoch advancement is a stateful operation |
| `metrics.py` | **Class** (`MetricsLogger`) + Plain functions | Holds persistent metrics path state for CSV appends; retains stateless metric helpers. |
| `reward_model_manager.py` | **Class** (`RewardModelManager`) | Holds training state (datasets, model, optimizer, scheduler, masking schedule), epoch loop, checkpointing; owns dev eval helpers |
| `reward_model_main.py` | Plain script, DDP entry point | CLI parsing, logging setup, runtime init; instantiates `RewardModelManager` and delegates training |
| `calibrate.py` | **Class** (`TemperatureCalibrator`) + Plain script entry point | Coordinates dev-split inference and temperature scaling with a minimal stateful wrapper |
| `inference.py` | **Class** (`RewardModelInference`) | Loaded once, called repeatedly per RL step; holds frozen weights and calibration params in memory |
| `validate_contract.py` | Plain script, CLI tool | One-shot assertion run; exits with code 0 (pass) or 1 (fail) |
| `export_model.py` | Plain script, CLI tool | One-shot serialisation; no shared state |

### Encapsulation rules

All helper functions are module-private (prefixed `_`). The public interface of each pipeline module is its `run(config)` function or class constructor. `reward_model_main.py` owns CLI parsing, logging setup, and runtime initialisation, then delegates to the `RewardModelManager` class in `reward_model_manager.py`, which owns dataset loading/broadcast, model/optimizer/scheduler construction, and training/eval loops. Config is always passed as a validated Pydantic model object, not a raw dict.

### File naming conventions

`*.py` for pipeline step modules, class-only modules for each class, `reward_model_utils.py` for shared helpers/re-exports, `*_job.sh` for SLURM scripts, `submit_*.sh` for submission orchestrators. All names use `snake_case`.

---

## 3. Cross-Cutting Concerns

---

### 3.1 Configuration Loading

`config/reward_model.yaml` is the single source of truth for all parameters. It is loaded and validated by any CLI entry point using the Pydantic model defined in `reward_model_config.py`. The Pydantic model enforces types, required vs optional fields, and value constraints at startup — a misconfigured schedule or invalid path raises before any computation begins.

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
| `WARNING` | Fallback used (e.g. single-GPU mode when fewer than 2 GPUs are available) |
| `ERROR` | A validation assertion failed but the process can report it before exiting |
| `CRITICAL` | Unrecoverable — let the exception propagate |

Log format set by the entry point: `%(asctime)s  %(levelname)-8s  %(name)s  %(message)s`

---

### 3.3 DDP Process Group and Rank Conventions

`torchrun` spawns one process per GPU and injects environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`) into each process before it starts. `WORLD_SIZE` equals `--nproc_per_node` (the number of GPUs requested). Each process has a `rank` (its global index) and a `local_rank` (its GPU index on the current node); for single-node runs these are identical.

The process group is initialised with the `nccl` backend at the start of `reward_model_main.py`. If fewer than 2 GPUs are available or `WORLD_SIZE <= 1`, `init_process_group` is skipped and training proceeds in single-process mode without DDP wrapping. This fallback is detected automatically and logged at `WARNING`.

**Rank 0 is the designated I/O rank.** Only rank 0 writes checkpoints, `training_metrics.csv`, and `calibration_params.json`. Only rank 0 computes `pos_weight` values (inside `RewardModelManager.__init__`) and broadcasts them to other ranks. All ranks participate in forward passes, backward passes, and all-reduce operations. Barrier synchronisation (`dist.barrier()`) is used at three points: after checkpoint load, after `pos_weight` broadcast, and before the final checkpoint write at training completion.

---

### 3.4 Shared Utilities (`reward_model_utils.py`)

| Function | Purpose |
|----------|---------|
| `get_device(local_rank)` | Return `torch.device('cuda', local_rank)` or `cpu` with CUDA availability check |
| `unwrap_ddp(model)` | Return `model.module` if wrapped in DDP, else `model` directly |
| `RewardModelManager.broadcast_tensor(tensor, src_rank)` | Broadcast a scalar tensor from `src_rank` to all ranks via process group |

`reward_model_utils.py` contains only these two shared helpers. All classes live in their own modules and are imported directly. Functions used only once remain private to their module with a `_` prefix.

---

## 4. Feature Index Map

The feature index map defines the start and end index within the flat D-dimensional input tensor for each feature slot. It is derived at load time by the dataset-specific data loader from the ordered list of feature columns in the dataset file. The derivation algorithm iterates the ordered column list, skips metadata columns (e.g. primary key, split column) and label columns, and for each remaining feature column assigns a start index equal to the running offset and an end index equal to start plus the column's declared dimension. The result is a dict mapping column name to a `(start, end)` tuple.

The number and breakdown of feature slots, their column names, and their declared dimensions are dataset-specific. For the MIMIC-IV deployment (56 slots, 1 structured vector at 8 dims, 55 BERT embeddings at EMBEDDING_DIM each, D=42,248 at 768 dims or 7,048 at 128 dims) see `mimic4_feature_set.md` and Section 11.2 of this document.

The dataset-specific data loader validates the expected slot count and per-group breakdown and raises `ValueError` referencing the upstream data model document if any count does not match. This guards against silent miscounting if the upstream dataset schema changes.

The map is constructed once by the data loader, stored in the returned `DatasetBundle`, and passed explicitly to `masking.py` and `reward_model_manager.py`.

The map is never written to a standalone config or YAML file — a separate file would duplicate the upstream data model document and create a consistency risk.

---

## 5. Module Implementation

---

### `data_loader.py`

Defines the `DataLoader` abstract base class that establishes the contract for all dataset-specific loaders. Provides a `load(config)` template method that orchestrates the open → validate → build index map → split → bundle sequence via abstract hooks. Concrete subclasses implement the dataset-specific hooks; the template method ensures consistent ordering and error handling.

Public interface: `load(config) -> DatasetBundle`. All other methods are abstract or private.

---

### `checkpoint_manager.py`

Centralises all checkpoint read/write operations. `CheckpointManager` is the sole class permitted to write `best_model_train.pt` and `best_model.pt`.

**Key responsibilities:**
- `save_train_checkpoint(model, optimizer, epoch, best_dev_loss)` — writes model and optimizer state to `CHECKPOINT_DIR/best_model_train.pt`. Overwritten on every call; only the latest training state is kept.
- `save_best_model(model, epoch, best_dev_loss)` — writes model state only to `CHECKPOINT_DIR/best_model.pt`. Called only when dev loss improves.
- `find_latest() -> Optional[Path]` — returns the path to `best_model_train.pt` if it exists, else `None`.
- `load(path) -> dict` — loads and returns the checkpoint dict from the given path.

---

### `mimic4_data_loader.py`

Houses the MIMIC-IV `Mimic4DataLoader` implementation built on the generic `DataLoader` template. `DataLoader.load()` orchestrates open/validate/read/bundle steps via template hooks; dataset-specific logic lives in subclasses. The dataset is never fully materialised into a float32 tensor — batches are read lazily from disk by `ParquetDataset.__getitem__` at training time.

**Algorithm (Mimic4DataLoader implementation):**
1. Read `full_cdss_dataset.parquet` from `DATASET_PATH` using `pyarrow`. Do not use `fastparquet` — the fastparquet/pyarrow serialisation incompatibility observed in the preprocessing pipeline applies here.
2. Validate column presence and order via `Mimic4DataLoader._validate_schema()`; assert all 61 expected columns are present in canonical order from `PREPROCESSING_DATA_MODEL.md` Section 3.12.
3. Validate labels: `y1_mortality` dtype int8/float32 and non-null; `y2_readmission` dtype float32 with NaN for deceased rows and non-null for survivors (checked inside `_validate_labels`).
4. Build the feature index map via `Mimic4DataLoader._build_feature_index_map()`. This also validates the 55-embedding count and per-group breakdown — see Section 4.
5. Read the `split` column only (lightweight — metadata column) to determine row indices for each split. Produce three lists of row indices: `train_indices`, `dev_indices`, `test_indices`.
6. Compute per-target positive class weights from the training split rows via `Mimic4DataLoader._compute_pos_weights()` unless overridden by `POS_WEIGHTS` in config.
7. Instantiate three `ParquetDataset` objects — one per split — passing `config`, the open PyArrow file handle, the split's row index list, the feature index map, and label columns. Each `ParquetDataset` holds only the file handle and index metadata in memory. No feature data is read at this step.
8. Return a `DatasetBundle` named tuple: `train_dataset`, `dev_dataset`, `test_dataset` (each a `ParquetDataset`), `feature_index_map`, `pos_weights`, `label_names`.

**Config keys used:** `DATASET_PATH`, `FEATURES_DIM` (assertion only — must equal D as derived from the column schema), `DATASET_ROW_GROUP_CACHE_SIZE`.

**Memory note:** The dataset is never fully resident in RAM. At any given time, only the LRU-cached row groups are held in memory — at most `DATASET_ROW_GROUP_CACHE_SIZE × row_group_size × D × 4` bytes. At the default cache size of 2, a typical row group size of ~400 rows, and D=42,248, this is approximately **135 MB**. Combined with DataLoader prefetch workers (each holding one batch buffer of ~42 MB), total dataset-related RAM is well under 1 GB — comfortably within the 64 GB SLURM allocation alongside the model, optimizer state, and OS overhead.

---

### `reward_model.py`

Defines the `RewardModel` class — a feedforward MLP with a gradual funnel and T independent sigmoid output heads, one per classification target.

**Class: `RewardModel(nn.Module)`**

The constructor accepts `config: RewardModelConfig`. It reads `INPUT_DIM`, `LAYER_WIDTHS`, `DROPOUT_RATES`, `ACTIVATION`, and `NUM_TARGETS` directly from config. It builds a sequence of `Linear` → `BatchNorm1d` → `Activation` → `Dropout` blocks for each layer, then T independent `Linear(layer_widths[-1], 1)` heads producing raw logits, one per target.

`forward(x)` returns a tuple of T tensors `(logits_y1, ..., logits_yT)`, each of shape `(batch_size, 1)`. **Sigmoid is not applied inside `forward`** — raw logits are returned so that `BCEWithLogitsLoss` can be used for numerical stability in training. Sigmoid is applied only at inference time in `inference.py`.

The model is not DDP-aware — DDP wrapping is performed by `RewardModelManager` during construction.

**Config keys used:** `INPUT_DIM`, `LAYER_WIDTHS`, `DROPOUT_RATES`, `ACTIVATION`, `NUM_TARGETS`.

---

### `masking.py`

Implements the `MaskingSchedule` class, which maintains the curriculum state and applies the correct masking mode to each mini-batch.

**Class: `MaskingSchedule`**

The constructor accepts `config: RewardModelConfig` and `feature_index_map`. It reads all masking parameters directly from config and partitions the feature index map into `_always_visible_slots` (first `NUM_ALWAYS_VISIBLE_FEATURES` entries) and `_maskable_slots` (all remaining entries). M denotes the count of maskable slots.

`get_mode_probabilities(epoch)` computes the current `(p_random, p_adversarial, p_none)` tuple for the given epoch using a sigmoid crossover curve inline.

`sample_mode(epoch)` draws a masking mode string — `'random'`, `'adversarial'`, or `'none'` — according to the current probabilities, with explicit normalisation to guard against floating-point drift.

`_sample_k(min_fraction, max_fraction)` draws an integer k independently per sample from `Uniform(floor(min_fraction × M), ceil(max_fraction × M))`. After rounding, the lower bound is clamped to at least 1 and the upper bound to at most M−1. If lower still exceeds upper after clamping (possible only when M is very small and the fraction range is tight), lower is set equal to upper as a final safety guard.

`apply_random_mask(X)` draws k independently per sample via `_sample_k(random_k_min_fraction, random_k_max_fraction)`, selects k maskable slots uniformly at random per sample without replacement, then zeros the corresponding index ranges to 0.0. Returns a cloned masked tensor — the original is not modified.

`apply_adversarial_mask(X, grad_X)` draws k independently per sample via `_sample_k(adversarial_k_min_fraction, adversarial_k_max_fraction)`. For each sample it computes the RMS gradient magnitude per maskable slot: `importance[i, f] = ||grad_X[i, start_f:end_f]||_2 / sqrt(slot_dim_f)`. Dividing by the square root of slot dimension normalises for slot size, making `demographic_vec` (8 dims) and embedding slots (768 dims) directly comparable. Sorts all maskable slots by importance score descending and zeros the top k slots per sample. Returns a cloned masked tensor. The gradient computation is the responsibility of `RewardModelManager` — see Section 7.

**Config keys used:** `MASKING_START_RATIOS`, `MASKING_END_RATIOS`, `MASKING_TRANSITION_MIDPOINT_EPOCH`, `MAX_EPOCHS`, `MASKING_RANDOM_K_MIN_FRACTION`, `MASKING_RANDOM_K_MAX_FRACTION`, `MASKING_ADVERSARIAL_K_MIN_FRACTION`, `MASKING_ADVERSARIAL_K_MAX_FRACTION`, `NUM_ALWAYS_VISIBLE_FEATURES`.

---

### `metrics.py`

Contains pure metric helpers plus the `MetricsLogger` class.

`compute_metrics(logits_list, labels_list, masked=False)` applies sigmoid to each logits tensor, constructs per-target valid-sample masks, and computes AUROC, AUPRC, and ECE for each target using scikit-learn. Returns a dict keyed by target index: `{0: {'auroc': ..., 'auprc': ..., 'ece': ...}, 1: {...}, ...}`. The `masked` flag is included in the returned dict as a tag so callers can distinguish masked-input metrics from unmasked-input metrics in the epoch log. ECE uses equal-mass binning (adaptive bin edges via `np.percentile`) rather than equal-width binning, to produce reliable calibration estimates under class imbalance. Called on the dev split by rank 0 at epoch end using unmasked inputs. Also called on training batches for diagnostic AUROC tracking — see Section 6.4.

`MetricsLogger(Path, label_names)` owns the persistent `training_metrics.csv` path and appends rows using the standard `csv.DictWriter`. Creates the file with a header row if it does not exist. Rank 0 only. Columns: `epoch`, `time(seconds)`, `loss_total`, then for each label: `loss_<name>`, `auroc_<name>`, `auprc_<name>`, `ece_<name>`.

**Config keys used:** None directly; the caller supplies already-computed values.

---

### `reward_model_manager.py`

Defines the `RewardModelManager` class, which owns all training state and encapsulates the epoch loop.

- `__init__(config, rank, local_rank, world_size, is_ddp, device)` loads the dataset via `_load_datasets_and_weights()` (rank 0 reads from disk; broadcast to others), stores the feature index map and pos-weights, builds `RewardModel` (DDP-wrapped when applicable), and constructs the optimizer. It also initialises checkpoint/metrics paths and helper handles.
- `setup_training_state(ckpt_state, start_epoch)` builds the `MaskingSchedule`, `DataLoader` (with `RowGroupBlockSampler`), and LR scheduler (warmup + cosine). If `ckpt_state` is provided, restores model and optimizer state, then fast-forwards the scheduler by `start_epoch × steps_per_epoch` steps to match the resumed position.
- `_run_train_batch(X, labels, epoch)` executes the masking-aware mini-batch (random/adversarial/none). Adversarial mode clones `X`, sets `requires_grad_(True)`, does the first backward under `no_sync()` (when DDP) to obtain gradients, reapplies adversarial masks, then performs the second backward/step.
- `train_epochs(start_epoch, best_dev_loss)` drives the epoch loop: sets sampler epoch, iterates batches (calling `_run_train_batch` and stepping the scheduler), runs dev evaluation on rank 0, logs metrics, performs checkpoint/early-stop decisions, and broadcasts stop signals across ranks.
- `compute_loss(logits_list, labels_list)` is a method on the manager that applies per-target dynamic NaN masking and weighted BCE with `pos_weight`. Returns `(total_loss, component_losses)` for epoch logging.

**Config keys used:** `LEARNING_RATE`, `WEIGHT_DECAY`, `LR_WARMUP_EPOCHS`, `LR_MIN`, `ADAM_BETA1`, `ADAM_BETA2`, `MAX_EPOCHS`, `BATCH_SIZE_PER_GPU`, `EARLY_STOPPING_PATIENCE`, `CHECKPOINT_DIR`, `METRICS_PATH`, all masking keys, all loss keys, all model architecture keys.

---

### `reward_model_main.py`

Plain script and DDP entry point launched by `torchrun`. It wires CLI/runtime concerns and delegates training to `RewardModelManager`.

**Algorithm:**
1. Parse CLI arguments: `--config`, `--resume`.
2. Load and validate config via `load_and_validate_config()`.
3. Call `_init_runtime()` to set up ranks, process group (`nccl`), and device selection. DDP is enabled only when `torch.cuda.device_count() >= 2` and `WORLD_SIZE > 1`; otherwise single-process mode is used automatically.
4. Instantiate `RewardModelManager(config, rank, local_rank, world_size, is_ddp, device)`.
5. If `--resume`, locate latest checkpoint via `CheckpointManager.find_latest()`, load on rank 0, broadcast to all ranks, and recover `start_epoch`/`best_dev_loss`.
6. Call `manager.setup_training_state(ckpt_state, start_epoch)`.
7. Execute `manager.train_epochs(start_epoch, best_dev_loss)`.
8. Destroy the process group if DDP is active.

---

### `calibrate.py`

Temperature-scaling calibration is owned by the `TemperatureCalibrator` class. Single GPU — no DDP.

**Class: `TemperatureCalibrator`**

- `__init__(config, device)`: Loads `best_model.pt` from `CHECKPOINT_DIR`, reconstructs `RewardModel(config)`, loads the state dict, moves it to `device`, sets eval mode, and freezes gradients. Loads the dev split via `Mimic4DataLoader(config)` and stores `dev_dataset` and `feature_index_map` from the bundle.
- `_collect_logits()`: Builds a single-worker `DataLoader` on the dev split and, under `torch.no_grad()`, runs a full forward pass to collect per-head logits and labels as flat tensors. Returns `(logits_list, labels_list)` (lists of length T).
- `calibrate_and_save()`: Calls `_collect_logits()`, then for each target i: constructs mask `~np.isnan(labels_i)`, logs pre-calibration ECE via `_compute_ece_from_logits(logits_i, labels_i, T=1.0, mask)`, fits `T_i` via `_fit_temperature` (log-space L-BFGS), logs post-calibration ECE, and writes `{f"T_{i}": float(T_i)}` to `CALIBRATION_PARAMS_PATH` as JSON (directory created if missing).

**Pure helpers (module-level, unchanged):**
- `_compute_ece_from_logits(logits, labels, T, mask)`: Expected Calibration Error with adaptive binning.
- `_fit_temperature(logits, labels, mask)`: Log-space scalar temperature fitting with L-BFGS, clamped to `1e-8` minimum.

**Entry point:**
`main()` parses `--config`, sets up logging, selects `torch.device("cuda" if available else "cpu")`, instantiates `TemperatureCalibrator`, and calls `calibrate_and_save()`.

**Config keys used:** `CHECKPOINT_DIR`, `DATASET_PATH`, `CALIBRATION_PARAMS_PATH`, `BATCH_SIZE_PER_GPU`.

---

### `inference.py`

Provides the `RewardModelInference` class consumed by the RL agent. Loaded once per RL session, called once per episode step.

**Class: `RewardModelInference`**

The constructor accepts `config: RewardModelConfig`, `checkpoint_path` (path to `frozen_model.pt` written by `export_model.py`), and an optional `device`. It loads calibration temperatures and the feature index map from the exported artefact, reconstructs `RewardModel(config)`, loads the state dict, moves the model to the target device, calls `model.eval()`, and freezes all parameters. No gradient computation ever occurs in this class.

`predict(X)` accepts a float32 tensor of shape `(N, input_dim)`. Under `torch.no_grad()`, it runs a forward pass and returns a tuple of T tensors, each of shape `(N, 1)`, containing calibrated probabilities from `sigmoid(logit_i / T_i)` for each target i. Temperature values are clamped to a minimum of 1e-8 at load time to prevent division by zero.

`get_feature_index_map()` returns the feature index map loaded from the exported artefact so the RL agent can construct correctly masked input tensors for each episode step without needing access to the dataset.

**Config keys used:** `INPUT_DIM`, `LAYER_WIDTHS`, `DROPOUT_RATES`, `ACTIVATION`, `NUM_TARGETS`.

---

### `validate_contract.py`

Standalone CLI tool. Runs only the schema assertions from `Mimic4DataLoader.load()` steps 2–5 without constructing tensors or loading data into memory. Intended to be run before submitting a training job.

**Algorithm:**
1. Load config from `--config` argument.
2. Read `full_cdss_dataset.parquet` column names and dtypes only (schema metadata, no data materialisation).
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
3. Load the feature index map from `Mimic4DataLoader(config).load()`.
4. Construct an export dict: model state dict (DDP prefix stripped if present), feature index map, T calibration temperatures, `INPUT_DIM`, and `NUM_TARGETS`.
5. Write to `EXPORT_PATH` as a PyTorch `.pt` file.
6. Log the export path and total model parameter count.

**Config keys used:** `CHECKPOINT_DIR`, `CALIBRATION_PARAMS_PATH`, `EXPORT_PATH`.

---

## 6. DDP Training Implementation

---

### 6.1 Process Launch and Initialisation

`torchrun` is the DDP launcher, invoked from `reward_job.sh` with `--nproc_per_node` matching the number of GPUs requested via `--gres`. It sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` before spawning each worker process. `reward_model_main.py` reads these at startup and calls `dist.init_process_group(backend='nccl')`. Each process calls `torch.cuda.set_device(LOCAL_RANK)` to bind to its assigned GPU.

If `torch.cuda.device_count() < 2` or `WORLD_SIZE <= 1` at runtime, `init_process_group` is skipped. The model is not wrapped in DDP and `RowGroupBlockSampler` handles sharding with `world_size=1`. This single-GPU fallback is logged at `WARNING`.

The `nccl` backend is used for all GPU-to-GPU communication. `nccl` is the only supported backend for CUDA DDP — `gloo` is not suitable for large all-reduce operations at the gradient sizes involved here (~1.4 GB per step at full model dimensionality).

---

### 6.2 Data Sharding

Only rank 0 loads the dataset from `full_cdss_dataset.parquet` inside `RewardModelManager.__init__`. Rank 0 calls `Mimic4DataLoader(config).load()`, which returns three `ParquetDataset` objects — one per split. These are lightweight wrappers holding a PyArrow file handle and row index lists; no feature data is materialised at load time. Non-rank-0 processes do not touch disk — they receive the `DatasetBundle` object and broadcast `pos_weight` scalars from rank 0 and wait at a barrier while rank 0 loads.

Training data is sharded across GPUs via `RowGroupBlockSampler`. The sampler shuffles at the row group level rather than globally across individual row indices, preserving Parquet row-group locality and preventing I/O thrashing on the `ParquetDataset` LRU cache. Row groups are partitioned round-robin across DDP ranks, ensuring each rank processes a non-overlapping subset of admissions per epoch. Each rank's `DataLoader` calls `ParquetDataset.__getitem__` for its assigned indices, which reads the corresponding row groups lazily from disk via the LRU cache and materialises individual batch tensors on demand. With `BATCH_SIZE_PER_GPU = 1024` and N GPUs, the effective batch size is `1024 × N`.

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

`training_metrics.csv` is appended per epoch by rank 0 only via `MetricsLogger`.

---

## 7. Adversarial Masking Gradient Computation

The gradient computation step in `RewardModelManager` feeds `masking.py apply_adversarial_mask()` with `∂L/∂X`.

**The problem:** Identifying the worst-case feature slot per sample requires the gradient of the loss with respect to the input tensor. PyTorch does not compute this by default because `X` (the data tensor loaded from the dataset) is not a leaf tensor with `requires_grad=True`.

**The mechanism:**

Before the first forward pass in an adversarial batch, `RewardModelManager` clones the batch input tensor and calls `requires_grad_(True)` on the clone. This ensures the original data tensor in the `DataLoader` is not modified. The first forward pass runs on this clone. The resulting loss is computed via `RewardModelManager.compute_loss()`. `loss.backward()` is called inside `model.no_sync()` to suppress DDP all-reduce. After the backward pass, the clone's `.grad` attribute holds `∂L/∂X` of shape `(batch_size, D)`.

**Importance scoring per feature slot (RMS normalisation):**

For each sample and each maskable feature slot, the importance score is the RMS gradient magnitude: `importance[i, f] = ||grad[i, start_f : end_f]||_2 / sqrt(slot_dim_f)`. Dividing by the square root of slot dimension normalises for slot size — `demographic_vec` has 8 dimensions while all embedding slots have 768. Without this normalisation, the L2 norm of a 768-dim gradient vector would almost always exceed an 8-dim vector regardless of actual feature importance, biasing adversarial selection toward embedding slots.

**Top-k slot selection:**

After computing importance scores for all maskable slots, sort them descending. Zero the top k slots per sample, where k is drawn independently per sample via `_sample_k(adversarial_k_min_fraction, adversarial_k_max_fraction)`. This replaces the earlier single-slot argmax with a sorted selection over the configured fraction range.

**Gradient cleanup:** After the first pass, the clone tensor is discarded. `optimizer.zero_grad()` is called before the second forward pass to clear any accumulated gradients. The second pass uses the original (non-clone) batch input tensor, which has `requires_grad = False`.

---

## 8. Checkpoint and Resume

Two checkpoint files are maintained in `CHECKPOINT_DIR`:

**`best_model_train.pt`** — training checkpoint, overwritten on every dev-loss improvement:
- `model_state_dict` — model weights, unwrapped from DDP via `unwrap_ddp()`
- `optimizer_state_dict` — AdamW state for resume
- `epoch` — last completed epoch number
- `best_dev_loss` — best dev loss seen so far (used for early stopping)

**`best_model.pt`** — inference checkpoint, overwritten on every dev-loss improvement:
- `model_state_dict` — model weights only (no optimizer state)
- `epoch`
- `best_dev_loss`

Checkpoint ownership is centralised in `CheckpointManager`. The LR scheduler is not saved — on resume it is rebuilt from config and fast-forwarded by `start_epoch × steps_per_epoch` steps. The masking schedule is also not saved — it is always rebuilt from config on resume. `config/reward_model.yaml` is always the authoritative source for architecture.

**Resume mechanism:**

On `--resume`, `best_model_train.pt` is loaded on rank 0. Model and optimizer state are restored. The scheduler is rebuilt and fast-forwarded. Under DDP, the checkpoint dict is broadcast to all ranks via `dist.broadcast_object_list()` and a `dist.barrier()` ensures all ranks proceed together.

---

## 9. Configuration Reference

All keys defined in `config/reward_model.yaml`. Loaded and validated by `load_and_validate_config()`. ★ marks the primary tuning knob for each performance or capacity constraint.

### Model architecture

| Key | Default | Used by | Description |
|-----|---------|---------|-------------|
| `INPUT_DIM` | `7048` (reduced dataset) / `42248` (full dataset) | `reward_model.py` | Input dimensionality; must match the actual feature index map derived from the dataset |
| `LAYER_WIDTHS` | `[8192, 2048, 512, 128]` | `reward_mode.py` | Hidden layer output sizes; length determines depth |
| `★ DROPOUT_RATES` | `[0.4, 0.3, 0.3, 0.2]` | `reward_mode.py` | Per-layer dropout probabilities; must match length of `LAYER_WIDTHS`. A single float is also accepted and broadcast to all layers for backward compatibility |
| `ACTIVATION` | `relu` | `reward_mode.py` | Activation function; supports `relu`, `leaky_relu` |
| `NUM_TARGETS` | `2` | `reward_mode.py`, `reward_model_manager.py`, `calibrate.py`, `inference.py` | Number of classification targets T; determines number of output heads |

### Training

| Key | Default | Used by | Description |
|-----|---------|---------|-------------|
| `MAX_EPOCHS` | `100` | `reward_model_manager.py` | Maximum training epochs |
| `★ BATCH_SIZE_PER_GPU` | `1024` | `reward_model_manager.py` | Samples per GPU per step; effective batch = this × number of GPUs allocated |
| `LEARNING_RATE` | `1e-4` | `reward_model_manager.py` | Initial AdamW learning rate |
| `WEIGHT_DECAY` | `1e-5` | `reward_model_manager.py` | AdamW weight decay |
| `ADAM_BETA1` | `0.9` | `reward_model_manager.py` | AdamW first moment decay |
| `ADAM_BETA2` | `0.999` | `reward_model_manager.py` | AdamW second moment decay |
| `LR_WARMUP_EPOCHS` | `5` | `reward_model_manager.py` | Linear warmup before cosine decay |
| `LR_MIN` | `1e-6` | `reward_model_manager.py` | Minimum LR at end of cosine decay |
| `EARLY_STOPPING_PATIENCE` | `10` | `reward_model_manager.py` | Epochs without dev loss improvement before stopping |

### Loss

| Key | Default | Used by | Description |
|-----|---------|---------|-------------|
| `LOSS_WEIGHTS` | `[0.75, 0.25]` (MIMIC-IV) | `reward_model_manager.py` | List of T normalised weights, one per target; `RewardModelConfig` validates they sum to 1.0 at startup |
| `POS_WEIGHTS` | (computed) | `reward_model_manager.py` | List of T positive class weights; each computed from the applicable (non-NaN) training rows for that target if omitted |

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
| `DATASET_PATH` | (required) | `mimic4_data_loader.py`, `calibrate.py`, `validate_contract.py` | Absolute path to `full_cdss_dataset.parquet` |
| `★ DATASET_ROW_GROUP_CACHE_SIZE` | `2` | `parquet_dataset.ParquetDataset` | Number of decompressed Parquet row groups held in the LRU cache per `ParquetDataset` instance; higher values increase RAM usage but reduce re-reads when the DataLoader accesses rows from the same row group across consecutive batches |

### Paths

| Key | Default | Used by | Description |
|-----|---------|---------|-------------|
| `DATASET_PATH` | (required) | `mimic4_data_loader.py`, `calibrate.py`, `validate_contract.py` | Absolute path to `full_cdss_dataset.parquet` |
| `CHECKPOINT_DIR` | `data/reward_model/checkpoints` | `reward_model_manager.py`, `calibrate.py`, `export_model.py` | Checkpoint directory |
| `METRICS_PATH` | `data/reward_model/training_metrics.csv` | `reward_model_manager.py` | Per-epoch metrics CSV output (`.csv` suffix substituted automatically if another extension is given) |
| `CALIBRATION_PARAMS_PATH` | `data/reward_model/calibration_params.json` | `calibrate.py`, `inference.py` | Temperature scaling output |
| `EXPORT_PATH` | `data/reward_model/frozen_model.pt` | `export_model.py` | Self-contained frozen model artefact |

---

## 10. Memory Requirements

Variables: **A** = total samples (e.g. ~546k admissions for MIMIC-IV). **B** = `BATCH_SIZE_PER_GPU` (currently 1024). **D** = input dimensionality (7,048 for reduced dataset; 42,248 for full dataset). **P** = total model parameters (~364M at default widths with D=42,248; ~61M with D=7,048). **C** = `DATASET_ROW_GROUP_CACHE_SIZE`. **R** = Parquet row group size (rows per group, typically ~400 at default PyArrow settings).

### Dataset RAM (rank 0 only)

The dataset is never fully materialised into a float32 tensor. `ParquetDataset` reads lazily from disk via an LRU row group cache. At any given time, the dataset-related RAM on rank 0 is bounded by the cache contents plus DataLoader prefetch buffers.

| Component | Formula | Estimated size |
|-----------|---------|----------------|
| LRU row group cache | C × R × D × 4 bytes | C=2, R=400, D=7,048 → **~23 MB** |
| DataLoader prefetch buffers | `num_workers` × B × D × 4 bytes | 2 workers, B=1024, D=7,048 → **~58 MB** |
| PyArrow file handle + metadata | Fixed | ~50 MB |
| **Total dataset RAM** | | **~131 MB** |

This is well within the 64 GB SLURM allocation. No eager loading, no memmap file, and no PCA reduction is required.

### Per-GPU memory during training (reduced dataset, D=7,048)

| Component | Formula | H200 |
|-----------|---------|------|
| Model weights | P × 4 B | ~0.23 GB |
| AdamW state | P × 8 B | ~0.46 GB |
| Gradients | P × 4 B | ~0.23 GB |
| Activations (fwd+bkwd) | ~B × layers × 4 B | ~0.5 GB |
| Input batch | B × D × 4 B | ~29 MB |
| CUDA context + fragmentation | Fixed | ~1.5 GB |
| **Total estimated at B=1024** | | **~3.0 GB** |

Adversarial masking adds one extra clone tensor of size B × D × 4 bytes per adversarial batch (~29 MB at B=1024, D=7,048). This is negligible.

### Calibration and inference

| Phase | GPU VRAM | Notes |
|-------|---------|-------|
| `calibrate.py` | ~0.23 GB | Weights only; no optimiser state |
| `inference.py` | ~0.23 GB | Frozen model; `torch.no_grad()` eliminates gradient tensors |

### Recommended allocation summary

| Job | RAM | GPU config | Notes |
|-----|-----|-----------|-------|
| `reward_job.sh` | 64 GB | 1 × H200 | Dataset loaded lazily; ~131 MB dataset RAM |
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

Y2 = NaN for deceased patients is enforced upstream by `extract_y_data.py` and validated by `Mimic4DataLoader._validate_labels()`. The dynamic NaN mask in `RewardModelManager.compute_loss()` handles this per batch without removing rows from the dataset.

### 11.2 Feature Slot Summary

| Slots | Count | Always visible | Dim each | Total dims (EMBEDDING_DIM = 768) | Total dims (EMBEDDING_DIM = 128) |
|-------|-------|----------------|----------|-------------------------------|-------------------------------|
| F1 (`demographic_vec`) | 1 | Yes | 8 | 8 | 8 |
| F2–F5 (history/triage embeddings) | 4 | Yes | EMBEDDING_DIM | 3,072 | 512 |
| F6–F18 (lab group embeddings) | 13 | No (maskable) | EMBEDDING_DIM | 9,984 | 1,664 |
| F19 (radiology embedding) | 1 | No (maskable) | EMBEDDING_DIM | 768 | 128 |
| F20–F56 (microbiology panel embeddings) | 37 | No (maskable) | EMBEDDING_DIM | 28,416 | 4,736 |
| **Total** | **56** | **5 always visible** | | **42,248** | **7,048** |

`NUM_ALWAYS_VISIBLE_FEATURES = 5`. Maskable slot count M = 51.

### 11.3 Default Config Values

| Key | MIMIC-IV Default | Notes |
|-----|-----------------|-------|
| `NUM_TARGETS` | `2` | Y1 mortality + Y2 readmission |
| `EMBEDDING_DIM` | `768` | Matches full dataset; set to 128 for reduced dataset |
| `INPUT_DIM` | `42248` (full) / `7048` (reduced) | Derived at runtime; validation only |
| `NUM_ALWAYS_VISIBLE_FEATURES` | `5` | F1–F5 positional |
| `LOSS_WEIGHT_Y1` | `0.75` | 3:1 ratio favouring mortality |
| `LOSS_WEIGHT_Y2` | `0.25` | |
| `LAYER_WIDTHS` | `[8192, 2048, 512, 128]` | Gradual funnel from D=42,248 (full) or proportionally generous for reduced |
| `DROPOUT_RATES` | `[0.4, 0.3, 0.3, 0.2]` | Heavier on wider early layers |

---

> See `reward_model_architecture.md` for system context, design rationale, pipeline overview, and the upstream pipeline that produces `full_cdss_dataset.parquet`.
