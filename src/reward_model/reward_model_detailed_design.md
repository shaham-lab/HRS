CDSS-ML Reward Model — Detailed Design

See reward_model_architecture(2).md for the high-level architecture, target definitions, and hardware execution model.
Table of Contents

1. Identifier Hierarchy

The reward model dataset is flat (one row per admission), but maintains identifiers for traceability:

    hadm_id: Primary key for all records. Used for index alignment during tensor creation.

    subject_id: Patient identifier. Used upstream for data splits; retained here purely for audit and RL trajectory alignment.

Null handling: Neither hadm_id nor subject_id can be null. Any missing IDs at load time raise a ValueError.
2. Module Decomposition

Module Boundaries:

    Data loading (load_dataset.py), the neural network (model.py), and masking logic (masking.py) are strictly decoupled. The network has no awareness of masking; the masking logic has no awareness of network weights.

    Training loop logic (train.py) orchestrates the components and handles all PyTorch DistributedDataParallel (DDP) synchronization.

Class vs. Script:

    Classes: Used only for RewardModel (maintains PyTorch nn.Module state) and FeatureMasker (if maintaining schedule state).

    Scripts: All executable steps (train.py, calibrate.py, validate_contract.py) are plain scripts with a run(config) entry point.

Encapsulation:

    Configuration is passed as a pure dictionary to run(config). Modules never call yaml.safe_load.

    Inter-module data exchange (e.g., passing the trained model to calibrate.py) happens via .pt files on disk, not in-memory object passing.

3. Cross-Cutting Concerns

Configuration:
Loaded once by the entry point script using yaml.safe_load(open(args.config)).
Required keys are validated at the top of run(config):

Logging:
Each module acquires a logger via logger = logging.getLogger(__name__).
Log format is enforced by the orchestrator:
%(asctime)s  %(levelname)-8s  %(name)s  %(message)s
Inner loops (e.g., batch iteration) use tqdm with leave=False. Outer loops (e.g., epochs) use INFO logging.

Shared Utilities (utils.py):

    set_seed(seed: int): Forces deterministic behavior across PyTorch, NumPy, and random.

    setup_ddp(): Initializes the NCCL process group and returns local rank.

    cleanup_ddp(): Destroys the process group.

4. Data Contract & Feature Mapping

This domain logic is required by all downstream modules.

The Y2 NaN Contract:
The dataset must encode deceased patients (y1_mortality == 1) with y2_readmission = NaN (float32). This is non-negotiable. It ensures the loss function computes conditional readmission dynamically.

Feature Index Map Discovery:
The 42,248-dimensional input vector is assembled dynamically at load time. The boundaries of each feature slot (for masking) are derived by scanning the parquet columns:

    Read demographic_vec → assign indices [0:8].

    Iterate through all columns ending in _embedding in the exact order they appear in the parquet.

    Assign each a 768-width block (e.g., diag_history_embedding → [8:776]).

    Return a dictionary: feature_index_map: dict[str, tuple[int, int]].

5. Module Implementation
validate_contract.py

Validates the input dataset against hard constraints without allocating GPU memory or starting a training loop.

Algorithm:

    Load final_cdss_dataset.parquet using pyarrow.parquet / pandas.

    Assert y1_mortality contains only 0 or 1.

    Assert y2_readmission is NaN for all rows where y1_mortality == 1.

    Assert y2_readmission is not NaN for all rows where y1_mortality == 0.

    Assert all *_embedding columns exist and are float32 arrays of length 768.

    Exit 0 if all pass; raise AssertionError with descriptive text if any fail.

load_dataset.py

Converts the parquet dataset into PyTorch tensors and computes class weights.

Algorithm:

    Load dataset; separate into Train, Dev, Test based on the split column.

    Build feature_index_map via column scanning (see Section 4).

    Horizontally concatenate demographic_vec and all 55 *_embedding columns into X_train (float32, shape [N, 42248]). Repeat for Dev/Test.

    Extract y1_train, y2_train.

    Compute class weights exclusively from the Train split:

        pos_weight_y1 = (Count y1==0) / (Count y1==1)

        pos_weight_y2 = (Count y2==0 where y1==0) / (Count y2==1 where y1==0)

    Return X, Y, feature_index_map, and pos_weights.

model.py

Defines the RewardModel class (inherits from nn.Module).

Algorithm:

    Build nn.Sequential block for the shared MLP funnel:

        Linear(42248, 8192) → BatchNorm1d → ReLU → Dropout

        Linear(8192, 2048) → BatchNorm1d → ReLU → Dropout

        Linear(2048, 512) → BatchNorm1d → ReLU → Dropout

        Linear(512, 128) → BatchNorm1d → ReLU → Dropout

    Build two independent output heads:

        head_y1 = Linear(128, 1)

        head_y2 = Linear(128, 1)

    forward(x): Pass x through shared MLP, return head_y1(x) and head_y2(x) as raw logits.

masking.py

Contains functional logic for zeroing out feature slots.

Algorithm:

    apply_random_mask(X, feature_index_map, k):

        Randomly select k keys from feature_index_map (excluding demographic_vec F1).

        For each selected key, look up (start, end) indices and set X[:, start:end] = 0.0.

    apply_adversarial_mask(X, feature_index_map, grads):

        grads is ∂L/∂X (shape [batch_size, 42248]).

        For each feature slot, compute L2 norm of gradients over its index range.

        Identify the slot with the highest norm per sample.

        Zero out the highest-norm slot for each sample independently.

loss.py

Computes the weighted dual-head loss with dynamic NaN masking.

Algorithm:

    Compute loss_y1 = BCEWithLogitsLoss(logits_y1, y1, pos_weight=pos_weight_y1).

    Compute survivor_mask = ~torch.isnan(y2).

    If survivor_mask.sum() == 0: loss_y2 = torch.tensor(0.0).

    Else: loss_y2 = BCEWithLogitsLoss(logits_y2[survivor_mask], y2[survivor_mask], pos_weight=pos_weight_y2).

    Return (w1 * loss_y1) + (w2 * loss_y2).

train.py

Main DDP execution script.

Algorithm:

    Initialize NCCL process group.

    Call load_dataset.py (each GPU loads data into RAM, but uses DistributedSampler to shard mini-batches).

    Initialize RewardModel, wrap in DistributedDataParallel.

    Initialize AdamW optimizer and CosineAnnealingLR scheduler.

    Loop over Epochs:

        Compute current curriculum probabilities (Random, Adversarial, None) via sigmoid interpolation.

        Loop over mini-batches:

            Sample masking mode.

            If Adversarial: compute gradient norms via first pass (see Section 6), apply mask.

            If Random: apply random mask.

            Zero grad → Forward pass → loss.py → Backward pass → Optimizer step.

        End of epoch: Run validation on Dev split (No Masking).

    Rank 0 checks early stopping and writes best_model.pt and training_metrics.parquet.

    Cleanup DDP.

calibrate.py

Learns temperature scaling parameters T_y1, T_y2 on the Dev split.

Algorithm:

    Load best_model.pt on a single GPU. Set to eval() mode.

    Forward pass full Dev split (No Masking) to get raw logits.

    Define T_y1 = nn.Parameter(torch.ones(1)) and T_y2 = nn.Parameter(torch.ones(1)).

    Use LBFGS optimizer to minimize Negative Log Likelihood loss against ground truth y1 and y2 (using survivor_mask for y2).

    Write calibration_params.json containing the learned floats.

inference.py

Class wrapper provided to the RL agent.

Algorithm:

    Load best_model.pt and calibration_params.json.

    predict(X):

        Move X to device.

        logits_y1, logits_y2 = model(X)

        Return sigmoid(logits_y1 / T_y1) and sigmoid(logits_y2 / T_y2).

6. Adversarial Masking under DDP

Problem: Standard PyTorch DDP automatically synchronizes gradients (all-reduce) during the backward() call. Adversarial masking requires a "dummy" backward pass just to find gradient norms before applying the mask. If DDP syncs during this dummy pass, it wastes network bandwidth and corrupts the optimizer state.
Solution: Use the model.no_sync() context manager to disable gradient communication for the first pass.
7. Final Output / Assembly

The final artifact used by the RL agent requires two files:

best_model.pt structure:
8. Configuration Reference
9. Memory Requirements

Calculated for float32 precision. 4 bytes per float.

Hardware Note: A 24 GB GPU (L4, A100, or equivalent) is strictly required to accommodate the 14-18 GB VRAM footprint per worker during adversarial masking (which requires retaining the computational graph across passes).