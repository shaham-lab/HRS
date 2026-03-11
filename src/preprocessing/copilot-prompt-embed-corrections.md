# Correction Prompt: Fix `embed_features.py` and Shell Scripts

The previously generated `embed_features.py`, `check_embed_status.py`, and `submit_all.sh`
contain several bugs. Apply the following targeted corrections without changing any logic
that is already correct.

---

## Fix 1 — Multi-GPU: slice-split BEFORE LPT, not instead of it

**Problem:** The current code likely assigns whole features to GPUs via LPT, with each
GPU worker seeing the full slice (~40k admissions). This is wrong — both workers end up
loading the full slice, doubling memory, and the slice is never divided.

**Correct design:** Two separate steps:

**Step A — split the slice's `hadm_ids` evenly between workers:**
```python
slice_hadm_ids = sorted(slice_hadm_ids)   # deterministic order
mid = len(slice_hadm_ids) // 2
gpu_hadm_ids = [
    set(slice_hadm_ids[:mid]),             # GPU 0 gets first half (~20k)
    set(slice_hadm_ids[mid:]),             # GPU 1 gets second half (~20k)
]
```

**Step B — for each GPU worker, apply LPT to balance the 18 features across its own
`hadm_id` half.** Each worker receives its own `hadm_id` set and a task list that covers
all 18 features, but each feature's texts are already pre-filtered to that worker's
`hadm_id` set before computing LPT costs:

```python
# Build per-GPU task lists using LPT on each worker's own text counts
for g in range(n_gpus):
    worker_tasks = []
    for feature in all_features:
        texts = {hid: text_map[hid] for hid in gpu_hadm_ids[g] if hid in text_map}
        cost  = len(texts) * feature["max_length"]
        worker_tasks.append({**feature, "texts": texts, "cost": cost})

    worker_tasks.sort(key=lambda t: t["cost"], reverse=True)
    # LPT is not needed across workers here — each worker processes all 18 features
    # for its own hadm_id half. The purpose is ordering within a worker so the most
    # expensive features are started first (better progress visibility).
    gpu_task_lists[g] = worker_tasks
```

> Note: with 2 GPU workers each processing all 18 features on their own ~20k hadm_id
> half, LPT ordering within a worker is optional but good practice. The key correctness
> requirement is that **each worker only embeds its own hadm_id half** — the two workers
> never touch the same admission rows.

Each worker appends its results independently to the shared output parquets. Because
slices run sequentially (enforced by SLURM `--dependency=afterok`), there is no
concurrent write conflict between slices. Within a single slice job, the two workers
must write their per-feature results sequentially (worker 0 finishes a feature, then
worker 1 appends, or vice versa) — or write to separate temporary files and merge.
The simplest safe approach: worker 0 writes first, then the main process signals
worker 1 to append, using a `multiprocessing.Barrier` or by running workers sequentially
per feature. **Do not let both workers append to the same parquet file simultaneously.**

---

## Fix 2 — Atomic writes apply only to the FIRST write; appends are direct

**Problem:** The current code may apply `.tmp` → `os.replace()` to every checkpoint
append, which corrupts fastparquet's row-group index on subsequent appends.

**Correct rule:**
- **First write for a feature** (output parquet does not yet exist): write to
  `{output_path}.tmp`, then `os.replace(tmp_path, output_path)`. This protects against
  a killed job leaving a half-written file that passes the existence check.
- **Subsequent checkpoint appends** (output parquet already exists and is valid):
  call `fastparquet.write(..., append=True)` directly. No `.tmp` needed — a partial
  append leaves an incomplete row group that fastparquet will ignore on the next read.

```python
import fastparquet as fp
from pathlib import Path

def _append_batch(output_path: Path, df: pd.DataFrame) -> None:
    tmp_path = output_path.with_suffix(".tmp")
    if not output_path.exists():
        # First write — use atomic rename
        fp.write(str(tmp_path), df, compression="snappy")
        os.replace(tmp_path, output_path)
    else:
        # Subsequent appends — direct, no rename
        fp.write(str(output_path), df, compression="snappy", append=True)
```

---

## Fix 3 — Dynamic batch size formula: cap the scaling

**Problem:** The formula `effective_batch_size = base_batch_size * (8192 // max_length_cap)`
produces absurdly large batches for short-capped features. For `chief_complaint`
(cap=64): `32 * (8192 // 64) = 32 * 128 = 4096` — this will OOM on any GPU.

**Correct formula:** Scale relative to a reference length of 512 tokens, with a maximum
cap of `4 * base_batch_size`:

```python
_REFERENCE_LENGTH = 512

def _effective_batch_size(base: int, max_length_cap: int) -> int:
    scaled = base * (_REFERENCE_LENGTH // max(max_length_cap, 1))
    return max(1, min(scaled, base * 4))   # never exceed 4× base
```

This gives sensible values across the full range of caps:

| Feature             | Cap   | base=32 → effective |
|---------------------|-------|---------------------|
| `chief_complaint`   | 64    | min(256, 128) = 128 |
| `triage`            | 256   | min(64, 128)  = 64  |
| `diag_history`      | 512   | min(32, 128)  = 32  |
| `radiology`         | 1024  | min(16, 128)  = 16  |
| all lab groups      | 2048  | min(8, 128)   = 8   |
| `discharge_history` | 4096  | min(4, 128)   = 4   |

---

## Fix 4 — `check_embed_status.py`: derive expected parquet list from `lab_panel_config.yaml`

**Problem:** The current code likely hardcodes 18 parquet names or a fixed count of 18.
The number of lab groups is dynamic and must be read from `lab_panel_config.yaml`.

**Correct implementation:**

```python
import yaml, sys, os
import pandas as pd

def get_expected_parquet_names(config: dict) -> list[str]:
    """Return the list of expected embedding parquet filenames."""
    # Fixed text features (always 5)
    names = [
        "diag_history_embeddings.parquet",
        "discharge_history_embeddings.parquet",
        "triage_embeddings.parquet",
        "chief_complaint_embeddings.parquet",
        "radiology_embeddings.parquet",
    ]
    # Dynamic lab groups from lab_panel_config.yaml
    lab_config_path = os.path.join(
        config["CLASSIFICATIONS_DIR"], "lab_panel_config.yaml"
    )
    if not os.path.exists(lab_config_path):
        print(f"ERROR: lab_panel_config.yaml not found at {lab_config_path}", file=sys.stderr)
        sys.exit(2)
    with open(lab_config_path) as f:
        lab_groups = yaml.safe_load(f)
    for group_name in lab_groups:
        names.append(f"lab_{group_name}_embeddings.parquet")
    return names
```

Then use `get_expected_parquet_names(config)` wherever you currently have a hardcoded
list or count of 18.

---

## Fix 5 — `submit_all.sh`: compute `n_slices` dynamically from config

**Problem:** The loop `for i in $(seq 0 13)` hardcodes 14 slices. If
`BERT_SLICE_SIZE_PER_GPU` is changed in `preprocessing.yaml`, the script submits the
wrong number of jobs.

**Correct implementation:** compute `n_slices` at runtime by calling a small Python
helper before the loop:

```bash
# Compute number of slices from config at runtime
N_SLICES=$(python3 - <<'EOF'
import yaml, math, sys
with open("config/preprocessing.yaml") as f:
    cfg = yaml.safe_load(f)
slice_size = cfg.get("BERT_SLICE_SIZE_PER_GPU", 20000)
n_gpus     = cfg.get("BERT_MAX_GPUS", 2) or 2
import pandas as pd
splits = pd.read_parquet(cfg["PREPROCESSING_DIR"] + "/data_splits.parquet",
                         columns=["hadm_id"])
total = splits["hadm_id"].nunique()
print(math.ceil(total / (slice_size * n_gpus)))
EOF
)

if [ -z "$N_SLICES" ] || [ "$N_SLICES" -lt 1 ]; then
    echo "ERROR: could not compute n_slices" >&2
    exit 1
fi

echo "Submitting $N_SLICES embed slice jobs..."
for i in $(seq 0 $(( N_SLICES - 1 ))); do
    # ... submit embed_job.sh $i with dependency chain
done
```

> If `data_splits.parquet` does not yet exist (status code 2 path), `N_SLICES` cannot be
> computed from it. In that case, fall back to computing from config alone using the
> known total of 546,028:
> `n_slices = ceil(546028 / (BERT_SLICE_SIZE_PER_GPU * n_gpus))`
> or simply defer the slice loop submission until after the pipeline job completes by
> using a second `submit_all.sh` invocation as a SLURM job step.

---

## Fix 6 — `BERT_FORCE_REEMBED` must bypass all three resume levels

**Problem:** The current code may not implement `BERT_FORCE_REEMBED` at all, or may only
apply it at one level.

**Correct behaviour:** when `config.get("BERT_FORCE_REEMBED", False)` is `True`:
- Skip the feature-level existence check entirely (always proceed)
- Skip the record-level `already_done` set (embed all rows in the slice, not just pending)
- Delete any existing `.tmp` file before starting

```python
force = config.get("BERT_FORCE_REEMBED", False)

# Feature-level check
if not force and output_path.exists():
    existing_ids = set(pd.read_parquet(output_path, columns=["hadm_id"])["hadm_id"])
    if slice_hadm_ids.issubset(existing_ids):
        logger.info("[SKIP slice=%d feature=%s]", slice_index, feature_name)
        continue

# Record-level resume
if not force and output_path.exists():
    existing_ids = set(pd.read_parquet(output_path, columns=["hadm_id"])["hadm_id"])
    already_done = existing_ids & slice_hadm_ids
else:
    already_done = set()

pending_ids = slice_hadm_ids - already_done
```

---

## Fix 7 — `pipeline_job.sh`: remove `--gpus=0`

**Problem:** Some SLURM versions reject `--gpus=0` as an invalid directive.

**Fix:** Remove the `--gpus` directive entirely from `pipeline_job.sh` — omitting it
means no GPU is requested, which is the correct behaviour for a CPU-only job.

```bash
#SBATCH --partition=L4-12h
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
# No --gpus directive — this is a CPU-only job
```

---

## Fix 8 — Suppress HuggingFace warnings

**Problem:** The current code likely does not suppress HuggingFace's verbose startup
logging, which pollutes SLURM job logs.

**Fix:** Add the following immediately after imports, before any model loading:

```python
import transformers
import transformers.utils.logging as hf_logging
hf_logging.set_verbosity_error()
```

---

## Summary of changes by file

| File | Fixes to apply |
|------|---------------|
| `embed_features.py` | Fix 1 (slice-split before LPT), Fix 2 (atomic write scope), Fix 3 (batch size formula), Fix 6 (BERT_FORCE_REEMBED), Fix 8 (HF logging) |
| `check_embed_status.py` | Fix 4 (dynamic parquet list from lab_panel_config.yaml) |
| `submit_all.sh` | Fix 5 (dynamic n_slices) |
| `pipeline_job.sh` | Fix 7 (remove --gpus=0) |
| `embed_job.sh` | No changes needed |
| `combine_job.sh` | No changes needed |
