# Refactor: True Parallel GPU Execution in `embed_features.py`

## Problem

Currently `_worker` processes are spawned **sequentially** — worker 0 is joined
(fully waited for) before worker 1 is spawned. This means only one GPU is active
at a time, giving no parallelism benefit from the 2-GPU allocation:

```python
# CURRENT — sequential, wrong
for rank, (device, partition) in enumerate(zip(devices, partitions)):
    p = ctx.Process(target=_worker, ...)
    p.start()
    p.join()   # ← blocks until worker N fully finishes before spawning N+1
```

Both workers write to the same output parquets, which is why sequential
execution was used — to avoid concurrent fastparquet append conflicts.

## Solution

Run both workers **in parallel** by having each worker write to its own
per-worker temporary parquet file. The main process merges the per-worker
files into the final output parquets after all workers complete.

```
Worker 0 → {output_path}.worker0   (parallel)
Worker 1 → {output_path}.worker1   (parallel)
    ↓ both finish
Main process merges → {output_path}  (atomic rename)
```

---

## Changes required

### 1. `_worker()` — write to per-worker temp path instead of shared output path

In `_worker()`, every reference to `output_path` for writing must be replaced
with a per-worker path: `worker_path = output_path + f".worker{rank}"`.

Reading (for resume checks) still uses the final `output_path`.

**Find the section that computes `output_path` and add `worker_path`:**
```python
output_path   = task["output_path"]
embedding_col = task["embedding_col"]
# ADD THIS:
worker_path   = output_path + f".worker{rank}"
```

**Feature-level resume check — reads from final output_path (unchanged):**
```python
# Resume check still reads the merged final output
already_done: set = set()
if os.path.exists(output_path) and not force_reembed:
    try:
        existing = pd.read_parquet(output_path, columns=["hadm_id"])
        already_done = set(existing["hadm_id"].tolist()) & slice_hadm_ids
    except Exception:
        already_done = set()
```

**Checkpoint writes — use `worker_path` instead of `output_path`:**

Replace:
```python
is_first_write = not os.path.exists(output_path)
```
With:
```python
is_first_write = not os.path.exists(worker_path)
```

Replace the atomic first-write block:
```python
# BEFORE
if is_first_write:
    tmp_path = output_path + ".tmp"
    fp.write(tmp_path, batch_sh, compression="snappy")
    os.replace(tmp_path, output_path)
    is_first_write = False
else:
    fp.write(output_path, batch_sh, compression="snappy", append=True)
```
With:
```python
# AFTER — write to per-worker path
if is_first_write:
    tmp_path = worker_path + ".tmp"
    fp.write(tmp_path, batch_sh, compression="snappy")
    os.replace(tmp_path, worker_path)
    is_first_write = False
else:
    fp.write(worker_path, batch_sh, compression="snappy", append=True)
```

**`result_queue` payload — include the worker_path so the main process knows
where to find each worker's output:**

Replace:
```python
result_queue.put({"rank": rank, "completed": completed, "failed": failed})
```
With:
```python
result_queue.put({
    "rank": rank,
    "completed": completed,   # list of final output_paths (not worker_paths)
    "failed": failed,
    "worker_paths": {          # map from output_path → worker_path for merging
        task["output_path"]: task["output_path"] + f".worker{rank}"
        for task in feature_tasks
        if task["output_path"] in [p for p, _ in failed] or
           task["output_path"] in completed
    }
})
```

Simpler alternative — just have the main process reconstruct worker paths
from the known pattern `output_path + f".worker{rank}"` rather than passing
them through the queue. Either approach is acceptable.

---

### 2. `run()` — spawn all workers in parallel, then join all

**Replace the sequential spawn loop:**
```python
# BEFORE — sequential
results = []
for rank, (device, partition) in enumerate(zip(devices, partitions)):
    p = ctx.Process(target=_worker, ...)
    p.start()
    p.join()          # ← blocks here
    result = result_queue_mp.get()
    results.append(result)
```

**With a parallel spawn + join-all pattern:**
```python
# AFTER — parallel
processes = []
for rank, (device, partition) in enumerate(zip(devices, partitions)):
    p = ctx.Process(
        target=_worker,
        args=(rank, device, partition, config, result_queue_mp, slice_index),
        name=f"embed-worker-{rank}",
    )
    p.start()
    logger.info(
        "Spawned worker %d on %s (pid=%d) — %d admissions, %d features",
        rank, device, p.pid, len(gpu_hadm_ids[rank]), len(partition),
    )
    processes.append(p)

# Wait for all workers to finish
for p in processes:
    p.join()
    logger.info("Worker %s (pid=%d) finished with exit code %d",
                p.name, p.pid, p.exitcode)

# Drain the result queue
results = []
while not result_queue_mp.empty():
    results.append(result_queue_mp.get())
results.sort(key=lambda r: r["rank"])
```

---

### 3. `run()` — add merge step after all workers complete

After collecting all results, add a merge step that combines per-worker
parquets into the final output parquets. This runs in the main process,
sequentially across features (no concurrency needed here).

Add this after the results collection, before the summary logging:

```python
# ------------------------------------------------------------------ #
# Merge per-worker parquets into final output parquets               #
# ------------------------------------------------------------------ #
logger.info("Merging per-worker parquets into final outputs…")

# Collect all unique output paths from successful tasks
all_output_paths = set()
for r in results:
    for path in r["completed"]:
        all_output_paths.add(path)

for output_path in sorted(all_output_paths):
    worker_paths = [
        output_path + f".worker{rank}"
        for rank in range(n_gpus)
        if os.path.exists(output_path + f".worker{rank}")
    ]

    if not worker_paths:
        logger.warning("No worker files found for %s — skipping merge",
                       os.path.basename(output_path))
        continue

    if len(worker_paths) == 1:
        # Single worker (or only one succeeded) — rename directly
        tmp_path = output_path + ".tmp"
        os.replace(worker_paths[0], tmp_path)
        os.replace(tmp_path, output_path)
    else:
        # Merge all worker parquets
        dfs = [pd.read_parquet(wp) for wp in worker_paths]
        merged = pd.concat(dfs, ignore_index=True)
        tmp_path = output_path + ".tmp"
        merged.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, output_path)
        # Clean up worker files
        for wp in worker_paths:
            os.remove(wp)

    logger.info("  Merged %d worker file(s) → %s (%d rows)",
                len(worker_paths), os.path.basename(output_path),
                len(pd.read_parquet(output_path, columns=["hadm_id"])))
```

---

### 4. Resume: handle interrupted merge on restart

On startup of `_worker`, before the feature loop, clean up any stale
`.worker{rank}` files for features whose final output already exists and
is complete (i.e. feature-level resume would skip them). This prevents a
re-run from attempting to merge a stale per-worker file with new data.

Add at the top of the `_worker` feature loop, inside the `for task in feature_tasks:` block, after computing `slice_hadm_ids`:

```python
# Clean up stale per-worker temp if final output is already complete
worker_path = output_path + f".worker{rank}"
if os.path.exists(worker_path) and not force_reembed:
    if os.path.exists(output_path):
        try:
            existing = pd.read_parquet(output_path, columns=["hadm_id"])
            if slice_hadm_ids.issubset(set(existing["hadm_id"].tolist())):
                os.remove(worker_path)
                worker_logger.info(
                    "[GPU %d] Removed stale worker temp for completed feature %s",
                    rank, embedding_col,
                )
        except Exception:
            pass  # if we can't read the final output, leave the temp alone
```

---

## What NOT to change

- `_embed_texts()` — unchanged
- `_build_feature_tasks()` — unchanged
- `_effective_batch_size()` — unchanged
- The single-GPU / CPU path — it already runs in-process with no multiprocessing;
  leave it as-is (no per-worker files needed for single-GPU)
- SLURM scripts — no changes needed
- The feature-level and record-level resume logic that **reads** from
  `output_path` — only the **write** side changes to `worker_path`

---

## Summary of all write-path changes

| Location | Before | After |
|----------|--------|-------|
| `is_first_write` check | `not os.path.exists(output_path)` | `not os.path.exists(worker_path)` |
| Atomic first write `.tmp` | `output_path + ".tmp"` | `worker_path + ".tmp"` |
| `fp.write` first write | `fp.write(tmp_path, ...)` → `os.replace(tmp, output_path)` | `fp.write(tmp_path, ...)` → `os.replace(tmp, worker_path)` |
| `fp.write` append | `fp.write(output_path, ..., append=True)` | `fp.write(worker_path, ..., append=True)` |
| Worker spawn loop | Sequential: `p.start(); p.join()` | Parallel: spawn all, then join all |
| After workers finish | Nothing | Merge `worker_path × n_gpus` → `output_path` |
