# Task: Move Preprocessing Shell Scripts into `src/preprocessing/`

Move the four SLURM shell scripts from the project root into `src/preprocessing/`,
alongside the Python modules they invoke. Update all internal `cd` paths accordingly.
Do not change any SBATCH directives, logic, or comments unless specified below.

---

## Step 1 — Move the files

Move (git mv) the following files:

| From | To |
|------|----|
| `pipeline_job.sh` | `src/preprocessing/pipeline_job.sh` |
| `embed_job.sh` | `src/preprocessing/embed_job.sh` |
| `combine_job.sh` | `src/preprocessing/combine_job.sh` |
| `submit_all.sh` | `src/preprocessing/submit_all.sh` |

```bash
git mv pipeline_job.sh src/preprocessing/pipeline_job.sh
git mv embed_job.sh    src/preprocessing/embed_job.sh
git mv combine_job.sh  src/preprocessing/combine_job.sh
git mv submit_all.sh   src/preprocessing/submit_all.sh
```

---

## Step 2 — Update `cd` path in each script

All four scripts currently contain:

```bash
cd ~/Python/HRS/
```

Since the scripts now live two levels deeper (`src/preprocessing/`), the working
directory for the SLURM job must still be `~/Python/HRS/` — the project root where
`config/`, `src/`, and `data/` live. The `cd` line is correct as-is and **must not
change**. No path updates are needed inside the scripts themselves.

> Confirm: grep each moved file for `cd ~/Python/HRS` and verify it is present.

---

## Step 3 — Update `submit_all.sh`: script invocation paths

`submit_all.sh` calls `sbatch` on the other three scripts by filename. After the move,
all four scripts are in the same directory (`src/preprocessing/`), so the calls must
use paths relative to the project root (where the script `cd`s to before running).

Find these `sbatch` calls in `src/preprocessing/submit_all.sh` and update them:

| Old | New |
|-----|-----|
| `sbatch ... pipeline_job.sh` | `sbatch ... src/preprocessing/pipeline_job.sh` |
| `sbatch ... embed_job.sh "$i"` | `sbatch ... src/preprocessing/embed_job.sh "$i"` |
| `sbatch ... combine_job.sh` | `sbatch ... src/preprocessing/combine_job.sh` |

Also update the `check_embed_status.py` invocation path (it is already in
`src/preprocessing/` but may be referenced by a relative path from the old root):

| Old | New |
|-----|-----|
| `python src/preprocessing/check_embed_status.py ...` | unchanged — already correct |

---

## Step 4 — Update `submit_all.sh`: `mkdir -p logs`

The `logs/` directory is relative to wherever the script runs. Since all scripts `cd`
to `~/Python/HRS/` first, `mkdir -p logs` correctly creates `~/Python/HRS/logs/` and
does **not** need to change.

> No action required for this step — confirm only.

---

## Step 5 — Verify nothing else references the old root-level paths

Search the entire repository for any remaining references to the scripts at the project
root and update them if found:

```bash
grep -r "pipeline_job\.sh\|embed_job\.sh\|combine_job\.sh\|submit_all\.sh" \
     --include="*.py" --include="*.sh" --include="*.md" --include="*.yaml" \
     --include="*.txt" .
```

Expected hits after the move:
- `src/preprocessing/submit_all.sh` — the `sbatch` calls updated in Step 3 ✓
- Any README or documentation files — update those paths too if found

---

## Summary of all changes

| File | Change |
|------|--------|
| `pipeline_job.sh` → `src/preprocessing/pipeline_job.sh` | Move only, no content change |
| `embed_job.sh` → `src/preprocessing/embed_job.sh` | Move only, no content change |
| `combine_job.sh` → `src/preprocessing/combine_job.sh` | Move only, no content change |
| `submit_all.sh` → `src/preprocessing/submit_all.sh` | Move + update 3 `sbatch` call paths |
