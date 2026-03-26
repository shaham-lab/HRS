# Exploration Notebooks — How to Run

This guide explains how to execute the MIMIC-IV exploration notebooks either locally (no SLURM) or on the SLURM cluster using A100 GPUs. The SLURM scripts mirror the style used in preprocessing and reward model jobs.

## Notebook list

Located in `notebooks/mimic4/`:

- `mimic4_data_exploration.ipynb` — high-level cohort/visit counts and table coverage.
- `mimic4_chartevents_exploration.ipynb` — vital signs/charted measurements exploration.
- `mimic4_labevents_exploration.ipynb` — laboratory result distributions and missingness.
- `mimic4_microbiology_exploration.ipynb` — cultures, organisms, and sensitivities overview.
- `demographic_feature_exploration.ipynb` — demographics feature ranges and sparsity.
- `notes_exploration.ipynb` — clinical note counts and length/section statistics.

## Local (no SLURM)

Prereqs:
- Conda env `hrs` (or your env with project deps).
- From repo root: `cd /home/runner/work/HRS/HRS`.

Run Jupyter:
```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hrs
jupyter lab notebooks/mimic4
```

Execute a notebook headless:
```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hrs
cd /home/runner/work/HRS/HRS
jupyter nbconvert --to notebook --execute notebooks/mimic4/mimic4_data_exploration.ipynb \
  --output notebooks/mimic4/mimic4_data_exploration_executed.ipynb \
  --ExecutePreprocessor.timeout=-1
```

## SLURM (A100, one GPU per notebook)

Scripts (in `notebooks/mimic4/`):
- `exploration_job.sh`: runs one notebook on 1× A100 via `nbconvert`, saves executed copy to `notebooks/mimic4/executed/`, logs to `logs/`.
- `submit_exploration.sh`: CLI wrapper to submit one job per requested notebook.

### Run all exploration notebooks
```bash
cd ~/Python/HRS
bash notebooks/mimic4/submit_exploration.sh --all
```

### Run selected notebooks
You can pass notebook paths or basenames. Paths are used as given; if you provide only a file
name it is resolved in your current working directory (e.g., `notebooks/mimic4/` if you `cd`
there first):
```bash
cd ~/Python/HRS
bash notebooks/mimic4/submit_exploration.sh notebooks/mimic4/mimic4_data_exploration.ipynb mimic4_labevents_exploration.ipynb
```

Each job:
- Partition: `A100-4h`
- Resources: `--gres=gpu:1`, `--cpus-per-task=4`, `--mem=24G`, `--time=4:00:00`
- Job name: `explore_<notebook_basename>`
- Logs: `logs/explore_<job>.out/err`
- Executed notebook: `notebooks/mimic4/executed/<name>_executed.ipynb`

Monitor:
```bash
squeue -u $USER
```
