#!/bin/bash
#SBATCH --job-name=hrs_reduce
#SBATCH --output=logs/hrs_reduce_%j.out
#SBATCH --error=logs/hrs_reduce_%j.err
#SBATCH --partition=cpu1T-24h
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --mail-user=eli.kazum@biu.ac.il
#SBATCH --mail-type=END,FAIL

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hrs

cd ~/Python/HRS/
mkdir -p logs

# Run only reduce_dataset — full_cdss_dataset.parquet already written by combine_job.sh
python ./src/preprocessing/run_pipeline.py \
    --config config/preprocessing.yaml \
    --reduce_dataset

echo "End: $(date)"
