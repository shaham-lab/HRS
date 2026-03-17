#!/bin/bash
#SBATCH --job-name=hrs_combine
#SBATCH --output=logs/hrs_combine_%j.out
#SBATCH --error=logs/hrs_combine_%j.err
#SBATCH --partition=L4-12h
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=eli.kazum@biu.ac.il
#SBATCH --mail-type=END,FAIL

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hrs

cd ~/Python/HRS/
mkdir -p logs

# Run only combine_dataset — all embeddings already written by embed_job.sh
python ./src/preprocessing/run_pipeline.py \
    --config config/preprocessing.yaml \
    --modules combine_dataset

echo "End: $(date)"
