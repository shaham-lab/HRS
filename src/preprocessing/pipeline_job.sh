#!/bin/bash
#SBATCH --job-name=hrs_preprocessing
#SBATCH --output=logs/hrs_preprocessing_%j.out
#SBATCH --error=logs/hrs_preprocessing_%j.err
#SBATCH --partition=L4-12h
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=eli.kazum@biu.ac.il
#SBATCH --mail-type=END,FAIL

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Start: $(date)"

nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hrs

cd ~/Python/HRS/
mkdir -p logs

# Run all pipeline steps except embed_features
# embed_features runs separately via embed_job.sh with 2 GPUs
python ./src/preprocessing/run_pipeline.py \
    --config config/preprocessing.yaml \
    --all \
    --skip-modules embed_features

echo "End: $(date)"
