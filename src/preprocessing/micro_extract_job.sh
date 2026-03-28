#!/bin/bash
#SBATCH --job-name=hrs_micro_extract
#SBATCH --output=logs/hrs_micro_extract_%j.out
#SBATCH --error=logs/hrs_micro_extract_%j.err
#SBATCH --partition=cpu1T-24h
#SBATCH --cpus-per-task=48
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

python ./src/preprocessing/run_pipeline.py \
    --config config/preprocessing.yaml \
    --extract_microbiology \
    --force

echo "End: $(date)"
