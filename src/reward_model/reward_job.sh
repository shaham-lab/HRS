#!/bin/bash
#SBATCH --job-name=reward_train
#SBATCH --output=logs/reward_train_%j.out
#SBATCH --error=logs/reward_train_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --mail-user=eli.kazum@biu.ac.il
#SBATCH --mail-type=END,FAIL

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Start: $(date)"

nvidia-smi

cd ~/Python/HRS
mkdir -p logs

# Load cluster modules — adjust module names for your cluster.
# module load cuda/12.x python/3.11

# Activate conda environment.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hrs
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/hrs/lib:$LD_LIBRARY_PATH"
export NCCL_BUFFSIZE=268435456
export NCCL_DEBUG=INFO



# Launch DDP training.
# --nproc_per_node must equal the number of GPUs requested above (--gres=gpu:N).
# --rdzv_backend=c10d uses the built-in rendezvous; no external store required.
# "$@" forwards any arguments passed to sbatch (e.g. --resume).
torchrun \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    src/reward_model/reward_model_main.py \
    --config config/reward_model.yaml \
    "$@"

echo "End: $(date)"