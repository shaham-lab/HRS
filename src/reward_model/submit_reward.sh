#!/bin/bash
# Submit the reward model training job, then chain the calibration job as an
# afterok dependency so calibration only runs after a successful training run.
# See Architecture §14 (Job Chain).
#
# Usage:
#   bash src/reward_model/submit_reward.sh          # fresh run
#   bash src/reward_model/submit_reward.sh --resume  # resume from checkpoint

set -euo pipefail

cd ~/Python/HRS
mkdir -p logs

RESUME_FLAG="${1:-}"

echo "Submitting training job..."
TRAIN_JOB_ID=$(sbatch --parsable src/reward_model/reward_job.sh ${RESUME_FLAG})
echo "  Training job submitted: ${TRAIN_JOB_ID}"

echo "Submitting calibration job (afterok:${TRAIN_JOB_ID})..."
CALIB_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:"${TRAIN_JOB_ID}" \
    src/reward_model/calibrate_job.sh)
echo "  Calibration job submitted: ${CALIB_JOB_ID}"

echo ""
echo "Job chain:"
echo "  [${TRAIN_JOB_ID}] reward_job.sh"
echo "  [${CALIB_JOB_ID}] calibrate_job.sh  (runs after ${TRAIN_JOB_ID} succeeds)"
echo ""

echo "log:  reward_train_${TRAIN_JOB_ID}.err"
