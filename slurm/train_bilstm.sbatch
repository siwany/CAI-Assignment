#!/bin/bash
#
# Ready to run.
# Submits a Part B BiLSTM training run on PACE-ICE using course-provided `train.py` CLI.
#
# Usage: sbatch slurm/train_bilstm.sbatch <train_fraction>
#   e.g. sbatch slurm/train_bilstm.sbatch 0.25

#SBATCH --job-name=a1a-bilstm
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

module load anaconda3
conda activate CAI_SO_FUN

FRACTION="${1:-1.00}"
DROPOUT="${2:-0.0}"
HIDDEN_DIM="${3:-128}"

RUN_NAME="bilstm_f${FRACTION}_d${DROPOUT}_h${HIDDEN_DIM}"

echo "Training configuration:"
echo "  fraction   = ${FRACTION}"
echo "  dropout    = ${DROPOUT}"
echo "  hidden_dim = ${HIDDEN_DIM}"
echo "  run_name   = ${RUN_NAME}"

python train.py \
    --model bilstm \
    --train-fraction "${FRACTION}" \
    --checkpoint-dir "checkpoints/${RUN_NAME}" \
    --seed 42 \
    --dropout "${DROPOUT}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --tensorboard-logdir "runs/${RUN_NAME}"
