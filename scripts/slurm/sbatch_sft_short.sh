#!/bin/bash
#SBATCH --job-name=agent-sft-align
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm_%x_%j.out

# Short 4-hour job window maximizes SLURM backfill priority on congested university clusters.
# `run_sft.py` automatically detects and resumes from the latest checkpoint in `--output-dir`.

set -euo pipefail
mkdir -p logs

MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
TASK=${TASK:-"alfworld"}
VARIANT=${VARIANT:-"ours"}
REWRITTEN_JSONL=${REWRITTEN_JSONL:-"outputs/rewritten_data/alfworld_rewritten.jsonl"}
OUT_DIR=${OUT_DIR:-"checkpoints/${TASK}_${VARIANT}"}

torchrun --nproc_per_node=2 scripts/run_sft.py \
  --model "${MODEL}" \
  --task "${TASK}" \
  --variant "${VARIANT}" \
  --rewritten-jsonl "${REWRITTEN_JSONL}" \
  --output-dir "${OUT_DIR}" \
  --per-device-batch-size 2 \
  --grad-accum 4 \
  --save-steps 100
