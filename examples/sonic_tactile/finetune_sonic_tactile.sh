#!/usr/bin/env bash
# Tactile (skin-suit) finetune for the unitree_g1_sonic embodiment.
#
# Wraps examples/finetune.sh with:
#   --modality-config-path  -> adds the `tactile` modality + fixes the stereo
#                              video keys (see examples/sonic_tactile/modality_config.py)
#   -- --use_tactile         -> builds the tactile encoder + touch-dreaming heads
#                              and adds the touch-dreaming auxiliary loss
#
# Requirements:
#   * HF_TOKEN exported (base model nvidia/GR00T-N1.7-3B, ~6GB, auto-downloads on
#     first run; it is NOT gated). Or set BASE_MODEL_PATH to a local checkpoint.
#   * One GPU. Override which one with CUDA_VISIBLE_DEVICES.
#
# Tunable via env (sensible defaults shown):
#   CUDA_VISIBLE_DEVICES=1 MAX_STEPS=10000 GLOBAL_BATCH_SIZE=16 USE_WANDB=0 \
#   OUTPUT_DIR=... BASE_MODEL_PATH=... bash examples/sonic_tactile/finetune_sonic_tactile.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

NUM_GPUS=1 \
USE_WANDB="${USE_WANDB:-0}" \
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}" \
MAX_STEPS="${MAX_STEPS:-10000}" \
SAVE_STEPS="${SAVE_STEPS:-1000}" \
bash examples/finetune.sh \
  --base-model-path "${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}" \
  --dataset-path "${DATASET_PATH:-$REPO/outputs/carry-bucket-stereo}" \
  --embodiment-tag unitree_g1_sonic \
  --modality-config-path "$REPO/examples/sonic_tactile/modality_config.py" \
  --output-dir "${OUTPUT_DIR:-$REPO/outputs/sonic_tactile_ft}" \
  -- --use_tactile
