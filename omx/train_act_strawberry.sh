#!/usr/bin/env bash
# ACT training on tkoo/strawberry_train_1013 (lecture 09, section 1.3).
# Episode 0 (598-frame test recording from 1013_2200) is excluded.
# Usage: omx/train_act_strawberry.sh [smoke]
set -euo pipefail

source ~/venv/il/bin/activate
cd ~/il_ws/src/lerobot

DATASET=tkoo/strawberry_train_1013
EPISODES="[$(seq -s, 1 20)]"

if [[ "${1:-}" == "smoke" ]]; then
  OUT=outputs/train/smoke_act; JOB=smoke_act
  EXTRA=(--steps=200 --save_freq=100 --log_freq=20)
else
  OUT=outputs/train/omx_act_strawberry_1013; JOB=act_strawberry_1013
  EXTRA=(--steps=100000 --save_freq=10000)
fi

lerobot-train \
  --dataset.repo_id="${DATASET}" \
  --dataset.root="/srv/il/lerobot_data/${DATASET}" \
  --dataset.episodes="${EPISODES}" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --output_dir="${OUT}" \
  --job_name="${JOB}" \
  --batch_size=16 \
  --save_checkpoint=true \
  "${EXTRA[@]}"
