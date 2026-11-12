#!/usr/bin/env bash
# Predict the evaluation images with a trained run and submit to Kaggle (after a y/N confirmation).
#
#   scripts/submit.sh RUN "message"            e.g. scripts/submit.sh final_a "resnet_a 200 epochs"
#   scripts/submit.sh RUN "message" --dry-run  write and validate submission.csv without submitting
#   TTA=flip_shift scripts/submit.sh ...       test-time views: none, flip (default) or flip_shift
#
# Needs runs/RUN/final.pt locally (python scripts/colab.py fetch RUN) and a signed-in Kaggle CLI
# (kaggle auth login). Each team gets 5 submissions per day.
set -euo pipefail

COMPETITION=cifar-10-res-net-architecture-challenge-dl
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}"

run="${1:?usage: scripts/submit.sh RUN \"message\" [--dry-run]}"
message="${2:?missing submission message}"
dry_run="${3:-}"

"$PYTHON" -m src.predict --ckpt "runs/$run/final.pt" --out submission.csv --tta "${TTA:-flip}"

if [ "$dry_run" = "--dry-run" ]; then
  echo "Dry run: submission.csv written, nothing submitted."
  exit 0
fi

read -r -p "Submit submission.csv to Kaggle as \"$message\"? [y/N] " answer
if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
  echo "Not submitted."
  exit 0
fi

kaggle competitions submit "$COMPETITION" -f submission.csv -m "$message" --wait
kaggle competitions submissions "$COMPETITION" | head -8
