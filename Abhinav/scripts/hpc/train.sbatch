#!/bin/bash
# Train one run on NYU HPC Cloud Bursting, inside the Singularity + overlay environment made by setup_env.sh.
# Submit from an Open OnDemand Jupyter terminal, in the project folder:
#
#   sbatch --job-name=RUN scripts/hpc/train.sbatch <src.train args>
#
# The run writes to runs/RUN/ (train.log, history.json, last.pt, final.pt). It uses --resume, so a requeued or
# resubmitted job continues from runs/RUN/last.pt. Course account and the 1x A100 partition are the defaults; override
# them on the sbatch command line, e.g. --partition=g2-standard-12 for an L4.
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --export=ALL
#SBATCH --output=runs/slurm-%x-%j.out

set -euo pipefail
PROJ=${PROJ:-$SLURM_SUBMIT_DIR}
ENV=${ENV:-$PROJ/env}
ARGS=$(printf '%q ' "$@")

cd "$PROJ"
mkdir -p "runs/$SLURM_JOB_NAME"
echo "$(date) job $SLURM_JOB_ID on $(hostname): $ARGS" >> "runs/$SLURM_JOB_NAME/slurm_jobs.txt"
# Host the queue babysitter here (one pass of scripts/hpc/babysit.sbatch every 5 min): CPU nodes get reclaimed after
# about 30 min, GPU jobs run for hours. One host acts at a time: runs/babysit/host holds its job ID, and a host that
# has not passed for 10 min is taken over.
(
    set +e
    while true; do
        h=runs/babysit/host
        if [ ! -f "$h" ] || [ "$(cat "$h")" = "$SLURM_JOB_ID" ] || [ $(( $(date +%s) - $(stat -c %Y "$h") )) -gt 600 ]; then
            echo "$SLURM_JOB_ID" > "$h"
            sleep 2
            [ "$(cat "$h")" = "$SLURM_JOB_ID" ] && BABYSIT_ONCE=1 bash scripts/hpc/babysit.sbatch >> runs/babysit/hosted.log 2>&1
        fi
        sleep 300
    done
) &
BABYSIT_PID=$!
singularity exec --nv --overlay "$ENV/overlay.ext3:ro" "$ENV/image.sif" /bin/bash -c \
    "source /ext3/env.sh && cd '$PROJ' && python -u -m src.train --out 'runs/$SLURM_JOB_NAME' --resume \
     --num-workers $((SLURM_CPUS_PER_TASK - 1)) $ARGS >> 'runs/$SLURM_JOB_NAME/train.log' 2>&1; \
     echo \$? > 'runs/$SLURM_JOB_NAME/exit_code'"
kill "$BABYSIT_PID" 2>/dev/null || true
