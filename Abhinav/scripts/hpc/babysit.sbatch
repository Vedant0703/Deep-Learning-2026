#!/bin/bash
#SBATCH --job-name=h_babysit
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=runs/babysit-%j.out
# Keeps every run listed in runs/jobs.txt queued until it has runs/NAME/final.pt. Cloud nodes that fail to boot get
# their jobs cancelled by Slurm, so this resubmits them (train.sbatch resumes from last.pt). Line format:
#   NAME|SBATCH_OPTS|TRAIN_ARGS|FILES_NEEDED_FIRST      (@C and @T expand to the shared FM-student / teacher flags)
# Without an interactive terminal: one-off commands dropped into runs/inbox/*.sh run once per pass (output in
# runs/inbox/done/NAME.out), and the queue is written to runs/babysit/squeue.txt every pass.
cd /scratch/as21153/dl1
# Spot preemption cancels CPU-partition jobs instead of requeueing them, so hand over to a fresh copy on SIGTERM.
trap 'echo "$(date) SIGTERM: resubmitting myself: $(sbatch --parsable scripts/hpc/babysit.sbatch)"; exit 0' TERM
others=$(squeue --me -h -n h_babysit -t R -o %i | grep -vx "$SLURM_JOB_ID")
if [ -n "$others" ]; then echo "$(date) babysitter $others is already running; exiting"; exit 0; fi
C="--ema 0.999 --student-noise 0.04 --mix-alpha 1.0 --mix-prob 0.5"
T="--model wrn28_10 --epochs 400 --ema 0.999 --label-smoothing 0 --sam-rho 0.05"
mkdir -p runs/babysit runs/inbox/done
while true; do
    for f in runs/inbox/*.sh; do
        [ -f "$f" ] || continue
        b=$(basename "$f" .sh); mv "$f" "runs/inbox/done/$b.sh"
        echo "$(date) inbox: running $b"
        bash "runs/inbox/done/$b.sh" > "runs/inbox/done/$b.out" 2>&1
    done
    queued=$(squeue --me -h -o %j)
    { date; squeue --me -o "%.8i %.20j %.3t %.10M %.24R"; } > runs/babysit/squeue.txt
    while IFS='|' read -r name opts args needs; do
        if [ -z "$name" ] || [ "${name:0:1}" = "#" ]; then continue; fi
        [ -f "runs/$name/final.pt" ] && continue
        grep -qx "$name" <<<"$queued" && continue
        ok=1; for f in $needs; do [ -f "$f" ] || ok=0; done
        [ $ok = 1 ] || continue
        n=$(cat "runs/babysit/$name.count" 2>/dev/null || echo 0)
        [ "$n" -ge 8 ] && continue
        echo $((n + 1)) > "runs/babysit/$name.count"
        args=${args//@C/$C}; args=${args//@T/$T}
        echo "$(date) submit $name (#$((n + 1))): $(sbatch --parsable --job-name=$name $opts scripts/hpc/train.sbatch $args)"
    done < runs/jobs.txt
    sleep 300 & wait $!  # wait (not a foreground sleep) lets the TERM trap run immediately
done
