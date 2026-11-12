#!/bin/bash
# One command to set up and launch the long runs on NYU HPC Cloud Bursting.
# Run it in an Open OnDemand Jupyter terminal, in the folder where dl1_bundle.tar was uploaded:
#
#   tar -xOf dl1_bundle.tar scripts/hpc/bootstrap.sh | bash
#
# It unpacks the bundle (code, data.zip, teachers) into $PROJ, builds the environment (setup_env.sh, about 15 min),
# checks that PyTorch imports, then submits the four long jobs. SUBMIT=0 skips the submission.
set -euo pipefail
BUNDLE=${BUNDLE:-$PWD/dl1_bundle.tar}
if [ -z "${PROJ:-}" ]; then
    if [ -d "/scratch/$USER" ] && [ -w "/scratch/$USER" ]; then PROJ=/scratch/$USER/dl1; else PROJ=$HOME/dl1; fi
fi
echo "project folder: $PROJ"
mkdir -p "$PROJ"
tar -C "$PROJ" -xf "$BUNDLE"
cd "$PROJ"
mkdir -p data runs
[ -f data/train_labels.csv ] || python3 -m zipfile -e data.zip data || unzip -q data.zip -d data
echo "data: $(ls data/train_images | wc -l) train / $(ls data/test_images | wc -l) test images"

PROJ=$PROJ bash scripts/hpc/setup_env.sh
singularity exec --overlay env/overlay.ext3:ro env/image.sif /bin/bash -c \
    "source /ext3/env.sh && python -c 'from src.models import build_model, count_params; \
     print(\"resnet_c:\", count_params(build_model(\"resnet_c\")))'"

if [ "${SUBMIT:-1}" = 1 ]; then
    common=(--ema 0.999 --student-noise 0.04 --mix-alpha 1.0 --mix-prob 0.5)
    # Function matching for 800 epochs: 45k twin (validation) and the all-50k final.
    sbatch --job-name=h_fm800 scripts/hpc/train.sbatch --model resnet_c --epochs 800 "${common[@]}" \
        --teacher runs/teacher_wrn300/final.pt
    sbatch --job-name=h_final800 scripts/hpc/train.sbatch --model resnet_c --epochs 800 "${common[@]}" \
        --teacher runs/teacher50/final.pt --val-size 0
    # Stronger teachers: WRN-28-10 with SAM, 400 epochs (45k validated, 50k for finals).
    sbatch --job-name=h_teacher_sam45 scripts/hpc/train.sbatch --model wrn28_10 --epochs 400 --ema 0.999 \
        --label-smoothing 0 --sam-rho 0.05
    sbatch --job-name=h_teacher_sam50 scripts/hpc/train.sbatch --model wrn28_10 --epochs 400 --ema 0.999 \
        --label-smoothing 0 --sam-rho 0.05 --val-size 0
fi
squeue --me
echo "check progress any time with:  bash $PROJ/scripts/hpc/status.sh"
