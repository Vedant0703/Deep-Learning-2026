#!/bin/bash
# One-time environment for scripts/hpc/train.sbatch: a CUDA Singularity image plus an ext3 overlay holding Miniconda,
# PyTorch and the project's dependencies. It follows the course's "Singularity with Conda" instructions and uses the
# templates in /share/apps. Run inside an Open OnDemand Jupyter terminal, from the project folder. It takes about 10-15
# minutes and is safe to re-run: finished steps are skipped.
set -euo pipefail
PROJ=${PROJ:-$PWD}
ENV=${ENV:-$PROJ/env}
mkdir -p "$ENV"

if [ ! -f "$ENV/overlay.ext3" ]; then
    template=$(ls /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz 2>/dev/null || ls /share/apps/overlay-fs-ext3/overlay-*GB*.ext3.gz | head -1)
    echo "overlay template: $template"
    cp "$template" "$ENV/overlay.ext3.gz"
    gunzip "$ENV/overlay.ext3.gz"
fi
if [ ! -e "$ENV/image.sif" ]; then
    image=$(ls /share/apps/images/cuda11.8*cudnn*ubuntu22*.sif 2>/dev/null | head -1 || true)
    [ -n "$image" ] || image=$(ls /share/apps/images/cuda*cudnn*ubuntu*.sif | sort -V | tail -1)
    echo "image: $image"
    ln -s "$image" "$ENV/image.sif"
fi

singularity exec --overlay "$ENV/overlay.ext3:rw" "$ENV/image.sif" /bin/bash -c '
set -e
if [ ! -d /ext3/miniconda3 ]; then
    cd /ext3
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
    rm Miniconda3-latest-Linux-x86_64.sh
fi
# Miniconda-latest ships a Python with no cu118 PyTorch wheels (3.14 in Nov 2026), so use a Python 3.11 env.
[ -d /ext3/py311 ] || /ext3/miniconda3/bin/conda create -y -q -p /ext3/py311 -c conda-forge --override-channels python=3.11
cat > /ext3/env.sh <<"EOS"
#!/bin/bash
unset -f which
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
conda activate /ext3/py311
EOS
source /ext3/env.sh
python -c "import torch, torchvision" 2>/dev/null || pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -q numpy pandas pillow
python -c "import torch, torchvision; print(\"torch\", torch.__version__, \"torchvision\", torchvision.__version__)"
'
echo "environment ready in $ENV"
