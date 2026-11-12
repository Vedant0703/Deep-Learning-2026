# Training on NYU HPC Cloud Bursting (Fall 2026)

Long runs (teachers, 800-epoch students) run here as SLURM batch jobs. They keep running after the browser tab or the
Mac is closed, and the course allocation is free: account `cs_gy_6923-2026fa`, 300 GPU hours.

- **Access:** only through the Cloud Bursting **Open OnDemand** web portal, with NYU VPN off campus. Course work must
  not use the Torch cluster.
- **Partitions:**

  | Partition | GPUs |
  |---|---|
  | `c12m85-a100-1` (the default here) | 1 × A100 40 GB |
  | `c24m170-a100-2` | 2 × A100 |
  | `g2-standard-12` / `g2-standard-24` | 1 / 2 × L4 |
  | `n2c48m24` | CPU only |
  | `interactive` | interactive jobs |

- **Shutdown rule:** instances with low GPU use for 20 minutes are shut down automatically. Training keeps the GPU busy.

## 1. On the Mac: build the upload bundle
```
COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -cf hpc_upload/dl1_bundle.tar \
    src scripts requirements.txt data.zip runs/teacher_wrn300/final.pt runs/teacher50/final.pt
```
That is code, data and two teachers, about 420 MB.

## 2. In Open OnDemand
1. Start a **Jupyter** session: account `cs_gy_6923-2026fa`, a CPU partition (`n2c48m24` or `interactive`), 2-3 hours.
2. Upload `dl1_bundle.tar` with the file browser, into your home or scratch folder.
3. In the Jupyter **terminal**, from that folder, run:
   ```
   tar -xOf dl1_bundle.tar scripts/hpc/bootstrap.sh | bash
   ```
   This unpacks into `/scratch/$USER/dl1` (or `~/dl1`) and builds the Singularity + conda environment from
   `/share/apps` (about 15 min). It then submits four jobs to the A100 partition:

   | Job | What | Time |
   |---|---|---|
   | `h_fm800` | resnet_c, function matching, 800 epochs, 45k twin | about 2.5 h |
   | `h_final800` | the same on all 50k | about 2.7 h |
   | `h_teacher_sam45` | WRN-28-10 + SAM, 400 epochs, 45k | about 2.7 h |
   | `h_teacher_sam50` | the same on 50k | about 3 h |

## 3. Check and collect
```
bash /scratch/$USER/dl1/scripts/hpc/status.sh     # queue + last log line of each run
bash /scratch/$USER/dl1/scripts/hpc/collect.sh    # results_<time>.tar with history, logs, final.pt
```
Download the results tar with the file browser and extract it inside the repo on the Mac (`tar -xf results_*.tar`).

To submit another run from a Jupyter terminal in the project folder:
```
sbatch --job-name=NAME scripts/hpc/train.sbatch <src.train arguments>
```
- Jobs use `--resume`, so resubmitting the same command continues from `runs/NAME/last.pt`.
- The walltime default is 8 h.
