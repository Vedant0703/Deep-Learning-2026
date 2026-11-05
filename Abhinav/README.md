# Efficient ResNet on CIFAR-10 (at most 5M parameters)

NYU Deep Learning, Fall 2026, Project 1. We train a modified ResNet **from scratch** on the CIFAR-10 training data
provided by the course, under a budget of 5,000,000 trainable parameters, for the Kaggle competition
[cifar-10-res-net-architecture-challenge-dl](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl).

## Results

Every submission is in [`submissions/`](submissions/), one folder each (`Abhinav_Run-D` … `Abhinav_Run-M`, plus
unsubmitted `candidates/`). Each folder has the model (`model.pt`), the submitted CSV, a class-balanced CSV, the logs,
the 45k twin used to validate it, and the exact training command.

| Run | Recipe (all `resnet_c` 4.62M unless noted, trained from scratch) | Twin val (flip) | Public |
|---|---|---|---|
| D | KD from WRN-28-10 + input noise, 300 epochs | 97.50 | 0.891 |
| E | Run D's 45k twin, submitted directly | 97.50 | 0.888 |
| F | function-matching KD (MixUp/CutMix) from WRN-28-10, 300 epochs | 97.78 | **0.901** |
| G | same as F with Vedant's PreAct 4.98M | 97.58 | **0.901** |
| H | F at 800 epochs (NYU HPC) | 98.00 | 0.897 |
| I | H with a diverse 4-teacher ensemble (2x WRN-28-10 SAM, WRN-28-10 AutoAugment, WRN-40-10) | 97.82 | 0.900 |
| J | Run I's model, flip_shift TTA + class-balanced decoding (no retraining) | 97.96 (fs + bal) | 0.902 |
| K | Run H's model, flip_shift TTA + class-balanced decoding (no retraining) | 97.98 (fs + bal) | 0.894 |
| L | Run I's model fine-tuned 100 epochs with on-the-fly hard-factor augmentation (`src/hard.py`), fs + bal | 97.86 (fs + bal) | 0.906 |
| M | Run I's recipe from scratch with on-the-fly hard-factor augmentation, 600 epochs, fs + bal | 97.96 (fs + bal) | **0.910** |

Function matching was the big lever. After it, every change ties within the ±0.2 twin and ±0.95 public-LB noise.
Class-balanced decoding (`src.predict --tta flip_shift --balance`, approved by the professor) adds about +0.26 on the
shifted val suite; on the 1,000 public images it moved I by +2 images (J) and H by -3 (K), which is within noise.
Round 9's on-the-fly hard-factor augmentation (low resolution, zoom in/out, background clutter; the teacher labels the
clean image, and augmented images get a 0.5 true-label weight) is the first change since then to move the public
score: it ties on clean val, lifts a hard-factor suite from 88.4 to 93.7, and gave Run M 0.910 (+0.8 over J).
The full story is in `notes/key_takeaways.md`, and the HPC workflow (Slurm scripts, babysitter) in `scripts/hpc/`.

## Repository layout

```
src/models.py      configurable ResNet (BasicBlock: conv-BN-ReLU-conv-BN + skip), candidate configs, param counting;
                   WideResNet-28-10 used only as a distillation teacher
src/data.py        loads the competition PNG/CSV files, train/val split, augmentation
src/train.py       training loop (SGD + warmup/cosine, AMP, EMA, knowledge distillation, checkpoint/resume)
src/evaluate.py    validation accuracy, clean and under brightness/contrast changes, to pick the submitted model
src/predict.py     TTA inference on the evaluation images, writes and validates submission.csv
notebooks/project1.ipynb   end-to-end Colab notebook: data, param counts, training, curves, submission
scripts/colab.py   runs training on a Colab GPU from a local terminal via the Google Colab CLI
scripts/submit.sh  predicts locally and submits to Kaggle after a confirmation
scripts/hpc/       NYU HPC Cloud Bursting: env setup, train.sbatch, the babysitter that resubmits jobs
submissions/       one folder per Kaggle submission: model.pt, CSVs, logs, 45k twin, README
```

## Reproducing the results

### Google Colab (recommended)
Open `notebooks/project1.ipynb` in Colab with a GPU runtime and run all cells. The notebook clones this repo,
downloads the competition data with the Kaggle API (paste your Kaggle API token when prompted),
saves checkpoints to Google Drive, and writes `submission.csv`.

### Colab GPU from a local editor (Cursor / VS Code terminal)
The [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) can start a Colab GPU and run code on it
from a local terminal. `scripts/colab.py` wraps it, and `.vscode/tasks.json` exposes each step under
*Tasks: Run Task* in Cursor.

```bash
uv tool install google-colab-cli && colab usage        # one-time install + Google sign-in
uv tool install kaggle --with six && kaggle auth login  # one-time install + Kaggle sign-in

python scripts/colab.py up                 # start an A100 session (COLAB_GPU=L4 for an L4), upload code + data
python scripts/colab.py screen 40          # 40-epoch screening of all candidate models, concurrently on one GPU
python scripts/colab.py train final_a --model resnet_a --epochs 200   # `start` launches without following
python scripts/colab.py status final_a     # progress; works from any terminal while training runs
python scripts/colab.py eval runs/final_a/final.pt   # clean + brightness/contrast-shifted val accuracy
python scripts/colab.py fetch final_a      # download logs and weights to runs/final_a/
python scripts/colab.py down               # stop the session so it stops using compute units
scripts/submit.sh final_a "resnet_a 200 epochs"   # predict locally, confirm, submit, show the score
```
Training runs as a detached process on the VM, so stopping the log stream does not stop training;
`python scripts/colab.py follow RUN` reattaches. Run `python scripts/colab.py sync` after every code change.

### Command line
```bash
pip install -r requirements.txt
kaggle competitions download cifar-10-res-net-architecture-challenge-dl -p data
unzip -q data/cifar-10-res-net-architecture-challenge-dl.zip -d data

python -m src.models                                    # print parameter counts
python -m src.train --model resnet_a --epochs 40 --out runs/screen_a                     # screening (45k/5k split)
python -m src.train --model resnet_a --epochs 200 --out runs/final_a                     # final run
python -m src.train --model resnet_a --epochs 200 --out runs/final_a --resume            # continue after a disconnect
python -m src.predict --ckpt runs/final_a/final.pt --out submission.csv
```

Each run directory has `config.json` (all hyperparameters and the parameter count), `history.json`
(per-epoch loss, accuracy, and learning rate), `last.pt` and `best.pt` (resumable checkpoints),
and `final.pt` (final weights).

## Training configuration (defaults in `src/train.py`)

| Setting | Value |
|---|---|
| Optimizer | SGD, momentum 0.9, Nesterov; weight decay 5e-4 on conv/linear weights only |
| Learning rate | peak 0.1, 3-epoch linear warmup, then cosine decay to 0 (per step) |
| Batch size / epochs | 256 / 40 for screening, 200 for the final run |
| Loss | cross-entropy with label smoothing 0.1 |
| Augmentation | random crop (pad 4, reflect) + horizontal flip + TrivialAugmentWide + random erasing (p=0.25) |
| Precision | mixed precision (fp16 autocast) and channels_last on CUDA |
| Inference | average of softmax over the image and its horizontal flip |
| Validation | stratified 5,000-image split of the training set (500 per class), seed 0 |

## Data and rule compliance
- Training uses only the 50,000 labeled training images supplied with the competition.
- Model selection uses a held-out split of those training images. The 2,000 unlabeled evaluation images are used only to
  produce `submission.csv`; they are never used for training, validation, or normalization statistics.
- No pretrained weights or external datasets are used.

## References
- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- kuangliu/pytorch-cifar, https://github.com/kuangliu/pytorch-cifar (ResNet for CIFAR reference implementation).
- S. G. Müller, F. Hutter. *TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation.* ICCV 2021.
- E. D. Cubuk et al. *AutoAugment: Learning Augmentation Policies from Data.* CVPR 2019.
- Z. Zhong et al. *Random Erasing Data Augmentation.* AAAI 2020. T. DeVries, G. Taylor. *Improved Regularization of CNNs with Cutout.* 2017.
- C. Szegedy et al. *Rethinking the Inception Architecture for Computer Vision* (label smoothing). CVPR 2016.
- I. Loshchilov, F. Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts* (cosine schedule). ICLR 2017.
- P. Goyal et al. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour* (warmup, zero-init residual BN). 2017.
