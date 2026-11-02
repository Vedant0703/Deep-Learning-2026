# Efficient ResNet on CIFAR-10 (at most 5M parameters)

NYU Deep Learning, Fall 2026, Project 1. We train a modified ResNet **from scratch** on the CIFAR-10 training data
provided by the course, under a budget of 5,000,000 trainable parameters, for the Kaggle competition
[cifar-10-res-net-architecture-challenge-dl](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl).

## Results

| Model | Channels | Blocks | Params | Val acc (5k held out) | Kaggle public |
|---|---|---|---|---|---|
| resnet_a | 64-128-256-512 | 2-1-1-1 | 4,977,226 | TBD | TBD |
| resnet_b | 80-160-320 | 2-2-2 | 4,337,370 | TBD | TBD |
| resnet_c | 64-128-256 | 3-4-3 | 4,623,178 | TBD | TBD |

## Repository layout

```
src/models.py      configurable ResNet (BasicBlock: conv-BN-ReLU-conv-BN + skip), candidate configs, param counting
src/data.py        loads the competition PNG/CSV files, train/val split, augmentation
src/train.py       training loop (SGD + warmup/cosine, AMP, checkpoint/resume), logs history.json
src/predict.py     flip-TTA inference on the evaluation images, writes and validates submission.csv
notebooks/project1.ipynb   end-to-end Colab notebook: data, param counts, training, curves, submission
scripts/colab.py   runs training on a Colab GPU from a local terminal via the Google Colab CLI
scripts/submit.sh  predicts locally and submits to Kaggle after a confirmation
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
python scripts/colab.py train final_a --model resnet_a --epochs 200
python scripts/colab.py status final_a     # progress; works from any terminal while training runs
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
