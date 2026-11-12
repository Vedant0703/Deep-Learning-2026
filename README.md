# Efficient ResNet on CIFAR-10 (at most 5M parameters)

NYU Deep Learning, Fall 2026, Project 1, team **Survey Corps**: Abhinav Sivakumar
([ebbynav](https://github.com/ebbynav)), Vedant Pradhan ([Vedant0703](https://github.com/Vedant0703)) and
Dhanushkumar Jayakumar ([Dhanush010](https://github.com/Dhanush010)). We train a single ResNet with at most 5,000,000 trainable parameters
**from scratch** on the CIFAR-10 training images provided by the course, for the Kaggle competition
[cifar-10-res-net-architecture-challenge-dl](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl).

**Final result:** Run M, a 4.62M-parameter ResNet, scored 97.96% on our 5,000-image held-out split, 0.910 on the
public leaderboard (18th) and **0.893 on the private leaderboard, 6th of 81 teams**.

## Final model (Run M)

Everything for it is in [`submissions/Abhinav_Run-M/`](submissions/Abhinav_Run-M/): the weights (`model.pt`), the
submitted predictions, the training log, the 45k twin used to validate it and the exact training command.
[`notebooks/final_model.ipynb`](notebooks/final_model.ipynb) loads it and prints its parameter count, held-out and
final test accuracy, reproduces the submitted CSV and plots its training curves (outputs saved).

| | |
|---|---|
| Architecture | `resnet_c` (`src/models.py`): post-activation ResNet, 64/128/256 channels, 3-4-3 BasicBlocks, 4,623,178 parameters |
| Distillation | function matching from a diverse ensemble of 4 teachers (2x WRN-28-10 + SAM, WRN-28-10 + SAM + AutoAugment, WRN-40-10 + SAM); the teacher labels the same MixUp/CutMix batch the student sees |
| Hard-factor augmentation | 30% of each batch gets low resolution, zoom in/out or background clutter (`src/hard.py`); the teacher sees the clean image, and augmented images get a 0.5 true-label weight |
| Training | 600 epochs on all 50,000 images, SGD with Nesterov momentum, lr 0.1 with cosine decay, batch 256, EMA 0.999 |
| Decoding | flip-and-shift test-time augmentation (10 views) plus class-balanced decoding (Sinkhorn, T = 1) |

Predict the evaluation images with the submitted weights:

```bash
pip install -r requirements.txt
kaggle competitions download cifar-10-res-net-architecture-challenge-dl -p data
unzip -q data/cifar-10-res-net-architecture-challenge-dl.zip -d data
python -m src.predict --ckpt submissions/Abhinav_Run-M/model.pt --out submission.csv --tta flip_shift --balance
```

## Results

Each leaderboard half has about 1,000 images, so a single score is only good to about ±1 point. Every Kaggle
submission, with its held-out numbers, is in [`submissions/README.md`](submissions/README.md).

| Run | By | What changed | Held-out (flip) | Public | Private |
|---|---|---|---|---|---|
| A | Vedant | 4.98M PreAct wide ResNet (`preact_v`), 300 epochs, EMA, 45k images | 97.08 | 0.875 | 0.865 |
| B | Vedant | Run A's recipe on all 50k images | – | 0.892 | 0.873 |
| D | Abhinav | distillation from a WRN-28-10 teacher + input noise | 97.50 | 0.891 | 0.865 |
| F | Abhinav | function-matching distillation (the teacher labels the mixed images) | 97.78 | 0.901 | 0.870 |
| I | Abhinav | 4-teacher ensemble, 800 epochs | 97.82 | 0.900 | 0.876 |
| J | Abhinav | Run I's model + flip-and-shift TTA + class-balanced decoding | 97.96 (fs + bal) | 0.902 | 0.884 |
| M | Abhinav | Run I's recipe + hard-factor augmentation, 600 epochs | 97.96 (fs + bal) | **0.910** | **0.893** |
| N | Abhinav | Run M's recipe on Vedant's `preact_v` | 98.20 (fs + bal) | 0.909 | 0.886 |

What we learned, in short:
- Function matching was the big training lever (0.891 to 0.901 public). After it, more epochs, stronger teachers and
  other architectures all tied within noise.
- Turning an error analysis into augmentation (hard-factor augmentation) was the only later training change that
  moved the leaderboard.
- Class-balanced decoding looked like noise on the public half (+2 and -3 images) but gained +8 and +16 images on
  the private half. That is why we dropped the least of the leading teams from public to private and moved from
  18th to 6th.

The full story is in [`notes/key_takeaways.md`](notes/key_takeaways.md) (summary) and
[`notes/training_log.md`](notes/training_log.md) (the full log, written while the runs trained).

## Repository layout

```
src/            training, evaluation and prediction code (python -m src.train / src.evaluate / src.predict)
  models.py     configurable ResNets (resnet_c, preact_v, ...), parameter counting; WRN teachers for distillation
  data.py       loads the competition PNG/CSV files, the stratified 45k/5k split, augmentation
  train.py      training loop: SGD + warmup/cosine, AMP, EMA, distillation, function matching, hard-factor aug, SAM
  hard.py       hard-factor augmentation (low resolution, zoom in/out, background clutter)
  evaluate.py   held-out accuracy: clean, under brightness/contrast/noise shifts, and on the hard-factor suite
  predict.py    test-time augmentation and class-balanced decoding; writes and validates submission.csv
scripts/        Colab runner (colab.py), Kaggle submit (submit.sh), run comparison, model soups, hard-suite figure
  hpc/          NYU HPC Cloud Bursting: environment setup, train.sbatch, the job babysitter, job queues
notebooks/      final_model.ipynb: the submitted model's parameters, accuracy, predictions and training curves
                project1.ipynb: end-to-end Colab notebook (data, parameter counts, training, curves, submission)
  vedant/       Vedant's notebooks for Runs A and B, Run C (robust augmentation) and Run D (ensemble distillation)
submissions/    one folder per Kaggle submission: model.pt, submitted CSV, logs, 45k twin, README with the command
  candidates/   trained but not submitted
runs/           config, per-epoch history and training log of every run (teachers, twins, finals, screens)
experiments/    run_g_tta/: Dhanush's translation-TTA study of Run G
notes/          key_takeaways.md and training_log.md
report/         final-report figures (figures/), the script that draws them, leaderboard data (data/),
                and the Nov 5 interim team write-up (PDF)
```

## Team contributions

- **Vedant Pradhan** designed the `preact_v` architecture and the baseline recipe, trained Runs A to C and the
  ensemble-distilled Run D ([`notebooks/vedant/`](notebooks/vedant/),
  [`submissions/Vedant_Runs-A-D/`](submissions/Vedant_Runs-A-D/)), and set up the team repository and HPC notebooks.
- **Dhanushkumar Jayakumar** trained a scratch baseline and audited Run G's inference: his translation-and-flip TTA lifted Run G
  from 0.901 to 0.904 public without retraining ([`experiments/run_g_tta/`](experiments/run_g_tta/),
  [`submissions/Dhanush_Run-G-TTA/`](submissions/Dhanush_Run-G-TTA/)).
- **Abhinav Sivakumar** wrote the training and evaluation pipeline (`src/`, `scripts/`), the twin protocol and the hard-factor
  suite, trained the teachers and Runs D to P, and ran the leaderboard analysis (`report/`).

## Reproducing the results

### Command line

```bash
python -m src.models                                                  # print parameter counts
python -m src.train --model resnet_c --epochs 40 --out runs/screen_c  # screening run (45k/5k split)
python -m src.train --model resnet_c --epochs 40 --out runs/screen_c --resume   # continue after a disconnect
python -m src.evaluate runs/screen_c/final.pt --tta none flip         # clean + shifted held-out accuracy
python -m src.predict --ckpt runs/screen_c/final.pt --out submission.csv
```

Every submission folder's README has the exact command that trained it. Distillation needs the teachers first: their
settings are in `runs/<teacher>/config.json`, and two WRN-28-10 teachers (`teacher50`, `teacher_wrn300`) are
attached to the [`teachers-v1`](https://github.com/Vedant0703/Deep-Learning-2026/releases/tag/teachers-v1) release.
The teachers are over the 5M budget and are never submitted.

Each run directory has `config.json` (all hyperparameters and the parameter count), `history.json` (per-epoch loss,
accuracy and learning rate) and `train.log`. Weights (`last.pt`, `best.pt`, `final.pt`) stay local, except the
submitted models, which are in `submissions/`.

### Google Colab

Open `notebooks/project1.ipynb` in Colab with a GPU runtime and run all cells. The notebook clones this repo,
downloads the competition data with the Kaggle API (paste your Kaggle API token when prompted), saves checkpoints to
Google Drive and writes `submission.csv`.

### Colab GPU from a local editor (Cursor / VS Code terminal)

The [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) can start a Colab GPU and run code on it from
a local terminal. `scripts/colab.py` wraps it, and `.vscode/tasks.json` exposes each step under *Tasks: Run Task*.

```bash
uv tool install google-colab-cli && colab usage        # one-time install + Google sign-in
uv tool install kaggle --with six && kaggle auth login  # one-time install + Kaggle sign-in

python scripts/colab.py up                 # start an A100 session (COLAB_GPU=L4 for an L4), upload code + data
python scripts/colab.py train final_c --model resnet_c --epochs 200   # `start` launches without following
python scripts/colab.py status final_c     # progress; works from any terminal while training runs
python scripts/colab.py fetch final_c      # download logs and weights to runs/final_c/
python scripts/colab.py down               # stop the session so it stops using compute units
scripts/submit.sh final_c "resnet_c 200 epochs"   # predict locally, confirm, submit, show the score
```

Training runs as a detached process on the VM, so stopping the log stream does not stop training;
`python scripts/colab.py follow RUN` reattaches. Run `python scripts/colab.py sync` after every code change.

### NYU HPC

The long runs (800-epoch students, teachers) ran on NYU HPC Cloud Bursting A100s with `scripts/hpc/train.sbatch`;
`scripts/hpc/babysit.sbatch` resubmits preempted jobs from a queue file. See
[`scripts/hpc/README.md`](scripts/hpc/README.md).

## Default training configuration (`src/train.py`)

The final recipe above adds distillation, function matching and hard-factor augmentation to these defaults.

| Setting | Value |
|---|---|
| Optimizer | SGD, momentum 0.9, Nesterov; weight decay 5e-4 on conv/linear weights only |
| Learning rate | peak 0.1, 3-epoch linear warmup, then cosine decay to 0 (per step) |
| Batch size | 256 |
| Loss | cross-entropy with label smoothing 0.1 |
| Augmentation | random crop (pad 4, reflect) + horizontal flip + TrivialAugmentWide + random erasing (p = 0.25) |
| Precision | mixed precision (fp16 autocast) and channels_last on CUDA |
| Validation | stratified 5,000-image split of the training set (500 per class), seed 0 |

## Data and rule compliance

- Training uses only the 50,000 labeled training images supplied with the competition. No pretrained weights or
  external datasets are used.
- Every recipe is trained twice: a 45k twin, validated on the 5,000 held-out training images, and a 50k final that is
  submitted. Models are chosen on the held-out split, never on the evaluation images.
- The 2,000 unlabeled evaluation images are used only to produce `submission.csv`. Class-balanced decoding assumes
  equal class counts and rebalances the predicted probabilities over those images at test time, without labels; the
  professor approved it.

## References

- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- S. Zagoruyko, N. Komodakis. *Wide Residual Networks.* BMVC 2016.
- G. Hinton, O. Vinyals, J. Dean. *Distilling the Knowledge in a Neural Network.* 2015.
- L. Beyer et al. *Knowledge Distillation: A Good Teacher is Patient and Consistent* (function matching). CVPR 2022.
- S. G. Müller, F. Hutter. *TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation.* ICCV 2021.
- H. Zhang et al. *mixup: Beyond Empirical Risk Minimization.* ICLR 2018. S. Yun et al. *CutMix.* ICCV 2019.
- P. Foret et al. *Sharpness-Aware Minimization for Efficiently Improving Generalization.* ICLR 2021.
- kuangliu/pytorch-cifar, https://github.com/kuangliu/pytorch-cifar (ResNet for CIFAR reference implementation).
