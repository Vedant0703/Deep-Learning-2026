# Abhinav_Run-N: Run M's recipe on Vedant's PreAct 4.98M (preact_v): FM KD from the diverse 4-teacher ensemble + on-the-fly hard augmentation, 600 epochs, all 50k (NYU HPC)

| | |
|---|---|
| Kaggle public | **0.909** (submitted 2026-11-06 11:34 EST; Run M 0.910) |
| Architecture | `preact_v` (`src/models.py`), 4,980,392 trainable parameters |
| Training data | all 50,000 images, from scratch |
| Twin val, flip_shift TTA (none / balanced) | 98.08 / **98.20** (Run M's twin: 98.00 / 97.96) |
| Twin corruption suite, flip_shift + balanced | **97.03** (Run M 96.88) |
| Twin hard-factor suite, flip_shift | **93.87** (Run M 93.68); its gain is on the `ambiguous` row (83.2 vs 81.8), not the synthetic factors |
| Paired test vs Run M's twin (balanced) | 31 vs 19 images, McNemar p = 0.12 (not significant); error Jaccard 0.59 |

- The only candidate that beat Run M on every twin metric.
- Its errors overlap Run M's least of all our models, so {Run M, Run N} is the team's most decorrelated final pair.
- On random 1,000-image subsets of the held-out set (the public LB's size), it beats Run M 75% of the time, ties
  10% and trails 15%. The 0.909 vs 0.910 public gap (1 image) is inside that noise.

## Files

- `model.pt`: the submitted weights (EMA). Load with `src.train.load_model`.
- `submission.csv`: the predictions that were submitted: flip_shift TTA (10 views) + class-balanced decoding
  (`src.predict --tta flip_shift --balance`, approved by the professor on Nov 4).
- `config.json`: all hyperparameters.
- `twin/`: the same recipe trained on 45k with 5k held out (`runs/h_fm600_pre_haug`), used to validate it. It has
  `model.pt`, `config.json` and `eval.log` (the src.evaluate output above). Use it for honest val numbers;
  `model.pt` has seen every training image.
- `train.log` / `history.json` for both runs are on the NYU HPC at
  `/scratch/as21153/dl1/runs/{h_final600_pre_haug,h_fm600_pre_haug}/`.

## Teacher(s)

The same four as Run M: `h_teacher_sam50`, `h_teacher_sam50_s2`, `h_teacher_aa50` and `h_teacher_w40_50` (all 50k).
The twin used their 45k versions. Teacher weights (146-224 MB each) are over GitHub's 100 MB limit. They are on the
NYU HPC at `/scratch/as21153/dl1/runs/<name>/final.pt`; ask Abhinav for a copy.

## Reproduce

```
python -m src.train --out runs/h_final600_pre_haug --model preact_v --epochs 600 --ema 0.999 --student-noise 0.04 --mix-alpha 1.0 --mix-prob 0.5 --val-size 0 --hard-aug 0.3 --hard-label-weight 0.5 --teacher runs/h_teacher_sam50/final.pt runs/h_teacher_sam50_s2/final.pt runs/h_teacher_aa50/final.pt runs/h_teacher_w40_50/final.pt
```

## Predict

```
python -m src.predict --ckpt submissions/Abhinav_Run-N/model.pt --out submission.csv --tta flip_shift --balance
```
