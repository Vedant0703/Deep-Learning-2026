# Abhinav_Run-O: Run N's recipe + SAM (rho 0.05) on the student, 600 epochs, all 50k (NYU HPC)

| | |
|---|---|
| Kaggle public | **0.905** (submitted 2026-11-06 21:35 EST "just to see" after it failed the gate below; Run N 0.909, Run M 0.910) |
| Architecture | `preact_v` (`src/models.py`), 4,980,392 trainable parameters |
| Training data | all 50,000 images, from scratch |
| Twin val (none / flip TTA, from train.log) | 97.96 / 98.12 (Run N's twin: 98.06 / 98.06) |
| Twin val, flip_shift TTA (none / balanced) | 98.12 / 98.12 (Run N 98.08 / **98.20**) |
| Twin corruption suite, flip_shift + balanced | 97.02 (Run N 97.03) |
| Twin hard-factor suite, flip_shift | 93.73 (Run N **93.87**); lowres16 93.94 vs 94.72 is the largest drop |
| Paired test vs Run N's twin (balanced) | 18 vs 22 images, McNemar p = 0.64; error Jaccard 0.64 |

- Gate for a new PreAct-family pick: clean ≥ 98.10, hard ≥ 93.87 and no significant McNemar loss vs Run M. It passes
  the clean bar but misses the hard bar, and it is below Run N on both.
- On random 1,000-image subsets of the held-out set (the public LB's size), it beats Run N 30% of the time, ties 16%
  and trails 54% (mean gap -0.08 points). A tie with Run N, so it is not a final pick. Its public 0.905 (4 images
  below Run N) agrees with that.
- SAM led Run N's twin by about 0.15-0.25 val points through epoch 433. The lead was gone by epoch 600, as in Round 1.
- The final model's predictions differ from Run N's submission on 68 of 2,000 hidden images and from Run M's on 70.

## Files

- `model.pt`: the 50k weights (EMA). Load with `src.train.load_model`.
- `submission.csv`: the predictions that were submitted: flip_shift TTA (10 views) + class-balanced decoding
  (`src.predict --tta flip_shift --balance`, approved by the professor on Nov 4).
- `config.json`: all hyperparameters.
- `twin/`: the same recipe trained on 45k with 5k held out (`runs/h_fm600_pre_haug_sam`), used to validate it. It
  has `model.pt`, `config.json`, `eval.log` (src.evaluate with `--hard-suite --baseline` Run N's twin) and
  `paired_vs_run_N.log` (McNemar, error groups by class and confusion, 1,000-image subset gaps).
- `train.log` / `history.json` for both runs are on the NYU HPC at
  `/scratch/as21153/dl1/runs/{h_final600_pre_haug_sam,h_fm600_pre_haug_sam}/`.

## Teacher(s)

The same four as Runs M and N: `h_teacher_sam50`, `h_teacher_sam50_s2`, `h_teacher_aa50` and `h_teacher_w40_50`
(all 50k). The twin used their 45k versions. Teacher weights (146-224 MB each) are over GitHub's 100 MB limit. They
are on the NYU HPC at `/scratch/as21153/dl1/runs/<name>/final.pt`; ask Abhinav for a copy.

## Reproduce

```
python -m src.train --out runs/h_final600_pre_haug_sam --model preact_v --epochs 600 --ema 0.999 --student-noise 0.04 --mix-alpha 1.0 --mix-prob 0.5 --val-size 0 --sam-rho 0.05 --hard-aug 0.3 --hard-label-weight 0.5 --teacher runs/h_teacher_sam50/final.pt runs/h_teacher_sam50_s2/final.pt runs/h_teacher_aa50/final.pt runs/h_teacher_w40_50/final.pt
```

## Predict

```
python -m src.predict --ckpt submissions/Abhinav_Run-O/model.pt --out submission.csv --tta flip_shift --balance
```
