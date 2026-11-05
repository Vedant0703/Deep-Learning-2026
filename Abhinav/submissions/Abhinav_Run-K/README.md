# Abhinav_Run-K: Run H's model with flip_shift TTA + class-balanced decoding (no retraining)

| | |
|---|---|
| Kaggle public | **0.894** (submitted 2026-11-05 02:25 EST) |
| Architecture | `resnet_c` (`src/models.py`), 4,623,178 trainable parameters |
| Training data | all 50,000 images, from scratch |
| Validation (45k twin, none / flip TTA) | 97.84 / 98.00 |
| Twin val with class-balanced decoding (flip_shift TTA) | 97.98 (suite mean 96.79) |

## Files

- `model.pt`: the submitted weights (EMA). Load with `src.train.load_model`.
- `submission.csv`: the predictions that were submitted: flip_shift TTA (10 views) + class-balanced decoding (`src.predict --tta flip_shift --balance`, approved by the professor on Nov 4).
- `config.json`, `history.json`, `train.log`: all hyperparameters, per-epoch metrics, the log.
- `twin/`: the same recipe trained on 45k with 5k held out (`runs/h_fm800`), used to validate it. Use it for honest val numbers; `model.pt` has seen every training image.

## Teacher(s)

- `teacher50`: WRN-28-10, 300 epochs, all 50k (50k version of teacher_wrn300)

Teacher weights (146-224 MB each) are over GitHub's 100 MB file limit. They are on the NYU HPC at `/scratch/as21153/dl1/runs/<name>/final.pt` (some 45k teachers are also on Abhinav's laptop); ask Abhinav for a copy.

## Reproduce

```
python -m src.train --out runs/h_final800 --model resnet_c --epochs 800 --ema 0.999 --student-noise 0.04 --mix-alpha 1.0 --mix-prob 0.5 --teacher runs/teacher50/final.pt --val-size 0
```

## Predict

```
python -m src.predict --ckpt submissions/Abhinav_Run-K/model.pt --out submission.csv --tta flip_shift --balance
```
