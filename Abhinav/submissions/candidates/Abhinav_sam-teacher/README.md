# Abhinav_sam-teacher: function-matching KD from one WRN-28-10 SAM teacher, 800 epochs, all 50k

| | |
|---|---|
| Kaggle public | not submitted (a candidate for the final picks) |
| Architecture | `resnet_c` (`src/models.py`), 4,623,178 trainable parameters |
| Training data | all 50,000 images, from scratch |
| Validation (45k twin, none / flip TTA) | 97.86 / 97.78 |
| Twin val with class-balanced decoding (flip TTA) | 97.82 (suite mean 96.47 → 96.71) |

## Files

- `model.pt`: the weights (EMA). Load with `src.train.load_model`.
- `submission.csv`: flip-TTA predictions (not submitted).
- `submission_balanced.csv`: class-balanced decoding (`--tta flip --balance`; Runs J and K submitted the `--tta flip_shift --balance` version). Not submitted.
- `config.json`, `history.json`, `train.log`: all hyperparameters, per-epoch metrics, the log.
- `twin/`: the same recipe trained on 45k with 5k held out (`runs/h_fm800_sam`), used to validate it. Use it for honest val numbers; `model.pt` has seen every training image.

## Teacher(s)

- `h_teacher_sam50`: WRN-28-10 + SAM 0.05, 400 epochs, all 50k (45k version h_teacher_sam45: 98.02 flip)

Teacher weights (146-224 MB each) are over GitHub's 100 MB file limit. They are on the NYU HPC at `/scratch/as21153/dl1/runs/<name>/final.pt` (some 45k teachers are also on Abhinav's laptop); ask Abhinav for a copy.

## Reproduce

```
python -m src.train --out runs/h_final800_sam --model resnet_c --epochs 800 --ema 0.999 --student-noise 0.04 --mix-alpha 1.0 --mix-prob 0.5 --teacher runs/h_teacher_sam50/final.pt --val-size 0
```

## Predict

```
python -m src.predict --ckpt submissions/candidates/Abhinav_sam-teacher/model.pt --out submission.csv --tta flip   # add --balance for class-balanced decoding
```
