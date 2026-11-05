# Abhinav_ens3: function-matching KD from a 3-teacher ensemble (WRN-28-10 + 2x WRN-28-10 SAM), 800 epochs, all 50k

| | |
|---|---|
| Kaggle public | not submitted (a candidate for the final picks) |
| Architecture | `resnet_c` (`src/models.py`), 4,623,178 trainable parameters |
| Training data | all 50,000 images, from scratch |
| Validation (45k twin, none / flip TTA) | 97.72 / 97.80 |
| Twin val with class-balanced decoding (flip TTA) | 97.84 (suite mean 96.61 → 96.77) |

## Files

- `model.pt`: the weights (EMA). Load with `src.train.load_model`.
- `submission.csv`: flip-TTA predictions (not submitted).
- `submission_balanced.csv`: class-balanced decoding (`--tta flip --balance`; Runs J and K submitted the `--tta flip_shift --balance` version). Not submitted.
- `config.json`, `history.json`, `train.log`: all hyperparameters, per-epoch metrics, the log.
- `twin/`: the same recipe trained on 45k with 5k held out (`runs/h_fm800_ens3`), used to validate it. Use it for honest val numbers; `model.pt` has seen every training image.

## Teacher(s)

- `teacher50`: WRN-28-10, 300 epochs, all 50k (50k version of teacher_wrn300)
- `h_teacher_sam50`: WRN-28-10 + SAM 0.05, 400 epochs, all 50k (45k version h_teacher_sam45: 98.02 flip)
- `h_teacher_sam50_s2`: WRN-28-10 + SAM, seed 2, all 50k (45k sibling h_teacher_sam45_s1: 98.22 flip)

Teacher weights (146-224 MB each) are over GitHub's 100 MB file limit. They are on the NYU HPC at `/scratch/as21153/dl1/runs/<name>/final.pt` (some 45k teachers are also on Abhinav's laptop); ask Abhinav for a copy.

## Reproduce

```
python -m src.train --out runs/h_final800_ens3 --model resnet_c --epochs 800 --ema 0.999 --student-noise 0.04 --mix-alpha 1.0 --mix-prob 0.5 --teacher runs/teacher50/final.pt runs/h_teacher_sam50/final.pt runs/h_teacher_sam50_s2/final.pt --val-size 0
```

## Predict

```
python -m src.predict --ckpt submissions/candidates/Abhinav_ens3/model.pt --out submission.csv --tta flip   # add --balance for class-balanced decoding
```
