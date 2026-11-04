# Abhinav_Run-F: function-matching KD (MixUp/CutMix) from WRN-28-10, 300 epochs, all 50k

| | |
|---|---|
| Kaggle public | **0.901** (submitted 2026-11-03 15:39 EST) |
| Architecture | `resnet_c` (`src/models.py`), 4,623,178 trainable parameters |
| Training data | all 50,000 images, from scratch |
| Validation (45k twin, none / flip TTA) | 97.66 / 97.78 |
| Twin val with class-balanced decoding (flip) | 97.78 (suite mean 96.28 → 96.42) |

## Files

- `model.pt`: the submitted weights (EMA). Load with `src.train.load_model`.
- `submission.csv`: the predictions that were submitted (flip TTA).
- `submission_balanced.csv`: the same model with class-balanced decoding (`src.predict --balance`, approved by the professor on Nov 4). Not yet submitted.
- `config.json`, `history.json`, `train.log`: all hyperparameters, per-epoch metrics, the log.
- `twin/`: the same recipe trained on 45k with 5k held out (`runs/r1_fm`), used to validate it. Use it for honest val numbers; `model.pt` has seen every training image.

## Teacher(s)

- `teacher50`: WRN-28-10, 300 epochs, all 50k (50k version of teacher_wrn300)

Teacher weights (146-224 MB each) are over GitHub's 100 MB file limit. They are on the NYU HPC at `/scratch/as21153/dl1/runs/<name>/final.pt` (some 45k teachers are also on Abhinav's laptop); ask Abhinav for a copy.

## Reproduce

```
python -m src.train --out runs/r2_final_H --model resnet_c --epochs 300 --ema 0.999 --student-noise 0.04 --mix-alpha 1.0 --mix-prob 0.5 --teacher runs/teacher50/final.pt --val-size 0
```

## Predict

```
python -m src.predict --ckpt submissions/Abhinav_Run-F/model.pt --out submission.csv --tta flip   # add --balance for class-balanced decoding
```
