# Abhinav_Run-E: the 45k twin of Run D, submitted directly to measure the val-to-hidden gap

| | |
|---|---|
| Kaggle public | **0.888** (submitted 2026-11-03 11:06 EST) |
| Architecture | `resnet_c` (`src/models.py`), 4,623,178 trainable parameters |
| Training data | 45k (5k held out), from scratch |
| Validation (45k twin, none / flip TTA) | 97.30 / 97.50 (its own val) |

## Files

- `model.pt`: the submitted weights (EMA). Load with `src.train.load_model`.
- `submission.csv`: the flip-TTA predictions, regenerated from `model.pt` (the same procedure reproduced Run F's submitted file byte for byte).
- `config.json`, `history.json`, `train.log`: all hyperparameters, per-epoch metrics, the log.

## Teacher(s)

- `teacher_wrn`: WRN-28-10, 200 epochs, 45k (val 97.56 flip)

Teacher weights (146-224 MB each) are over GitHub's 100 MB file limit. They are on the NYU HPC at `/scratch/as21153/dl1/runs/<name>/final.pt` (some 45k teachers are also on Abhinav's laptop); ask Abhinav for a copy.

## Reproduce

```
python -m src.train --out runs/twin_kdn --model resnet_c --epochs 300 --ema 0.999 --student-noise 0.04 --teacher runs/teacher_wrn/final.pt
```

## Predict

```
python -m src.predict --ckpt submissions/Abhinav_Run-E/model.pt --out submission.csv --tta flip   # add --balance for class-balanced decoding
```
