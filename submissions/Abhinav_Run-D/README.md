# Abhinav_Run-D: KD from WRN-28-10 + input-noise robustness, 300 epochs, all 50k

| | |
|---|---|
| Kaggle public | **0.891** (submitted 2026-11-02 19:53 EST) |
| Architecture | `resnet_c` (`src/models.py`), 4,623,178 trainable parameters |
| Training data | all 50,000 images, from scratch |
| Validation (45k twin, none / flip TTA) | 97.30 / 97.50 |

## Files

- `model.pt`: the submitted weights (EMA). Load with `src.train.load_model`.
- `submission.csv`: the flip-TTA predictions, regenerated from `model.pt` (the same procedure reproduced Run F's submitted file byte for byte).
- `config.json`, `history.json`, `train.log`: all hyperparameters, per-epoch metrics, the log.
- `twin/`: the same recipe trained on 45k with 5k held out (`runs/twin_kdn`), used to validate it. Use it for honest val numbers; `model.pt` has seen every training image.

## Teacher(s)

- `teacher_wrn`: WRN-28-10, 200 epochs, 45k (val 97.56 flip)

Teacher weights (146-224 MB each) are over GitHub's 100 MB file limit. They are on the NYU HPC at `/scratch/as21153/dl1/runs/<name>/final.pt` (some 45k teachers are also on Abhinav's laptop); ask Abhinav for a copy.

## Reproduce

```
python -m src.train --out runs/final_kdn --model resnet_c --epochs 300 --ema 0.999 --student-noise 0.04 --teacher runs/teacher_wrn/final.pt --val-size 0
```

## Predict

```
python -m src.predict --ckpt submissions/Abhinav_Run-D/model.pt --out submission.csv --tta flip   # add --balance for class-balanced decoding
```
