# Abhinav_Run-G: PreAct ResNet 4.98M (Vedant's preact_v), function-matching KD from WRN-28-10, 300 epochs, all 50k

| | |
|---|---|
| Kaggle public | **0.901** (submitted 2026-11-03 19:00 EST) |
| Architecture | `preact_v` (`src/models.py`), 4,980,392 trainable parameters |
| Training data | all 50,000 images, from scratch |
| Validation (45k twin, none / flip TTA) | 97.40 / 97.58 |
| Twin val with class-balanced decoding (flip TTA) | 97.56 (suite mean 96.23 → 96.30) |

## Files

- `model.pt`: the submitted weights (EMA). Load with `src.train.load_model`.
- `submission.csv`: the predictions that were submitted (flip TTA).
- `submission_balanced.csv`: class-balanced decoding (`--tta flip --balance`; Runs J and K submitted the `--tta flip_shift --balance` version). Not submitted.
- `config.json`, `history.json`, `train.log`: all hyperparameters, per-epoch metrics, the log.
- `twin/`: the same recipe trained on 45k with 5k held out (`runs/r2_twin_S`), used to validate it. Use it for honest val numbers; `model.pt` has seen every training image.

## Teacher(s)

- `teacher50`: WRN-28-10, 300 epochs, all 50k (50k version of teacher_wrn300)

Teacher weights (146-224 MB each) are over GitHub's 100 MB file limit. They are on the NYU HPC at `/scratch/as21153/dl1/runs/<name>/final.pt` (some 45k teachers are also on Abhinav's laptop); ask Abhinav for a copy.

## Reproduce

```
python -m src.train --out runs/r2_final_S --model preact_v --epochs 300 --ema 0.999 --student-noise 0.04 --mix-alpha 1.0 --mix-prob 0.5 --teacher runs/teacher50/final.pt --val-size 0
```

## Predict

```
python -m src.predict --ckpt submissions/Abhinav_Run-G/model.pt --out submission.csv --tta flip   # add --balance for class-balanced decoding
```

## Dhanush translation-TTA follow-up

The same weights with 10-view translation/flip probability TTA, without balancing, scored **0.90400**. See [Dhanush_Run-G-TTA](../Dhanush_Run-G-TTA/) for reproduction instructions, validation report and predictions. The original flip-only result above remains 0.90100.
