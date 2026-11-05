# Abhinav_Run-L: Run I's model fine-tuned for 100 epochs with on-the-fly hard augmentation (low resolution, zoom in/out, background clutter; the teacher sees the clean image; true-label weight 0.5 on augmented images)

| | |
|---|---|
| Kaggle public | **0.906** (submitted 2026-11-05 13:12 EST) |
| Architecture | `resnet_c` (`src/models.py`), 4,623,178 trainable parameters |
| Training data | all 50,000 images, from scratch |
| Validation (45k twin, none / flip TTA) | 97.62 / 97.70 |
| Twin val with class-balanced decoding (flip_shift TTA) | 97.86 (suite mean 96.74; hard-factor suite 93.0 vs Run I's 88.4) |

## Files

- `model.pt`: the submitted weights (EMA). Load with `src.train.load_model`.
- `submission.csv`: the predictions that were submitted: flip_shift TTA (10 views) + class-balanced decoding (`src.predict --tta flip_shift --balance`, approved by the professor on Nov 4).
- `config.json`, `history.json`, `train.log`: all hyperparameters, per-epoch metrics, the log.
- `twin/`: the same recipe trained on 45k with 5k held out (`runs/c_ft100_haug`), used to validate it. Use it for honest val numbers; `model.pt` has seen every training image.

## Teacher(s)

- `h_teacher_sam50`: WRN-28-10 + SAM 0.05, 400 epochs, all 50k (45k version h_teacher_sam45: 98.02 flip)
- `h_teacher_sam50_s2`: WRN-28-10 + SAM, seed 2, all 50k (45k sibling h_teacher_sam45_s1: 98.22 flip)
- `h_teacher_aa50`: WRN-28-10 + SAM + AutoAugment, all 50k (45k version h_teacher_aa45: 97.88 flip)
- `h_teacher_w40_50`: WRN-40-10 + SAM, 300 epochs, all 50k (45k version h_teacher_w40_45: 98.14 flip)

Teacher weights (146-224 MB each) are over GitHub's 100 MB file limit. They are on the NYU HPC at `/scratch/as21153/dl1/runs/<name>/final.pt` (some 45k teachers are also on Abhinav's laptop); ask Abhinav for a copy.

## Reproduce

```
python -m src.train --out runs/h_ftfinal100_haug --model resnet_c --epochs 100 --ema 0.999 --student-noise 0.04 --mix-alpha 1.0 --lr 0.02 --warmup-epochs 1 --hard-aug 0.3 --hard-label-weight 0.5 --mix-prob 0.5 --teacher runs/h_teacher_sam50/final.pt runs/h_teacher_sam50_s2/final.pt runs/h_teacher_aa50/final.pt runs/h_teacher_w40_50/final.pt --val-size 0 --init runs/h_final800_div/final.pt
```

## Predict

```
python -m src.predict --ckpt submissions/Abhinav_Run-L/model.pt --out submission.csv --tta flip_shift --balance
```
