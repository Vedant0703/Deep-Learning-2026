# Baseline model: Run F (0.901 public)

The team's best submission so far, shared as a starting point.

| | |
|---|---|
| File | `final_model.pt` (EMA weights, 18.6 MB) |
| Architecture | `resnet_c`: 3 stages, 64/128/256 channels, 3-4-3 BasicBlocks (`src/models.py`) |
| Trainable parameters | 4,623,178 |
| Training data | all 50,000 training images, from scratch |
| Public score | **0.901** (`submission.csv` here, flip TTA) |
| Validation | 97.66% / 97.78% with flip TTA. This is the 45k/5k twin `runs/r1_fm`, same recipe; the 50k model has no held-out set |

## Recipe

- **Teacher:** knowledge distillation from a WRN-28-10 teacher (36.5M params, 300 epochs, trained from scratch on all
  50k, `runs/teacher50`). The teacher is not submitted and is too large for GitHub.
- **Function matching** (Beyer et al., CVPR 2022): MixUp/CutMix (α 1.0) on half of the batches, applied *before* both
  the teacher and the student see the batch, so the teacher labels the mixed images.
- **Student training:** KD α 0.9 with temperature 4, plus label-smoothed CE; Gaussian noise (σ up to 0.04) on the
  student's input; TrivialAugment + random erasing; SGD with Nesterov momentum, lr 0.1, cosine schedule, 300 epochs,
  batch 256; EMA 0.999.

Command that trained it (run from this folder's parent, on a GPU):

```
python -m src.train --out runs/r2_final_H --model resnet_c --epochs 300 --ema 0.999 \
    --teacher runs/teacher50/final.pt --student-noise 0.04 --mix-alpha 1.0 --mix-prob 0.5 --val-size 0
```

## Use it

Predict the evaluation images (needs the competition data in `data/`):

```
python -m src.predict --ckpt final_model/final_model.pt --out submission.csv --tta flip
```

Load it in Python, for example to fine-tune it or to use it as a teacher:

```python
import torch
from src.train import load_model
model = load_model("final_model/final_model.pt", torch.device("cpu"))  # eval mode
```

Score it on the validation split with the corruption suite. Note that it has seen all 50k images, so its val numbers
are not held-out; use `runs/r1_fm` for honest validation.

```
python -m src.evaluate final_model/final_model.pt --tta none flip
```

What worked, what didn't, and ideas for improving on it: `notes/key_takeaways.md`.
