# Abhinav_confuse-mix-ft: Run N fine-tuned 60 epochs with confusion-aware MixUp/CutMix (NYU HPC, Nov 6 night)

| | |
|---|---|
| Kaggle public | not submitted: significantly worse than Run N on the twin |
| Architecture | `preact_v` (`src/models.py`), 4,980,392 trainable parameters |
| Training data | all 50,000 images (fine-tune of Run N's final) |
| Idea | half of the mixed images get a partner from a confusable class in the batch (cat-dog, airplane-ship, airplane-bird, deer-dog, automobile-truck: the twins' top validation confusions), so the teachers' soft labels on mixed cat/dog images teach the cat/dog boundary |

| Twin (flip_shift TTA + balanced) | Clean | Corruption suite | Hard-factor suite | vs Run N's twin (McNemar) |
|---|---|---|---|---|
| Run N (`h_fm600_pre_haug`) | **98.20** | **97.03** | **93.87** | – |
| Plain fine-tune (control, `control_twin/`) | 97.90 | 96.74 | 93.26 | 21 vs 29, p = 0.32 |
| Confusion-aware fine-tune (`twin/`) | 97.86 | 96.69 | 93.08 | 10 vs 27, p = 0.008 |

- **The fine-tune itself costs about 0.3 points:** a 60-epoch restart at lr 0.02 knocks N off its optimum and doesn't
  fully recover. The best val checkpoint of both fine-tunes is their starting point (98.06).
- **Confusion-aware mixing adds nothing on top:** 19 vs 21 vs the control (p = 0.88, error Jaccard 0.68). It doesn't
  reduce cat/dog errors either (12 cat/dog errors only it makes vs 8 only the control makes).
- On 1,000-image subsets it trails Run N 89% of the time (mean gap −0.34 points).
- A from-scratch run (600 epochs) with `--mix-confuse` would be the fair test; it didn't fit before the deadline.

## Sinkhorn analysis (M vs N twins), `sinkhorn_transitions_M_vs_N.log`

- On the twin val, balancing changes only 19-20 predictions per model. Its main move is dog→cat, because both
  over-predict dog (508 / 512 for 500 dogs). The net is noise-sized (M −2, N +6 images).
- After balancing, M and N disagree on 56 images, 31 of them true cat/dog. Of the 71 images both get wrong, 34 are
  cat/dog. That was the motivation for this run.

## Files

- `model.pt`: the 50k fine-tune (`h_ftcfinal60_pre_haug`). `submission_balanced.csv`: its flip_shift + balanced
  predictions (not submitted; they differ from Run N's submission on 54 of 2,000 images).
- `twin/`: the 45k fine-tune of Run N's twin (`h_ftc60_pre_haug`), its `eval.log`, and the paired tests vs Run N's
  twin and vs the control.
- `control_twin/`: the same fine-tune without confusion-aware mixing (`h_ft60_pre_haug`) and its `eval.log`.
- `paired.py`, `sinkhorn_transitions.py`: the analysis scripts (run from the project folder).

## Reproduce

Code: `src/train.py` (`ConfusionMixer`, `--mix-confuse`, `--mix-confuse-pairs`; off by default).

```
python -m src.train --out runs/h_ftcfinal60_pre_haug --model preact_v --epochs 60 --lr 0.02 --warmup-epochs 1 --hard-aug 0.3 --hard-label-weight 0.5 --ema 0.999 --student-noise 0.04 --mix-alpha 1.0 --mix-prob 0.5 --mix-confuse 0.5 --val-size 0 --init runs/h_final600_pre_haug/final.pt --teacher runs/h_teacher_sam50/final.pt runs/h_teacher_sam50_s2/final.pt runs/h_teacher_aa50/final.pt runs/h_teacher_w40_50/final.pt
```

The twin uses `--init runs/h_fm600_pre_haug/final.pt`, the 45k teachers and the default `--val-size 5000`. The
control drops `--mix-confuse 0.5`.
