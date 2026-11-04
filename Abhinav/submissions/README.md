# Abhinav's submissions

One folder per Kaggle submission, plus unsubmitted candidates. Each has the model (`model.pt`, under 5M parameters), the submitted CSV, a class-balanced CSV where measured, the training logs, its 45k twin (`twin/`) and a README with the exact training command.

| Run | Public | Twin val (none / flip) | Twin val balanced (flip) | Recipe | Folder |
|---|---|---|---|---|---|
| Abhinav_Run-D | 0.891 | 97.30 / 97.50 | – | KD from WRN-28-10 + input-noise robustness, 300 epochs, all 50k | [`Abhinav_Run-D`](Abhinav_Run-D/) |
| Abhinav_Run-E | 0.888 | 97.30 / 97.50 (its own val) | – | the 45k twin of Run D, submitted directly to measure the val-to-hidden gap | [`Abhinav_Run-E`](Abhinav_Run-E/) |
| Abhinav_Run-F | 0.901 | 97.66 / 97.78 | 97.78 (suite mean 96.28 → 96.42) | function-matching KD (MixUp/CutMix) from WRN-28-10, 300 epochs, all 50k | [`Abhinav_Run-F`](Abhinav_Run-F/) |
| Abhinav_Run-G | 0.901 | 97.40 / 97.58 | 97.56 (suite mean 96.23 → 96.30) | PreAct ResNet 4.98M (Vedant's preact_v), function-matching KD from WRN-28-10, 300 epochs, all 50k | [`Abhinav_Run-G`](Abhinav_Run-G/) |
| Abhinav_Run-H | 0.897 | 97.84 / 98.00 | 97.92 (suite mean 96.49 → 96.67) | function-matching KD from WRN-28-10, 800 epochs, all 50k (NYU HPC) | [`Abhinav_Run-H`](Abhinav_Run-H/) |
| Abhinav_Run-I | 0.900 | 97.92 / 97.82 | 97.90 (suite mean 96.59 → 96.77) | function-matching KD from a diverse 4-teacher ensemble (2x WRN-28-10 SAM, WRN-28-10 AutoAugment, WRN-40-10), 800 epochs, all 50k (NYU HPC) | [`Abhinav_Run-I`](Abhinav_Run-I/) |
| Abhinav_1600-epochs | – | 97.86 / 97.86 | 97.92 (suite mean 96.55 → 96.71) | function-matching KD from WRN-28-10, 1600 epochs, all 50k (NYU HPC) | [`candidates/Abhinav_1600-epochs`](candidates/Abhinav_1600-epochs/) |
| Abhinav_ens3 | – | 97.72 / 97.80 | 97.84 (suite mean 96.61 → 96.77) | function-matching KD from a 3-teacher ensemble (WRN-28-10 + 2x WRN-28-10 SAM), 800 epochs, all 50k | [`candidates/Abhinav_ens3`](candidates/Abhinav_ens3/) |
| Abhinav_sam-teacher | – | 97.86 / 97.78 | 97.82 (suite mean 96.47 → 96.71) | function-matching KD from one WRN-28-10 SAM teacher, 800 epochs, all 50k | [`candidates/Abhinav_sam-teacher`](candidates/Abhinav_sam-teacher/) |

Takeaways:

- Function matching (the teacher labels the same MixUp/CutMix batch) was the big lever: 0.891 → 0.901.
- After that, more epochs, stronger or more diverse teachers and other architectures all tie at 97.8-98.0 twin val and 0.897-0.901 public. The public LB has 1,000 images, so its noise is about ±0.95 points.
- Class-balanced decoding (`--balance`) adds +0.07 to +0.24 on the shifted val suite. Every model under-predicts airplane and over-predicts automobile on the hidden set.
- Details: `../notes/key_takeaways.md` (distilled) and `../notes/training_log.md` (full log).
