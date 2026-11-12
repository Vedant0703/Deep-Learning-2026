# Team submissions

One folder per Kaggle submission, plus unsubmitted `candidates/`. Each of Abhinav's folders has the model
(`model.pt`, under 5M parameters), the submitted CSV, a class-balanced CSV where measured, the training logs, its 45k
twin (`twin/`) and a README with the exact training command.

Public and private are the two halves of the 2,000 hidden images (about 1,000 each), so each score is good to about
±1 point. The team's final picks were Run M and Run N; Run M's private 0.893 placed 6th of 81 (18th on public).
"Twin val" is the 45k twin's accuracy on the 5,000 held-out training images; fs + bal is flip_shift TTA with
class-balanced decoding.

## Kaggle submissions, in the order they were made

| Run | By | Public | Private | Twin val (none / flip) | Twin val balanced (TTA in the README) | Recipe | Folder |
|---|---|---|---|---|---|---|---|
| Run A | Vedant | 0.875 | 0.865 | – / 97.08 | – | 4.98M PreAct wide ResNet (`preact_v`), 300 epochs, EMA, flip TTA, 45k images | [`Vedant_Runs-A-D`](Vedant_Runs-A-D/) |
| Run B | Vedant | 0.892 | 0.873 | – | – | Run A's recipe on all 50k images | [`Vedant_Runs-A-D`](Vedant_Runs-A-D/) |
| Run C | Vedant | 0.881 | 0.868 | – | – | robust augmentation (blur, JPEG, noise), 400 epochs, all 50k | [`Vedant_Runs-A-D`](Vedant_Runs-A-D/) |
| Abhinav_Run-D | Abhinav | 0.891 | 0.865 | 97.30 / 97.50 | – | KD from WRN-28-10 + input-noise robustness, 300 epochs, all 50k | [`Abhinav_Run-D`](Abhinav_Run-D/) |
| Abhinav_Run-E | Abhinav | 0.888 | 0.870 | 97.30 / 97.50 (its own val) | – | the 45k twin of Run D, submitted directly to measure the val-to-hidden gap | [`Abhinav_Run-E`](Abhinav_Run-E/) |
| Run D (Vedant) | Vedant | 0.895 | 0.872 | – | – | distilled from the averaged predictions of Runs A, B and C, 300 epochs, all 50k | [`Vedant_Runs-A-D`](Vedant_Runs-A-D/) |
| Abhinav_Run-F | Abhinav | 0.901 | 0.870 | 97.66 / 97.78 | 97.78 (suite mean 96.28 → 96.42) | function-matching KD (MixUp/CutMix) from WRN-28-10, 300 epochs, all 50k | [`Abhinav_Run-F`](Abhinav_Run-F/) |
| Scratch baseline | Dhanush | 0.878 | 0.858 | – | – | Dhanush's from-scratch baseline | not in the repo |
| Abhinav_Run-G | Abhinav | 0.901 | 0.870 | 97.40 / 97.58 | 97.56 (suite mean 96.23 → 96.30) | PreAct ResNet 4.98M (Vedant's preact_v), function-matching KD from WRN-28-10, 300 epochs, all 50k | [`Abhinav_Run-G`](Abhinav_Run-G/) |
| Abhinav_Run-H | Abhinav | 0.897 | 0.873 | 97.84 / 98.00 | 97.92 (suite mean 96.49 → 96.67) | function-matching KD from WRN-28-10, 800 epochs, all 50k (NYU HPC) | [`Abhinav_Run-H`](Abhinav_Run-H/) |
| Abhinav_Run-I | Abhinav | 0.900 | 0.876 | 97.92 / 97.82 | 97.90 (suite mean 96.59 → 96.77) | function-matching KD from a diverse 4-teacher ensemble (2x WRN-28-10 SAM, WRN-28-10 AutoAugment, WRN-40-10), 800 epochs, all 50k (NYU HPC) | [`Abhinav_Run-I`](Abhinav_Run-I/) |
| Abhinav_Run-J | Abhinav | 0.902 | 0.884 | 97.92 / 97.82 | 97.96 (suite mean 96.88) | Run I's model with flip_shift TTA + class-balanced decoding (no retraining) | [`Abhinav_Run-J`](Abhinav_Run-J/) |
| Abhinav_Run-K | Abhinav | 0.894 | 0.889 | 97.84 / 98.00 | 97.98 (suite mean 96.79) | Run H's model with flip_shift TTA + class-balanced decoding (no retraining) | [`Abhinav_Run-K`](Abhinav_Run-K/) |
| Abhinav_Run-L | Abhinav | 0.906 | 0.887 | 97.62 / 97.70 | 97.86 (suite mean 96.74; hard-factor suite 93.0 vs Run I's 88.4) | Run I's model fine-tuned for 100 epochs with on-the-fly hard augmentation (low resolution, zoom in/out, background clutter; the teacher sees the clean image; true-label weight 0.5 on augmented images) | [`Abhinav_Run-L`](Abhinav_Run-L/) |
| Abhinav_Run-M | Abhinav | **0.910** | **0.893** | 97.80 / 97.94 | 97.96 (suite mean 96.88; hard-factor suite 93.7 vs Run I's 88.4) | Run I's recipe from scratch with on-the-fly hard augmentation (30% of each batch: low resolution, zoom in/out, background clutter; the teacher sees the clean image; true-label weight 0.5), 600 epochs, all 50k (NYU HPC) | [`Abhinav_Run-M`](Abhinav_Run-M/) |
| Dhanush_Run-G-TTA | Dhanush | 0.904 | 0.873 | – / 97.58 | 97.76 (flip_shift, no balancing) | Abhinav's Run G weights with 10-view translation + flip probability TTA, no balancing (no retraining) | [`Dhanush_Run-G-TTA`](Dhanush_Run-G-TTA/) |
| Abhinav_Run-N | Abhinav | 0.909 | 0.886 | 98.06 / 98.06 | 98.20 (suite mean 97.03; hard-factor suite 93.9) | Run M's recipe on Vedant's PreAct 4.98M (preact_v): the only model to beat Run M on every twin metric, and the least correlated with it (McNemar 31 vs 19, p = 0.12) | [`Abhinav_Run-N`](Abhinav_Run-N/) |
| Abhinav_Run-O | Abhinav | 0.905 | 0.888 | 97.96 / 98.12 | 98.12 (suite mean 97.02; hard-factor suite 93.7) | Run N's recipe + SAM (rho 0.05) on the student: ties Run N on the twin (McNemar 18 vs 22, p = 0.64) and misses its hard-suite bar; submitted to see | [`Abhinav_Run-O`](Abhinav_Run-O/) |
| Abhinav_Run-P | Abhinav | 0.904 | 0.886 | 97.62 / 97.78 | 97.86 (suite mean 96.69; hard-factor suite 93.1) | Run N fine-tuned 60 epochs with confusion-aware MixUp/CutMix (cat-dog etc. partners): worse than Run N (McNemar 10 vs 27, p = 0.008) and tied with a plain fine-tune control | [`Abhinav_Run-P`](Abhinav_Run-P/) |

Two more team submissions on Nov 4 (public / private 0.887 / 0.873 and 0.890 / 0.870) have no description on Kaggle
and no files here.

## Candidates (trained, not submitted)

| Run | Twin val (none / flip) | Twin val balanced (TTA in the README) | Recipe | Folder |
|---|---|---|---|---|
| Abhinav_1600-epochs | 97.86 / 97.86 | 97.92 (suite mean 96.55 → 96.71) | function-matching KD from WRN-28-10, 1600 epochs, all 50k (NYU HPC) | [`candidates/Abhinav_1600-epochs`](candidates/Abhinav_1600-epochs/) |
| Abhinav_ens3 | 97.72 / 97.80 | 97.84 (suite mean 96.61 → 96.77) | function-matching KD from a 3-teacher ensemble (WRN-28-10 + 2x WRN-28-10 SAM), 800 epochs, all 50k | [`candidates/Abhinav_ens3`](candidates/Abhinav_ens3/) |
| Abhinav_sam-teacher | 97.86 / 97.78 | 97.82 (suite mean 96.47 → 96.71) | function-matching KD from one WRN-28-10 SAM teacher, 800 epochs, all 50k | [`candidates/Abhinav_sam-teacher`](candidates/Abhinav_sam-teacher/) |
| Abhinav_mixoff-600 | 97.76 / 97.84 | 97.86 (suite mean 96.82) | Run I's recipe at 600 epochs with mixing off for the last 30 (Vedant's trick) | [`candidates/Abhinav_mixoff-600`](candidates/Abhinav_mixoff-600/) |
| Abhinav_dist-600 | 97.54 / 97.82 | 97.76 (suite mean 96.61) | Run I's recipe at 600 epochs with the DIST loss (kd-alpha 0.67) instead of KL | [`candidates/Abhinav_dist-600`](candidates/Abhinav_dist-600/) |
| Abhinav_preact-600 | 97.78 / 97.88 | 97.94 (suite mean 96.74; hard-factor suite 88.3) | Run I's recipe at 600 epochs on Vedant's PreAct 4.98M (preact_v) | [`candidates/Abhinav_preact-600`](candidates/Abhinav_preact-600/) |

## Takeaways

- Function matching (the teacher labels the same MixUp/CutMix batch) was the big lever: 0.891 → 0.901.
- After that, more epochs, stronger or more diverse teachers and other architectures all tie at 97.8-98.0 twin val and 0.897-0.901 public.
- Class-balanced decoding (`--balance`) with flip_shift TTA adds about +0.26 on the shifted val suite. Every model under-predicts airplane and over-predicts automobile on the hidden set. On the public half it moved Run I by +2 images (Run J) and Run H by -3 (Run K), which is noise; on the private half it gained +8 and +16. It is the main reason the team moved from 18th on public to 6th on private.
- Round 8 (one change at a time to Run I's recipe: mixing off at the end, DIST loss, KD temperature 1) all tie or lose on the twin.
- Round 9: on-the-fly hard-factor augmentation (`--hard-aug 0.3 --hard-label-weight 0.5`, `src/hard.py`) tied Run I's twin on clean val and lifted a hard-factor suite (low resolution, scale, background) from 88.4 to 93.7. It gave the best scores: Run M 0.910 public / 0.893 private (from scratch, 600 epochs) and Run L 0.906 (fine-tune of Run I). A fixed set of 2,000 pre-made hard images did nothing.
- Run N (Run M's recipe on `preact_v`) beat Run M on every twin metric but not on either leaderboard half. SAM on the student (Run O) and confusion-aware mixing (Run P) did not help.
- Details: `../notes/key_takeaways.md` (distilled) and `../notes/training_log.md` (full log).
