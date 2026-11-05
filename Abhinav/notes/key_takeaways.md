# Key takeaways: CIFAR-10 ResNet challenge (NYU DL Fall 2026, Project 1)

A distilled summary of what we learned from every run, for planning the next rounds and writing the report.
- **Detail:** the epoch-by-epoch observations behind each point are in `notes/training_log.md`.
- **Comparing runs:** `python scripts/compare_runs.py RUN ...` lines runs up by fraction of their schedule.
- **Robustness and TTA:** `python -m src.evaluate CKPT ... --tta none flip` gives clean and corruption-suite
  accuracy.

Last updated: 2026-11-03 01:30 EST, after Phase 3. The second submission was deliberately held (section 3, item 6).

---

## 1. The one-paragraph summary

The hidden evaluation set is a **natural distribution shift**: newly collected CIFAR-style images, like CIFAR-10.1.
- It is not a synthetic corruption. Models at 97-97.5% on our 5k validation split score only 87.5-89.2% public.
- **Public accuracy follows clean validation accuracy along a line** (probit slope about 1.2). Corruption robustness
  does not move a model off that line.
- So the lever is clean accuracy, and the most efficient way we found to raise it under the 5M-parameter cap is
  **knowledge distillation from a large WRN-28-10 teacher** (+0.5-0.7 over a label-only student).
- With a 97.4% teacher, the student matched it within 0.1.
- But **stronger teachers did not give a better student:**
  - a 98.0% teacher (trained 300 epochs) and a 97.7% 3-teacher ensemble both gave students statistically tied with
    the 97.4%-teacher student, at about 97.3 clean / 97.4-97.5 flip;
  - more student epochs (500 vs 300) did nothing either.
- **The 4.6M student, or the way it absorbs the teacher, is now the bottleneck.** Next-round ideas should change how
  the student learns (function matching with mixup, distillation hyperparameters, architecture), not the teacher.

---

## 2. Results at a glance

All val numbers are on the same stratified 5k held-out split (split seed 0). TTA "flip" averages the image and its
mirror. The suite mean covers 11 generic corruptions.

### Students (resnet_c: 3 stages, 64/128/256 channels, 3-4-3 blocks, 4,623,178 params)

| Run | Recipe | Val | Val (flip) | Suite mean (flip) | Noise 0.04 (flip) | Public |
|---|---|---|---|---|---|---|
| `base_c` | 200 ep, EMA, TrivialAugment, label smoothing 0.1 | 96.64 | 97.00 | 94.34 | 87.78 | – |
| `twin_kdn` | 300 ep, KD from `teacher_wrn` + student noise 0.04 | 97.30 | **97.50** | 96.03 | 96.14 | – |
| `final_kdn` (Run D) | same, all 50k | (twin) | (twin) | – | – | **0.891** |
| `twin_kdn500` | 500 ep, otherwise as `twin_kdn` | 97.36 | 97.26 | 96.13 | 96.56 | – |
| `twin_ens` | 300 ep, KD from a 3-teacher ensemble (97.70) | 97.20 | 97.34 | 96.10 | 96.26 | – |
| `twin_strong` | 300 ep, KD from `teacher_wrn300` (97.96) | 97.22 | 97.38 | 96.08 | 96.38 | – |

### Teachers (WRN-28-10, 36.5M params, EMA 0.999, no label smoothing; 45k unless noted)

| Teacher | Recipe | Val | Val (flip) |
|---|---|---|---|
| `teacher_wrn` | 200 ep, seed 0, TrivialAugment | 97.42 | 97.54 |
| `teacher_s1` | 200 ep, seed 1, TrivialAugment | 97.28 | 97.42 |
| `teacher_s2` | 200 ep, seed 2, AutoAugment | 97.06 (best 97.36) | 97.50 |
| ensemble {wrn, s1, s2} | averaged softmax | 97.70 | 97.78 |
| **`teacher_wrn300`** | **300 ep**, seed 0, TrivialAugment | **97.96** (best 98.06) | 97.90 |
| best ensemble with it, {wrn300, wrn, s2} | averaged softmax | – | 98.08 |
| `teacher50` | 300 ep, all 50k (for the final only; cannot be validated) | – | – |

### Team leaderboard (public)

| Run | Description | Public |
|---|---|---|
| A | Vedant: 4.98M wide ResNet, 300 ep, EMA, flip TTA, val 97.08% (45k) | 0.875 |
| B | Vedant: same recipe, all 50k | **0.892** |
| C | robust augmentation, 400 ep, all 50k | 0.881 |
| D | ours: KD from WRN-28-10 + noise robustness, 300 ep, all 50k | 0.891 |
| E | ours: D's 45k twin (`twin_kdn`), flip TTA, val 97.50% (analytics) | 0.888 |

Run E is the first submission with directly measured val accuracy: 97.50 flip maps to 0.888. D and E agree on 96.2% of
the hidden images and differ by 3 public images. All five team scores are within the public noise of each other.

The legitimate top of the leaderboard is about 0.914; the #1 entry at 1.000 is not credible.

---

## 3. Takeaways about the hidden set and the leaderboard

1. **Validation accuracy does not equal the leaderboard, but it predicts it.** 97.08 maps to 87.5 and 97.50 to 89.1.
   - A probit-linear fit through A and D (slope about 1.2) predicts 98.0 val → about 91.1 and 98.2 → about 91.9.
   - It is only two points; treat it as a rough guide.
2. **Synthetic robustness did not show on the leaderboard.**
   - D is +8.4 points at noise sigma 0.04 and +1.7 on the corruption mean over a `base_c`-like model, yet it scored
     exactly what its clean accuracy predicts.
   - This matches Taori et al. (NeurIPS 2020) and Miller et al. (ICML 2021, "Accuracy on the line"): synthetic
     robustness does not transfer to natural shifts.
3. **Pixel statistics of the evaluation images** (unlabeled, compared with training images, used only to choose
   which corruption families to test):
   - brighter upper tail;
   - slightly higher contrast with more clipped pixels;
   - some images more washed out;
   - more high-frequency detail;
   - no JPEG blockiness.
   Brightness, contrast and saturation robustness were never the problem (all within 0.5 of clean).
4. **The public score is noisy.** It is computed on at most 2,000 images (SE about 0.7), or about 1,000 if Kaggle
   splits public and private (SE about 1.0).
   - B (0.892) and D (0.891) are tied, 2 images apart.
   - **Choose final submissions by validation evidence; ignore public gaps under about 1.5 points.**
5. **Train every final model on all 50k images.** A to B (same recipe, +5k images) was +1.7 public. That is partly
   noise, but more data means more clean accuracy, which is the lever.
6. **Don't spend a submission on a model that validation says is tied with one already submitted.** After Phase 3
   all three student routes tied (McNemar p >= 0.43), so a new final would most likely score about 0.891 again, and the
   public noise would hide any difference. We held the second submission and saved about 7-10 units.

## 4. Takeaways about distillation (KD)

6. **Distillation is the biggest clean-accuracy lever we found:** +0.5 to +1.0 over a label-only student at matched
   points, and the final `twin_kdn` beat `base_c` by +0.66 (+0.50 with flip).
7. **Distillation front-loads learning:** +18-20 points at epochs 10-15. Its advantage narrows as training converges
   (+2.2 at 20% of the schedule, +0.8 at 80%). Less is left for the low-LR phase (+0.22 vs +0.84 for `base_c` over
   the last 30%).
8. **The student matches its teacher closely:** 97.30 / 97.50 flip vs the teacher's 97.42 / 97.54. Once that
   happens, **the teacher is the ceiling.**
9. **The distillation loss scale differs from cross-entropy** (0.9 x T^2 x KL with T = 4). Train losses of
   distillation runs can't be compared with those of label-only runs. A softer target (an ensemble) gives a lower
   loss without meaning faster learning.
10. **Run-to-run noise for an identical recipe is about 0.2-0.3 points** of val accuracy (nondeterministic GPU
    kernels, same seed). That is the floor for calling any difference real.

## 5. Takeaways about training length

11. **Longer student training did not help** when the teacher was the ceiling.
    - 500 vs 300 epochs: 97.36 vs 97.30 (no TTA), 97.26 vs 97.50 (flip).
    - Paired McNemar: p = 0.91 without TTA; with flip, 300 was slightly ahead, p = 0.21.
    - The 500-epoch run led by 0.3-0.4 mid-way, but that lead was gone by 80%. It was only +0.1-0.2 more robust,
      at 67% more compute.
12. **Longer teacher training helped a lot:** 300 vs 200 epochs gave 97.96 vs 97.42 (+0.54). Measured against the
    mean of the three 200-epoch teachers (97.25), it is +0.7.
    - The gain came in the last 25% (+0.70 as the LR went below about 0.01; the 200-epoch teacher gained +0.20 there).
    - The 36.5M-param teacher still had capacity to learn from the data. **Spend extra epochs on the teacher, not
      the student.**
13. **Compare runs of different lengths at the same fraction of their schedule.** That means the same LR under
    cosine decay. Early on, val accuracy depends mainly on epochs seen; the payoff of a longer schedule (if any)
    shows only in its low-LR phase.
14. **Cosine runs follow a typical shape:** a fast rise, then a mid-training plateau at high LR (val flat for 30+
    epochs while val loss still falls), then a final climb as the LR approaches zero.
    - Val loss falling while val accuracy is flat means better calibration, not more correct answers.

## 6. Takeaways about teachers and ensembles

15. **A 3-teacher ensemble of 200-epoch WRNs gains only +0.28** over its best member (97.70 vs 97.42; flip 97.78 vs
    97.54).
    - Members disagree on only 2.0-2.3% of images, and any two share 82-91 errors: same architecture, correlated
      mistakes.
16. **Adding weaker teachers to a strong one barely helps or hurts:** the best is +0.18 ({wrn300, wrn, s2}); adding s1
    is -0.08; all 4 is -0.02. An ensemble only pays off if its members are about equally strong.
17. **Augmentation diversity decorrelates errors more than seed diversity.** The AutoAugment teacher shares the
    fewest errors with the others (65 vs 78-91) and gives the highest "any member right" bound.
18. **AutoAugment + WRN-28-10 + no label smoothing overfits at the end.** Augmented-train accuracy reached 97.8%,
    above val; val loss rose from 0.105 to 0.113 over the last 50 epochs, and val accuracy fell from 97.36 to 97.06.
    - If using AutoAugment for a teacher: add label smoothing, keep `best.pt`, or train for fewer epochs.
19. **Flip TTA stops helping well-trained models:** +0.12 for `teacher_wrn`, -0.06 for `teacher_wrn300`, -0.10 for
    `twin_kdn500`. Pick the TTA mode per model on val.
20. **A better teacher did NOT give a better student.**
    - 97.42 teacher: student 97.30 / 97.50 flip.
    - 97.70 ensemble: 97.20 / 97.34.
    - 97.96 teacher: 97.22 / 97.38.
    - All tied (McNemar p >= 0.43 in every pair and TTA mode); suite means within 0.15.
    - The teacher-student gap grew from 0.1 to 0.5-0.7 as the teacher got stronger, so **the student's absorption is
      the limit.**
    - The stronger-teacher students trailed for most of training. The sharper 98% teacher gives a larger KL while
      the student is weak, and the student never pulled ahead.
    - Possible reasons:
      - student capacity;
      - the distillation recipe (no mixup-style function matching; T = 4 with alpha 0.9 plus noise);
      - the teacher-student size gap (Mirzadeh et al., AAAI 2020).

## 7. Takeaways about robustness training

21. **Student noise (Gaussian noise on the student's input; the teacher sees the clean image) works as designed.**
    - At noise 0.04: 96.1 vs 87.8. The student is more noise-robust than its own teacher (91.7).
    - Blur, sharpen and desaturation each improved by 0.4-1.1.
22. **Student noise costs about 0.5 clean val at the same epoch with distillation** (about 0 without it), but that gap
    closes over 300 epochs at matched fractions. It did not buy anything visible on the leaderboard (takeaway 2).
    Dropping it is a candidate simplification.
23. **JPEG robustness stays the weakest spot:** 86.2 at quality 50, 11 below clean. The hidden set shows no JPEG
    artifacts, though, so it probably doesn't matter for the score.
24. **AugMix is too slow on Colab's 12-CPU VMs:** about 60 s/epoch, CPU-bound in PIL. It would need a GPU
    implementation.
25. **Brightness/contrast jitter was not needed:** `base_c` was already within 0.2 of clean under ±20% brightness and
    contrast.

## 8. Takeaways about method and validation

26. **Twin runs:** every model trained on all 50k (no held-out set) gets a 45k/5k twin with the same recipe; the twin's
    val numbers are reported for it. The two runs' train losses must match (they did: 0.433 vs 0.433).
27. **No model in a validated comparison may have seen the val images, including teachers.** A teacher trained on 50k
    can only be used for a final model, never to validate.
28. **Bug found and fixed:** `--seed` also reshuffled the 45k/5k split.
    - A teacher with `--seed 1` would have trained on 4,526 of our 5,000 val images.
    - `--split-seed` (default 0) now fixes the split independently. Check this whenever a new seed is introduced.
29. **Use paired tests for close calls:** McNemar on the same 5k images, plus the corruption suite (11 x 5k images),
    which is more stable than one val number.
30. **Leakage audit (base_c), all clean:**
    - the 45k/5k split is disjoint and stratified;
    - the loaders hold exactly those images;
    - 0 exact duplicates within train, val to train, or test to train;
    - an independent re-evaluation matched the logged accuracy;
    - no overfitting (clean train 99.5% vs val 96.0%, val loss falling).
31. **Rules we followed:**
    - nothing trained, validated or selected on the 2,000 evaluation images; no pseudo-labels; no test-time BN
      adaptation;
    - no ensembles at submission, only single models under 5M params;
    - no external data or pretrained weights;
    - teachers trained from scratch on the allowed data.

## 9. Infrastructure lessons (Colab CLI + Kaggle)

32. **Colab prunes a CLI VM about 4 minutes after its kernel goes idle,** even while detached training processes run.
    Fix: a supervisor cell keeps the kernel busy, and jobs are queued as uploaded files (`scripts/colab.py`).
33. **The VM also dies when the Mac sleeps:** closing the lid on battery at 16:01 lost three runs (epochs 169, 145 and
    138 of 200) and about 20 units.
    - Fixes: `follow` backs up `last.pt` every 10-20 minutes; `put` uploads large files in chunks (to resume
      elsewhere); `caffeinate` blocks idle sleep.
    - **Keep the lid open and the charger in during runs.**
34. **`colab upload` of a 125 MB file fails with HTTP 500;** upload in 16 MB chunks with a checksum. `colab download`
    can hang, so every call has a timeout.
35. **zsh does not word-split `$ARGS`:** an unquoted variable holding several flags reached `train.py` as one argument.
    Write flags out, or use arrays.
36. **`--wait-for`** queues a run that starts as soon as its teachers' `final.pt` exists. That keeps the GPU busy
    without manual handoffs.
37. **GPU packing:**
    - One A100 40GB fits 4 WRN-28-10 teachers (about 3.8 GB each with AMP) at about 46 s/epoch each.
    - Packing 3 jobs is about 18% cheaper per epoch than 2, and units are billed per VM-hour.
    - A distillation student runs at 11 s/epoch alone, about 20 s with 3 jobs.
38. **Credentials:** the Colab CLI logs OAuth tokens at DEBUG level to `~/.config/colab-cli/colab.log` (chmod 600; never
    print it). Revoke "Google Cloud SDK" access after the project. The Kaggle CLI needs `six` installed.

## 9b. Round 4 (Nov 3): function matching breaks the student ceiling

39. **The diagnostic that explained the plateau:** on its 45k training images, every student agreed with its teacher
    100.00% (KL about 0.0003), and both teachers were 100% correct there.
    - The stronger teacher's generalization advantage was invisible in the signal the student learned from: the
      teacher had memorized the transfer set. It was not student capacity.
40. **Function matching fixed it** (Beyer et al., CVPR 2022): mixup/CutMix on the batch that both teacher and student
    see, so the teacher labels images it never memorized. Flags: `--mix-alpha 1.0 --mix-prob 0.5`.
    - The student went from 97.22 / 97.38 flip to **97.66 / 97.78 flip** (McNemar p = 0.026 / 0.015 vs the same
      recipe without it; p = 0.042 vs our previous best, `twin_kdn`).
    - The student now follows the 98% teacher more closely on unseen images: val agreement 98.36% vs 97.98%, KL
      0.046 vs 0.064.
41. **Other one-factor results** (all with the 98% teacher, 300 epochs):
    - Vedant's pre-activation 4.98M net: +0.20 (adopt).
    - SAM (ρ = 0.05): +0.11 (tie).
    - DIST loss: -0.05 (tie).
    - Mixing every batch: +0.31, not better than half the batches.
42. **Run E (`twin_kdn` submitted directly, 97.50 flip val):** 0.888 public vs D's 0.891. All five team submissions sit
    within the public noise of each other (87.5-89.2, SD 0.7).
43. **Test-time BN adaptation is allowed** (confirmed by the professor), but measured harmful.
    - The hidden set's BN-input statistics are only mildly shifted in early layers and most shifted in the last
      (semantic) layers.
    - On the held-out twin, adapting BN costs about 0.5 on clean val; it helps only for blur/JPEG-type shifts.
    - Not used.

## 9c. Round 5 (Nov 3 evening): architecture levers under FM, and HPC

44. **More FLOPs per parameter does not help under function matching** (300 epochs, same teacher, vs `r1_fm` 97.66 /
    97.78 flip):
    - strides (1,2,1), last stage at 16×16: -0.66 / -0.34, McNemar p = 0.001;
    - strides (1,1,2), stages 1-2 at 32×32, 3× the FLOPs: -0.30 / -0.20;
    - squeeze-and-excitation: -0.24 / -0.24.

    All three are also worse on every corruption family. s112 has the lowest val loss but not the best accuracy.
    Plain resnet_c stays the student.
45. **Two architectures with FM both score 0.901 public:** Run F (resnet_c, twin 97.78) and Run G (PreAct 4.98M, twin
    97.58). FM is the lever; architecture is second-order, and the 0.2 gap is below public-LB resolution.
46. **Patience and teacher strength look like the remaining levers.** At the same fraction of the schedule:
    - FM at 800 epochs runs +0.29 above the 300-epoch run at 52% of training;
    - a SAM-trained WRN-28-10 teacher runs +0.19 above `teacher_wrn300` at 31%.
47. **NYU HPC Cloud Bursting lessons:**
    - Access is OOD only.
    - Ship npz arrays, not 52k PNGs (NFS crawls).
    - Miniconda-latest has Python 3.14, which has no cu118 PyTorch wheel, so use a 3.11 env.
    - One A100 per run is about 4x faster than Colab's 4-per-GPU packing: 8 s/epoch for an FM student, 22 s/epoch
      for WRN-28-10 + SAM.
    - The L4 partition never booted (three CF → requeue cycles).
    - Interactive Jupyter sessions get preempted. Monitor through the dashboard file viewer instead; the batch jobs are
      independent of it.

## 9d. Round 6 (Nov 4): the FM student has reached its teacher

48. **One-factor results vs `h_fm800`** (97.84 / 98.00 flip, suite mean 96.49 flip):
    - pure KD (`--kd-alpha 1.0`): -0.20 / -0.18;
    - an equal-strength SAM teacher: +0.02 / -0.22, after leading by +0.28 mid-run;
    - no student noise: 0.00 / -0.10;
    - **1600 epochs (`h_fm1600`): +0.02 / -0.14, suite mean +0.16 / +0.06, McNemar p = 1.0 / 0.41.** 800 → 1600
      epochs is a tie, so `h_final1600` is not adopted.

    All four are within the ±0.2 run-to-run noise. Mid-run leads keep vanishing in the low-LR phase, so only a final
    paired comparison counts.
49. **`h_fm800` matches its teacher** (98.9% agreement, 98.00 vs `teacher_wrn300` 97.90 flip). Neither more epochs nor
    the student settings move it. At the time this pointed to the teacher as the binding constraint; takeaway 51
    disproves that.
50. **Ensembles of same-recipe models saturate at about 98.16** (2 teachers 98.08; 4 students 98.16; 2 + 2 98.16)
    because their errors are correlated. The remaining training lever is a **diverse** teacher set (AutoAugment,
    WRN-40-10).
51. **Diverse teachers raise the target but not the student.**
    - Ensembles that include AutoAugment teachers reach 98.38-98.40, against 98.18 for the same-recipe ones. Adding
      2 more diverse teachers (6 in all) stays at 98.38.
    - Yet `h_fm800_div` ties `h_fm800`: 97.92 / 97.82 flip, McNemar p 0.71 / 0.32. Its 50k final (Run I) scores 0.900
      public, against 0.901 for F and G.
    - Every 800-epoch FM student lands at 97.78-98.00 flip whatever its target. The 4.62M student recipe is the
      ceiling; the extra correct answers of an ensemble sit at the decision margin and are not inherited.
    - What remains is prediction-time (class-balanced decoding and kNN smoothing, if the professor allows them) and
      the choice of the final 2 picks.

52. **Class-balanced decoding is approved by the professor** (Nov 4) and measured on every twin (flip TTA).
    - It is ±0.08 on clean val, which is already balanced, and +0.07 to +0.24 on the suite mean. JPEG gains +1.1 to
      +2.2.
    - It changes 61-78 of 2,000 hidden predictions per model. Every model under-predicts airplane (168-177) and
      over-predicts automobile (217-221) before balancing.
    - kNN smoothing was not covered by the approval and is not used.

53. **The best decoding is flip_shift TTA (10 views) + balancing at T=1**, chosen as one rule for every model (mean of
    7 twins, clean / suite): 97.87 / 96.72, against 97.80 / 96.46 for plain flip (what was submitted).
    - The suite gain holds on all 7 twins (+0.20 to +0.35).
    - Temperature does not matter (T = 0.1, the near-exact 200-per-class assignment, ties T = 1).
    - EM prior estimation (no balance assumption) is worse than plain, as expected if the skew comes from the shift.

54. **Switching mixing off at the end does nothing for the FM student**, tested twice.
    - Colab, F's recipe at 300 epochs with mixing off for the last 15: 97.58 / 96.46 vs 97.70 / 96.50.
    - HPC, `div` at 600 epochs with mixing off for the last 30: 97.86 / 96.82 vs `div`'s 97.96 / 96.88 at 800
      (McNemar p = 1.0). Train accuracy jumps to about 94% once mixing stops; val does not move.
    - Vedant's runs use it with hard labels, where it may matter more; under FM the teacher already gives clean
      targets.

55. **Round 8: no one-factor change to the `div` recipe beats it** (600-epoch twins, fs + bal vs `div`'s 97.96 / 96.88
    at 800): mix-off 97.86 / 96.82, KD temperature 1 97.90 / 96.78, DIST 97.76 / 96.61 (the worst suite).
    - DIST's softmax gives Sinkhorn little to work with: balancing moved 35 hidden predictions, against 58-86 for KL
      students, and left 185-218 per class.
    - The 4.62M FM student is at its ceiling (about 97.9 clean, 96.8 suite with balancing); the submission choice is
      now about decoding and hedging, not training.

56. **Distillation helps every architecture, and the best baseline is not the best student** (Round 9, 300 epochs,
    flip TTA). Baselines: preact_v 97.26, resnet_b 97.22, resnet_c 97.12, resnet_a 96.20. With FM KD: resnet_c 97.78,
    then preact_v / resnet_b / resnet_c_s112 all at 97.58. Gains range from +0.32 (preact_v) to +0.92 (resnet_c_se).
    resnet_c stays the student.
57. **Hard-factor augmentation works when it is on the fly, with the teacher on the clean image** (Round 9).
    - A fixed set of 2,000 hard images (×3) did nothing: twin hard mean 88.0 vs 88.4.
    - `--hard-aug 0.3 --hard-label-weight 0.5` from scratch (600 epochs) tied Run I's twin on clean / suite
      (fs + bal 97.96 / 96.88, McNemar p = 0.68) and lifted the hard suite from 88.4 to 93.7 (low-res 16 px
      80.8 → 94.7, zoom-out 86.2 → 96.2, background 80.5 → 95.6).
    - Why: each image gets a new transform every epoch, and the student learns the teacher's answer for the clear
      image, so the teacher's own weakness on hard images is not copied.
    - **On Kaggle it is the first training change since Round 6 to move the public score:** Run M (from scratch)
      0.910 and Run L (fine-tune of Run I) 0.906, vs Run J's 0.902 with the same decoding. The hidden set contains
      images with these factors.

## 10. Ideas for the next round (ranked by expected value per compute unit)

Phase 3 moved the bottleneck from the teacher to the student, so student-side ideas come first. Teachers already
trained and available: `teacher_wrn` (97.42), `teacher_wrn300` (97.96), `teacher_s1`, `teacher_s2` (45k), and
`teacher50` (50k). No new teacher compute is needed to test ideas 1-4.

1. **Mixup/CutMix during distillation ("function matching", Beyer et al., CVPR 2022):** the teacher labels the same
   mixed images the student sees. This is reported as the key trick for absorbing a strong teacher into a small
   student, and it addresses takeaway 20 directly. The code already supports it (`--mix-alpha`, `--mix-prob`). About
   1 GPU-hour per 300-epoch student; test it with `teacher_wrn300`.
2. **Distillation hyperparameters,** three students in parallel on one A100:
   - pure distillation (`--kd-alpha 1.0`);
   - temperature 2 vs 6;
   - no student noise (it bought nothing on the leaderboard and costs about 0.5 mid-training).
3. **Student architecture within 5M params:** resnet_c (64/128/256, 3-4-3) was chosen from 40-epoch label-only
   screening. A wider or deeper variant may have more headroom under distillation. Re-screen 2-3 shapes with
   distillation, not label-only.
4. **Teacher-assistant distillation** (Mirzadeh et al., 2020): distill the 98% teacher into a mid-size network (for
   example WRN-16-8, about 11M params), then into the student. Only worth trying if 1 and 2 don't move the student.
5. **Teacher improvements (500 epochs, SAM, equal-strength ensembles, other architectures):** deprioritized. The
   student did not benefit from a teacher going from 97.4 to 98.0, so a stronger teacher is unlikely to help until
   the student can absorb it.

## 11. Compute ledger

- **Start:** 143 Colab compute units on 2026-11-02. A100 at 6.73 units/hour.
- **Balance:** **67.9 units at 01:25 on 2026-11-03,** after Phase 3, with the VM stopped.
- **Big costs:**
  - lost VMs, about 20 units plus earlier screening;
  - Phase 2 (`final_kdn`, `twin_kdn`, `twin_kdn500`), about 17;
  - Phase 3 (4 teachers + 2 student twins), about 35.
- **Plan for the next round:**
  - each 300-epoch distillation student costs about 1 GPU-hour (about 7 units), less when 3 share an A100;
  - a full student round (3 twins + 1 final) is about 20-25 units;
  - keep about 10 units in reserve for a final 50k model before the Nov 6 deadline.
