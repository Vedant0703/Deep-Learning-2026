# Training log: observations and inferences

A running record of what the training curves show, written while runs train. It is meant for choosing the next
round's changes and for the report. Each round should start from the "Open questions" section at the bottom.

Curves are compared with `python scripts/compare_runs.py [--remote] RUN ...`. Runs are compared at the same
fraction of their schedule (cosine decay over the whole run), not at the same epoch.

## How to read the numbers

- **Training-batch numbers:** `train loss` and `train acc` are measured on training batches after augmentation
  (TrivialAugment, random erasing) and, with `--student-noise`, after added pixel noise.
  - They sit *below* val accuracy for the whole run. That is expected with this much augmentation, and it is not
    underfitting in the usual sense.
- **Distillation loss scale:** with distillation, `train loss = 0.1 * CE(label-smoothed) + 0.9 * T^2 * KL(teacher || student)`,
  where T = 4, so T^2 = 16.
  - It starts around 12 and cannot be compared with the loss of a run without distillation.
- **EMA:** val numbers come from the EMA weights (decay 0.999, which averages over about 1,000 steps, roughly 6
  epochs).
  - For the first few epochs val accuracy is far below train accuracy because the EMA still includes early weights.
- **Noise level of val accuracy:** the 5k val split has a standard error of about 0.24 points at 97%. A difference
  under about 0.3 points between two runs is not evidence by itself. The corruption suite (`src/evaluate.py`, 11 x 5k
  images) gives more stable comparisons.

## Runs in this round (Phase 2, started 2026-11-02 18:02 EST)

| Run | Data | Epochs | Recipe | Purpose |
|---|---|---|---|---|
| `final_kdn` | all 50k | 300 | resnet_c, EMA, KD from `teacher_wrn` (alpha 0.9, T 4), student noise 0.04, TrivialAugment | submission candidate |
| `twin_kdn` | 45k / 5k val | 300 | same | validates the submission recipe (clean val + corruption suite) |
| `twin_kdn500` | 45k / 5k val | 500 | same | does a longer schedule help? |

Teacher: `teacher_wrn` (WRN-28-10, 36.5M params, 200 epochs, EMA, no label smoothing): val 97.42%, flip TTA 97.54%.

## Evidence from earlier runs

Sources: `runs/base_c` (200 epochs, complete), and the Phase 1 runs `kd_c`, `trn_c` and `kdn_c` (200 epochs each).
The Phase 1 runs were lost with the VM at epochs 169, 145 and 138; their curves were recovered from the follow log.

Val accuracy relative to `base_c` at the same fraction of training:

| Run | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% |
|---|---|---|---|---|---|---|---|---|
| `kd_c` (KD) | +9.00 | +2.24 | +1.82 | +1.72 | +1.10 | +1.26 | +1.18 | +0.78 |
| `kdn_c` (KD + student noise) | +9.26 | +2.30 | +1.46 | +1.22 | +0.74 | +0.64 | | |
| `trn_c` (noise in the input pipeline) | +0.86 | -0.72 | -0.62 | -0.04 | -0.26 | -0.14 | +0.06 | |

Inferences:

1. **Distillation's advantage is largest early and shrinks as training converges:** +2.2 at 20%, about +1.2 at
   50-70%, +0.8 at 80%. The final gain is probably about +0.4 to +0.6 points.
   - Distillation mostly speeds up learning. A longer schedule narrows its advantage rather than widening it.
2. **`kd_c` plateaued at about 97.0% from 66% of training onward** (best 97.10 at epoch 133), while its val loss
   kept falling (0.118 to 0.115).
   - A falling loss with flat accuracy means better-calibrated confidence on images it already gets right, not
     more correct answers.
   - The student is within about 0.4 points of the teacher (97.42%). **The teacher is now the ceiling**: a better
     teacher is a more direct lever than more student epochs.
3. **Noise alone costs nothing at convergence.** `trn_c` lost 0.6-0.7 points at 20-30% of training, then caught up
   (+0.06 at 70%).
4. **Noise costs more with distillation than without:** `kdn_c` trails `kd_c` by 0.4-0.6 points at 30-60% of training.
   - The student has to reproduce the teacher's clean-image prediction from a noisy image, a harder target than a
     one-hot label.
   - Whether this closes by the end is the first thing to check in `twin_kdn`.
5. **`base_c` gained only +0.02 val accuracy over its last 10% of training,** while its val loss kept falling slowly.
   - Augmented-train accuracy (93.0%) stayed 3.7 points below val accuracy (96.6%), so the model never overfit.
   - Longer training may add a little. The 500-epoch twin tests it.
6. **Speed and cost:** one A100 runs 2 of these jobs at about 15.5 s/epoch each, or 3 at about 19.7 s/epoch each.
   - That is 0.13 vs 0.15 epochs/s in total. Packing 3 runs per VM is about 18% cheaper per epoch, and compute
     units are billed per VM-hour.

## Observations during Phase 2

- **18:03:** Warmup is correct. LR logged at the start of epochs 1-4 is 0.0002, 0.033, 0.067, 0.100, then cosine
  decay (checked against the formula by `compare_runs.py`).
  - Epoch-1 loss is 12.8 (the distillation scale).
  - EMA lag as expected: epoch 4 shows train acc 53.5% vs val 24.3%; by epoch 13 val (76.3%) has passed train (73.3%).
- **18:08:** All three runs are at about 19.7 s/epoch.
  - Expected finish: `twin_kdn` and `final_kdn` about 19:40; `twin_kdn500` about 20:20 (it speeds up once the others
    finish).
  - At epoch 12, `final_kdn` (50k) and `twin_kdn` (45k) have nearly the same train loss (3.33 vs 3.32). The 50k run
    shows no sign of a problem it couldn't reveal without a val split.
- **18:12, `twin_kdn` at 10% (epoch 30):** val 93.24%, train acc (augmented) 79.7%, train loss falling at
  0.038/epoch. LR is on schedule for all three runs.
  - **The relaunch reproduces the lost run.** At the same absolute epoch, while all runs are still near the peak LR:

    | Epoch | `base_c` | lost `kdn_c` | `twin_kdn` | `twin_kdn500` |
    |---|---|---|---|---|
    | 10 | 38.6 | 59.4 | 57.7 | 59.1 |
    | 20 | 80.8 | 90.1 | 89.9 | 89.9 |
    | 30 | 89.7 | 93.5 | 93.2 | |

    - Run-to-run spread for the identical recipe is about 0.2-0.3 points by epoch 20-30. It comes from
      nondeterministic GPU kernels; the seed is the same.
    - That spread is the floor for calling any later difference real.
  - **Distillation accelerates early learning enormously:** +20 points at epoch 10 and +18 at epoch 15 over
    `base_c`, which learns only from labels.
    - The soft targets carry much more information per image than a one-hot label.
  - **The 50k run's train loss is slightly lower** at the same epoch (2.16 vs 2.26 at epoch 29-30). It takes 11% more
    optimizer steps per epoch (195 vs 175), so this is expected and not a warning sign.
  - **Speed:** 19.7 s/epoch for the twins and 21.0 s/epoch for the 50k run. ETA about 19:45 for both 300-epoch runs.
- **18:22, `twin_kdn` at 20% (epoch 60): val 95.28%; `twin_kdn500` at 10% (epoch 50): val 95.06%.**
  - **Schedule length doesn't matter yet.** At the same absolute epoch, all distilled runs are within 0.1 points:

    | Run | Epoch 40 | Epoch 50 |
    |---|---|---|
    | lost `kdn_c` (200 epochs) | 94.60 | 95.14 |
    | `twin_kdn` (300 epochs) | 94.58 | 95.18 |
    | `twin_kdn500` (500 epochs) | 94.64 | 95.06 |

    - Their LRs differ only slightly at this point (0.087-0.098).
    - So matched-fraction comparisons this early mostly reflect "more epochs so far": the 500-epoch run is +1.8
      at 10% because it has had 50 epochs vs 30.
    - **300 vs 500 can only be judged in the last third of training,** when the LR anneals and the accuracy gains
      happen (`base_c` went from 95.2 at 50% to 96.6 at 100%).
  - **Val accuracy has entered its high-LR plateau:** `twin_kdn` reached its best of 95.46 at epoch 55, then 95.28 at
    epoch 60; val loss is flat at about 0.209.
    - At LR around 0.09 the weights keep moving, and the EMA can only partly smooth that out.
    - The next big gains should come from 50% of training onward.
  - **Train loss keeps falling** (1.74 for `twin_kdn`, -0.016/epoch). Augmented train accuracy is 82.8%, still
    12.5 points under val. No sign of overfitting.
  - **The 50k and 45k runs remain in step** (train loss 1.753 vs 1.758 at epochs 56/59).
- **18:32, `twin_kdn` at 30% (epoch 90): val 95.82%** (+1.96 over `base_c` at 30%).
  - **The noise cost under distillation is reproducible.** At the same absolute epoch, `kd_c` (no noise) leads both
    noise runs by about 0.5 points; the two noise runs are identical at epoch 90:

    | Epoch | `kd_c` | lost `kdn_c` | `twin_kdn` |
    |---|---|---|---|
    | 60 | 95.68 | 95.32 | 95.28 |
    | 70 | 96.16 | 95.46 | 95.40 |
    | 80 | 96.20 | 95.70 | 95.44 |
    | 90 | 96.36 | 95.82 | 95.82 |

    - So the noise penalty is about 0.5 points of clean val in mid-training, not run-to-run noise.
    - The question is whether the robustness it buys is worth more than that on the hidden set. The `twin_kdn`
      corruption suite will tell.
  - **A higher LR at the same epoch hasn't cost accuracy.** `twin_kdn` at epoch 90 has LR 0.081 vs 0.060 for
    `kdn_c`, yet the same val. In this range the EMA val accuracy is insensitive to the LR.
  - **Train loss is flattening:** -0.005/epoch for `twin_kdn` and -0.0025/epoch for `final_kdn` over the last 10 epochs.
    This is typical of the high-LR plateau; it should fall again as the LR decays.
  - **Backups work:** every 10 minutes, `runs/{final_kdn,twin_kdn,twin_kdn500}/last.pt` (55.6 MB each) is refreshed
    locally and checked for completeness.
- **18:38, `twin_kdn500` at 20% (epoch 100): val 95.66%.**
  - **The LR is starting to separate the runs.** At epoch 100, `twin_kdn` (LR 0.076) has val 96.20% and val loss
    0.186; `twin_kdn500` (LR 0.091) has 95.66% and 0.203.
    - At epoch 60 they were tied (95.28 vs 95.20).
    - The run with the faster-decaying LR now converts training into val accuracy sooner. That's the expected cosine
      behaviour, not a sign that the 500-epoch run is worse; it should catch up when its own LR decays.
  - **Train loss tracks the LR the same way:** 1.47 vs 1.59 at epoch 100. A lower LR means less weight noise and a
    lower loss on the same data.
- **18:43, `twin_kdn` at 40% (epoch 120): val 96.50%** (best 96.52 at epoch 117), +2.02 over `base_c` at 40%.
  - **First evidence that a longer schedule helps.** At matched fractions, `twin_kdn` (300 epochs) leads the lost
    `kdn_c` (200 epochs, same recipe) by +0.68 at 20%, +0.50 at 30% and +0.80 at 40%.
    - That is above the 0.2-0.3 run-to-run spread.
    - If it holds to the end, 200 to 300 epochs is worth a few tenths of a point. That is a hint for 300 to 500, but
      gains usually shrink as schedules lengthen.
  - **The noise cost holds at about 0.4-0.6 points** at the same epoch: epoch 120 gives `kd_c` 96.92 vs `twin_kdn`
    96.50 and `kdn_c` 96.30.
  - **Val loss ticked up** from 0.1675 (epoch 114) to 0.1725 (epoch 120), while val accuracy stayed at about 96.5.
    That is a normal high-LR fluctuation; the 10-epoch rising-loss flag has not triggered.
  - **Augmented-train accuracy is 85.5%, 11 points under val.** The gap is narrowing steadily (13.6 at 10%,
    12.6 at 20%, 11.7 at 30%) as the model fits the augmented data better, but it is still far from overfitting.
- **18:52, `twin_kdn` at 50% (epoch 150): val 96.28%** (+1.08 over `base_c` at 50%).
  - **Mid-training plateau:** val accuracy has stayed between 96.16 and 96.54 for 30 epochs (epochs 121-150), and
    the change over the last 10% is -0.20.
    - Val loss, meanwhile, reached a new low (0.1630 at epoch 150).
    - The same pattern appeared in `kd_c`: confidence calibrates before more images flip to correct. The accuracy
      gains should come in the last 40%, as the LR goes from 0.05 to 0.
  - **The 300-vs-200 lead shrank to +0.34 at 50%** (it was +0.68, +0.50, +0.80 at 20-40%).
    - With swings like this the lead is within noise. The 300-epoch advantage over 200 is not established.
  - **At the same absolute epoch the two twins are still identical:** epoch 143 gives `twin_kdn` 96.32
    (LR 0.055) and `twin_kdn500` 96.28 (LR 0.085).
    - Through the first half of training, val accuracy depends on epochs seen, not on the LR.
    - Any benefit of the longer schedule must come from its longer low-LR phase at the end.
  - **Projection for `twin_kdn`:** about 97.0% (range 96.9-97.2) without TTA at the end, plus about 0.3 with flip TTA.
    This assumes a final-half gain like `kdn_c`'s (+0.8 from 50% to 69%) and `kd_c`'s plateau.
  - **The augmented-train gap is 9.3 points** (train acc 87.0%), still narrowing at about 1.5 points per 10% of the
    schedule.
- **18:55, `twin_kdn500` at 30% (epoch 150): val 96.12%,** val loss 0.183, LR 0.080.
  - Still level with `twin_kdn` at the same absolute epoch (96.28 at epoch 150, within noise).
  - At matched fractions it leads (+0.30 at 30%), only because it has seen more epochs.
  - No anomalies: LR on schedule, no loss spikes, val loss at a new low.
- **19:02, `twin_kdn` at 60% (epoch 180): val 96.54%** (best 96.78 at epochs 174-175), +0.88 over `base_c` at 60%.
  - **The plateau broke as predicted:** val rose from 96.24 (epoch 151) to 96.78 (epoch 174) as the LR fell from
    0.051 to 0.039, then dipped to 96.54.
  - **Val loss bump to watch:** lowest 0.1484 at epoch 175, now 0.1556 (+5%), while train loss keeps falling
    (0.96, -0.0045/epoch).
    - Five epochs is too short to call it a trend; `compare_runs.py` flags it if the rise lasts 10+ epochs.
    - If it persists, it would suggest the student is starting to fit the augmented training set more than the
      teacher's general signal. Check at 70%.
  - **The lead over the lost 200-epoch `kdn_c` is now +0.24 at matched fraction,** within noise. So far, 200 to 300
    epochs shows no clear benefit at matched fractions.
  - **The augmented-train gap is 8.0 points** (train acc 88.6%), steadily narrowing.
  - **The 500-epoch twin at 35% (epoch 173):** 96.38, versus `twin_kdn` at the same epoch, 96.64-96.78.
    - This is the first time the shorter schedule pulls ahead at the same absolute epoch (about +0.3). Its LR is
      now 0.039 vs 0.073, and the low-LR harvest has started for the 300-epoch run.
- **19:07:** `twin_kdn` is at epoch 197, val 96.80%. Val loss is back down to 0.151 (0.148 at epoch 200), so the
  epoch-175-180 bump was a fluctuation, not a trend.
- **19:08, `twin_kdn500` at 40% (epoch 200): val 96.30%.**
  - **Same LR, same accuracy.** Matched fraction means matched LR (both runs use cosine from the same peak), so this
    is the cleanest 300-vs-500 check so far:

    | Run | Epoch | LR | Train loss | Val loss | Val acc |
    |---|---|---|---|---|---|
    | `twin_kdn` | 120 | 0.067 | 1.344 | 0.1725 | 96.50 |
    | `twin_kdn500` | 200 | 0.066 | 1.280 | 0.1694 | 96.30 |

    - At the same LR, the 80 extra epochs bought a slightly lower train loss and val loss, but no val accuracy
      (-0.20, within noise).
    - Early evidence that 500 epochs will not beat 300 by much.
- **19:11, `twin_kdn` at 70% (epoch 210): val 97.08%, a new best,** with val loss at a new low of 0.1383.
  - It is +1.28 over `base_c` at 70%.
  - It is already equal to Vedant's 97.08%, which was measured with flip TTA; ours is without TTA, with 30% of
    training left.
  - **At matched fraction, the noise penalty has closed:** `twin_kdn` (KD + noise, 300 epochs) is now +0.10 ahead of
    the lost `kd_c` (KD only, 200 epochs) at 70%.
    - At the same absolute epoch, noise still cost about 0.5 at epochs 60-120.
    - Reading: the longer schedule paid for the noise. The student needs more epochs to learn the
      clean-prediction-from-noisy-input mapping, and 300 epochs gives it that.
  - **Train loss is falling faster again** (-0.008/epoch, vs -0.0045 at 60%): the low-LR harvest phase. Augmented
    train accuracy is 90.4%, 6.7 points under val.
  - **Projection revised up:** about 97.3% without TTA at the end (range 97.1-97.5).
- **19:21, `twin_kdn` at 80% (epoch 240): val 97.14%,** a new best, with val loss at a new low of 0.1246.
  - +0.94 over `base_c` at 80%, and +0.16 over `kd_c` at 80% (within noise, but consistently positive since 70%).
  - **Second plateau, at about 97.0:** val stayed between 96.84 and 97.14 over epochs 211-240, while val loss fell
    from 0.138 to 0.125.
    - `kd_c` showed the same plateau at about 97.0 from 66%. That looks like the ceiling this student/teacher pair
      reaches on clean val (the teacher is at 97.42).
    - The last 20% (LR 0.01 to 0) added +0.44 for `base_c`; a similar small gain is expected here.
  - **`final_kdn` (50k) remains in step:** train loss 0.738 vs `twin_kdn`'s 0.741 at 70%.
  - **Inference for the next round:** clean val is close to saturating for this recipe at about 97.0-97.3.
    - More clean accuracy needs a stronger teacher, or a student that matches it better (for example longer KD
      or a higher temperature).
    - The leaderboard gap, though, is set by robustness. The `twin_kdn` corruption suite shows how much was bought.
- **19:25, `twin_kdn500` at 50% (epoch 250): val 96.72%,** with val loss 0.1503 and LR 0.051.
  - **At matched LR (50% of each run), the 500-epoch run now leads:**

    | Run | Epoch | Val acc | Val loss | Train loss |
    |---|---|---|---|---|
    | `twin_kdn` | 150 | 96.28 | 0.1630 | 1.158 |
    | `twin_kdn500` | 250 | 96.72 | 0.1503 | 1.098 |

    - A +0.44 accuracy lead is borderline against the 0.2-0.3 noise level. The val loss gap (-8%) is the more
      convincing signal.
    - At 40% it was -0.20 in accuracy, but the val loss was already slightly better.
  - **Revised expectation:** the 500-epoch run may end a few tenths above `twin_kdn`.
- **19:31, `twin_kdn` at 90% (epoch 270): val 97.24%** (new best), val loss 0.1173 (new low), LR 0.0027.
  - It is +0.10 over the 80% mark, the small final-phase gain expected.
  - Augmented-train accuracy is 92.8%, 4.4 points under val, so no overfitting even this late.
- **19:41, `twin_kdn` finished: val 97.30%, flip TTA 97.50%** (best 97.34%). Final val loss is 0.1121, the lowest of
  the run; val loss fell monotonically through the whole low-LR phase.
  - **Versus `base_c`:** +0.66 at 100% (97.30 vs 96.64), +0.50 with flip TTA (97.50 vs 97.00). Val loss is 40% lower
    (0.112 vs 0.187), so the model is much better calibrated.
  - **Gains over the last 30%:** +0.22 (97.08 at 70% to 97.30). For `base_c` it was +0.84 over the same stretch.
    Distillation front-loads learning, so less is left for the low-LR phase.
  - **The projection held:** about 97.3 was forecast at 70%.
  - **Final augmented-train accuracy is 93.2%,** 4.1 points under val. The model never overfit the training set.
  - It is within 0.12 of the teacher (97.42; flip 97.54 vs 97.50). On clean images the student has essentially matched
    its teacher.
- **19:44, corruption suite (`src/evaluate.py`, 5k val):**

  | Model | TTA | Clean | Suite mean | Worst (JPEG 50) | Noise 0.02 | Noise 0.04 | Blur | Sharpen |
  |---|---|---|---|---|---|---|---|---|
  | `twin_kdn` | none | 97.30 | 95.71 | 85.46 | 96.64 | 95.86 | 96.08 | 96.52 |
  | `twin_kdn` | flip | **97.50** | 96.03 | 86.16 | 96.96 | 96.14 | 96.64 | 96.82 |
  | `twin_kdn` | flip_shift | 97.34 | **96.08** | **86.38** | 96.96 | 96.22 | 96.60 | 96.84 |
  | `base_c` | flip | 97.00 | 94.34 | 82.86 | 94.90 | 87.78 | 95.58 | 95.90 |
  | `teacher_wrn` (earlier) | flip | 97.54 | 95.16 | 81.2 | | 91.7 | | |

  - **Student noise worked:** at noise 0.04, +8.4 over `base_c` (96.14 vs 87.78).
    - The student is now **more noise-robust than its own teacher** (91.7): it learned the teacher's clean prediction
      from noisy inputs.
  - **Broad gains beyond noise:** blur +1.1, sharpen +0.9, noise 0.02 +2.1, saturation 0.5 +0.4.
    - The suite mean is up +1.7 (96.03 vs 94.34) and clean +0.5.
    - Brightness and contrast were already fine and stay within 0.2 of clean.
  - **JPEG remains the weak spot:** 86.2, -11 from clean, though +3.3 over `base_c`. The evaluation images show no
    JPEG blockiness, so this may matter less for the leaderboard, but it is the largest remaining gap in the suite.
  - **TTA choice: flip.** It gives the best clean accuracy (97.50 vs 97.34). `flip_shift` is only +0.05 better on the
    suite mean, which is within noise, and it costs 0.16 clean.
- **19:47, `final_kdn` (all 50k) finished.**
  - Final train loss is 0.433 with augmented-train accuracy 93.0%, identical to `twin_kdn` (0.433, 93.2%), so the
    twin's val numbers apply to it.
  - **Dry-run submission** (flip TTA): 2,000 rows, ids in sample order.
    - Class counts are airplane 176, automobile 216, bird 199, cat 210, deer 195, dog 212, frog 184, horse 205,
      ship 203, truck 200. They are balanced; airplane is lowest, as it was for every model so far.
  - **Agreement on the 2,000 unlabeled evaluation images** (a sanity check only, not used to choose): 96.2% with
    `twin_kdn` and 94.0% with `base_c`. The new recipe changes about 120 of `base_c`'s 2,000 predictions.
  - **Team leaderboard so far:** Run A 0.875, Run B 0.892 (same recipe as A, all 50k), Run C 0.881 (robust
    augmentation, 400 epochs).
- **19:44, `twin_kdn500` at 60% (epoch 300): val 96.84%,** val loss 0.1515.
  - At matched LR it leads `twin_kdn` (96.54 and 0.1556 at 60%) by +0.30 in accuracy, with a slightly lower loss.
  - So it has led at 50% and 60%, by about 0.3-0.4. It now runs at 16 s/epoch because `twin_kdn` has finished.
- **19:58, `twin_kdn500` at 80% (epoch 400): val 97.14%,** val loss 0.1213.
  - **Tied with `twin_kdn` at 80%** (97.14, 0.1246). The 0.3-0.4 lead at 50-60% has gone; only a slightly lower val
    loss (-3%) remains.
  - It runs at 11.1 s/epoch alone on the GPU. One distillation student per A100 is about 1.8x faster per epoch than
    three, but costs more per epoch (3 runs: 0.15 epochs/s in total; 1 run: 0.09). The end-of-run corruption suite
    will separate a real gain from noise better than a single 5k val number.

## Leaderboard result and what it implies (2026-11-02 19:54 EST)

**Run D** (`final_kdn`, flip TTA, description "Run D: KD from WRN-28-10 + noise robustness, 300 epochs, all 50k
images") scored **0.891 public**.

| Run | Recipe | Val (flip TTA) | Public |
|---|---|---|---|
| A | Vedant: 4.98M wide ResNet, 300 epochs, EMA, 45k | 97.08 | 0.875 |
| B | Vedant: same recipe, all 50k | not reported | 0.892 |
| C | robust augmentation, 400 epochs, all 50k | not reported | 0.881 |
| D | ours: KD + student noise, 300 epochs, all 50k | 97.50 (twin) | 0.891 |

1. **B and D are tied.** A public score is measured on at most 2,000 images (SE about 0.7 points), or about 1,000 if
   Kaggle splits public and private (SE about 1.0). The 0.001 gap is 2 images.
   - Even A vs D (1.6 points) is only about 1.5-2 standard errors. **Choose final submissions by validation
     evidence, not by public-score gaps under about 1.5 points.**
2. **Synthetic robustness did not buy a visible leaderboard gain.** D is +8.4 at noise 0.04 and +1.7 on the
   corruption mean over a `base_c`-like model, yet it lands where its clean val predicts.
   - This matches the robustness literature: robustness to synthetic corruptions does not transfer to natural
     distribution shifts (Taori et al., NeurIPS 2020). Accuracy on a naturally shifted test set is mostly a linear
     function, in probit space, of in-distribution accuracy (Recht et al., ICML 2019, CIFAR-10.1; Miller et al.,
     ICML 2021, "Accuracy on the line").
   - The hidden set behaves like a CIFAR-10.1-style natural shift (newly collected images), not like noise.
3. **The trend line, with a large caveat** (two points, A and D, both noisy):
   - The probit slope is about 1.2, so clean val predicts public as 97.8 → about 90.3, 98.0 → about 91.1,
     98.2 → about 91.9.
   - The legitimate leaderboard top (about 91.4) would correspond to about 98.1% clean val. **The lever for the next
     round is clean accuracy,** not more corruption robustness.
4. **Training on all 50k images matters more than expected:** A to B was +1.7 public for the same recipe. That is
   partly noise, but it also follows the line: more data means higher clean accuracy, which means more on the
   hidden set. Every final model should be trained on 50k.

## 300 vs 500 epochs: answered (2026-11-02 20:45 EST)

`twin_kdn500` finished at val 97.36% (flip TTA 97.26%), best 97.38%. `twin_kdn` (300 epochs) finished at 97.30%
(flip 97.50%).

Paired McNemar test on the same 5k val images (run on the Mac):

| TTA | Only 500 right | Only 300 right | p |
|---|---|---|---|
| none | 40 | 38 | 0.91 |
| flip | 26 | 37 | 0.21, slightly favouring 300 |

- **Conclusion: 500 epochs is no better than 300** for this student and teacher. The 0.3-0.4 lead at 50-60% of the
  schedule was gone by 80%.
- The final val loss is slightly lower (0.109 vs 0.112), meaning slightly better calibration, which does not change
  accuracy.
- This fits inference 2 above: the student was already at the teacher's level, so extra epochs had nothing left to
  learn from that teacher.
- **Corruption suite** (mean over 11 corruptions):

  | TTA | 500 epochs | 300 epochs | Clean, 500 vs 300 |
  |---|---|---|---|
  | none | 95.93 | 95.71 | 97.34 vs 97.30 |
  | flip | 96.13 | 96.03 | 97.26 vs 97.50 |
  | flip_shift | 96.12 | 96.08 | 97.44 vs 97.34 |

  - 500 epochs is +0.1 to +0.2 more robust, with clean accuracy mixed. That is not worth 67% more compute.
- **Decision (user, 20:50):** the Phase 3 student twins were switched from 500 to 300 epochs before they started. That
  saves about 17 units and 2 hours. **The baseline for Phase 3 is now `twin_kdn`** (300 epochs: 97.30, flip 97.50,
  suite mean 96.03 with flip).

## Phase 3: stronger teacher vs teacher ensemble (started 2026-11-02 20:10 EST)

Goal: raise clean accuracy through a better distillation target. Two routes are compared head to head with 500-epoch
students. The baseline is `twin_kdn500`: the same student recipe distilled from the single 45k `teacher_wrn`.

**Fairness rule:** no teacher used in a validated comparison has seen the 5k val images.
- Every comparison uses 45k teachers; the 50k teacher is only used for a final submission model, like the 50k
  students.
- Bug found and fixed first: `--seed` also reshuffled the 45k/5k split. With `--seed 1`, 4,526 of our 5,000 val
  images would have been training data for that teacher. `--split-seed` (default 0) now fixes the split
  independently of `--seed`.

| Run | Data | Epochs | Recipe | Role |
|---|---|---|---|---|
| `teacher_s1` | 45k | 200 | WRN-28-10, seed 1, TrivialAugment | ensemble member |
| `teacher_s2` | 45k | 200 | WRN-28-10, seed 2, AutoAugment (different augmentation, for diversity) | ensemble member |
| `teacher_wrn300` | 45k | 300 | WRN-28-10, seed 0 | strong-teacher route, validated version |
| `teacher50` | all 50k | 300 | WRN-28-10, seed 0 | strong-teacher route, final version |
| `twin_ens` | 45k | 500 | resnet_c student; KD from {`teacher_wrn`, `teacher_s1`, `teacher_s2`} (averaged softened predictions) | ensemble route |
| `twin_strong` | 45k | 500 | resnet_c student; KD from `teacher_wrn300` | strong-teacher route |

- **Shared settings:** all teachers use EMA 0.999 and no label smoothing. All students use EMA 0.999 and student
  noise 0.04, the same as `twin_kdn500`, so only the teacher differs.
- **Scheduling:** both twins were queued with `--wait-for` and start automatically when their teachers' `final.pt`
  exists.
- **Decision rule:** choose the route by the twins' clean val and corruption suite against `twin_kdn500`. Then
  train only the winner's 50k final:
  - ensemble route: same 45k teachers, student on 50k;
  - strong route: `teacher50`, student on 50k.
- **Expected costs:** the ensemble student runs three teacher forward passes per batch, about 1.7x the cost of a
  single-teacher student. Phase 3 total is about 65 compute units (balance 103 at the start) and about 9-10 hours
  of wall-clock time.
- **GPU memory at the start** (3 teachers + `twin_kdn500`): 13.3 of 40 GB, about 3.8 GB per WRN-28-10 with AMP,
  GPU at 100%.
- **20:50:** twins re-queued at 300 epochs (see the 300 vs 500 section); they had not started. The baseline is
  `twin_kdn`.
- **20:47, teachers at 25% (epoch 50 of 200), about 46 s/epoch with 4 teachers sharing the A100:**
  - `teacher_s1` (seed 1): val 95.16%. `teacher_s2` (seed 2, AutoAugment): 95.02%. The original `teacher_wrn` had
    94.98% at the same epoch, so the new seeds are on the same track.
  - **`teacher_s2`'s train loss is much lower** (0.35 vs 0.51) at the same val accuracy. AutoAugment's CIFAR policy
    is milder than TrivialAugment's random op strengths, so its training batches are easier.
    - That is a different view of the data, which is what makes an ensemble member useful: it should make
      different mistakes.
- **21:13, epoch 72:** `teacher_s2` (AutoAugment) leads with 96.04 vs `teacher_s1` 95.66 and `teacher_wrn300` 95.74.
- **21:15, `teacher_wrn300` at 25% (epoch 75): val 95.96%,** LR 0.087.
  - Against the 200-epoch `teacher_wrn`: +0.98 at the same fraction (94.98 at epoch 50), but -0.18 at the same
    epoch (96.14 at epoch 75, LR 0.071).
  - This is the same pattern as the students: early on, accuracy follows epochs seen, and a longer schedule pays
    off (if at all) only in its low-LR phase.
- **21:22, `teacher50` at 25% (epoch 75):** train loss 0.4659 and augmented-train accuracy 83.66%. Its 45k twin
  `teacher_wrn300` had 0.4656 and 83.71% at the same epoch, so the 50k teacher tracks it exactly.
  - It runs at 49.6 s/epoch vs 46.4 (11% more images). ETA about 01:00.
- **21:36, ensemble members at 50% (epoch 100):**

  | Teacher | Val acc | Val loss | Train loss | Augmented-train acc |
  |---|---|---|---|---|
  | `teacher_s1` | 96.30 | 0.111 | 0.391 | |
  | `teacher_s2` | 96.40 | 0.114 | 0.234 | 92.0 |
  | original `teacher_wrn` (same epoch) | 96.42 | 0.107 | 0.387 | |

  - All three are within 0.12 points: each new member is as strong as the original.
  - `teacher_s2` fits its (AutoAugment) training data far more closely, at the same val accuracy. It generalizes
    from a different, easier training distribution, which is the source of ensemble diversity.
  - After they finish, the useful numbers are the 3-teacher ensemble's own val accuracy (via `src.evaluate` with
    `a+b+c`) and how often the members disagree.
- **22:12, ensemble members at 75% (epoch 150, LR 0.016):**

  | Teacher | Val acc | Val loss |
  |---|---|---|
  | `teacher_s1` | 96.84 | 0.099 |
  | `teacher_s2` (epoch 149) | 97.30 | 0.105 |
  | original `teacher_wrn` (epoch 150) | 97.22 | 0.093 |

  - `teacher_s2` (AutoAugment) is the strongest of the three so far. `teacher_s1` trails by about 0.4, which is
    seed variation; it should narrow as the LR goes to zero.
  - `teacher_s2`'s augmented-train accuracy is 95.8%, only 1.5 points under val, much closer to fitting its
    training data than the TrivialAugment teachers (91.6%).
- **22:15, `teacher_wrn300` at 50% (epoch 150): val 96.70%,** val loss 0.0994.
  - At the same fraction, it is +0.28 over the 200-epoch `teacher_wrn` (96.42 at epoch 100), with a lower val loss
    (0.099 vs 0.107).
  - The real test is the last 30%: `teacher_wrn` ended at 97.42. The longer teacher needs to end clearly above that,
    around 97.7 or more, for the strong-teacher route to be worth it.
- **22:23, `teacher50` at 50%:** train loss 0.363 vs 0.360 for `teacher_wrn300` at the same epoch, so it is on track.
- **22:44, near the end of the ensemble members (epoch 189-190 of 200, LR below 0.001):**
  - `teacher_s1`: val 97.26%, val loss 0.092, falling. Healthy.
  - **`teacher_s2` (AutoAugment) is overfitting:**
    - Val loss rose from 0.1048 (epoch 150) to 0.1128 (epoch 189), and val accuracy dipped from 97.30 to 97.14.
    - Augmented-train accuracy (97.8%) is now *above* val. With AutoAugment's milder policy and no label
      smoothing, WRN-28-10 nearly memorizes its training views by the end.
    - **Consequence for distillation:** a teacher that has memorized its training images gives near one-hot
      targets on them, which carry less "dark knowledge".
    - It is mitigated here because the student's batches use TrivialAugment views, which `teacher_s2` never
      trained on.
    - For future teachers with AutoAugment: use label smoothing, or select by `best.pt`, or train for fewer epochs.
  - `teacher_wrn300` (63%): 97.18%, val loss 0.093, still improving.
- **22:55, ensemble members finished:** `teacher_s1` 97.28% (flip 97.42%); `teacher_s2` 97.06% (flip 97.50%, best
  97.36% before its late overfitting).
- **Ensemble check** (on the Mac, 5k val, averaged softmax):

  | | No TTA | Flip TTA |
  |---|---|---|
  | `teacher_wrn` | 97.42 | 97.54 |
  | `teacher_s1` | 97.28 | 97.42 |
  | `teacher_s2` | 97.06 | 97.50 |
  | **Ensemble of 3** | **97.70** | **97.78** |
  | "Any member right" upper bound | 98.64 | 98.58 |

  - **The ensemble gains +0.28 (no TTA) and +0.24 (flip) over the best single teacher.** That is the new ceiling the
    `twin_ens` student can approach, up from 97.42. On the VM, `twin_ens` measured the ensemble at 97.72 (CUDA vs MPS
    numerics).
  - **Disagreement:** members disagree on 2.0-2.3% of images without TTA; `teacher_s2` (AutoAugment) is the most
    different (2.30% and 2.34% vs 2.00% between the two TrivialAugment teachers).
    - Each pair is wrong together on 82-91 images (1.6-1.8%), and those are mostly the same hard images.
    - That shared-error floor is why the 3-teacher ensemble gains only about 0.3: same architecture and similar
      recipes means correlated mistakes.
  - **Inference:** a more diverse ensemble would raise the ceiling more, for example a different architecture family
    (ResNeXt, PyramidNet, DenseNet), or teachers trained with different strong augmentations. The 98.6 upper bound
    shows that only about a third of the members' disagreements are resolved correctly by averaging.
  - **Expected effect if the student matches the ensemble as closely as it matched `teacher_wrn`:** clean val
    about 97.6-97.7 (flip about 97.8). On the A/D trend line that is about +0.6 to +1.0 public, a modest step.
- **23:06, `teacher_wrn300` at 75% (epoch 225): val 97.32%,** val loss 0.0859.
  - At the same fraction, `teacher_wrn` had 97.22 and 0.0934: +0.10 accuracy and 8% lower loss.
  - Speed is up to 34.7 s/epoch now that `teacher_s1` and `teacher_s2` are done. ETA about 00:05.
  - **Outlook:** it would need about +0.3 in the last 25% to reach about 97.7. `teacher_wrn` gained +0.20 over its last
    25% (97.22 to 97.42), so expect about 97.5-97.6: probably weaker than the 3-teacher ensemble (97.70).
- **23:13, `twin_ens` at 10% (epoch 30): val 93.30%,** the same as `twin_kdn` at epoch 30 (93.24).
  - **Its train loss is lower** (2.107 vs 2.260). The averaged ensemble target is softer (higher entropy) than a
    single teacher's, so the KL term is smaller for the same student. This is not a sign of faster learning.
  - It runs at 40 s/epoch while sharing the GPU with two teachers. ETA about 02:00.
- **23:16, `teacher50` at 75%:** train loss 0.232 vs 0.230 for `teacher_wrn300` at the same epoch, so it is on track.
  `twin_ens` at 20% (epoch 60): 95.16, level with `twin_kdn` (95.28).
- **23:44, `teacher_wrn300` at epoch 291 of 300: val 98.02%. My 75% forecast (97.5-97.6) was wrong.**

  | Epoch | 225 | 235 | 245 | 255 | 265 | 270 | 275 | 280 | 285 | 291 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Val acc | 97.32 | 97.46 | 97.66 | 97.64 | 97.70 | 97.74 | 97.92 | 97.96 | 98.02 | 98.02 |
  | Val loss | 0.0859 | 0.0855 | 0.0852 | 0.0863 | 0.0821 | 0.0816 | 0.0792 | 0.0787 | 0.0773 | 0.0770 |

  - **It gained +0.70 over its last 25%,** a steady climb as the LR went below about 0.01. Over the same stretch,
    `teacher_wrn` (200 epochs) gained only +0.20, from 97.22 to 97.42, and was flat over its last 30 epochs.
  - **Inference: the longer schedule matters for the big teacher, but not for the small student.** The 36.5M-param
    WRN still had capacity to keep learning from the data. The 4.6M student was limited by its teacher (it matched
    `teacher_wrn` within 0.1), so 500 student epochs added nothing.
    - **Rule for this project: spend extra epochs on the teacher, not on the student.**
  - **One teacher at 98.0 beats the 3-teacher ensemble (97.70).** A better-trained teacher beat averaging three
    weaker, correlated ones.
    - The `twin_strong` vs `twin_ens` comparison now tests whether a 0.3-point-better teacher gives a better student.
    - The 5k val SE is about 0.2, so 98.02 vs 97.42 is about 3 SE; a paired test follows when it finishes.
  - **What this implies for `teacher50`** (same recipe, all 50k): it should be at least as strong on unseen images.
    On our trend line, a student near 97.8-97.9 clean would map to about 90.3-91% public.
- **00:00, `teacher_wrn300` final: 97.96%, flip TTA 97.90%** (best 98.06%).
  - That is +0.54 / +0.36 over `teacher_wrn`. Flip TTA does not help it, as with `twin_kdn500`: longer-trained models
    gain less from view averaging.
  - `twin_strong` started at 28 s/epoch (sharing the GPU with `twin_ens` and `teacher50`).
- **Ensembles that include the 98% teacher** (Mac, flip TTA, 5k val, averaged softmax):

  | Ensemble | Acc | Any member right |
  |---|---|---|
  | `teacher_wrn300` alone | 97.90 | |
  | + `teacher_wrn` | 98.00 | 98.44 |
  | + `teacher_s1` | 97.82 | 98.34 |
  | + `teacher_s2` | 97.94 | 98.70 |
  | + `teacher_wrn` + `teacher_s2` | **98.08** (best) | 98.78 |
  | + `teacher_wrn` + `teacher_s1` | 97.84 | 98.62 |
  | all 4 teachers | 97.88 | 98.92 |

  - **Adding weaker teachers to the strong one gains at most +0.18 and can lose** (+`teacher_s1`: -0.08; all 4:
    -0.02).
    - Averaging pulls the strong member's correct answers toward the weaker members' mistakes.
    - The best combination (+0.18) is within noise, and as a distillation target it would triple the teacher cost
      per student step.
  - **`teacher_s2` (AutoAugment) is the most complementary member:** it shares the fewest errors with `teacher_wrn300`
    (65 vs 78-83 for the others) and gives the highest "any member right" bound. Augmentation diversity
    decorrelates errors more than seed diversity does.
  - **`twin_ens` at 30% (epoch 90):** 95.72, level with `twin_kdn` (95.82).
  - **`twin_strong` at 10% (epoch 30):** 92.72, about 0.5 behind `twin_ens` (93.30) and `twin_kdn` (93.24), with a
    higher train loss (2.48 vs 2.11-2.26).
    - Likely reason: the 300-epoch teacher fits its training images more closely (augmented-train accuracy 95.3%
      vs about 94% for `teacher_wrn`), so its targets on training batches are sharper. The KL term is then larger
      while the student is weak.
    - Check at 50-100% whether the gap closes.
  - **`teacher50` at epoch 299:** train loss 0.127 and augmented-train accuracy 95.4%, matching its 45k twin
    (0.133, 95.3%).
  - **00:12, `twin_ens` at 40% (epoch 120): 96.08, -0.42 vs `twin_kdn` (96.50)** at the same epoch and LR. It was
    -0.10 at 30%.
    - A possible cause is the softer averaged target (lower train loss, weaker gradient signal per image) while the
      LR is high.
    - Watch whether it catches up in the low-LR phase, as `twin_kdn` did from its plateau.
  - **Inference: an ensemble is only worth it if its members are about equally strong.** For a future ensemble teacher,
    use two or more 300-epoch teachers (ideally one with AutoAugment and one with TrivialAugment), not 200-epoch
    ones.

## Phase 3 result: the student is the bottleneck (2026-11-03 01:25 EST)

Final numbers (5k val; corruption suite of 11 x 5k on the VM):

| Student (300 ep, KD + student noise) | Teacher val | Clean | Flip | Flip+shift | Suite mean (none / flip / flip+shift) |
|---|---|---|---|---|---|
| `twin_kdn` | 97.42 (`teacher_wrn`) | 97.30 | **97.50** | 97.34 | 95.71 / 96.03 / 96.08 |
| `twin_ens` | 97.70 (ensemble of 3) | 97.20 | 97.34 | **97.44** | 95.85 / 96.10 / **96.17** |
| `twin_strong` | 97.96 (`teacher_wrn300`) | 97.22 | 97.38 | 97.34 | 95.76 / 96.08 / 96.16 |

Paired McNemar on the same 5k images (Mac):

| TTA | `twin_kdn` vs `twin_ens` | `twin_kdn` vs `twin_strong` | `twin_ens` vs `twin_strong` |
|---|---|---|---|
| none | 39 vs 35, p=0.73 | 52 vs 49, p=0.84 | 44 vs 45, p=1.00 |
| flip | 32 vs 25, p=0.43 | 41 vs 36, p=0.65 | 38 vs 40, p=0.91 |
| flip_shift | 25 vs 29, p=0.68 | 35 vs 36, p=1.00 | 43 vs 40, p=0.83 |

(Each cell: images only the first model gets right vs only the second.)

- **All three students are statistically tied** (every p >= 0.43), on clean accuracy and on the suite (within 0.15).
- **A teacher that is 0.3-0.55 points better did not make a better 4.6M student.** The student stops at about 97.2-97.3
  without TTA and about 97.4-97.5 with flip, whatever the teacher above 97.4.
  - With `teacher_wrn`, the student matched its teacher (gap 0.1).
  - With the 98.0 teacher the gap is 0.7, and with the ensemble 0.5. **The student's ability to absorb the teacher
    is now the limit,** not the teacher.
- **Mid-training curves foreshadowed this:** both new students trailed `twin_kdn` by 0.2-0.7 at the same epoch for most
  of training and never pulled ahead.
  - `twin_strong` started 0.5 behind (its sharper teacher gives a larger KL loss) and caught up only to level.
  - `twin_ens` swung between -0.42 and +0.14.
- **Possible reasons** (to test next round, not established):
  1. **Capacity:** 4.6M params may be near this architecture's ceiling at about 97.3-97.5.
  2. **The distillation recipe limits transfer:** T = 4, alpha = 0.9 with 10% label-smoothed CE, student noise on
     half the batch, 300 epochs. Function matching (Beyer et al., 2022) argues that small students need mixup-style
     views and long schedules to absorb a strong teacher.
  3. **A capacity gap:** distillation is known to work less well when the teacher is much larger than the student
     (Mirzadeh et al., AAAI 2020, "teacher assistant"). The 98% teacher may be too far from the student.
- **Decision (user, 01:25):** hold the second submission and stop the VM. A new 50k final would most likely score about
  Run D's 0.891, and the public score's ±0.7-1.0 noise would not reveal a difference. Compute balance after Phase 3:
  67.9 units.

## Round 4: making the student absorb a strong teacher (started 2026-11-03 10:25 EST)

Full plan and analysis: `~/.claude/plans/the-high-level-goal-rosy-star.md`. Summary of the analysis:
- **The public LB uses 1,000 of the 2,000 hidden images:** all 50 teams' scores are exact multiples of 0.001.
  - The SE per score is about 1.0, and the private 1,000 decide the ranking.
  - Teams at 0.906-0.914 report their best of many tries, so they are probably about 90-90.5 true.
- **Any two team submissions agree on only 92.7-93.7% of the hidden images.** All of them under-predict airplane
  (170-177) and frog (180-184). It is a harder natural shift: newly collected images, CIFAR-10.2-like.
- **Vedant's recipe:**
  - pre-activation ResNet, widths 59/118/236, blocks 4/4/4, stem 29, 4,980,392 params;
  - Nesterov SGD, bs 128, label smoothing 0.1, 300 epochs;
  - TrivialAugment + erasing + MixUp(0.2)/CutMix(1.0) at p 0.5, off for the last 15 epochs; EMA.
  - Results: 96.86 / flip 97.08 on his own split (seed 42, so not comparable to ours). Run C (+blur/JPEG/noise)
    fell 1.1 public.
- **Two rival explanations for "a better teacher gives no better student":**
  1. **The teacher memorized the transfer set:** students are distilled on the 45k images the teacher was trained on.
  2. **Student capacity:** every student ends at about 93.2% augmented-train accuracy.

**Round 1 (one factor per arm, 300 epochs, 45k, KD from `teacher_wrn300`, student noise 0.04; baseline
`twin_strong`: 97.22 / 97.38 flip):**

| Arm | Change | Tests |
|---|---|---|
| `r1_fm` | `--mix-alpha 1.0 --mix-prob 0.5` | function matching (Beyer et al., CVPR 2022): the teacher labels the same mixed image |
| `r1_fm1` | `--mix-alpha 1.0 --mix-prob 1.0` | function matching on every batch |
| `r1_pre` | `--model preact_v` | Vedant's architecture: full 4.98M params, pre-activation, deeper, same FLOPs |
| `r1_dist` | `--kd-loss dist --kd-alpha 0.67` | DIST loss (Huang et al., NeurIPS 2022), for stronger teachers |
| `r1_sam` | `--sam-rho 0.05` | SAM optimizer (Foret et al., ICLR 2021), about 1.35x cost |

- **Decision rule:**
  - Δ = mean of the no-TTA and flip deltas vs `twin_strong`.
  - **Adopt** if Δ ≥ +0.15 (suite mean no more than 0.3 lower, NLL no more than 5% higher); **strong** if
    Δ ≥ +0.35 or McNemar p < 0.1; **drop** if Δ ≤ -0.15.
  - On ties, keep function matching (free) and drop DIST and SAM.
- **New code** (all off by default):
  - `--kd-loss dist`, `--sam-rho`, `--mix-off-epochs`, `--model preact_v`;
  - `src.evaluate --baseline` (exact McNemar), `--reference` (agreement/KL), `--split train`, `--no-suite`.
- **Smoke tests passed:** DIST(z, z) = 0, the SAM step has norm ρ, weights restore exactly, and BN statistics stay
  frozen during the second pass. `--baseline` reproduces the earlier McNemar (52 vs 49, p = 0.84).
- **Step 0 diagnostic (Mac, clean images, no TTA):**

  | | 45k training images (the transfer set) | 5k val |
  |---|---|---|
  | `teacher_wrn300` / `teacher_wrn` accuracy | 100.00 / 100.00 | 97.96 / 97.42 |
  | `twin_strong` / `twin_kdn` accuracy | 100.00 / 100.00 | 97.22 / 97.28 |
  | Student-teacher top-1 agreement | 100.00% / 100.00% | 97.98% / 98.26% |
  | KL(teacher ‖ student) | 0.0002 / 0.0004 | 0.064 / 0.051 |

  - **Verdict: the transfer set is memorized, not a capacity limit.** On the images it is distilled on, the student
    reproduces the teacher exactly. But both teachers are 100% correct and near one-hot there, so the stronger
    teacher's generalization advantage simply isn't in the signal the student learns from.
  - The stronger teacher is followed *less* closely on val (97.98% vs 98.26% agreement), which is consistent with that.
  - **Priority: function matching** (the teacher labels mixed images it has never seen), then SAM (generalization).
    Architecture is a lower priority.

### Round 1 progress
- **About 10:55, all arms at 10% (epoch 30)** vs `twin_strong` (92.72):

  | Arm | Val | Δ | Speed |
  |---|---|---|---|
  | `r1_sam` | 92.92 | +0.20 | 41 s/epoch |
  | `r1_pre` | 92.60 | | 39 s/epoch |
  | `r1_fm` | 92.58 | | 35 s/epoch |
  | `r1_fm1` | 92.40 | | 35 s/epoch |
  | `r1_dist` | 92.40 | | 36 s/epoch |

  All level within noise. SAM costs about 1.17x per epoch here and `preact_v` about 1.1x (5 jobs share the GPU).
- **About 11:12, all arms at 20% (epoch 60)** vs `twin_strong` (94.86, val loss 0.221):

  | Arm | Val | Δ | Val loss |
  |---|---|---|---|
  | `r1_sam` | 95.26 | +0.40 | 0.215 |
  | `r1_fm1` | 95.14 | +0.28 | **0.190** |
  | `r1_pre` | 95.04 | +0.18 | 0.230 |
  | `r1_fm` | 94.92 | +0.06 | 0.202 |
  | `r1_dist` | 94.68 | -0.18 | **0.184** |

  - Mid-training gaps of ±0.4 have reversed before (the 500-epoch run), so this is not yet evidence.
  - **Lower val loss for the function-matching and DIST arms** suggests better-calibrated students, the effect their
    papers describe.
- **About 11:37, all arms at 30% (epoch 90)** vs `twin_strong` (95.54, val loss 0.216); `twin_kdn` had 95.82:

  | Arm | Val | Δ | Val loss |
  |---|---|---|---|
  | `r1_fm1` | **96.16** | **+0.62** | **0.159** |
  | `r1_pre` | 95.78 | +0.24 | 0.199 |
  | `r1_fm` | 95.64 | +0.10 | 0.176 |
  | `r1_sam` | 95.64 | +0.10 | 0.192 |
  | `r1_dist` | 95.38 | -0.16 | 0.165 |

  - **`r1_fm1` (function matching on every batch) leads,** its lead grew from 20% to 30%, and it has the lowest val loss.
  - That fits Step 0: the teacher labels mixed images it never memorized, so the student gets real dark knowledge
    on every batch.
  - By epoch 105 the high-LR plateau narrowed the lead in accuracy (96.04 vs 95.98 for `r1_fm`), but its val loss stays
    clearly the best (0.151).
- **About 12:05, all arms at 40% (epoch 120)** vs `twin_strong` (95.84, val loss 0.183); `twin_kdn` had 96.50:

  | Arm | Val | Δ | Val loss |
  |---|---|---|---|
  | `r1_fm1` | **96.36** | **+0.52** | **0.143** |
  | `r1_sam` | 96.34 | +0.50 | 0.163 |
  | `r1_fm` | 96.16 | +0.32 | 0.157 |
  | `r1_pre` | 96.14 | +0.30 | 0.181 |
  | `r1_dist` | 95.60 | -0.24 | 0.156 |

  - **Four of five arms lead the baseline by 0.3-0.5.** That is now consistent across 30-40%, with the right
    mechanisms leading (function matching and SAM).
  - DIST trails in accuracy despite a low loss.
  - Note `twin_kdn` (the weaker 97.4 teacher) had 96.50 here: the 98% teacher's students are only now catching up to
    it mid-training. The final 30% decides whether they pass it.
- **About 12:20, all arms at 50% (epoch 150)** vs `twin_strong` (96.26, val loss 0.170); `twin_kdn` had 96.28:

  | Arm | Val | Δ | Val loss |
  |---|---|---|---|
  | `r1_fm1` | **96.58** | +0.32 | **0.134** |
  | `r1_sam` | 96.54 | +0.28 | 0.154 |
  | `r1_fm` | 96.48 | +0.22 | 0.145 |
  | `r1_pre` | 96.30 | +0.04 | 0.182 |
  | `r1_dist` | 96.14 | -0.12 | 0.150 |

  - Function matching (both strengths) and SAM have led for three straight milestones and are now also ahead of
    `twin_kdn`. That is the first time any 98%-teacher student has beaten the 97.4-teacher recipe mid-training.
  - `r1_pre` gave back its 40% lead. `r1_dist` still trails.
- **About 12:40, all arms at 60% (epoch 180)** vs `twin_strong` (96.56, val loss 0.161); `twin_kdn` had 96.54:

  | Arm | Val | Δ | Val loss |
  |---|---|---|---|
  | `r1_fm` | **96.88** | +0.32 | 0.127 |
  | `r1_fm1` | **96.88** | +0.32 | **0.121** |
  | `r1_pre` | 96.76 | +0.20 | 0.163 |
  | `r1_sam` | 96.72 | +0.16 | 0.142 |
  | `r1_dist` | 96.26 | -0.30 | 0.149 |

- **At 70% (epoch 210),** `r1_fm` reached 97.36 and `r1_fm1` 97.34, with val loss 0.115 / 0.110.
  - `twin_strong` had 96.88 at the same epoch (+0.46-0.48), and `twin_kdn` 97.08 (+0.26-0.28).
  - Both function-matching arms are already above where `twin_kdn` *ended* (97.30), with 30% of training to go.
  - **Function matching is the winner so far,** at both mixing strengths. The two are tied, with `fm1` slightly lower
    in loss.
- **About 13:05, the other arms at 70% (epoch 210)** vs `twin_strong` (96.88): `r1_sam` 97.04 (+0.16, val loss 0.130),
  `r1_pre` 96.66 (-0.22, 0.159), `r1_dist` 96.40 (-0.48).
- **80% (epoch 240)** vs `twin_strong` (96.98, val loss 0.135); `twin_kdn` had 97.14:

  | Arm | Val | Δ | Val loss |
  |---|---|---|---|
  | `r1_fm1` | **97.40** | **+0.42** | **0.101** |
  | `r1_fm` | 97.28 | +0.30 | 0.109 |
  | `r1_dist` | 97.08 | +0.10 | 0.139 |

  - **Function matching on every batch is pulling ahead of mixing half the batches** (+0.12, lower loss). Beyer et al.
    likewise mix aggressively.
  - DIST jumped late, from -0.48 to +0.10, so it may tie rather than lose.
  - `r1_pre` faded in the low-LR phase.

### Round 1 final results (13:35-14:05)

| Arm | Clean | Flip | Δ vs `twin_strong` (mean of none and flip) | Val loss |
|---|---|---|---|---|
| **`r1_fm`** (function matching, p 0.5) | **97.66** | **97.78** | **+0.42** | 0.096 |
| `r1_fm1` (function matching, p 1.0) | 97.48 | 97.74 | +0.31 | **0.094** |
| `r1_dist` | 97.18 | 97.32 | -0.05 | 0.136 |
| `twin_strong` (baseline) | 97.22 | 97.38 | – | 0.115 |
| `twin_kdn` (previous best, teacher 97.42) | 97.30 | 97.50 | – | 0.112 |

**Paired McNemar on the 5k val** (images only the arm gets right vs only the other model):

| Arm | vs `twin_strong`, none | vs `twin_strong`, flip | vs `twin_kdn`, none | vs `twin_kdn`, flip |
|---|---|---|---|---|
| `r1_fm` | 56 vs 34, **p = 0.026** | 41 vs 21, **p = 0.015** | 49 vs 30, **p = 0.042** | 37 vs 22, p = 0.067 |
| `r1_fm1` | 48 vs 35, p = 0.19 | 51 vs 33, p = 0.063 | 47 vs 37, p = 0.33 | 43 vs 30, p = 0.16 |

- **Function matching is the first statistically significant student improvement since distillation itself:** +0.4
  over the baseline, and +0.3 over the previous best recipe.
- **The mechanism is confirmed by teacher agreement on val:**
  - `r1_fm` agrees with `teacher_wrn300` on 98.36% of images (flip: 98.76%), vs 97.98% for `twin_strong`.
  - KL(teacher ‖ student) fell from 0.064 to 0.046 (flip: 0.033).
  - The student now reproduces the teacher on images it never trained on, exactly what Step 0 said was missing.
- **Mixing every batch is not better than half the batches** (-0.18 clean, -0.04 flip, slightly lower loss). By the rule
  (choose p 1.0 only if ≥ 0.15 better), **Round 2 uses `--mix-prob 0.5`.**
- **DIST: drop** (a tie).
- **The last two arms:**
  - `r1_pre` (`preact_v`): 97.38 / flip 97.62, Δ = +0.20 → **adopt** (a small gain from the full 4.98M budget and
    pre-activation, at about 1.1x cost).
  - `r1_sam`: 97.38 / flip 97.44, Δ = +0.11 → **tie, drop**. Its val loss is good (0.099), but the accuracy gain is
    below threshold, at 1.17x cost.

**Corruption suite** (VM, flip TTA; suite mean | JPEG 50):

| Arm | Suite mean | JPEG 50 |
|---|---|---|
| `r1_fm` | 96.28 (+0.20 vs `twin_strong` 96.08) | 85.66 |
| `r1_fm1` | 96.23 | 85.02 |
| `r1_sam` | 96.28 | **87.08** |
| `r1_pre` | 96.16 | 85.36 |
| `r1_dist` | 96.04 | 87.02 |
| `twin_strong` | 96.08 | 86.24 |

- No adopted factor costs robustness (the rule allowed a drop of up to 0.3 in the suite mean).
- Function matching is about 0.6-1.2 weaker on JPEG only; the hidden set shows no JPEG artifacts.
- SAM gives the most JPEG robustness but no clean gain.
- Exact McNemar on the VM matched the Mac numbers: `r1_pre` p = 0.48 / 0.20; `r1_sam` p = 0.43 / 0.81;
  `r1_dist` p = 0.92 / 0.82.

**Round 1 verdict:** function matching (p 0.5) +0.42, strong; `preact_v` +0.20, adopt; SAM +0.11 tie; DIST -0.05;
`fm1` +0.31 but not better than p 0.5. **S = `preact_v` + function matching (mix-alpha 1.0, mix-prob 0.5)**, keeping
student noise 0.04, KD α 0.9 and T 4.

## Round 2: stack + finals (launched 2026-11-03 about 13:49 EST)

| Run | Recipe | Purpose |
|---|---|---|
| `r2_twin_S` | S, 45k, `teacher_wrn300` | validates the stack (factors may not add up) |
| `r2_final_S` | S, all 50k, `teacher50` | Pick 1 candidate |
| `r2_final_H` | resnet_c + function matching (the `r1_fm` recipe), all 50k, `teacher50` | hedge / Pick 2; its twin is `r1_fm` |

- **Rule:** `final_S` is Pick 1 if `twin_S` ≥ (`r1_fm` − 0.1), otherwise `final_H`.

### Run F: `r2_final_H` submitted (2026-11-03 15:40 EST) → **0.901 public, new team best**
- **Recipe:** resnet_c (4.62M), KD from `teacher50` with function matching (mix-alpha 1.0, p 0.5), student noise 0.04,
  300 epochs, all 50k, flip TTA. Twin `r1_fm`: 97.66 / 97.78 flip.
- **Final train loss** 0.704, vs 0.710 for the twin; the two stayed in step all run.
- **Class counts:** airplane 177 (still the lowest, as for every model), automobile 217, bird 205, cat 211, deer 195,
  dog 205, frog 185, horse 203, ship 208, truck 194.
- **vs Run D (0.891, same pipeline without function matching): +1.0 public (about +10 of the 1,000 images)** for
  +0.28 flip val (twin vs twin).
  - That is about 3.5 public points per val point, in line with the accuracy-on-the-line slope predicted near 89%
    (about 3).
  - vs Run E (0.888): +1.3.
- **Leaderboard:** team "Survey Corps" is now **#20 of about 50** (tied with #18-19 at 0.901). The top-20% cut (10th) is
  0.907. The legitimate top is 0.914.
  - "Baba Tillu" jumped to 0.972 today, which is not credible for ≤5M from-scratch models.
- **A teammate submitted "Run D: distilled from Runs A, B and C, 300 epochs, all 50k images" → 0.895** (19:27 UTC).
  That was Vedant's own ensemble distillation; note the clashing name "Run D".
- **Submissions today (UTC):** 4 of 5 used (our D at 00:53 UTC, E, Vedant's, F). One left before the 19:00 EST reset,
  reserved for Run G (`r2_final_S`).

### `r2_twin_S` result: the stack does not add up (about 16:00)
- `r2_twin_S` (`preact_v` + function matching): **97.40 / flip 97.58**, val loss 0.109.
  - It equals `r1_pre` (97.38 / 97.62, same architecture without function matching), and is below `r1_fm` (97.66 /
    97.78).
- **Paired McNemar:**

  | vs | none | flip |
  |---|---|---|
  | `r1_fm` | 31 vs 44, p = 0.17 | 15 vs 25, p = 0.15 |
  | `r1_pre` | 45 vs 44, p = 1.00 | 32 vs 34, p = 0.90 |

- **Function matching added nothing measurable on `preact_v`,** although it again raised teacher agreement on val
  (98.26% / flip 98.42%, vs 97.98% for `twin_strong`).
- **Honest reading:** with a run-to-run spread of about 0.2-0.3, `r1_fm`'s +0.42 was probably partly favourable noise
  and this run partly unfavourable. Averaged over the two architectures, function matching is worth about +0.2 val.
  - Factors don't simply add. The 4.98M pre-activation net may already get, from its depth, what function matching
    gives resnet_c.
- **Decision rule:** `twin_S` (mean 97.49) < `r1_fm` (mean 97.72) − 0.1, so **Pick 1 = Run F (`r2_final_H`)**.
  `r2_final_S` is still submitted (user approved both) as the different-architecture Pick 2 candidate, at 0.23 below
  Pick 1.

### Run G blocked by the daily limit (16:08 EST)
- `r2_final_S` finished: train loss 0.688, vs 0.685 for its twin; 4,980,392 params.
  - Its dry-run CSV is saved at `runs/r2_final_S/submission_flip.csv` (2,000 rows, sample order).
  - Class counts: airplane 176, automobile 217, bird 204, cat 206, deer 197, dog 216, frog 183, horse 201, ship 203,
    truck 197.
- **Kaggle returned 400 Bad Request.** A teammate had submitted `original_flip.csv` (no description) at 21:05 UTC, 3
  minutes earlier: **0.878**. That was the team's 5th submission of the UTC day.
  - Run G can go in after the reset at 00:00 UTC (19:00 EST).
  - **Lesson: coordinate the shared 5/day.**
- **VM stopped at 16:10. Balance 31.4 units** (Round 1 + Round 2 + setup used about 36).
- **Pick 2** is the other one: different architectures (pre-activation 4.98M vs post-activation 4.62M), so decorrelated
  errors.
- **Side question: test-time BN adaptation.** The professor confirmed it is allowed. My assessment, not yet tested:
  - It corrects low-level covariate shift; our hidden set looks like mostly a content shift.
  - Expected gain about 0 to +0.5 (+1 at best). Use full-set stats from all 2,000 images, with prior mixing and
    possibly early layers only.
  - Evaluate offline first: a clean-val sanity check, the corrupted-val simulation, and a layer-wise stat-shift
    diagnostic hidden vs train. Then at most one paired submission.

### Run E: an analytics submission (2026-11-03 11:06 EST)

`twin_kdn`, the 45k twin of Run D (KD from `teacher_wrn` + student noise, 300 epochs, flip TTA, val 97.30 /
**97.50 flip**), scored **0.888 public**. Run D (same recipe, all 50k) scored 0.891.

What it tells us:
- **It is the first submission whose val accuracy is measured directly,** not inferred from a twin. 97.50 flip val
  maps to 0.888, an 8.7-point gap.
- **D and E agree on 96.2% of the 2,000 hidden images** (about 76 differ, so about 38 on the public half). Their public
  scores differ by 3 images.
  - So 45k → 50k was worth +0.3 here, versus +1.7 for Vedant's A → B. Both differences are within the public noise.
  - The average data effect is about +1.0, but neither number is reliable alone.
- **All five team submissions sit within 87.5-89.2** (mean 88.5, SD 0.7), while their val spans about 97.1-97.5. The
  spread is about what 1,000-image sampling noise (SE about 1.0) alone would produce. The public LB cannot rank models
  this close together.
- 0.888 is again a multiple of 0.001, consistent with the 1,000-image public subset.

## Open questions for the next round

- ~~Does the noise penalty under distillation close by the end of training?~~ Yes, at matched fractions (+0.10 vs
  `kd_c` at 70%). It bought +8.4 at noise 0.04 and +3.3 on JPEG, but no visible leaderboard gain.
- Does 500 epochs beat 300 by more than the 0.3-point noise level? `twin_kdn500` vs `twin_kdn` answers this
  (finishing at about 20:20).
- **Raising clean accuracy toward 98%,** the main lever according to the leaderboard result. Candidates, most
  promising first:
  1. **A stronger teacher.** The student already matches the teacher within 0.1, so the teacher sets the ceiling.
     Options: WRN-28-10 trained on all 50k for longer; an ensemble of 2-3 teachers, averaging their soft labels
     (the submitted student is still one model under 5M, so this is allowed); SAM optimizer for the teacher.
  2. **A longer student schedule,** if `twin_kdn500` shows a real gain.
  3. **Drop the student noise** if it costs clean accuracy in the final comparison. It is neutral at matched
     fractions with 300 epochs, but it bought nothing visible on the leaderboard.

## Final push, Phase 1: architecture screens (launched 2026-11-03 16:48 EST, Colab)
Plan: `~/.claude/plans/the-high-level-goal-rosy-star.md`. The goal is to raise clean val of a single ≤5M student
toward 98.3-98.6.

**Measured first, ruling out test-time BN adaptation:**
- BN-input statistics on the hidden images vs clean val:
  - early layers: mean-shift 0.034 (floor 0.011), about as much as ±20% brightness, which costs about 0.2;
  - **last layers: 0.070 mean-shift / 0.138 log-var**, larger than any photometric corruption (0.01-0.04);
  - so the shift is **semantic**.
- TTBN on the held-out twin (`r1_fm`): clean -0.50, brightness -0.5 to -0.6, noise -0.65; blur +0.25, JPEG +1.75.
  It changes 80 of 2,000 hidden predictions. **Not used.**

**Phase 1 arms** (300 ep, 45k, FM α 1 / p 0.5, student noise 0.04, KD from `teacher_wrn300`; baseline `r1_fm`, 97.66 /
97.78 flip):

| Arm | Model | Params | GMACs | Tests |
|---|---|---|---|---|
| `p1_hr` | `resnet_c_hr`, strides (1,2,1), stages at 32/16/16 | 4,623,178 | 1.34 | resolution in the last stage |
| `p1_s112` | `resnet_c_s112`, strides (1,1,2), stages at 32/32/16 | 4,623,178 | 2.21 | resolution in stages 2-3 |
| `p1_se` | `resnet_c_se`, Squeeze-and-Excitation r = 16 in every block | 4,659,046 | 0.72 | channel attention |

- `p1_ens` (FM + 3-teacher ensemble) and `p1_sam` (FM + SAM) were started and stopped after about 1 minute, on the
  design review's advice:
  - the ensemble is at most +0.18 over `teacher_wrn300`, so its transfer is below what one run can resolve;
  - SAM already tied without FM.
- Long function matching (800 ep) and a SAM teacher go to NYU HPC (`scripts/hpc/`) once access works.

### NYU HPC Cloud Bursting set up (2026-11-03, about 17:45 EST)
- **Access, Fall 2026:**
  - Only through Open OnDemand at `https://ood.burst.hpc.nyu.edu` (NYU VPN + Microsoft SSO).
  - SSH to greene/burst is retired, and Torch is off-limits for coursework.
  - Course account `cs_gy_6923-2026fa`: 300 GPU hours.
  - Partitions: `c12m85-a100-1` (A100), `c24m170-a100-2`, `g2-standard-*` (L4), `n2c48m24` (CPU), `interactive`.
  - No time limit; overlay templates in `/share/apps/overlay-fs-ext3`, images in `/share/apps/images`.
- **Burst nodes are preemptible:**
  - The first Jupyter session was killed mid-unzip ("Terminated") and requeued on a new host.
  - Every job therefore uses `--requeue` plus `--resume` (checkpoints every 10 epochs).
- **Upload:** `hpc_upload/dl1_bundle.tar` (418 MB: code, data.zip, `teacher_wrn300`, `teacher50`), uploaded to
  `/scratch/as21153` via the OOD file browser.
- **Submitted from a Jupyter terminal** on node b-10-147:
  - `setup_job.sh` (job 20907, CPU): unzips the data, builds the Singularity/conda env with the CUDA 11.8 image and the
    15 GB overlay, and caches the dataset.
  - Training jobs, each with `--dependency=afterok:20907` on `c12m85-a100-1` (1 × A100):

    | Job | Run | What |
    |---|---|---|
    | 20908 | `h_fm800` | resnet_c, function matching, 800 epochs, 45k, `teacher_wrn300` |
    | 20909 | `h_final800` | the same, all 50k, `teacher50` |
    | 20910 | `h_teacher_sam45` | WRN-28-10 + SAM ρ 0.05, 400 epochs, label smoothing 0, 45k |
    | 20911 | `h_teacher_sam50` | the same, all 50k |
- **Resubmitted at 17:46 EST:**
  - Unzipping 52k PNGs onto burst `/scratch` (NFS) crawled at about 1,100 files/min, and node b-10-160 logged Slurm
    munge auth errors.
  - So jobs 20907-20911 were cancelled. **New `setup_job2.sh` (job 20914)** builds the env, then writes
    `data/cache/{train,test}.npz` straight from `data.zip` in memory, with no per-image files.
  - Training jobs **20915** `h_fm800`, **20916** `h_final800`, **20917** `h_teacher_sam45`, **20918**
    `h_teacher_sam50` depend on it (afterok).
  - **Lesson:** on burst, avoid many small files on /scratch; ship arrays, not image folders.
- **Setup finished interactively (18:00 EST), after two more problems:**
  - Job 20914 sat in node state CF (configuring) on b-10-102, so 20914-20918 were cancelled. `setup_job2.sh` ran
    directly in the Jupyter session instead.
  - `Miniconda3-latest` now ships **Python 3.14.9**, which has no cu118 PyTorch wheel ("No matching distribution").
    - Fix: a conda-forge Python 3.11 env at `/ext3/py311`, activated in `/ext3/env.sh`. It runs torch 2.7.1+cu118 and
      torchvision 0.22.1+cu118.
    - `scripts/hpc/setup_env.sh` was updated to match.
  - The npz cache built from `data.zip` in under 30 s ((50000, 32, 32, 3) / (2000, 32, 32, 3)), and `load_split`
    reads it. SETUP_OK.
  - CPU smoke test (`--subset 512`, 1 epoch, FM + `teacher_wrn300`):
    - loads the teacher (97.85 val);
    - resnet_c has 4,623,178 params;
    - the epoch runs end to end.
- **Submitted at 18:02 EST** (no dependencies; all on `c12m85-a100-1`):

  | Job | Run |
  |---|---|
  | 20929 | `h_fm800` |
  | 20930 | `h_final800` |
  | 20931 | `h_teacher_sam45` |
  | 20932 | `h_teacher_sam50` |

  - All start as PD (Resources/Priority) while cloud A100 nodes power up. The partition had 144 nodes powered off
    (`idle~`), 24 powering up and 29 allocated.

### Phase 1 at about 70% (18:03 EST): val acc vs `r1_fm` at the same fraction of the schedule
| Run | 10% | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---|---|---|---|---|---|---|
| `p1_hr` (1,2,1) | -0.84 | -0.58 | -0.32 | -0.46 | -0.48 | -0.52 | -0.72 |
| `p1_se` | -0.66 | -0.12 | +0.14 | -0.10 | +0.02 | -0.20 | |
| `p1_s112` (1,1,2) | -0.36 | +0.60 | +0.26 | +0.14 | +0.32 | | |

- **`p1_hr` is consistently behind.** It keeps the last stage at 16×16, which halves the receptive field of the
  final features relative to resnet_c. The extra 0.6 GMACs go to the wrong place.
- **`p1_s112`** keeps stages 1-2 at full 32×32 resolution and is consistently ahead (+0.14 to +0.60 since 20%), with
  the lowest val loss of the three (0.115).
  - Its early features see more spatial detail, and the last stage still downsamples, so the receptive field is kept.
- **`p1_se` is tied** within the noise.

### HPC jobs running (18:20 EST)
- **Speed on one A100 (`c12m85-a100-1`), one run per GPU, AMP:**
  - resnet_c + function matching with a WRN-28-10 teacher forward: **8.0-8.7 s/epoch**, so 800 epochs take about
    1.9 h. Colab packed 4 runs on one A100 at about 36 s/epoch each.
  - WRN-28-10 + SAM: **22 s/epoch**, so 400 epochs take about 2.4 h.
- **The A100 pool is congested; the L4 pool is free.**
  - A100: 23 jobs pending in `c12m85-a100-1`, and allocations hover near 29-31 nodes. That looks like a cloud quota
    cap, even though `sinfo` lists 144 powered-off (`idle~`) nodes. Jobs got nodes one at a time, about 2-8 min
    apart.
  - L4 (`g2-standard-12`: 12 CPUs, 47 GB, 1 × L4): 0 allocated, 190 `idle~`.
- **`p1_s112` is ahead of `r1_fm`, so the s112 runs that would be needed if it wins went to L4 right away**
  (`--partition=g2-standard-12 --cpus-per-task=10 --mem=40GB`):

  | Job | Run | What |
  |---|---|---|
  | 20938 | `h_s112_final300` | resnet_c_s112, FM, 300 epochs, all 50k, `teacher50` (the next submission if s112 wins) |
  | 20939 | `h_s112_fm800` | resnet_c_s112, FM, 800 epochs, 45k twin, `teacher_wrn300` |
  | 20940 | `h_s112_final800` | the same on all 50k, `teacher50` |
- **State at 18:20:**
  - Running: `h_fm800` (epoch 78/800, val 95.56), `h_final800` (55/800), `h_teacher_sam45` (2/400).
  - Pending: `h_teacher_sam50` (A100 queue).
  - Configuring: the three L4 jobs.
- **L4 boot failure (18:21 EST):**
  - All three L4 jobs sat in CF for about 7 min, then went back to PD (BeginTime) with Restarts=1, so the cloud nodes
    never came up. Ten `g2-standard-12` nodes are drained, and nobody else had a job there: the L4 pool looks broken,
    not free.
  - `scontrol update partition=` is refused for requeued jobs, and the submit plugin rejects multi-partition jobs
    ("please validate account ... or partition").
  - So 20938-20940 were cancelled (none had started training) and resubmitted:

    | Job | Run | Partition |
    |---|---|---|
    | 20952 | `h_s112_final300` | A100 |
    | 20953 | `h_s112_fm800` | A100 |
    | 20954 | `h_s112_final800` | L4 again, as a probe; it moves to A100 if it fails again |

### Phase 1 results (18:40 EST): `r1_fm` is 97.66 / 97.78 flip at 300 epochs
| Run | Final val | Flip | Δ vs `r1_fm` (none / flip) | Verdict |
|---|---|---|---|---|
| `p1_hr` (1,2,1) | 97.00 | 97.44 | -0.66 / -0.34 | drop: the last stage at 16×16 hurts |
| `p1_se` (SE r = 16) | 97.42 | 97.54 | -0.24 / -0.24 | drop: no gain from channel attention with FM |
| `p1_s112` (1,1,2) | pending | | | |

- **Smoothed deltas** (15-epoch windows from epoch 150) made both drops clear before the end: `p1_hr` stayed at -0.4 to
  -0.6, and `p1_se` slid from -0.05 to -0.47.
- **`p1_s112` fades towards the end:** +0.20, +0.12, +0.04, -0.06 over epochs 151-210, but its val loss stays about
  0.016 lower than `r1_fm`'s throughout.
- **HPC at 18:38 EST:** all six A100 jobs are running. `h_s112_final800` was cancelled after its third failed L4 boot
  (the L4 pool is unusable); it goes back on A100 if s112 wins.
  - `h_fm800` at epoch 229/800 (29%): val 96.36, vs `r1_fm` 95.60 at the same fraction (+0.76).
  - s112 on A100: about 10 s/epoch.

### Phase 1 final + suite (19:05 EST): every arm loses to `r1_fm`
| Run | val (none) | val (flip) | suite mean (flip) | worst (JPEG 50) | McNemar vs `r1_fm` (none / flip) |
|---|---|---|---|---|---|
| `r1_fm` | 97.66 | 97.78 | 96.28 | 85.66 | - |
| `p1_s112` | 97.38 | 97.58 | 95.99 | 83.88 | 28 vs 42, p = 0.12 / 26 vs 36, p = 0.25 |
| `p1_se` | 97.42 | 97.54 | 96.11 | 85.02 | 30 vs 42, p = 0.20 / 21 vs 33, p = 0.13 |
| `p1_hr` | 97.00 | 97.44 | 95.86 | 84.50 | 32 vs 65, p = 0.001 / 25 vs 42, p = 0.05 |

- **More FLOPs per parameter does not help under FM at 300 epochs.**
  - Higher-resolution stages (hr, s112) and SE attention all land 0.2-0.7 below plain resnet_c, on clean val and on
    every corruption family.
  - s112 has the lowest val loss (0.091) but not the best accuracy: better calibrated, not more accurate. It is also
    notably weaker on JPEG.
- **Recipe S\* = `r1_fm` unchanged** (resnet_c + FM + `teacher_wrn300`). The remaining levers are patience (800
  epochs) and a stronger teacher (SAM).
- **Colab VM shut down** (`colab.py down`) at 18:52. HPC is now the main compute.

### Run G submitted (19:01 EST, after the 00:00 UTC reset): public 0.901
- `runs/r2_final_S/submission_flip.csv` (PreAct 4.98M, FM, all 50k; twin 97.58 flip): **0.901**, identical to Run F
  (resnet_c FM, twin 97.78 flip).
- The 0.2 twin gap is below public-LB resolution (SE about 1 point on 1,000 images). Two different architectures with
  FM both land at 0.901, which supports FM as the lever and architecture as second-order.
- Team submissions today (UTC Nov 4): 1 of 5 used.

### HPC monitoring without a terminal (19:05 EST)
- **Jupyter session 20883 (b-10-147) was preempted again** ("DNS lookup failure for b-10-147"). The batch jobs were
  unaffected; they run on their own nodes.
- **Monitoring now goes through the OOD dashboard:**
  - Active Jobs: `/pun/sys/dashboard/activejobs?jobfilter=user`.
  - Raw log text: `/pun/sys/dashboard/files/fs//scratch/as21153/dl1/runs/RUN/train.log`, fetched from the page
    with JS.
  - No CPU node needed.
- **Progress vs baselines at the same fraction of the schedule:**
  - `h_fm800`: 413/800 (52%), val 96.88, vs `r1_fm` 96.59 (+0.29). Patience looks like it helps.
  - `h_s112_fm800`: 195/800 (24%), 96.24, vs 95.39 (expected to lead early: more epochs at each fraction).
  - `h_teacher_sam45`: 124/400 (31%), 96.40, vs `teacher_wrn300` 96.21 (+0.19).
  - `h_final800` at 366/800, `h_teacher_sam50` at 79/400, `h_s112_final300` at 182/300.

### HPC milestone (19:35 EST): leads narrowing as the runs converge
| Run | Epoch | Val | Same fraction of the baseline (11-epoch mean) | Δ |
|---|---|---|---|---|
| `h_fm800` | 649/800 (81%) | 97.50 | `r1_fm` 97.41 | +0.09 (was +0.29 at 52%, +0.24 at 67%) |
| `h_s112_fm800` | 393/800 (49%) | 96.92 | `r1_fm` 96.44 | +0.48 |
| `h_teacher_sam45` | 212/400 (53%) | 97.08 | `teacher_wrn300` 97.01 | +0.07 (was +0.41 at 42%) |

- `h_s112_final300` is done (300/300). It will not be submitted.
- **Pattern:** longer schedules (and SAM) lead mid-run but converge towards the baseline in the low-LR phase. The
  same happened with 500 vs 300 epochs without FM (a tie).
  - Final gains are likely ≤0.2. Single-epoch val on 5k has SD about 0.2, so wait for the final + flip + McNemar.
- `h_fm800` config is identical to `r1_fm` except epochs (checked by diffing the configs), so it is a clean one-factor
  test.

### Patience works: `h_fm800` 97.84 / 98.00 flip, the first student ≥98 (20:00 EST)
- **Result vs `r1_fm` (the same recipe at 300 epochs, 97.66 / 97.78):** +0.18 / +0.22.
  - McNemar: 38 vs 29, p = 0.33 (none); 27 vs 16, p = 0.13 (flip).
  - Suite mean (flip): 96.49 vs 96.28, better on 10 of 11 corruptions (JPEG 85.32 vs 85.66).
  - Agreement with `teacher_wrn300`: 98.90% (flip), KL 0.024.
  - **Adopted** by the Phase 1 rule (mean Δ 0.20 ≥ 0.15, no suite drop).
- **The curve:** it led by +0.29 at 52%, narrowed to +0.09 at 81%, then gained again in the last 20% (the final low-LR
  phase). Mid-run deltas under-predict the end.
- **`h_final800`** (all 50k, `teacher50`, 800 epochs) finished at 20:06, exit 0. Downloaded through the OOD file
  browser (a top-level navigation to `...?download=1`; scripted blob downloads get blocked after the first).
  - `runs/h_final800/submission_flip.csv`: 2,000 rows in sample order, balanced classes.
  - Agreement with Run F 95.80%, with Run G 95.90% (F vs G 95.45%).
  - **Ready to submit as Run H**, waiting for the user's OK.
- **Next round submitted at 20:10 EST** (new Jupyter session 21028 on b-10-70, after 20883 was preempted):

  | Job | Run | What |
  |---|---|---|
  | 21029 | `h_fm1600` | resnet_c FM 1600 epochs, 45k, `teacher_wrn300` (does patience keep paying?) |
  | 21030 | `h_final1600` | the same on all 50k, `teacher50` |
  | 21031 | `h_fm800_sam` | FM 800 epochs with the SAM teacher `h_teacher_sam45`; afterok:20931 |
  | 21032 | `h_final800_sam` | the same on 50k with `h_teacher_sam50`; afterok:20932 |

### Run H submitted (20:15 EST): public 0.897
- `runs/h_final800/submission_flip.csv` (resnet_c FM 800 epochs, all 50k, twin 98.00 flip): **0.897**, vs Run F 0.901
  (twin 97.78) and Run G 0.901 (twin 97.58).
- **Within noise:**
  - H and F disagree on 4.2% of the 2,000 hidden images, about 42 on the public half. -0.4 is 4 images.
  - Public SE is about 1 point (1,000 images). Three FM models with twins 97.58-98.00 score 0.897-0.901, so the
    public LB cannot rank them.
- **Final selection stays by twin val, per the plan.** Run H has the best twin (98.00 flip, p = 0.13 vs `r1_fm`,
  better on 10 of 11 corruptions). The private 1,000 images decide the ranking.
- Team submissions today (UTC Nov 4): 2 of 5 used (G, H).

### 20:36 EST
- **`h_fm1600` (21029) was cancelled by the system** (uid 0) after about 12 min in node configuration on b-19-25. It
  wrote no log, so the cloud node failed to boot. Resubmitted as **21053**. `h_final1600` (21030) is running (epoch
  81/1600, 8.5 s/epoch).
- **`h_teacher_sam45`** at 376/400: val 98.06; `teacher_wrn300` ended at 97.96.
- `h_s112_fm800` at 768/800: 97.70.

### SAM teacher and ensemble (20:50 EST)
| Teacher | val | flip | McNemar vs `teacher_wrn300` (none / flip) |
|---|---|---|---|
| `teacher_wrn300` (300 ep) | 97.96 | 97.90 | - |
| `h_teacher_sam45` (SAM ρ 0.05, 400 ep, LS 0) | 98.04 | 98.02 | 37 vs 33, p = 0.72 / 31 vs 25, p = 0.50 |
| `teacher_wrn300` + `h_teacher_sam45` (avg softmax) | 98.12 | 98.08 | 18 vs 10, p = 0.19 / 15 vs 6, p = 0.08 |

- **The SAM teacher is a tie** (+0.08), below the plan's 98.2 bar. Its single-teacher students are replicates.
  - `h_final800_sam` gets cancelled.
  - `h_fm800_sam` stays as a noise calibration: an equal-strength teacher plus a different run gives the run-to-run
    spread at 800 epochs.
- **`h_s112_fm800` final: 97.76 / 97.72.** s112 does not catch up with patience either (`h_fm800` 97.84 / 98.00).
- **The student has reached the teacher.** `h_fm800` is at 97.84 against the teacher's 97.96 (none), and with flip it
  exceeds it (98.00 vs 97.90), with 98.9% agreement. Under FM the teacher is now the bottleneck, which reverses the
  Round 3 finding (the student was the bottleneck under plain KD).
  - **Next lever: a stronger teacher target.** The 2-ensemble is +0.16 / +0.18. More seeds should push it towards
    about 98.3.
- **Planned jobs** (blocked: Jupyter 21028 on b-10-70 was preempted before the commands ran):
  - `h_fm800_ens2` (FM 800 with `teacher_wrn300` + `h_teacher_sam45`);
  - `h_fm800_a1` (kd-alpha 1.0);
  - `h_fm800_nn` (no student noise);
  - teachers `h_teacher_sam45_s1` and `_s2` (seeds 1 and 2, for a 4-teacher ensemble).
- **Preemption is widespread right now:**
  - `h_fm1600` (21053) was requeued after 9 min of running;
  - `h_fm800_sam` (21031) after 2 min;
  - both resume from `last.pt`.

### 21:07 EST: cloud node provisioning failing
- **Jupyter sessions:**
  - 21066 (`n2c48m24`) ended without starting.
  - The `interactive` partition rejects `--mem=4G` ("Memory specification can not be satisfied"), so 21068 was
    launched there with no memory option; still queued after 5 min.
- **Batch jobs:**
  - `h_fm1600` (21053) disappeared from the queue after a requeue, again without a log.
  - `h_fm800_sam` (21031) has used about 4 min on nodes across requeues and never wrote a log: its nodes fail during
    configuration.
  - Jobs already on healthy nodes are fine: `h_teacher_sam50` at 391/400, `h_final1600` at 293/1600.
- **Lessons:** keep long jobs running rather than resubmitting, and submit follow-ups with `--dependency` while a
  terminal is up (the terminal is the scarce resource).

### Round 6 submitted (21:45 EST), via the JupyterLab terminal API
- **Terminal access:**
  - Jupyter 21086 (`n2c48m24`, b-10-122) started after about 10 min. 21068 on `interactive` never left the queue
    and was cancelled.
  - Screenshots of the lab tab time out, so the terminal is driven from page JS:
    - `POST /node/HOST/PORT/api/terminals` creates a terminal;
    - a WebSocket to `/terminals/websocket/NAME` sends stdin and reads stdout;
    - the script was written with `PUT /api/contents/dl1/submit_round6.sh`.
  - This avoids keyboard-focus problems. A copy of the script is in `scripts/hpc/submit_round6.sh`.
- **Jobs:**

  | Job | Run | What |
  |---|---|---|
  | 21135 | `h_fm1600` | resubmitted twin, 1600 ep |
  | 21136 | `h_fm800_ens2` | FM 800 with `teacher_wrn300` + `h_teacher_sam45` (98.12 / 98.08 ensemble) |
  | 21137 | `h_final800_ens2` | 50k version with `teacher50` + `h_teacher_sam50` |
  | 21138 | `h_fm800_a1` | kd-alpha 1.0 (pure distillation, as in Beyer et al.) |
  | 21139 | `h_fm800_nn` | student noise 0 |
  | 21140-21141 | `h_teacher_sam45_s1/_s2` | WRN-28-10 SAM 400 ep, seeds 1-2, 45k |
  | 21142-21143 | `h_teacher_sam50_s1/_s2` | the same on 50k |
  | 21144 | `h_fm800_ens4` | FM 800 with 4 teachers (wrn300, sam45, sam45_s1, sam45_s2); afterok:21140:21141 |
  | 21145 | `h_final800_ens4` | 50k: teacher50, sam50, sam50_s1, sam50_s2; afterok:21142:21143 |

- **Still queued from before:** `h_fm800_sam` (21031, noise-calibration replicate) and `h_final800_sam` (21032).
- **Running:** `h_final1600` (21030).
- **Expected:**
  - if nodes come up, the 800-epoch one-factor runs finish about 2 h after start, and the teachers about 2.5 h;
  - the 4-teacher students (about 16 s/epoch, about 3.6 h) finish around 05:00-06:00 Nov 4.

### Run H post-mortem and two cheap levers (22:30 EST; read-only analysis on the hidden set + twin val)
- **Run H's 0.897 is most likely noise.**
  - H vs F disagree on 84 of the 2,000 hidden images, about 42 on the public half, so -0.4 is 4 net images (SD of the
    net change about 0.65 pt).
  - At the literature CIFAR-10 → CIFAR-10.1 probit slope (1.0-1.2), a 98.00 twin maps to about 90.7-90.9, so H was
    1.5-1.9 SD under expectation.
  - On the 84 disagreements, G sides with H 42 vs 33 and teacher50 41 vs 32. All share the teacher, so this is weak
    evidence.
- **Shared class bias on the hidden set:**

  | Model | airplane | frog | automobile | dog | Σ\|n−200\| |
  |---|---|---|---|---|---|
  | F | 177 | 185 | 217 | 205 | 98 |
  | G | 176 | 183 | 217 | 216 | 94 |
  | H | 177 | 179 | 220 | 212 | 114 |
  | teacher50 | 172 | 181 | 224 | 214 | 114 |

  - H copies the teacher more closely (agreement 95.6% vs F 95.15%), bias included.
  - Mean max-prob is 0.96 for all students.
- **Wider TTA (flip_shift, 10 views) vs flip on the twin val: a tie, not adopted.**
  - `h_fm800`: clean +0.04, suite mean +0.06.
  - `r1_fm`: clean -0.14, suite +0.03.
  - It helps only JPEG (+0.2 to +0.6).
- **Class-balanced decoding (Sinkhorn to a uniform class marginal), twin val (exactly 500 per class), `h_fm800` /
  `r1_fm`:**

  | Shift | Gain from balancing |
  |---|---|
  | clean | -0.08 / 0.00 |
  | sharpen | +0.22 / +0.16 |
  | blur | +0.08 / +0.06 |
  | JPEG | +1.38 / +1.14 |
  | suite mean | +0.15 / +0.12 |

  - The gain grows with prediction imbalance; the hidden set's imbalance is about 5% of N. Expected +0.3 to +0.7 pt,
    if the hidden set is class-balanced (2,000 images = CIFAR-10.1 v6 size, 200 per class).
  - Added as `src.predict --balance` (`balance_probs`), off by default. On Run H it changes 75 of 2,000 predictions.
  - **Not to be submitted until the professor approves** (the user is asking).

### 22:36 EST: the HPC cloud is still failing to start nodes
- **`h_final1600`** at 908/1600 (on b-19-44 since 20:10, unaffected); done about 00:20.
- **Gone again:** `h_fm1600` (21135) and `h_fm800_sam` (21031) vanished after failed node starts. Jupyter 21086 also
  ended.
- **Still queued:** all other Round 6 jobs. Several briefly got nodes (00:00:18 to 00:07:19 of time used) and were
  requeued without writing a log, so the nodes fail during configuration.
- **Next:**
  - New Jupyter 21249 was requested so `h_fm1600` and `h_fm800_sam` can be resubmitted.
  - The Colab fallback decision is at 23:30, per the plan.

### 22:52 EST: babysitter for jobs cancelled by Slurm
- **Diagnosis:** `h_fm800_a1` (21138) shows "CANCELLED by 0" after 11:51 in configuration on b-19-69, with no log.
  Slurm cancels (does not requeue) a job whose cloud node fails to boot within its resume timeout. The earlier losses
  of `h_fm1600` (twice) and `h_fm800_sam` were the same.
- **Fix: `scripts/hpc/babysit.sbatch` (job 21260)**, on a CPU node with 1 CPU, 2 GB, 48 h, requeue.
  - Every 5 min it reads `runs/jobs.txt` (local copy: `scripts/hpc/jobs_round6.txt`) and resubmits any run that is not
    queued and has no `runs/NAME/final.pt`. `--resume` continues from `last.pt`.
  - Each run is capped at 8 submissions.
  - Runs that need other runs first (the ens4 students) wait until those runs' final.pt files exist. The Slurm-dependency
    versions (21144/21145) were cancelled: a dependency on a cancelled teacher ID would never be satisfied.
- **Queue at 22:52:**
  - running: `h_final1600`, `h_teacher_sam50_s2`;
  - configuring: `h_final800_sam`, `h_fm800_nn`, `h_teacher_sam45_s1/_s2`, `h_teacher_sam50_s1`;
  - pending: `h_fm800_ens2`, `h_final800_ens2`;
  - the babysitter resubmits `h_fm1600`, `h_fm800_a1` and `h_fm800_sam`.
- Jupyter 21249 is on b-10-70.
- **22:54 EST, the babysitter works:**
  - In its first loop it resubmitted `h_fm1600` (21262), `h_fm800_a1` (21263) and `h_fm800_sam` (21264).
  - At 03:52 UTC it resubmitted `h_teacher_sam45_s1/_s2` (21268/9), `h_teacher_sam50_s1` (21270) and `h_fm800_nn`
    (21271), after the originals were cancelled following 14:50 in CF.
  - Running: `h_final1600` 1039/1600 (done about 00:15), `h_teacher_sam50_s2` 23/400 (24 s/epoch).

### 23:35 EST: babysitter preempted, and OOD can no longer launch interactive sessions
- **Babysitter 21260 killed by preemption:** "JOB 21260 ON b-10-122 CANCELLED AT 2026-11-04T04:21:01 DUE to SIGNAL
  Terminated". CPU-partition jobs are cancelled on spot preemption, not requeued, despite `--requeue`.
  - Before dying it resubmitted `h_fm1600` (#2, 21280), `h_fm800_ens2` (21281) and `h_final800_ens2` (21282).
  - Fix for its next start: `trap 'sbatch scripts/hpc/babysit.sbatch; exit 0' TERM` with `sleep 300 & wait`, so it
    hands over to a fresh copy.
- **Jupyter 21249 also ended.** Launching a new Jupyter or Code Server session now fails every time with "This app
  does not supply a sub app form file under the directory '/var/www/ood/apps/sys/burst_jupyter/local'" (and
  `.../burst_codeserver/local`). This is server-side, so there is no terminal and no way to submit until OOD recovers.
- **Running (6):**

  | Run | Epoch | Val |
  |---|---|---|
  | `h_final1600` | 1257/1600 | - |
  | `h_teacher_sam50_s2` | 99/400 | - |
  | `h_final800_sam` | 151/800 | - |
  | `h_fm800_a1` | 160/800 | 96.04 |
  | `h_teacher_sam45_s1` | 40/400 | 94.24 |
  | `h_fm800_sam` | 117/800 | 95.94 |

- **Queued with no babysitter:** `h_fm1600`, `h_fm800_ens2`, `h_final800_ens2`, `h_fm800_nn`, `h_teacher_sam45_s2`,
  `h_teacher_sam50_s1`.
- **Not yet submitted:** the ens4 students (they need the babysitter, or a terminal once their teachers exist).
- The OOD login session also expires every few minutes now, so a reload is needed before each log fetch.

### 23:58 EST: portal back, babysitter v2 running
- **OOD interactive apps work again.** Jupyter 21699 is on b-10-104.
- **Babysitter v2 (21713):**
  - A TERM trap resubmits it when its spot node is preempted.
  - `sleep 300 & wait` lets the trap fire immediately.
  - It exits if another copy is already running.
- **Nothing was lost during the outage.**
  - Running (6): `h_final1600` 1425/1600, `h_fm800_a1`, `h_fm800_sam`, `h_final800_sam`, `h_teacher_sam45_s1`,
    `h_teacher_sam50_s2`.
  - `h_teacher_sam50_s1` is in CF.
  - `h_fm1600`, `h_fm800_ens2`, `h_final800_ens2`, `h_fm800_nn` and `h_teacher_sam45_s2` are pending on Priority:
    fairshare now ranks us below others, since we hold 6 GPUs.
- **Early one-factor readings** vs `h_fm800` at the same epoch (11-epoch mean):
  - `h_fm800_a1` (pure KD) at 336: 96.44 vs 96.78 (-0.34);
  - `h_fm800_sam` (SAM teacher) at 293: 96.78 vs 96.53 (+0.25);
  - mid-run deltas under-predicted the end for `h_fm800`, so wait for the finals.

### 00:16 EST Nov 4: `h_final1600` done
- **`h_final1600`** (resnet_c FM 1600 epochs, all 50k, `teacher50`) finished with exit 0. Downloaded;
  `runs/h_final1600/submission_flip.csv` written.
  - Hidden-set class counts: airplane 176, automobile 224, frog 182; Σ\|n−200\| = 98.
  - Agreement with H 96.15%, F 95.90%, G 95.70%.
  - **Held until its twin `h_fm1600` (pending) confirms 1600 > 800.**
- **The babysitter (21713) works:** it resubmitted `h_teacher_sam50_s1` (#2, 21732) after another failed boot.
- **Pending on Priority** (fairshare): `h_fm800_ens2` / `h_final800_ens2` / `h_fm1600` at 12171 (top), `h_fm800_nn`
  at 12171, teachers at 12168-12169. `scontrol update nice` and `scontrol hold` are both refused, so own-job
  priorities cannot be reordered.
- **Running:**

  | Run | Epoch | Val | Done (EST) |
  |---|---|---|---|
  | `h_fm800_a1` | 514/800 | 97.04 | about 00:55 |
  | `h_fm800_sam` | 479/800 | 97.16 | about 01:00 |
  | `h_final800_sam` | 495/800 | - | about 01:00 |
  | `h_teacher_sam50_s2` | 222/400 | - | about 01:25 |
  | `h_teacher_sam45_s1` | 169/400 | 97.18 | about 01:40 |

### 01:00 EST Nov 4: first Round 6 one-factor finals (vs `h_fm800` 97.84 / 98.00)
| Run | Change | val | flip | Δ (none / flip) | Verdict |
|---|---|---|---|---|---|
| `h_fm800_a1` | kd-alpha 1.0 (pure KD) | 97.64 | 97.82 | -0.20 / -0.18 | drop: the CE term helps |
| `h_fm800_sam` | single SAM teacher (98.04 / 98.02) | 97.86 | 97.78 | +0.02 / -0.22 | tie: not adopted |

- **`h_fm800_sam` led by +0.28 at 60% and finished at a tie.** This is the third time a mid-run lead vanished in the
  low-LR phase.
- **It doubles as a noise calibration:** an equal-strength teacher gives a twin within ±0.2 of `h_fm800`. The 800 vs
  300 gain (+0.18 / +0.22) is the same size, so treat single-twin deltas under about 0.2 as unresolved.
- **`h_final800_sam`** (50k, SAM teacher) is done, exit 0, and downloaded. It is not submitted (its twin did not
  clear the bar). Keep it as a diverse option for the final 2 picks.
- **Started:** `h_fm1600` (85/1600 at 00:56, done about 04:30) and `h_fm800_nn` (97/800, done about 02:20).
- **Still pending:** `h_fm800_ens2`, `h_final800_ens2`, `h_teacher_sam45_s2`, `h_teacher_sam50_s1`.
- **Teachers:** `h_teacher_sam45_s1` at 283/400 (97.70), `h_teacher_sam50_s2` at 327/400.

### 01:50 EST Nov 4: diverse teachers queued, without a terminal
- **The babysitter's TERM trap worked:** 21713 was preempted, resubmitted itself as **21886**, and kept going. It also
  re-queued `h_fm800_ens2` as 21863.
- **Jupyter 21699 ended as well.** Files are now edited without any terminal, through the OOD Files API from the
  dashboard page: `PUT /pun/sys/dashboard/files/fs//scratch/...` with the page's `csrf-token`, read back to verify.
- **Done that way:**
  - `wrn40_10` (55.8M) added to `TEACHER_CONFIGS` in the HPC `src/models.py` (and locally);
  - 4 diverse teachers appended to `runs/jobs.txt` for the babysitter to submit.

    | Run | What |
    |---|---|
    | `h_teacher_aa45` / `_aa50` | WRN-28-10, AutoAugment (CIFAR-10 policy), SAM 0.05, 400 ep, LS 0, seed 3 |
    | `h_teacher_w40_45` / `_w40_50` | WRN-40-10, TrivialAugment, SAM 0.05, 300 ep, LS 0, seed 4 |

- **Why:** every ensemble of our current models saturates at 98.16. The models share the WRN-28-10 + TrivialAugment
  recipe, so their errors correlate. A different augmentation policy and a different depth should decorrelate them.
- **Per-class specialist ideas checked and rejected:**
  - teacher oracle per class 95 errors vs averaging 94;
  - student routing 99 vs best single 100.
- **Transductive kNN smoothing** (10 feature-space neighbours, w 0.5) on the shifted val: suite mean +0.18, +0.42
  together with balancing, clean ±0. It needs the same professor approval as balancing.
- **State 01:45:**
  - running: `h_fm1600` (21280), `h_fm800_nn` (21271), `h_teacher_sam45_s1` (21268), babysitter 21886;
  - queued: `h_fm800_ens2` (21863), `h_final800_ens2` (21282), `h_teacher_sam45_s2` (21269), `h_teacher_sam50_s1`
    (21732);
  - done: `h_teacher_sam50_s2`.
- **01:35 EST:** the babysitter submitted `h_teacher_aa45` / `_aa50` / `_w40_45` / `_w40_50` as jobs 21956-21959.
  - `h_teacher_sam45_s1` is at 387/400 with val **98.18** (the first SAM teacher ended at 98.04).
  - `h_teacher_sam50_s2` is done.
  - `h_fm800_nn` at 427/800 (97.00); `h_fm1600` at 367/1600 (96.40), done about 04:20.
- **01:50 EST: `h_teacher_sam45_s1` final 98.18 / 98.22 flip (best 98.24)**, the best single teacher so far
  (`teacher_wrn300` 97.96 / 97.90, `h_teacher_sam45` 98.04 / 98.02).
  - Added `h_fm800_ens3` (`teacher_wrn300` + `sam45` + `sam45_s1`) and `h_final800_ens3` (`teacher50` + `sam50` +
    `sam50_s2`) to `runs/jobs.txt`.
  - Their teachers all exist now, so they can start on the next free A100, ahead of ens4. ens4 still waits for
    `sam45_s2` / `sam50_s1`, which have not started.
  - `h_fm800_nn` at 546/800 (97.48); `h_fm1600` at 466/1600 (96.72).

### 02:20 EST Nov 4: `h_fm800_nn` result; queue re-prioritized
- **`h_fm800_nn`** (student noise 0): 97.84 / 97.90 vs `h_fm800` 97.84 / 98.00, a tie. Student noise does not matter
  under FM, so the recipe stays.
- **The babysitter was preempted again** (06:51 UTC) and resubmitted itself as 22019, which submitted `h_fm800_ens3`
  / `h_final800_ens3` (22023/22024).
- **The cluster is saturated:** only `h_fm1600` (697/1600) is running, with 10 of our jobs pending. Slurm runs our
  oldest pending jobs first, which would put the dominated ens2 and same-recipe seeds ahead of ens3 and the diverse
  teachers.
- **Decision:**
  - Disable (comment out in `runs/jobs.txt`) and cancel `h_fm800_ens2`, `h_final800_ens2`, `h_teacher_sam45_s2` and
    `h_teacher_sam50_s1`, plus the ens4 chain that depends on them.
  - **Keep:** `h_fm800_ens3` / `h_final800_ens3` (98.18 teacher included), then the diverse teachers `aa45` / `aa50` /
    `w40_45` / `w40_50`, then a diverse-ensemble student pair.
- The cancels need a terminal: Jupyter 22127 requested.
- **02:35 EST, terminal-free control:** Jupyter 22127 got b-10-10 ("DNS lookup failure"), then fell back to Queued.
  - Babysitter v3 was written to `scripts/hpc/babysit.sbatch` via the Files API. The running copy (22019, v2) has
    already parsed its loop, so it is unaffected; the next copy after a preemption runs v3.
  - **v3 adds:**
    - `runs/inbox/*.sh`: one-off commands, each run once per pass, with output in `runs/inbox/done/NAME.out`;
    - `runs/babysit/squeue.txt`: a queue snapshot every pass.
  - Dropped in `runs/inbox/cancel_ens2_seeds.sh` (`scancel 21863 21282 21269 21732`). The `runs/inbox` dir was
    created with `PUT ...?dir=true`.
- **03:00 EST: babysitter v3 is live** (its `runs/babysit/squeue.txt` exists). The inbox cancel ran: ens2 and the
  sam45_s2 / sam50_s1 seeds are gone.
  - **Queue:** `h_fm1600` running (993/1600, val 97.36, done about 04:20). Pending on Priority: `h_fm800_ens3` /
    `h_final800_ens3` (22023/4) and the 4 diverse teachers (21956-9).
  - **Reorder:** our own pending jobs start oldest-first, so `runs/inbox/reorder_teachers.sh` cancels 21956-9. The
    babysitter resubmits them in the same pass with newer IDs, putting the ens3 students first.
- **03:10 EST:** the reorder ran. Pending order is now `h_final800_ens3` / `h_fm800_ens3` (22023/4), then teachers
  `w40_50` / `w40_45` / `aa50` / `aa45` (22257-60). `h_fm1600` is running; babysitter 22153; Jupyter 22127 is finally
  up on b-10-107.
- **04:25 EST: `h_fm1600` finished, a tie, not adopted.**

  | Model | val none / flip | suite mean none / flip | JPEG 50 flip |
  |---|---|---|---|
  | `h_fm1600` | 97.86 / 97.86 | 96.41 / 96.55 | 86.00 |
  | `h_fm800` | 97.84 / 98.00 | 96.25 / 96.49 | 85.32 |

  - Paired disagreements: 31 vs 30 (none, p = 1.0) and 23 vs 30 (flip, p = 0.41).
  - Δ is +0.02 / -0.14 clean, under the +0.15 rule. **`h_final1600` is not submitted on merit.** It stays a candidate
    only as a decorrelated pick or a spare-slot reading.
  - **All 6 other GPU jobs are running** (09:12 UTC snapshot), with done times in EST:

    | Run | Progress | Val | Done |
    |---|---|---|---|
    | `h_fm800_ens3` | 265/800 | 96.48 | about 06:25 |
    | `h_final800_ens3` | 245/800 | – | about 06:40 |
    | `h_teacher_aa45` | 159/400 | 96.70 | about 05:50 |
    | `h_teacher_aa50` | 139/400 | – | about 06:05 |
    | `h_teacher_w40_45` | 94/300 | 96.28 | about 06:15 |
    | `h_teacher_w40_50` | 50/300 | – | about 06:50 |

  - The babysitter was preempted again and is now 22367. `w40_50` was resubmitted as 22352 after a failed boot.
- **05:50 EST: the first diverse teacher confirms the diversity hypothesis.**
  - `h_teacher_aa45` (WRN-28-10, AutoAugment, SAM, 400 epochs) scored 97.86 / 97.88 flip. It is the weakest teacher
    alone, but it disagrees with the TrivialAugment teachers on 1.40-1.84% of val images, against 0.98-1.26% within
    the family.
  - The new teacher-set ensembles (flip val, cached probs in the scratchpad `teacher_pool.py`):

    | Teacher set | acc | NLL |
    |---|---|---|
    | sam45 + sam45_s1 + aa45 | **98.38** | 0.0572 |
    | wrn300 + sam45 + sam45_s1 + aa45 | 98.26 | **0.0566** |
    | ens3 (wrn300 + sam45 + sam45_s1), without aa45 | 98.18 | 0.0570 |

  - So +0.1-0.2 comes from one diverse member. The subset maximum is optimistic: 11 subsets are scored on one 5k val,
    so NLL is the tie-break.
  - Waiting on `h_teacher_w40_45` (259/300, val 97.98) before choosing the set.
  - Others: `h_fm800_ens3` 665/800 (97.66); `aa50` 369/400; `w40_50` 202/300.
- **06:20 EST: the WRN-40-10 teacher is strong but not diverse; queued the diverse students (Round 7).**
  - `h_teacher_w40_45`: 98.00 / 98.14 flip at 300 epochs, NLL 0.061. It disagrees with the SAM/TrivialAugment
    WRN-28-10s on only 0.84-0.92% of images. Depth adds little diversity; the augmentation policy is what
    decorrelates errors (`aa45` is at 1.32-1.84%).
  - Teacher-pool ensembles (5 teachers, 26 subsets, flip val):

    | Teacher set | acc | NLL |
    |---|---|---|
    | sam45_s1 + aa45 + w40_45 | 98.40 | 0.0572 |
    | **sam45 + sam45_s1 + aa45 + w40_45** | **98.38** | **0.0562** |
    | all 5 | 98.28 | 0.0558 |
    | ens3 | 98.18 | 0.0570 |

  - **Chosen: the 4-teacher set.** It is top-tier on accuracy, second-best on NLL, and doesn't hinge on the single
    max subset. `teacher_wrn300` (no SAM, NLL 0.070) is dropped.
  - `runs/jobs.txt` (local copy `scripts/hpc/jobs_round6.txt`) got Round 7:
    - **`h_fm800_div`:** twin, 45k teachers sam45, sam45_s1, aa45, w40_45.
    - **`h_final800_div`:** 50k teachers sam50, sam50_s2, aa50, w40_50. It needs `w40_50`, done about 06:45.
    - **Second diverse teacher round:**
      - `h_teacher_aa45_s2` / `aa50_s2`: AutoAugment, seed 5;
      - `h_teacher_w40aa45` / `w40aa50`: WRN-40-10 + AutoAugment, seed 6.

      They are gated on `w40_50`, so the `div` final is submitted first and keeps priority. That feeds a possible
      `div2` student (6 teachers).
  - AugMix teachers are skipped because AugMix is CPU-bound (about 60 s/epoch on 12 CPUs).
  - OOD note: a `HEAD` on a missing file returns 200. Check existence with the directory listing (`Accept:
    application/json`).
- **06:20 EST, submission plan (the user's decision):** only 1 submission today, the `h_final800_div` final at about
  11:00 EST. Nothing is submitted before then; `ens3` is scored for reference only.
  - Team slots, Nov 4 UTC day: 3 of 5 used (Run G, Run H, and a teammate's `submission.csv` at 05:35 UTC, 0.887).
- **06:30 EST: `h_fm800_ens3` (3-teacher target 98.18) does not beat `h_fm800`.**

  | Model | val none / flip | suite mean none / flip | JPEG 50 flip |
  |---|---|---|---|
  | `h_fm800_ens3` | 97.72 / 97.80 | 96.45 / 96.61 | 86.62 |
  | `h_fm800` | 97.84 / 98.00 | 96.25 / 96.49 | 85.32 |

  - McNemar 29 vs 35 (p = 0.53) and 22 vs 32 (p = 0.22). Clean is -0.12 / -0.20 and the suite is +0.20 / +0.12, so it
    is not adopted.
  - **Student vs its own target** (flip val; "lost" = teacher right and student wrong, "gained" = the reverse):

    | Student | acc | target | agree | lost | gained | KL |
    |---|---|---|---|---|---|---|
    | `h_fm800` | 98.00 | 97.90 | 98.90% | 23 | 28 | 0.024 |
    | `h_fm1600` | 97.86 | 97.90 | 98.58% | 35 | 33 | 0.027 |
    | `h_fm800_ens3` | 97.80 | 98.18 | 99.02% | 32 | 13 | 0.017 |

  - The ens3 student matches its target more closely (KL 0.017) but does not inherit the ensemble's extra correct
    answers. Those sit at the decision margin, where the student's own errors return.
  - **The distillation temperature is not the cause.** Averaging the members at T = 1 / 2 / 4 gives the same target
    accuracy (ens3 98.18 / 98.16 / 98.16; div 98.38 at all three).
  - **Revised view:** every 800-epoch FM student lands at 97.78-98.00 flip (mean about 97.86), whether its target is
    97.90 or 98.18. `h_fm800`'s 98.00 is the high end of that noise, and the student recipe/capacity is the ceiling,
    not the teacher.
  - Expect `h_fm800_div` (target 98.38) at about 97.85-98.05, a likely tie. It is still a sound candidate for the
    11:00 submission: equal expected accuracy, and decorrelated from Runs F/G/H for the final 2 picks.
- `h_fm800_div` started 06:19 EST (job 22388, 19.0 s/epoch, done about 10:35). `h_final800_div` starts after `w40_50`
  (259/300), so it is done about 11:15.
- **06:55 EST:** `h_final800_div` is running (epoch 2, 20.7 s/epoch, done about 11:30). `h_fm800_div` is at 97/800 (val
  96.00, done about 10:30). The second-round teachers have started.
  - `h_final800_ens3` is downloaded, with predictions in `runs/h_final800_ens3/submission_flip.csv` (not submitted).
    Its class counts are as lopsided as the others (airplane 172, automobile 217).
  - Hidden-set disagreement between candidate finals: G vs H 82, H vs 1600 77, H vs ens3 77, 1600 vs ens3 68. The
    candidates differ on 3.4-4.4% of images.
- **08:45 EST midpoint:** all 6 jobs are running, with no preemptions.
  - `h_fm800_div` is at 452/800. Mean val over epochs 442-452:

    | Run | val |
    |---|---|
    | `h_fm800_div` | 96.98 |
    | `h_fm800_ens3` | 96.97 |
    | `h_fm800` | 96.99 |
    | `h_fm800_sam` | 97.18 |
    | `h_fm800_nn` | 97.15 |

    `div` is tracking the pack, so a tie is expected. Twin done about 10:35; `h_final800_div` (322/800) about 11:35.
  - Round 2 teachers: `aa45_s2` 304/400 (97.58), `aa50_s2` 278/400, `w40aa45` 202/300 (97.34), `w40aa50` 185/300.
- **10:40 EST: `h_fm800_div` (4-teacher diverse target, 98.38) ties `h_fm800`.**

  | Model | val none / flip | suite mean none / flip | JPEG 50 flip |
  |---|---|---|---|
  | `h_fm800_div` | 97.92 / 97.82 | 96.47 / 96.59 | 86.58 |
  | `h_fm800` | 97.84 / 98.00 | 96.25 / 96.49 | 85.32 |

  - McNemar 34 vs 30 (p = 0.71) and 28 vs 37 (p = 0.32).
  - It is the best no-TTA twin so far, but flip costs it 0.10 (noise). Under the agreed plan it is still today's
    submission (`h_final800_div`, flip TTA as for every run), when it lands about 11:30.
  - **Round 2 teachers:**
    - `aa45_s2`: 97.86 / 98.06 flip, NLL 0.071.
    - `w40aa45`: 97.58 / 97.70, NLL 0.074.
    - They disagree with the TrivialAugment teachers on 1.3-1.6%, but only 1.06-1.20% with `aa45`.
    - **The teacher pool saturates:** the best 6-teacher set is 98.38 and the best 4-teacher set 98.36-98.40, the same
      as `div`. No `div2` student is queued.
  - **Conclusion:** both ceilings are reached.

    | | Range |
    |---|---|
    | teacher ensembles | 98.4 |
    | 800-epoch FM students, any target | 97.78-98.00 flip |

    No training lever left is resolvable before the deadline. What remains is prediction-time (balancing/kNN,
    pending the professor) and the choice of the final 2 picks.
- **11:30 EST: Run I submitted (`h_final800_div`), public 0.900** (ref 57991519). The team has 4 of 5 slots used today,
  per the user's plan of one today.
  - Model: 4.62M resnet_c, FM from sam50 + sam50_s2 + aa50 + w40_50, 800 epochs, all 50k, flip TTA.
  - Checks: 2,000 rows in sample order, 4,623,178 params.
  - Class counts: airplane 174, automobile 219, frog 180; the same lopsidedness as every model.
  - It disagrees with H / G / ens3 / 1600 on 71 / 77 / 65 / 70 hidden images.
  - **Public scores:**

    | Run | Public |
    |---|---|
    | F | 0.901 |
    | G | 0.901 |
    | I | 0.900 |
    | H | 0.897 |

    All four sit within one public SE (0.95 pt) of each other, as the twin ties predicted.
  - The HPC is idle apart from the babysitter. Every `runs/jobs.txt` entry has its `final.pt`.
- **11:50 EST: the professor approved class-balanced prediction** (the Slack question covered prediction-time class-prior
  adjustment and assuming about equal images per class). kNN smoothing was not part of the question and stays unused.
  - Added `--balance` to `src.evaluate`: a second row per model with `balance_probs` decoding on clean val and on every
    suite set.
  - Twins, flip TTA (`scratchpad/balance_eval.log`):

    | Twin (final) | val | → balanced | suite mean | → balanced | JPEG 50 | → balanced |
    |---|---|---|---|---|---|---|
    | `h_fm800` (H) | 98.00 | 97.92 | 96.49 | 96.67 | 85.32 | 86.70 |
    | `h_fm800_div` (I) | 97.82 | **97.90** | 96.59 | **96.77** | 86.58 | 88.32 |
    | `h_fm800_ens3` | 97.80 | 97.84 | 96.61 | **96.77** | 86.62 | 88.06 |
    | `h_fm1600` | 97.86 | 97.92 | 96.55 | 96.71 | 86.00 | 87.40 |
    | `h_fm800_sam` | 97.78 | 97.82 | 96.47 | 96.71 | 86.40 | 88.60 |
    | `r1_fm` (F) | 97.78 | 97.78 | 96.28 | 96.42 | 85.66 | 86.80 |
    | `r2_twin_S` (G) | 97.58 | 97.56 | 96.23 | 96.30 | 85.38 | 86.66 |

  - Balancing is ±0.08 on clean val (already balanced) and +0.07 to +0.24 on the suite mean. Its gain is largest under
    JPEG, where predictions skew (+1.1 to +2.2). The 800-epoch FM students gain the most (+0.16 to +0.24).
  - **Best balanced: `div` (I)**, top on suite mean (tied with ens3) and on clean.
  - Balanced CSVs: `runs/{h_final800_div,h_final800,h_final800_ens3,h_final1600,h_final800_sam,r2_final_S,r2_final_H}/
    submission_flip_bal.csv`. They change 61-78 of 2,000 predictions each, and every class lands at 192-206.
- **The user's decision:** submit the balanced Run I (`runs/h_final800_div/submission_flip_bal.csv`) right after the
  19:00 EST slot reset, keeping today's last slot free for teammates. It is a clean A/B against Run I (0.900): only the
  70 balanced predictions differ.
- **12:30 EST: everything is shared on GitHub for the team.** Branch `abhinav` of `Vedant0703/Deep-Learning-2026`,
  commit `01aef2a`.
  - `Abhinav/submissions/Abhinav_Run-{D,E,F,G,H,I}/` and `candidates/{Abhinav_1600-epochs,Abhinav_ens3,
    Abhinav_sam-teacher}/`, each with:
    - `model.pt`;
    - the submitted `submission.csv` (D/E regenerated from their checkpoints; regeneration reproduces F's file byte
      for byte);
    - `submission_balanced.csv`;
    - logs and `twin/`;
    - a README with scores and the exact command.
  - Also `scripts/hpc/`, logs of every HPC run (`runs/h_*`, pulled as one JSON bundle through the browser because the
    babysitter's node died at 10:40 EST), and the latest `src/` and notes.
  - Teacher weights (146-224 MB) exceed GitHub's 100 MB limit and stay on the HPC.
- **12:45 EST: Round 8, the push before the 19:00 reset** (the user: train new approaches, add prediction-time
  class-prior variants, pick the top 2-3 together, then submit).
  - **Teammates' work:** Vedant's `origin/main` has 3 notebooks.
    - Model: PreAct 4.98M, 59/118/236 channels, 4-4-4 blocks.
    - Recipe: TrivialAugment + erasing, MixUp/CutMix with hard labels, EMA, flip TTA.
    - Runs A-D scored 0.875 / 0.892 / 0.881 / 0.895.
    - The one new idea: mixing OFF for the last 15 of 300 epochs.
    - His CSVs disagree with ours on 102-147 hidden images. Dhanush has no project code on GitHub.
  - **Training arms** (`runs/jobs.txt`, all on the `div` 4-teacher recipe, 600 epochs, twin + final):
    1. `mixoff`: `--mix-off-epochs 30`.
    2. `dist`: `--kd-loss dist --kd-alpha 0.67`.
    3. `pre`: `--model preact_v`.
    4. `t1`: `--kd-temp 1`.
  - **Job submission:** the babysitter died, so Jupyter 22754 (CPU) was launched to `sbatch` it again.
  - **Prediction-time variants added:**
    - `balance_probs(..., temp)` is now log-domain Sinkhorn on p^(1/T). T=1 matches the old version to 3e-15; T→0
      gives the exact-count assignment (T=0.05: 198-202 per class).
    - `em_prior_probs`: Saerens et al. EM prior estimation. On a skewed toy it amplifies the skew (275 → 350), as
      expected.
    - Flags: `src.predict --balance-temp`, `src.evaluate --balance-temps ... --em-prior`.
    - The sweep on 7 twins (flip + flip_shift) is running.
- **13:10 EST: HPC job submission restored.**
  - Jupyter sessions: 22754 (CPU) failed its boot after 13 min; 22767 (CPU) was held with "user env retrieval
    failed"; 22766 (A100) sat pending on Priority; 22776 (CPU) came up on b-10-83.
  - From its terminal (API + websocket):
    - babysitter v3 resubmitted (22780);
    - 22766 and 22767 cancelled;
    - the 8 Round 8 jobs submitted directly with the babysitter's own `@C` expansion: `h_fm600_*` /
      `h_final600_*` for `mixoff`, `dist`, `pre`, `t1` = 22781-22788. Their `.count` files are set to 1.
    - All 8 are pending on (Resources): the cloud nodes are being provisioned.
  - **Colab hedge** (16.87 units, A100): `c_fm300_mixoff` (twin, `teacher_wrn300`) and `c_final300_mixoff`
    (`teacher50`, all 50k) are running.
    - Recipe: Run F's exactly, plus `--mix-off-epochs 15` (Vedant's 15/300).
    - It is a clean one-factor test against `r1_fm` (97.66 / 97.78). Done about 14:40.
- **13:25 EST: the teachers are shared as a GitHub release** (the user chose a public release after being told the
  repo is public): <https://github.com/Vedant0703/Deep-Learning-2026/releases/tag/teachers-v1> (tag `teachers-v1` on
  `abhinav`).
  - `teacher_wrn300.pt` (45k, val 97.96 / 97.90) and `teacher50.pt` (all 50k), 146,039,914 bytes each.
  - MD5s match the HPC copies (`0af9a616…`, `929e4ba0…`). The notes give the recipe, the runs each teacher fed, and
    how to load and distil.
  - HPC: babysitter 22780 is running; the 8 Round 8 jobs are pending on Priority.
- **The user's direction (13:30):** do not shorten any run. Submit any time after 19:00, once the best candidates are
  ready; optimise for the best result, not the clock. Every Round 8 arm keeps its full 600 epochs, and the pending
  ones run whenever they get GPUs.
- **13:55 EST: decoding sweep done** (7 twins × {flip, flip_shift} × {plain, Sinkhorn T=2/1/0.5/0.25/0.1, EM};
  `scratchpad/decode_sweep.log`).

  | Decoding (mean over 7 twins) | clean val | suite mean |
  |---|---|---|
  | flip, plain (what was submitted so far) | 97.80 | 96.46 |
  | flip + balanced T=1 | 97.82 | 96.62 |
  | flip_shift, plain | 97.89 | 96.53 |
  | **flip_shift + balanced T=1** | **97.87** | **96.72** |
  | flip_shift + balanced T=0.1 (about exact 200 per class) | 97.87 | 96.70 |
  | EM prior (Saerens) | ≈ plain | 96.40 / 96.47 (worse, as expected) |

  - **Chosen for every candidate: flip_shift TTA + balancing at T=1.** It is +0.07 clean and +0.26 suite vs what was
    submitted.
    - The suite gain holds for all 7 models (+0.20 to +0.35).
    - Temperature doesn't matter (±0.02), and the exact-count version ties.
  - Per twin (fs + bal): `div` 97.96 / 96.88 (best suite), `h_fm800` 97.98 / 96.79, ens3 97.90 / 96.81, sam
    97.90 / 96.82, 1600 97.88 / 96.77, F's twin 97.70 / 96.50, G's 97.78 / 96.48.
  - **CSVs:** `runs/{h_final800_div,h_final800,h_final800_ens3,h_final1600,h_final800_sam,r2_final_H,r2_final_S}/
    submission_fs_bal.csv`. Each changes 58-86 predictions by balancing and has 195-207 per class.
  - I-fs-bal vs the submitted Run I: 81 predictions differ (vs I-flip-bal: 27). Pairwise disagreement between
    fs-bal candidates is 62-94.
  - **HPC:** `mixoff` twin 81/600 (val 95.62, done about 16:35); final 73/600 (done about 16:55). `dist` twin went
    back to pending after a failed boot; dist / pre / t1 are pending on Priority.
- **14:25 EST: the Colab mix-off test (Run F's recipe + `--mix-off-epochs 15`, 300 epochs) is a tie, not adopted.**

  | Model | val flip | suite mean flip | fs + bal (clean / suite) |
  |---|---|---|---|
  | `c_fm300_mixoff` | 97.62 | 96.31 | 97.58 / 96.46 |
  | `r1_fm` | 97.78 | 96.28 | 97.70 / 96.50 |

  - McNemar p = 0.36 (flip) and 1.0 (fs + bal).
  - Train accuracy jumps to 93.6% once mixing stops, but val does not move. Finishing on clean images does not help
    the FM student, which tempers expectations for the HPC `mixoff` arm.
  - Colab session stopped; 9.28 units left. Runs saved in `runs/c_fm300_mixoff`, `runs/c_final300_mixoff`.
- **14:35 EST HPC:** `mixoff` twin 232/600 (val 96.80, done about 16:25); final 211/600 (done about 16:40).
  - dist / pre / t1 are pending on Priority (each has a `train.log` header but no epochs; an earlier start was cut
    short).
  - Jupyter 22776 is gone (preempted). The babysitter resubmitted itself as 22810 and is running, so the queue is
    covered.
- **16:45 EST: the HPC `mixoff` twin (`div` recipe at 600 epochs + `--mix-off-epochs 30`) ties `div`; not adopted.**

  | Model (twin) | val none / flip | suite mean flip | fs + bal (clean / suite) | McNemar vs `div` (flip / fs) |
  |---|---|---|---|---|
  | `h_fm600_mixoff` | 97.76 / 97.84 | 96.65 | 97.86 / 96.82 | 28 vs 27, p = 1.0 / 21 vs 22, p = 1.0 |
  | `h_fm800_div` | 97.92 / 97.82 | 96.62 (bal) | 97.96 / 96.88 | baseline |

  - It matches `div` with 25% fewer epochs, but the gain is within noise (rule: +0.15). Second mix-off tie after the
    Colab one, so finishing on clean images does nothing for the FM student.
  - The `dist` twin (22783) started booting at 16:42; the queue seems to run two GPU jobs at a time. Expected order:
    `dist` pair done about 20:15, then `pre`, then `t1` (about 23:45 and 03:15).
- **19:05 EST: HPC is short of nodes, and `dist` has not started.**
  - The `dist` and `t1` boots failed (17:00-17:15) and the babysitter resubmitted them: `h_fm600_dist` 22955,
    `h_fm600_t1` 22956, `h_final600_t1` 22957. `h_final600_dist` 22784 is still pending.
  - Slurm's estimated starts: `h_final600_dist` 22:44 EST, `h_fm600_dist` 23:23 EST, so done about 02:30-03:00 Nov 5.
  - Only `h_final600_pre` (22786) is running: 292/600 at 23 s/epoch, done about 21:00. Its twin (22785) is pending,
    so it can't be validated tonight.
  - The babysitter is being preempted every 20-40 minutes but resubmits itself each time (now 23038).
  - Kaggle: a teammate submitted `submission (1).csv` at 23:55 UTC (0.890), the 5th slot of the Nov 4 UTC day. The
    Nov 5 UTC day (19:00 Nov 4 to 19:00 Nov 5 EST) has all 5 slots free.
- **21:05 EST: `h_final600_pre` (PreAct 4.98M, `div` recipe, 600 epochs, 50k) is done**, with no twin yet.
  - `runs/h_final600_pre/submission_fs_bal.csv`: balancing changed 72 predictions; 197-203 per class. It differs from
    I-fs-bal on 66 hidden images and from G-fs-bal (the same architecture at 300 epochs, 1 teacher) on 91.
  - `h_final600_dist` (22784) started at about 20:10: 149/600 at 20:57, done about 23:45.
  - Every twin is pending again after more failed boots (resubmitted as `h_fm600_dist` 23094, `h_fm600_pre` 23095,
    `h_fm600_t1` 23096, `h_final600_t1` 23097). Without a twin, a final can't be judged.
  - The OOD session expired (401) at about 21:00. One SSO redirect gave a 400 at `/oidc`; it worked after a retry.
- **23:45 EST: `h_final600_dist` is done; two twins are finally training.**
  - `runs/h_final600_dist/submission_fs_bal.csv`: balancing changed only 35 predictions, leaving 185-218 per class
    (automobile 218, frog 185). The DIST loss matches correlations, not probabilities, so its softmax is a poorer input
    for Sinkhorn at T=1. It differs from I-fs-bal on 80 hidden images.
  - The twins run on the same 600-epoch schedule as `h_fm600_mixoff`, which equals the `div` recipe until epoch 570:

    | Twin | Epoch | val (no TTA) | `h_fm600_mixoff` at the same epoch |
    |---|---|---|---|
    | `h_fm600_dist` (23094) | 385 | 96.90 | 97.12 |
    | `h_fm600_t1` (23096) | 454 | 97.30 | 97.46 |

    Both are slightly behind, within single-epoch noise. Expected done: `t1` about 00:30, `dist` about 00:50.
  - `h_fm600_pre` (23230) and `h_final600_t1` (23231) are still pending.
- **Nov 5, 01:00 EST: the `t1` and `dist` twins are done; neither beats `div`. Round 8 has no winner so far.**

  | Twin (600 epochs) | val none / flip | flip + bal (clean / suite) | fs + bal (clean / suite) | McNemar vs `div` (fs) |
  |---|---|---|---|---|
  | `h_fm800_div` (800) | 97.92 / 97.82 | 97.90 / 96.77 | **97.96 / 96.88** | baseline |
  | `h_fm600_mixoff` | 97.76 / 97.84 | 97.84 / 96.78 | 97.86 / 96.82 | 21 vs 22, p = 1.0 |
  | `h_fm600_t1` (`--kd-temp 1`) | 97.76 / 97.92 | 97.94 / 96.75 | 97.90 / 96.78 | 22 vs 24, p = 0.88 |
  | `h_fm600_dist` (DIST, α 0.67) | 97.54 / 97.82 | 97.80 / 96.55 | 97.76 / 96.61 | 19 vs 30, p = 0.15 |

  - `t1` ties. `dist` is worse on the suite (−0.27), and balancing helps it least (its softmax is poorly calibrated).
  - Still pending: `h_fm600_pre` (23277) and `h_final600_t1` (23278).
- **Nov 5, 02:25 EST: submitted Run J (I-bal) and Run K (H-bal) on the user's instruction.** These were the first 2
  of the Nov 5 UTC quota; 3 remain until 19:00 EST.

  | Run | File | Twin fs + bal (clean / suite) | Public | Same model, plain flip |
  |---|---|---|---|---|
  | **J** (ref 58014367) | `runs/h_final800_div/submission_fs_bal.csv` | 97.96 / 96.88 | **0.902** | Run I 0.900 |
  | K (ref 58014369) | `runs/h_final800/submission_fs_bal.csv` | 97.98 / 96.79 | 0.894 | Run H 0.897 |

  - **The team's best is now 0.902: #21 of 68.** The legit top 5 are 0.918, 0.915, 0.915, 0.914, 0.912.
  - On the public 1,000, decoding moved I by +2 images and H by −3, net −1 over the pair. Each changes about 80 of
    2,000 predictions, so about 40 public ones. That is within noise; the public set can't tell whether balancing
    helps.
- **02:40 EST GitHub:** `abhinav` commit 5c5f117 adds `Abhinav/submissions/Abhinav_Run-J`, `Abhinav_Run-K` and
  `candidates/{Abhinav_mixoff-600,Abhinav_dist-600,Abhinav_preact-600}`. It also has the Round 8 logs, the updated
  `src/predict.py` / `src/evaluate.py`, these notes and the README results table.

## Round 9 (Nov 5): baseline architecture study, and a hard-factor set built from our own images
Plan: `~/.claude/plans/the-high-level-goal-rosy-star.md`. Data rules (course): train only on the permitted training
images; augmentations of them are allowed; no external data or weights; no evaluation labels; reproducible.
- **03:07 EST: 9 baseline/KD jobs queued** (`runs/jobs.txt`, local copy `scripts/hpc/jobs_round9.txt`).
  - Seven architectures with no teacher at 300 epochs, `@C`: `h_base300_{c,pre,a,b,cse,chr,cs112}`.
  - Plus KD for the two without one: `h_kd300_{a,b}`, with `teacher_wrn300` as `r1_fm` / `p1_*` / `r2_twin_S`.
- **GPU usage:** 96.6 GPU-hours since Nov 1 (`sacct`). The account limit is `gres/gpu=18000` minutes (300 h).
- **`src/hard.py` (new):** factors made from our own labelled images, each keeping its source label:
  - `lowres` (down to 12-24 px and back);
  - `zoom_in` (1.2-1.5×);
  - `zoom_out` (0.6-0.85×, edge-replicate pad; reflect padding mirrored a second object into the frame);
  - `rotate` (15-30°);
  - `perspective` (0.3-0.5);
  - `background` (shrunk into the centre of another image);
  - `confuse` (a 0.65-0.8 blend with a confusable class, keeping the majority label).
  - **`src.evaluate --hard-suite`** runs each factor at one level on the 5k held-out images. It adds `ambiguous`:
    the 500 held-out images `teacher_wrn300` is least sure of.
  - **`--confusion`** prints the top mistakes per row.
- **03:15 EST: hard-suite diagnosis (flip TTA, twins):**

  | Model | Clean | lowres16 | lowres24 | zoom_in1.3 | zoom_out0.7 | rot20 | persp0.4 | bg0.7 | confuse0.7 | ambiguous | Mean |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | `h_fm800_div` (Run I) | 97.82 | 80.80 | 94.88 | 93.32 | 86.22 | 93.68 | 95.34 | 80.46 | 90.70 | 80.20 | 88.40 |
  | `h_fm800` (Run H) | 98.00 | 80.88 | 94.82 | 93.66 | 83.92 | 93.42 | 95.36 | 79.36 | 90.64 | 82.40 | 88.27 |
  | `h_fm600_t1` | 97.90 | 82.24 | 94.88 | 93.66 | 86.06 | 93.48 | 95.32 | 80.60 | 90.92 | 81.20 | 88.71 |
  | `r1_fm` (F, 300 ep) | 97.78 | 75.56 | 93.82 | 92.72 | 83.52 | 93.20 | 94.72 | 78.54 | 89.96 | 80.00 | 86.89 |
  | `r2_twin_S` (G, PreAct) | 97.58 | 77.58 | 93.58 | 92.80 | 85.72 | 92.84 | 94.94 | 79.26 | 90.12 | 78.00 | 87.20 |
  | `base_c` (no KD, 200 ep) | 97.00 | 69.68 | 92.10 | 90.32 | 81.26 | 90.60 | 93.66 | 75.76 | 88.12 | 74.20 | 83.97 |

  - **The big drops:**
    - background clutter (−17);
    - low resolution at 16 px (−17);
    - zoom-out (−12);
    - zoom-in (−4.5).
    Rotation (−4) and perspective (−2.5) are already handled by TrivialAugment.
  - **Systematic errors:**
    - a shrunk object becomes "cat" (zoom-out: frog→cat 188, dog→cat 106, deer→cat 103 for Run I);
    - clutter makes ships and planes "truck" (ship→truck 67, airplane→truck 47);
    - low resolution turns dogs and cats into "bird" (dog→bird 119, cat→bird 86);
    - everywhere else the dominant error is cat↔dog.
  - KD helps on every factor (+2.9 mean, `r1_fm` vs `base_c`), and longer FM adds about +1.5 more.
- **03:22 EST: hard training set and fine-tunes queued.**
  - **The set:** `scripts/make_hard.py --factors lowres zoom_in zoom_out background` writes
    `data/cache/hard_train2k.npz`.
    - 2,000 images, 200 per class, from the 45k side, 1-2 factors each.
    - Local MD5 `798495a0951cd17e1dcedd586dcca57c`. It is rebuilt on the HPC by an inbox script.
  - **The jobs** (100 epochs, lr 0.02, warmup 1, `div` teachers, extra set repeated 3×):
    - `h_ft100_hard` (twin, init `h_fm800_div`);
    - `h_ft100_ctrl` (the same, without the extra set);
    - `h_ftfinal100_hard` (final, init `h_final800_div`);
    - from scratch at 600 epochs: `h_fm600_hard` / `h_final600_hard`.
  - The baseline jobs get `Nice=1000`, so the fine-tunes run first.
  - **New flags in `src/train.py`:** `--extra-train/--extra-repeat` (with an assert that no source image is from the
    held-out split) and `--init` (our own `final.pt`).
- **03:45 EST: Round 9b/9c in flight, split between HPC and Colab** (the user asked for both, for speed).
  - **The user's follow-ups:**
    - Will fixing these weak spots raise the overall score? Only by the share of hidden images that look like them:
      0 to +0.5 expected, up to about +1 if the overlap is large.
    - Use augmentation, not only a fixed set.
    - Use the true labels on augmented images.
  - **`--hard-aug P --hard-factors ... --hard-label-weight W`** (`src/train.py`, `src/hard.py hard_batch`).
    - A fraction P of each batch gets one random factor on the GPU, *after* the teacher has seen the clean image.
    - Those images get W true-label weight (0.5 vs the usual 0.1), via per-image CE + KL.
    - Unchanged images stay bit-identical. The per-image KL mean equals the batchmean loss (checked).
  - **HPC:**
    - `h_ftfinal100_haug` (the final with on-the-fly aug) is queued.
    - `h_ft100_hard`, `h_ft100_ctrl`, `h_ftfinal100_hard` and `h_fm600_hard` / `h_final600_hard` are pending.
    - The baselines got nodes at 03:30: a, b, pre, `kd_a` and c are running.
    - Users can neither change Nice nor hold their own jobs here ("Unspecified error"), so the queue runs in
      submission order.
    - The HPC build of `hard_train2k.npz` matches the local MD5 (`798495a0951cd17e1dcedd586dcca57c`).
  - **Colab (A100, 9.28 units):** `c_ft100_haug` (the twin with on-the-fly aug, `div` 45k teachers, init
    `h_fm800_div`).
    - The teachers are uploaded in parallel; uploading 18.6 MB took 68 s.
    - **zsh lesson again:** an unquoted `$T` holding 4 paths went to `put` as a single path (takeaway 35).
- **04:20 EST: architecture baselines (no teacher, 300 epochs, `@C`, flip TTA; balanced in brackets):**

  | Arch | Params | Baseline none / flip (suite) | + KD (teacher_wrn300, FM) none / flip (suite) | KD gain (flip) | Hard mean, base → KD |
  |---|---|---|---|---|---|
  | resnet_c | 4.62M | 96.74 / 97.12 (95.73) | `r1_fm` 97.66 / 97.78 (96.28) | +0.66 | 84.1 → 86.9 |
  | preact_v | 4.98M | 96.78 / 97.26 (95.78) | `r2_twin_S` 97.40 / 97.58 (96.23) | +0.32 | 85.0 → 87.2 |
  | resnet_b | 4.34M | 96.94 / 97.22 (95.68) | `h_kd300_b` pending | | 83.3 → |
  | resnet_a | 4.98M | 95.84 / 96.20 (94.67) | `h_kd300_a` 96.80 / 96.88 (95.58) | +0.68 | 80.3 → 84.3 |

  - **Baseline ranking:** preact_v ≈ resnet_b ≈ resnet_c (97.1-97.3 flip, within noise), with resnet_a clearly last
    (one block per deep stage).
  - **With KD, resnet_c gains most and is best (97.78).** The PreAct baseline leads, but it gains only +0.32 from KD.
  - Still pending: `h_kd300_b`, `h_base300_{cse,chr,cs112}`.
- **04:45 EST: `c_ft100_haug` (Colab A100) works on the hard factors and ties on clean images.**
  - The run: Run I's twin fine-tuned for 100 epochs at lr 0.02 with `--hard-aug 0.3 --hard-label-weight 0.5`,
    `div` 45k teachers. It took 31 min at 18.7 s/epoch; the VM stopped with 4.35 units left.

  | Twin | val none / flip | fs + bal (clean / suite) | Hard mean (fs) | lowres16 | zoom_in | zoom_out | bg | rot | persp | confuse |
  |---|---|---|---|---|---|---|---|---|---|---|
  | `h_fm800_div` | 97.92 / 97.82 | 97.96 / 96.88 | 88.40 (flip) | 80.80 | 93.32 | 86.22 | 80.46 | 93.68 | 95.34 | 90.70 |
  | `c_ft100_haug` | 97.62 / 97.70 | 97.86 / 96.74 | **93.07** | **94.06** | **96.30** | **95.52** | **94.68** | 94.04 | 95.78 | 90.94 |

  - **McNemar vs `div`:** flip 30 vs 36, p = 0.54; fs 18 vs 27, p = 0.23. A tie on clean images.
  - The targeted factors gain +3 to +14. Rotation, perspective and confusable blends (not trained) are unchanged.
  - Against the rule: clean 97.86 (= the line); suite 96.74 (0.04 under the 96.78 line). The control
    `h_ft100_ctrl` (HPC, running) shows whether the small clean / suite dip comes from fine-tuning itself.
  - Balancing hurts the "ambiguous" row only because that row is a class-unbalanced subset (its balanced number is
    not meaningful).
- **05:20 EST: the control and fixed-set twins settle it: on-the-fly augmentation works, the fixed set does not.**

  | Twin (fs + bal) | Clean | Suite | Hard mean (fs) | lowres16 | zoom_out | bg | McNemar vs `div` (fs) |
  |---|---|---|---|---|---|---|---|
  | `h_fm800_div` (Run I) | 97.96 | 96.88 | 88.40 (flip) | 80.80 | 86.22 | 80.46 | – |
  | `h_ft100_ctrl` (fine-tune, nothing added) | 97.62 | 96.65 | 88.12 | 78.86 | 86.58 | 82.70 | 12 vs 26, **p = 0.03** |
  | `h_ft100_hard` (fixed 2k set ×3) | 97.80 | 96.75 | 88.04 | 76.78 | 85.94 | 82.08 | 8 vs 13, p = 0.38 |
  | `c_ft100_haug` (on-the-fly, teacher clean, label 0.5) | 97.86 | 96.74 | **93.07** | **94.06** | **95.52** | **94.68** | 18 vs 27, p = 0.23 |

  - Fine-tuning alone costs about 0.3 clean (p = 0.03), so 100 more epochs at lr 0.02 is slightly harmful.
  - **On-the-fly augmentation vs the control:** +0.24 clean, +0.09 suite, +5 on the hard suite. It is a tie with
    Run I's twin on clean images.
  - **The fixed set changed nothing on the hard factors.** Its images pass through crop / TrivialAugment / MixUp, and
    the teacher labels the hard version, so the student learns the teacher's (weak) answers there.
  - **Decision:** the fixed-set final and from-scratch jobs are cancelled. The from-scratch pair is now
    `h_fm600_haug` / `h_final600_haug` (`div` recipe, 600 epochs, `--hard-aug 0.3 --hard-label-weight 0.5`).
  - **Candidate for today's submission:** `runs/h_ftfinal100_haug/submission_fs_bal.csv`. Balancing changed 60
    predictions; 196-205 per class; it differs from Run J on 51 hidden images.
- **11:30 EST: the from-scratch hard-aug pair finished (HPC, 600 epochs, about 19-20 s/epoch, 3.3 h each).**
  - Train log: `h_fm600_haug` final val 97.80 (flip 97.94), best 97.84. `h_final600_haug` trained on all 50k.
  - Twin evaluation (`src.evaluate --tta flip flip_shift --balance --hard-suite --baseline runs/h_fm800_div/final.pt`):

  | Twin | flip | flip + bal | fs | fs + bal (clean / suite) | Hard mean (flip) | lowres16 | zoom_out | bg | zoom_in | McNemar vs `div` (flip) |
  |---|---|---|---|---|---|---|---|---|---|---|
  | `h_fm800_div` (Run I, 800 ep) | 97.82 | 97.90 | – | 97.96 / 96.88 | 88.40 | 80.80 | 86.22 | 80.46 | 93.32 | – |
  | `c_ft100_haug` (fine-tune) | 97.70 | 97.74 | – | 97.86 / 96.74 | 93.02 | 93.98 | 95.50 | 94.68 | 96.50 | 30 vs 36 |
  | **`h_fm600_haug` (from scratch, 600 ep)** | **97.94** | **98.06** | **98.00** | **97.96 / 96.88** | **93.58** | **94.62** | **96.20** | **95.66** | **96.74** | 33 vs 27, p = 0.52 |

  - **It passes every part of the rule:** clean 97.96 (≥ 97.86), suite 96.88 (≥ 96.78), the targeted factors
    +3.4 to +14 over `div`, and above the fine-tune on all of them.
  - It is the first change since Round 6 that ties the best clean / suite numbers *and* adds something measurable
    (hard mean +5.2). With 200 fewer epochs.
  - **Final:** `runs/h_final600_haug/submission_fs_bal.csv` (balancing changed 70; 197-206 per class). It differs
    from Run J on 69 hidden images and from the fine-tune (`h_ftfinal100_haug`) on 68.
- **11:38 EST: Round 9e added to the HPC queue file for Nov 6:** `h_fm800_haug` / `h_final800_haug` (Run I's full 800 epochs
  + hard aug 0.3) and `h_fm600_haug5` / `h_final600_haug5` (p = 0.5). Teachers and everything else unchanged.
- Also finished: `h_fm600_pre` (PreAct twin, for the `h_final600_pre` final), `h_kd300_b`, `h_base300_cse`,
  `h_base300_cs112`. Evaluating.
  - Flip_shift rows for `h_fm600_haug`: hard mean 93.68; McNemar vs `div` 28 vs 24, p = 0.68.
- **11:40 EST: the HPC babysitter is down.** It was preempted at 15:21 UTC (the 4th time today, about every 40-50
  min) and resubmitted itself (23548), but that job never ran. So Round 9e was not submitted.
  - Restarting it via an OOD Jupyter session on `n2c48m24` (23692, queued 16:41 UTC), then
    `sbatch scripts/hpc/babysit.sbatch` from its terminal.
  - The A100 partition has about 200 queued jobs from another user, so the 9e runs may wait for nodes.
- **11:50 EST: the baseline study is complete** (300 epochs, `@C`, flip TTA, suite in brackets; KD = FM from
  `teacher_wrn300`):

  | Arch | Params | Baseline flip (suite) | + KD flip (suite) | KD gain | Hard mean (flip), base → KD |
  |---|---|---|---|---|---|
  | **resnet_c** | 4.62M | 97.12 (95.73) | **97.78** (96.28) `r1_fm` | +0.66 | 84.1 → 86.9 |
  | preact_v | 4.98M | **97.26** (95.78) | 97.58 (96.23) `r2_twin_S` | +0.32 | 85.0 → 87.2 |
  | resnet_b | 4.34M | 97.22 (95.68) | 97.58 (96.17) `h_kd300_b` | +0.36 | 83.3 → 86.4 |
  | resnet_c_s112 | 4.62M | 97.12 (95.32) | 97.58 `p1_s112` | +0.46 | 81.9 → – |
  | resnet_c_se | 4.65M | 96.62 (94.95) | 97.54 `p1_se` | +0.92 | 84.1 → – |
  | resnet_c_hr | 4.62M | 96.70 (95.10) | 97.44 `p1_hr` | +0.74 | 81.9 → – |
  | resnet_a | 4.98M | 96.20 (94.67) | 96.88 (95.58) `h_kd300_a` | +0.68 | 80.3 → 84.3 |

  - **Every architecture gains +0.3 to +0.9 from distillation; resnet_c ends best by 0.2** (about 10 images; the
    next three tie at 97.58). The follow-up rule (another architecture ≥ 0.15 above `r1_fm` with KD) is not met, so
    resnet_c stays.
  - The baseline ranking does not predict the KD ranking: preact_v / resnet_b lead without a teacher but gain least.
  - KD also lifts the hard suite by 2.5-4 points without any hard augmentation (low-res and background most).
- **PreAct twin `h_fm600_pre`** (the check for the `h_final600_pre` final): fs + bal 97.94 / 96.74, hard mean 88.3
  (flip_shift); McNemar vs `div` 22 vs 24, p = 0.88. A tie on clean images, slightly lower suite, no hard gain.
  It is a decorrelated hedge only (`runs/h_final600_pre/submission_fs_bal.csv`); hard aug is the better candidate.
- **12:45 EST: cloud instability.** The 4 Round 9e jobs (23703-23706) sat in CF (node boot) for about 12 min, then
  ended without starting: the nodes failed to boot. The babysitter 23702 was preempted at 17:21 UTC after 30 min
  (resubmitted as 23737, queued), and the Jupyter session 23692 was preempted too.
  - Counts are at 1/8, so the babysitter resubmits all four when it next runs.
- **13:00 EST: resubmitted from a new Jupyter session (23765, b-10-119).**
  - sacct: 23703-23706 were "CANCELLED by 0" after 12-13 min in CF (cloud node boot failure).
  - The babysitter's own resubmit 23737 was stuck pending with "user env retrieval fail", so it would never start.
    It was cancelled and replaced by 23782.
  - The 4 Round 9e jobs are back in the queue as 23778-23781 (attempt 2/8). The A100 partition has 84 queued and 14
    running, and `n2c48m24` 77 queued and 5 running, so expect waits.
- **13:12 EST: submitted Runs M and L (the user confirmed; team count 2 of 5 before).**

  | Run | Model | Decoding | Twin fs + bal (clean / suite) | Hard mean | Expected | **Public** |
  |---|---|---|---|---|---|---|
  | J (before) | Run I's model | fs + bal | 97.96 / 96.88 | 88.4 | – | 0.902 |
  | **M** | `h_final600_haug` (from scratch, hard aug, 600 ep) | fs + bal | 97.96 / 96.88 | 93.6 | ≈ 0.904 (0.898-0.910) | **0.910** |
  | L | `h_ftfinal100_haug` (Run I + 100-ep hard-aug fine-tune) | fs + bal | 97.86 / 96.74 | 93.0 | ≈ 0.903 (0.897-0.909) | **0.906** |

  - **M is the team's best: 0.910, +0.8 over J** (+8 of the 1,000 public images). It differs from J on 69 hidden
    images, about 35 of them in the public half, and gets about 8 more of those right than J does. L is +0.4.
  - Both hard-aug models beat J, and the from-scratch one by more. Its twin tied J on clean val. **So the hidden set
    does contain images like the hard factors** (scale, clutter, low effective resolution): the first training change
    since Round 6 that moved the leaderboard.
  - Leaderboard: rank 13 of 69 (tied 11-13 at 0.910). The legit top 5 is 0.913-0.918 (two entries at 1.000 / 0.972
    are not credible), so M is 0.3-0.8 points from it.
  - 1 slot left before 19:00 EST.
- **13:50 EST: Run I's twin on the hard suite with flip_shift** (for like-for-like tables): no balancing, mean 88.78
  (lowres16 80.42, zoom_out 86.58, bg 82.68). So Run M's twin gains +4.9 on the mean, +14.3 on lowres16, +9.7 on
  zoom-out and +12.9 on background (all flip_shift).
- **Team write-up:** `SurveyCorps_CIFAR10_findings_Nov5.pdf` (6 pages: approach, baselines, KD, hard-image
  augmentation, results, ideas for parallel work, GitHub links), in `Abhinav/reports/` on the `abhinav` branch.
  Figure 1 comes from `scripts/hard_suite_figure.py`.
- **13:34 EST: Dhanush submitted "Run G, single 4.98M PreAct ResNet, 10-view translation + horizontal-flip probability
  TTA. Twin validation 97.76%. No class balancing." → 0.904** (ref 58032451), the team's third-best score after M
  0.910 and L 0.906. The letter clashes with Abhinav's Nov 4 Run G (PreAct FM KD, 0.901). Its code is not on GitHub.
  - That was the 5th team submission of the Nov 5 UTC day (J, K, M, L, G), so no slots remain until 19:00 EST.
  - Final-pick candidates: M plus L, or plus Dhanush's G (a different architecture, so a more decorrelated hedge).
  - The team PDF (v2) now credits runs by person (Vedant: A, B and his own Run D 0.895; teammate: C; Abhinav: D-M;
    Dhanush: the Nov 5 G) and adds Dhanush's result.

## Round 10 (Nov 5 afternoon to Nov 6): building on hard augmentation, robustly
- **Plan:** `~/.claude/plans/the-high-level-goal-rosy-star.md`.
  - New pairs: PreAct + hard aug, 6 factors (+ rotation, perspective), wide levels.
  - Scale-aware TTA tested on the twins under a pre-registered rule.
  - Selection by twin, decorrelation on held-out errors, and the final picks pre-registered so public noise
    can't drive them.
- **Do the report's approaches still hold?**
  - Function matching: yes.
  - resnet_c vs PreAct: resnet_c wins on the twin, but public is a tie (0.901-0.904), and PreAct had never had
    hard aug.
  - Longer training / diverse teachers: tie.
  - fs+bal: weak (twin suite +0.2; public ±0.3; Dhanush's 0.904 used none); kept as the pre-registered rule.
  - On-the-fly hard aug: strongly yes. It even lifts *untrained* factors a little (rot, persp, confuse,
    ambiguous: +0.4 on average), so it generalises.
  - Fixed hard set: no.
- **15:00 EST code** (`src/hard.py` 61908988a52768a7, `src/train.py` 220d94ca676bffca; both on the HPC via
  hash-verified patches):
  - `hard_batch` gains `rotate` / `perspective` (`ALL_BATCH_FACTORS`, both device-safe) and a level table
    `HARD_LEVELS` ({default, wide}; `--hard-levels`).
  - `tta_probs` gains `flip_shift_scale` (14 views: fs + both flips zoomed 1.15x and 0.875x); `TTA_CHOICES` is
    shared with evaluate and predict.
  - Checks:
    - the default path is bit-identical to Run M's `hard_batch` (same seed, same output);
    - the hard suite is unchanged (Figure 1 regenerates byte-identical);
    - 1-epoch smoke tests pass for preact_v + hard aug and for 6 factors + wide;
    - `config.json` records `hard_factors` and `hard_levels`.
- **Queue** (`runs/jobs.txt`, local copy `scripts/hpc/jobs_round10.txt`; job lines verified identical by hash):
  `h_{fm,final}600_pre_haug`, `h_{fm,final}600_haug6`, `h_{fm,final}600_haugw`, behind Round 9e. 9e status at
  19:12 UTC: `h_fm800_haug` epoch 65/800, `h_fm600_haug5` 65/600, `h_final600_haug5` 59/600, `h_final800_haug`
  pending.
- **Held-out error overlap with M's twin** (flip TTA; M has 103 errors):

  | Twin | Errors | Both wrong | Only M wrong | Only other wrong | Jaccard |
  |---|---|---|---|---|---|
  | L's twin `c_ft100_haug` | 115 | 86 | 17 | 29 | 0.65 |
  | PreAct-600 `h_fm600_pre` | 106 | 81 | 22 | 25 | 0.63 |
  | Run I `h_fm800_div` | 109 | 76 | 27 | 33 | 0.56 |
  | Run H `h_fm800` | 100 | 71 | 32 | 29 | 0.54 |
  | Abhinav's G `r2_twin_S` (PreAct 300) | 121 | 77 | 26 | 44 | 0.52 |

  - L is the *least* decorrelated from M, so it is a weak hedge for the private half.
- **15:00 EST: Dhanush's 0.904 is a re-decoding of Abhinav's Run G** (branch `dhanush/run-g-tta-0904`, folder
  `Abhinav/submissions/Dhanush_Run-G-TTA`).
  - Same `r2_final_S` weights (preact_v, FM KD from `teacher_wrn300`, 300 epochs) with `--tta flip_shift`, no
    balancing; 31 of 2,000 labels changed from Run G.
  - Its twin: 97.58 flip → 97.76 fs; public 0.901 → 0.904. So the three best team scores are all Abhinav's models
    (M, L, G).
  - Run G's twin is the most decorrelated from M's twin on held-out errors (Jaccard 0.52): the best hedge candidate.
  - The PDF (v3) is corrected to say so.
- **14:50 EST HPC:**
  - The babysitter now runs inside OOD Jupyter 24126 (`interactive` partition, b-9-10): `nohup bash
    scripts/hpc/babysit.sbatch &`. As a batch job it kept failing with "user env retrieval fail" (24144, 23737) or
    on CPU spot nodes.
  - A 12 h `interactive` Jupyter (b-9-21) is starting for the handover.
  - `h_final800_haug` (23779) had been "CANCELLED by 0" at 19:00 UTC before starting; resubmitted with the 6 Round 10
    jobs (24137-24143).
- **15:00 EST, A100 availability check** (the user asked to use other GPUs or CPU rather than wait):
  - `sbatch --test-only` estimates a new `c12m85-a100-1` job would start at 21:41 UTC (our queued jobs are ahead of
    that).
  - `c24m170-a100-2`, `a100-4-std` and `g2-standard-24` give "Unspecified error" (not allowed for our account).
  - `g2-standard-12` (1x L4) works with `--cpus-per-task=10 --mem=40GB` (start estimate 20:16 UTC).
  - **Insurance:** `h_fm600_pre_haug_l4` / `h_final600_pre_haug_l4` (24154/24155) on L4. Rule: for each run, the
    copy that starts training first is kept and the other cancelled, so we never choose between copies by score.
  - **CPU is not viable for training:** about 23 GFLOPs per image (4 WRN teachers dominate) × 45k images is about
    1 PFLOP per epoch, so 35-60 h for 600 epochs even on a 192-vCPU node. CPU (the Mac) is used for evaluation and
    prediction.
- **15:20 EST:**
  - The L4 insurance jobs (24154/24155) sat in CF for about 7 min, then went back to the queue (requeued): the L4
    pool still fails to boot, as on Nov 3.
  - Both `interactive` Jupyter sessions (24126 with the babysitter, 24147) ended after about 30 min, like every
    CPU node today, while the A100 jobs run for hours undisturbed.
  - Monitoring now uses the OOD Active Jobs page and the Files API. Resubmission needs a fresh Jupyter (5-15 min to
    start).
  - Our 7 A100 jobs are queued (Priority). The test-only start estimate for a new A100 job is 21:41 UTC.
  - Running: `h_fm800_haug`, `h_fm600_haug5`, `h_final600_haug5` (about 1h20m in; done about 17:00-18:00 EST).
- **15:40 EST: scale-aware TTA fails the pre-registered rule; dropped.**

  | Twin | fs+bal clean / suite | fs-scale+bal clean / suite | Δ clean | Δ suite |
  |---|---|---|---|---|
  | M (`h_fm600_haug`) | 97.96 / 96.88 | 98.00 / 97.01 | +0.04 | +0.13 |
  | L (`c_ft100_haug`) | 97.86 / 96.74 | 97.84 / 96.84 | −0.02 | +0.10 |

  - The rule needed clean ≥ +0.04 *and* suite ≥ +0.05 on both twins. L fails on clean, so fs+bal stays the decoding
    for every model (no per-model choice).
  - The suite gain is consistent (+0.1) but small; the hard suite does not move (M 93.59 vs 93.68).
  - The remaining fss evaluations (Run I, PreAct-600, Run G twins) were stopped to free the Mac.
- **16:45 EST, Round 11** (plan in `~/.claude/plans/the-high-level-goal-rosy-star.md`; the user: "don't disrupt the queue,
  add to the pipeline, maybe start on Colab"):
  - **Push-back recorded:**
    - an M + PreAct ensemble is not a legal submission (9.6M params, single model only), and the 2 final picks are a
      hedge, not an ensemble;
    - "M with 4.8M params" is low value: capacity isn't binding (every 4.3-5.0M student ties under FM; the student
      matches its teacher).
  - **A. Model soup** (`scripts/soup.py`, new): a uniform average of M + 4 hard-aug fine-tunes of M (`s_tw_a-d` from
    M's twin, `s_fn_a-d` from M's final; p 0.3 / 0.5 / 6 factors / wide; seeds 1-4), with BN recalibrated on 50
    training batches.
    - Checks: soup(M, M) == M exactly.
    - A soup of models from *different* starts (M + L's twin) scores 54%: soups need a shared start, which is why
      M and PreAct can't be averaged.
  - **B. Hard-view distillation** (`--hard-teacher`, `--hard-teacher-weight 0.5`; `src/train.py` aac624a243205cef on
    the HPC):
    - on hard-augmented images, the KD target mixes the teachers' clean-view answer with M's hard-view answer;
    - `h_fm600_hvd` / `h_final600_hvd`;
    - with weight 0 the loss is bit-identical to the run without it (the robust model is built inside `fork_rng`).
  - **C. Run G + L's recipe:** `h_ftG100_haug_tw` / `h_ftG100_haug` (preact_v, `--init` Run G's twin / final, 100 ep,
    lr 0.02, hard aug, 4 teachers). Run G's weights reached the HPC via `curl` from the public GitHub repo; SHA-256
    matches.
  - **C′ on Colab L4** (`c_ftG100_1t_tw` / `c_ftG100_1t`): C with Run G's own single teacher (`teacher_wrn300` /
    `teacher50`).
    - Both packed on one L4: 63-69 s/epoch, so about 1.9 h, about 3.8 of the 3.99 units.
    - `follow` backs up `last.pt` every 10 min in case the units run out.
- **16:40 EST: all 7 Round 10 jobs had been "CANCELLED by 0"** (20:56-21:10 UTC: nodes allocated, 14 min of boot, then
  cancelled), the L4 copies too; with no babysitter alive, nothing resubmitted them.
  - **Fix: the babysitter now runs inside the GPU jobs.**
    - `train.sbatch` (3cbb5af35074590c) starts a background loop that runs one pass of `babysit.sbatch`
      (`BABYSIT_ONCE=1`, e465dcd4dd1f67aa) every 5 min.
    - One host at a time: `runs/babysit/host` holds its job ID, and a host silent for 10 min is taken over.
    - The attempt cap is now 20 (was 8). `h_final600_t1`'s count was set to 99 so the old dropped run stays dead.
  - 21:40 UTC: one pass submitted all 19 missing jobs in queue-file order (Round 10's 7, then Round 11's 12:
    24302-24320). The same host loop runs in Jupyter 24297 until a GPU job takes over.
- **17:15 EST: `h_fm600_haug5` (p = 0.5) twin: a tie with M, not an improvement.**

  | Twin (fs+bal) | Clean | Suite | Hard mean (fs) | lowres16 | zoom_out | bg | McNemar vs M | Jaccard with M |
  |---|---|---|---|---|---|---|---|---|
  | M `h_fm600_haug` (p 0.3) | 97.96 | 96.88 | 93.68 | 94.74 | 96.24 | 95.58 | – | – |
  | `h_fm600_haug5` (p 0.5) | 97.84 | 96.81 | 93.77 | 95.20 | 96.38 | 95.66 | 23 vs 27, p = 0.67 | 0.60 |

  - Train log: final val 97.76, flip 97.86 (M: 97.80 / 97.94).
  - A stronger p buys +0.1 on the hard suite and costs about 0.1 clean. It misses the clean bar (97.86) by 0.02, so
    under rule D it doesn't qualify. p = 0.3 is about right.
  - The final `h_final600_haug5` was predicted anyway (`submission_fs_bal.csv`: 69 labels changed by balancing,
    196-206 per class; differs from M on 71, L on 76, J on 82). A spare, not a pick.
- **18:38 EST: `h_fm800_haug` (M's recipe, 800 epochs) twin: no gain from longer training.**

  | Twin (fs+bal) | Clean | Suite | Hard mean (fs) | lowres16 | rot20 | McNemar vs M | Jaccard with M |
  |---|---|---|---|---|---|---|---|
  | M `h_fm600_haug` (600 ep) | 97.96 | 96.88 | 93.68 | 94.74 | – | – | – |
  | `h_fm800_haug` (800 ep) | 97.90 | 96.87 | 93.37 | 94.28 | 94.08 | 18 vs 30, p = 0.11 | 0.67 |

  - It passes the bars (clean ≥ 97.86, suite ≥ 96.78, no significant loss) but doesn't beat M, and it is M's family,
    so rule D gives it no pick. The extra 200 epochs cost 0.3 on the hard suite: 600 epochs is the right length.
  - Its final `h_final800_haug` (24302) and `h_fm600_pre_haug` (24303) were "CANCELLED by 0" at boot (23:10 UTC).
    Jupyter 24435 ran one babysitter pass at 23:27 UTC: resubmitted as 24441 / 24442 (attempts 5 and 3). They now sit
    behind the other 17 pending jobs (lower age priority); nothing else was touched.
- **18:31 EST: Colab C′ finished** (`c_ftG100_1t_tw` val 97.74 / flip 97.74, vs Run G's twin 97.40 / 97.58).
  Both runs fetched; the VM is stopped. Colab balance: 0.11 units (no more Colab runs without buying units).
- **18:54 EST: C′ twin `c_ftG100_1t_tw` (Run G + hard aug + G's own teacher, 100 ep fine-tune):** better than Run G,
  still below M.

  | Twin | Clean fs | Clean fs+bal | Suite fs+bal | Hard mean (fs) | McNemar vs M | Flip errors | Jaccard with M |
  |---|---|---|---|---|---|---|---|
  | M `h_fm600_haug` | 98.00 | 97.96 | 96.88 | 93.68 | – | 103 | – |
  | C′ `c_ftG100_1t_tw` | 97.80 | 97.78 | 96.65 | 92.88 | 24 vs 34, p = 0.24 | 113 | 0.60 |
  | Run G `r2_twin_S` | 97.76 | 97.78 | 96.48 | 87.65 | C′ vs G: 24 vs 22, p = 0.88 | 121 | 0.52 |

  - Fails rule D's bars (clean 97.78 < 97.86, suite 96.65 < 96.78), so it can't be Pick 1.
  - As the PreAct hedge it is more accurate than G (113 vs 121 held-out errors) but less decorrelated from M (0.60
    vs 0.52): hard aug pulls it toward M.
  - Final `c_ftG100_1t/submission_fs_bal.csv`: 4,980,392 params, 74 labels changed by balancing, 197-203 per class;
    differs from M on 82, L on 94, Run G on 69.
  - **19:12 EST, Run G's twin on the same suite:** C′ ties G on clean (97.78 fs+bal both; McNemar p = 0.88) and
    gains +0.17 on the corruption suite and **+5.2 on the hard suite** (lowres16 77.8 → 93.5, background 81.4 → 94.3,
    zoom_out 85.9 → 95.1). The same treatment took Run I to L (+0.4 public).
  - So C′ dominates Run G as the PreAct hedge on the twin. It replaces G-TTA as the Pick 2 candidate if the team agrees;
    its public score is only a sanity check (no choosing on gaps < 0.004).
- **19:05 EST HPC:** all 19 of our jobs pending (Priority) in a congested A100 partition (about 150 jobs); none has
  started since 18:02. Jupyter 24435 has stopped responding, so until a GPU job starts there is no babysitter;
  checked by hand every 30-45 min.
- **Nov 6, 00:35 EST.** The user bought 100 Colab units (balance 100.11).
  - **HPC came back:** a batch of A100 nodes freed at about 22:58 EST. The GPU-hosted babysitter worked: at 03:58-04:13 UTC it resubmitted the 8 jobs cancelled at boot.
    - Done: `s_tw_a/b/c`, `s_fn_a/b/c` (soup ingredients; e.g. `s_tw_a` final val 97.46, flip 97.72) and `h_ftG100_haug` (C, full data).
    - Running: `h_fm600_pre_haug`, `h_final600_pre_haug`, `h_final600_haug6`, `h_final800_haug`, `h_fm600_haugw`.
    - Booting: `s_tw_d`, `s_fn_d`, `h_ftG100_haug_tw`, `h_fm600_haug6`, `h_final600_haugw`, `h_fm600_hvd`, `h_final600_hvd`.
  - **Colab A100 as insurance** for the jobs the HPC had not started (same flags as `runs/jobs.txt`; local script `colab_r11.sh`):
    - `c_s_tw_d`, `c_s_fn_d`, `c_fm600_hvd`, `c_final600_hvd`, all four on one A100 at about 38 s/epoch each with two running;
    - the full-data teachers T50 were downloaded from the HPC first.
  - **Partial-soup sanity check:** M + `s_tw_a/b/c`, BN recalibrated, scores flip 97.76 / suite 96.46, vs M 97.94 / 96.58.
    - This is a check only, not a selection: the composition stays the pre-registered M + a-d.
    - The ingredients drift below M during the lr-0.02 restart, and averaging doesn't recover it, so the soup is unlikely to beat M.
- **02:25 EST: twin soup `soup_twin`** (uniform average of M + `s_tw_a/b/c` + `c_s_tw_d`, BN recalibrated on 50
  training batches): **no gain.**

  | Twin (fs+bal) | Clean | Suite | Hard mean (fs) | McNemar vs M | Jaccard with M |
  |---|---|---|---|---|---|
  | M | 97.96 | 96.88 | 93.68 | – | – |
  | `soup_twin` | 97.88 | 96.83 | 93.40 | 12 vs 21, p = 0.16 | 0.74 |

  - It passes the bars but doesn't beat M, and it is M's family with the most correlated errors so far (0.74). Rule D gives it no pick.
  - Ingredient `c_s_tw_d` (wide levels): final val 97.74, flip 97.84.
  - The final soup is still built (pre-registered) once `c_s_fn_d` lands, as a spare.

## Round 12 (Nov 6, 02:00 EST): review of an outside "reach 92%" analysis
- Plan: `~/.claude/plans/the-high-level-goal-rosy-star.md`. User constraints: Colab only (no HPC changes), new
  runs after the current Colab work, in priority order.
- **Kept:** factor-weighted hard aug (#3), with weights from M's twin hard-suite excess error over clean:
  - lowres 0.35, zoom_in 0.15, zoom_out 0.20, background 0.30;
  - not the proposal's 35/10/30/25, which starves zoom_in, still at 3.6% error.
- **Dropped, with reasons:**
  - scale TTA: already failed its pre-registered rule;
  - exact-quota assignment: moves only about 5-10 labels, and relies on an external-dataset guess; only Sinkhorn is approved;
  - severity-aware soft targets: subsumed by hard-view distillation, which is running;
  - clean/hard consistency: redundant with clean-view KD;
  - hard-trained teachers: too costly;
  - the 800-epoch and p 0.5 runs: done, no gain;
  - PreAct + M's recipe: agreed, already running on the HPC.
- **Corrected one claim:** M's rot20 (94.22) is its worst single-factor row, so rotation is *not* "handled well".
- **Code checks:**
  - KD is KL × T² (`distillation_loss`);
  - TTA averages probabilities;
  - final class counts are 196-206.
- **`--hard-factor-weights`** (`src/train.py`) → `hard_batch(..., weights)` (`src/hard.py`):
  - With `weights=None`, the output is bit-identical to the old code (6 seeded batches, both factor sets).
  - 200k draws match the weights within 0.0003.
  - The 1-epoch MPS smoke test records the weights in `config.json`; a weight-count mismatch is rejected.
- **Colab queue** (watcher `colab_r12_watch.sh`):
  - `c_fm600_hfw` starts when `c_fm600_hvd` and `c_final600_hvd` are fetched;
  - `c_final600_hfw` follows only if the twin qualifies under rule D and its hard mean is ≥ 93.68.
  - If the HPC's own hvd pair is confirmed training first, the Colab duplicates are stopped (user's choice). SSH to
    the VM works for that: `ssh -o ProxyCommand="colab ssh --proxy-mode -s dl1" root@colab-dl1`.
