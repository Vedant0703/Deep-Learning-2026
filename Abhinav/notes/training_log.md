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
