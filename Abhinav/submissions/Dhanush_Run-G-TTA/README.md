# Dhanush: Run G with translation/flip TTA

Public Kaggle score: **0.90400**, confirmed by Dhanush's screenshot. This improves the original Run G flip-only public score of 0.90100 by 0.003 (0.30 percentage points). It is an inference experiment, not a newly trained model. Run M remains the team's best recorded public result at 0.91000.

Uses Abhinav's Run G EMA checkpoint, Vedant's preact_v architecture, 4,980,392 parameters. No retraining, class balancing, BN adaptation or model ensemble. Each image is evaluated at its original position and four cardinal +/-2-pixel shifts, with and without a horizontal flip (10 views, reflection padding). Softmax probabilities are averaged.

## Validation

The 45k twin, not the all-50k final model, evaluated the fixed 5k validation split:

| Condition | Flip | Translation + flip | Net correct |
|---|---:|---:|---:|
| Clean | 97.58% | 97.76% | +9 |
| Brightness 0.8 | 97.56% | 97.64% | +4 |
| Brightness 1.2 | 97.62% | 97.72% | +5 |
| Blur 0.6 | 96.82% | 96.72% | -5 |

Final-model flip predictions reproduced the original CSV exactly; translation TTA changed 31/2,000 test labels. This is modest development evidence, not a guarantee of higher private accuracy. The validation split was reused. No hidden labels were used.

`tta_report.json` records the HPC results, checkpoint hashes, and public score. `submission.csv` was reconstructed from the exact saved `test_predictions.npz` shift_prob argmax values because the original downloaded candidate CSV was no longer present locally; saved flip probabilities reproduce all original Run G labels and the reconstructed candidate has the recorded 31 differences. The user reports uploading that candidate under the filename submission.csv.

## Reproduce inference

From `Abhinav/`, with the original competition data extracted in DATA_DIR:

```bash
python -m src.predict --ckpt submissions/Abhinav_Run-G/model.pt --data-dir DATA_DIR --out reproduced_run_g_shift.csv --tta flip_shift
```

Do not add `--balance`. No teacher files are needed for inference. The Run G folder contains its training recipe and existing weights; this experiment does not duplicate checkpoints or include competition images.

To repeat the complete validation experiment on a GPU:

```bash
python experiments/run_g_tta/run_tta.py --data-dir DATA_DIR --out results/run_g_tta_new
```

The output directory must not exist. Alternatively open `experiments/run_g_tta/RunG_TTA_Comparison.ipynb` from that experiment directory and Run All. Set DATA_DIR to your location. The runner tests a fixed set of four TTA methods, applies the original validation screen, checks original submission reproduction, and never calls Kaggle.

Credit: Run G training/distillation pipeline by Abhinav; preact_v architecture by Vedant; TTA comparison and HPC/Kaggle evaluation by Dhanush, with Codex assistance. No GPU execution has been performed on this local Mac. Published scripts were syntax checked; the pre-publication HPC run produced the attached report.
