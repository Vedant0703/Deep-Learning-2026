# Vedant: Runs A to D

Vedant's first four Kaggle submissions (Nov 2-3), trained with the notebooks in
[`../../notebooks/vedant/`](../../notebooks/vedant/), which keep their execution output. Run A's 4.98M `preact_v`
architecture is the one Abhinav's Runs G and N also use.

| Run | Recipe | Public | Private | Notebook | Submitted CSV |
|---|---|---|---|---|---|
| A | 4.98M PreAct wide ResNet (`preact_v`), 300 epochs, EMA, flip TTA, 45k training images (held-out val 97.08%) | 0.875 | 0.865 | `run_ab_train.ipynb` | `run_A.csv` |
| B | Run A's recipe on all 50k images | 0.892 | 0.873 | `run_ab_train.ipynb` | `run_B.csv` |
| C | Run B plus robust augmentation (blur, JPEG, noise), 400 epochs, all 50k | 0.881 | 0.868 | `run_c_robust.ipynb` | `run_C.csv` |
| D | one new model distilled from the averaged predictions of Runs A, B and C, 300 epochs, all 50k | 0.895 | 0.872 | `run_d_distill.ipynb` | `run_D.csv` |

Each CSV is matched to its run by the predicted class counts that the notebooks print for that run (all ten counts
agree exactly). The model weights are not in the repository.
