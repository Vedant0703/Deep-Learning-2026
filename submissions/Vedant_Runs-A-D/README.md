# Vedant: Runs A to D

Vedant's first four Kaggle submissions (Nov 2-3), trained with the notebooks in
[`../../notebooks/vedant/`](../../notebooks/vedant/). Run A's 4.98M `preact_v` architecture is the one Abhinav's
Runs G and N also use.

| Run | Recipe | Public | Private | Notebook |
|---|---|---|---|---|
| A | 4.98M PreAct wide ResNet (`preact_v`), 300 epochs, EMA, flip TTA, 45k training images (held-out val 97.08%) | 0.875 | 0.865 | `run_ab_train.ipynb` |
| B | Run A's recipe on all 50k images | 0.892 | 0.873 | `run_ab_train.ipynb` |
| C | Run B plus robust augmentation (blur, JPEG, noise), 400 epochs, all 50k | 0.881 | 0.868 | `run_c_robust.ipynb` |
| D | one new model distilled from the averaged predictions of Runs A, B and C, 300 epochs, all 50k | 0.895 | 0.872 | `run_d_distill.ipynb` |

`submission_1.csv` to `submission_3.csv` were uploaded on Nov 2, after Runs A to C were submitted, and
`submission_4.csv` on Nov 3 with the Run D notebook. Which file belongs to which run was not recorded. The model
weights are not in the repository.
