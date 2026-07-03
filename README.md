# COO-tuned ANN for Sikkim mandarin yield prediction

Reproducible analysis for the manuscript _"A canine olfactory optimization–tuned
artificial neural network for predicting fruit yield of Sikkim mandarin
(Citrus reticulata Blanco) from fruit-set and fruit-quality traits."_

## What this does

For 30 treatments of Sikkim mandarin, the code relates fruit **yield (kg)** to
eight predictors — post-bloom fruit set (PBFS), total soluble solids (TSS), pH,
titratable acidity, fruit weight (FW), fruit diameter (FD), number of seeds (NS)
and pre-harvest fruit set (PHFS) — and:

1. computes descriptive statistics and a Pearson correlation matrix (with
   significance stars);
2. tunes an **artificial neural network (ANN)** with the **Canine Olfactory
   Optimization (COO)** metaheuristic (Garai et al. 2026), optimising the hidden
   architecture, L2 penalty and activation function;
3. benchmarks the COO-ANN against **multiple linear regression (MLR)** on an
   independent **train / validation / test** split and by **leave-one-out
   cross-validation (LOOCV)**;
4. ranks predictor importance by **permutation importance** on the best (COO) ANN.

COO's objective is the 5-fold cross-validated Q² **within the training set only**
— the validation and test sets are never seen during optimisation.

## Key results

COO selected a compact network: **1 hidden layer, 7 neurons, ReLU, L2 ≈ 26**.

| Model     | Subset     |  R² / Q²  | RMSE (kg) |
| --------- | ---------- | :-------: | :-------: |
| ANN (COO) | Training   |   0.573   |   2.48    |
| ANN (COO) | Validation |   0.659   |   2.27    |
| ANN (COO) | Test       |   0.539   |   2.52    |
| ANN (COO) | LOOCV      | **0.510** |   2.72    |
| MLR       | Test       |   0.438   |   2.78    |
| MLR       | LOOCV      |   0.388   |   3.04    |

The COO-tuned ANN **beat MLR on every held-out estimate**; the LOOCV Q² rose from
0.388 (MLR) to 0.510 (COO-ANN). Fruit **diameter** and **weight** were the
dominant predictors (~49% of total permutation importance).

## Repository layout

```
data/       data.csv        # input (30 x 9)
analysis.py                                      # main analysis (# %% cells)
figures/    Fig1..Fig3 (.png 600 dpi, .tiff)     # publication figures
outputs/    table1..table3 (.csv), table_permutation_importance.csv, coo_selected_model.csv
requirements.txt
```

## Run it

```bash
pip install -r requirements.txt        # installs coo-algorithm from PyPI
python analysis.py                     # regenerates tables + figures (COO ~40 s)
```

`analysis.py` is organised as `# %%` cells for Jupyter / VS Code. A fixed seed
(42) makes COO and the models reproducible. The COO optimiser is imported from
the PyPI package `coo-algorithm`;

## Notes / caveats

- n = 30 treatment-level means from a single dataset. The validation and test
  subsets contain only 5 observations each, so those subset R² values are
  high-variance; the **LOOCV Q²** (uses all data) is the most reliable summary
  and also favours the COO-ANN.
- Treat the models as decision-support / screening tools, not deployment-grade
  predictors, until validated on larger multi-season data.
