# Time-aware fusion of dense spectra and sparse assays in industrial fermentation

This is a small study on the IndPenSim benchmark: 100 simulated industrial-scale (100,000 L) fed-batch fermentations of the filamentous fungus *Penicillium chrysogenum*.

In each batch:
- Raman spectra and process signals are recorded every 12 minutes.
- Offline assays (biomass, penicillin, viscosity, PAA, NH3) arrive only about every 12 hours.

The question is how to combine a dense, high-dimensional stream with sparse lab measurements over a long run. Three things are tested:
- whether doing so by the real time of each measurement helps;
- whether the model's uncertainty can be trusted;
- whether the model's uncertainty responds when a new assay arrives.

## Data

The data are not included here. Download `100_Batches_IndPenSim.zip` (504 MB) from Mendeley Data, [doi:10.17632/pdnjz7zz5x.1](http://dx.doi.org/10.17632/pdnjz7zz5x.1) (Goldrick et al., 2019). Its SHA-256 is `bc434c6cf96444fee3c915d434ac84e2a9c6492be4daebcfa4aed9cdff181f00`.

Unzip it into `data_raw/`.

The header of `100_Batches_IndPenSim_V3.csv` is misaligned: the multi-line flag columns add extra names. Columns are therefore read by position in `code/01_convert.py`:
- positions 0–30 are process and offline variables;
- 31–36 are the fault, control, PAT and batch flags;
- 37–2236 are the 2,200 Raman channels;
- the last two columns are empty.

The simulator's own noise-free penicillin concentration is excluded from the inputs, since a plant would not measure it online.

## Running

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/01_convert.py data_raw/Mendeley_data/100_Batches_IndPenSim_V3.csv data
.venv/bin/python code/02_explore.py data data_raw/Mendeley_data/100_Batches_IndPenSim_Statistics.csv
.venv/bin/python code/03_prep.py data data_raw/Mendeley_data/100_Batches_IndPenSim_Statistics.csv data/hourly.npy
.venv/bin/python code/04_baselines.py data/hourly.npy results/baselines_yield.csv
.venv/bin/python code/05_fixed_target.py data/hourly.npy
# main experiment (about 25 minutes on a 4-core laptop CPU)
.venv/bin/python code/06_attention.py data/hourly.npy results/E5 --folds 10 --reps 3 --epochs 150 \
    --d 16 --dropout 0.2 --wd 1e-2 --lam 0.05 --anchor 1 --val_frac 0.15
.venv/bin/python code/07_uncertainty.py data/hourly.npy results/E5/traces.csv
.venv/bin/python code/08_paired.py results/E5/predictions.csv
```

Running `06_attention.py` with its default arguments reproduces the first version of the model (`results/E3_attention_v1`).

## Set-up

**Target.** The target is the offline penicillin concentration at 156 h, which every batch reaches. It is predicted from the data available up to horizon *k* = 24, 48, 72, 96, 120 and 144 h.

**Validation.** Batches are split by 10-fold cross-validation, stratified by control strategy and repeated three times with different splits. No batch appears in both training and test folds, and all scaling and PCA are fitted on the training folds only.

**Models.** Four models are compared:
- **time**: a time-aware cross-attention model. Hourly tokens combine process variables and 16 Raman principal components, and each assay is a separate token at its real time. Both carry learned continuous-time embeddings. Attention is bidirectional between the dense and assay tokens, and a query is placed at horizon *k*. A Normal-Inverse-Gamma (evidential) head gives the prediction and its uncertainty. The prediction is a residual over the last observed assay. An inner 15% validation split is used for early stopping and for scaling the predicted standard deviation.
- **grid**: the same network, with assays forward-filled onto the hourly grid instead of entering as time-stamped tokens.
- **dense**: the same network without assays.
- **ridge**: ridge regression on per-variable means, 6-hour means and slopes up to *k*, the last assay values and Raman PCA.

## Results (three repeats, mean ± sd)

| *k* (h) | time MAE % | grid MAE % | ridge MAE % | time R² | ridge R² | time 80% / 95% coverage |
|---|---|---|---|---|---|---|
| 24 | 11.2 ± 0.4 | 11.4 ± 0.2 | 11.4 ± 0.2 | 0.03 ± 0.17 | 0.17 ± 0.01 | 0.86 / 0.91 |
| 48 | 11.1 ± 0.5 | 11.6 ± 0.3 | 9.2 ± 0.2 | 0.07 ± 0.08 | 0.37 ± 0.04 | 0.81 / 0.89 |
| 72 | 10.0 ± 0.6 | 10.6 ± 0.7 | 10.4 ± 1.7 | 0.18 ± 0.11 | −0.14 ± 0.89 | 0.72 / 0.85 |
| 96 | 8.5 ± 0.6 | 9.3 ± 0.4 | 6.8 ± 0.2 | 0.43 ± 0.09 | 0.68 ± 0.02 | 0.68 / 0.79 |
| 120 | 5.4 ± 0.3 | 5.8 ± 0.5 | 5.1 ± 0.1 | 0.80 ± 0.03 | 0.77 ± 0.03 | 0.68 / 0.81 |
| 144 | 3.1 ± 0.2 | 3.7 ± 0.4 | 1.6 ± 0.1 | 0.95 ± 0.01 | 0.99 ± 0.00 | 0.82 / 0.93 |

MAE is given as a percentage of the mean target (22.4 g/L). Full tables are in `results/E5_attention_v2_3repeats/`.

**Target choice.** Final *total* yield could not be predicted from the first 24–120 h by any model (R² ≈ 0; `results/baselines_yield.csv`), because it depends on when each batch was stopped (167–290 h). A fixed-time target is predictable (`results/fixed_target_P156.txt`).

**Paired comparisons.** Per-batch absolute errors, averaged over repeats, were compared with Wilcoxon signed-rank tests, Holm-corrected over 18 tests (`paired.txt`):
- The time-aware model was better than the grid version (p = 0.011) and the dense version (p = 0.0002) only at 144 h. At 24–120 h the differences were not significant.
- Ridge was better than the time-aware model at 48 h and 144 h.

**Uncertainty when an assay arrives.** This was measured over 36,000 hourly prediction updates (`uncertainty_at_assay.txt`):
- The predicted standard deviation fell by 4.74% on average in hours when a new assay entered the model, against 0.66% in other hours.
- Absolute error fell by 0.26 g/L in assay hours and was essentially unchanged otherwise (+0.01 g/L).

**Calibration.** The first version (`E3_attention_v1`: larger, no early stopping, no interval scaling) was badly over-confident, with only about 11% of targets inside its 80% intervals. After regularisation, early stopping and scaling on the inner validation split, coverage was 0.68–0.86 (80%) and 0.79–0.93 (95%). It was still too narrow at 96–120 h.

## Limitations

- With 100 batches of mostly scalar measurements, a tuned linear model is hard to beat. Any gain from the time-aware design is modest and, per batch, significant only at the latest horizon.
- IndPenSim is a simulation of a fed-batch process. It says nothing about the biology of other organisms or about continuous culture.
- The fault labels mark whole batches, without onset times, so early fault detection was not evaluated here.

## Reference

Goldrick, S., Duran-Villalobos, C.A., Jankauskas, K., Lovett, D., Farid, S.S. and Lennox, B. (2019) 'Modern day monitoring and control challenges outlined on an industrial-scale benchmark fermentation process', *Computers & Chemical Engineering*, 130, 106471.
