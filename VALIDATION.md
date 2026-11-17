# Implementation verification

The implementation was verified against the original UCI HAR training partition.
The official test partition remains reserved; neither the smoke CV run nor the
model-freezing run evaluates it.

- Six automated tests passed: nested subject separation and row coverage,
  training-only transformations, model serialization consistency, all four
  classifier searches, compact feature selection, archive path protection, and
  an end-to-end CV/freeze/final workflow using synthetic data. Several checks
  share a test case.
- A separate two-process grouped search passed, checking the full configuration's
  parallel execution path.
- Real-data loading confirmed 7,352 training rows, 561 columns, 21 subjects, six
  classes, and 477 distinct feature names. Original IDs preserve repeated names.
- `configs/smoke.json` completed both grouped and random-window nested evaluation
  for all models on all training rows. The selector, scaling, fitting, scoring,
  timing, prediction export, and plotting paths were exercised.
- The protocol comparison and a grouped logistic-regression confusion matrix were
  visually inspected for readable labels and correct normalization.

## Smoke results (mean outer-fold macro-F1)

| Classifier | Grouped | Random windows |
|---|---:|---:|
| Majority baseline | 0.0535 | 0.0535 |
| Logistic regression | 0.9262 | 0.9814 |
| Random forest | 0.8919 | 0.9734 |
| RBF SVM | 0.9049 | 0.9732 |

These are reduced-search verification results, not final proposal findings.
Smoke mode uses two outer/inner folds, feature counts 20 and 561, fixed classifier
hyperparameters, and 50 forest trees. The best feature count is chosen using
inner-fold macro-F1 separately within each outer fold.

Local generated evidence lives under `results/smoke/` (including `metrics.csv`,
`summary.csv`, predictions, figures, models, and fold definitions). Training-only
smoke model freezing saves deployable pipelines under `results/smoke-frozen/`.
Results and downloaded data are intentionally excluded from git.

The full five-fold search, all five feature sizes, and expanded hyperparameter
grids are implemented in `configs/full.json` but have not been run. See README
for full-run commands and the separate final-test step.
