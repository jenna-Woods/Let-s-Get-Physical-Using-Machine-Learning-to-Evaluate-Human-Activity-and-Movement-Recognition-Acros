# Human activity recognition across individuals

Implements the CS 334 proposal with multinomial logistic regression, random forest,
RBF SVM, and a majority-class baseline on UCI HAR's 561 engineered features.
Random forest is used literally; XGBoost is a different algorithm and is not included.

## Setup (PowerShell, Python 3.11+)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m har.cli download
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

If your Python cannot bootstrap pip, use your existing pip:
`python -m pip --python .venv/Scripts/python.exe install -e .`.
All subsequent commands can use `.\.venv\Scripts\har.exe` (shown below as `har`).

`requirements-lock.txt` records the versions installed for verification. To recreate
that environment, install it with `pip install -r requirements-lock.txt` before
`pip install -e .`.

## Execution

```powershell
har audit --output results/audit
har cv --config configs/smoke.json --output results/smoke
har plot --results results/smoke
har cv --config configs/full.json --output results/full
har plot --results results/full
har freeze --config configs/full.json --output results/frozen
har final --frozen results/frozen --output results/final
har plot --results results/final
```

Each run requires a fresh output directory to protect previous results. Use
`--data "path/to/UCI HAR Dataset"` if you already downloaded the original archive.
Smoke mode uses all training rows with two folds, fewer feature counts, single
hyperparameter choices, and 50 forest trees. Its scores are verification results,
not the full proposal's findings. Full mode is substantially more expensive:
five outer folds, three inner folds, five feature counts, and all tuning choices.
It uses two search processes with one computational thread each by default.
Completed candidates are saved as the run proceeds; interrupted runs are not resumed.

## Evaluation contract

- `cv` loads only the official training partition (7,352 windows / 21 people).
  `freeze` also uses only that partition. The official 2,947-window / 9-person test
  is read by `final`, after a manifest and fitted models exist.
- Grouped validation uses StratifiedGroupKFold in both outer and inner loops;
  participant overlap and class coverage are checked explicitly. Random-window
  evaluation uses stratified folds at both levels. Every model shares the same
  folds within a protocol. Row indices, participant IDs, and inner splits are saved.
- SelectKBest with ANOVA F-scores ranks features inside each fit. Logistic
  regression and SVM standardize selected features inside the pipeline. Forest
  does not scale. Original feature IDs preserve duplicate feature names.
- For each fixed feature count, hyperparameters are selected with inner macro-F1
  and evaluated on outer folds. `best` selects feature count and hyperparameters
  together using inner scores. `compact` selects the smallest feature count within
  0.01 absolute inner macro-F1 of the best. Ties are deterministic.
- `freeze` repeats grouped inner validation on all training people, then fits
  best and compact variants of every model on the complete training partition.
  `final` evaluates these saved pipelines without fitting or choosing a winner.
  Run final evaluation only once the experiment design is settled. Do not change
  configurations in response to final-test performance.
- Macro-F1 is computed over all six fixed activity labels. Reports include
  accuracy, per-class metrics, confusion matrices, and per-person results.
  Prediction timing includes selection and scaling, uses a warmup and five
  repetitions, and reports median batch time and microseconds per window.
  Batch timings are not single-window real-time latency. Search time is separate
  from final pipeline fitting time; feature extraction from raw signals is excluded.
- Random windows overlap and can share people. Their performance is an optimistic
  illustration, not a deployment estimate. Differences between protocols also
  reflect different partitions, so do not attribute the entire difference to identity.
  Fold standard deviations are descriptive, not confidence intervals. Participants,
  rather than individual overlapping windows, are the appropriate sampling units
  for any later uncertainty analysis.

## Outputs

`metrics.csv` contains one row per fold/model/choice. `summary.csv` contains fold
means and standard deviations. Candidate folders hold predictions, per-person
metrics, classification reports, confusion matrices, feature scores and masks,
hyperparameters, and serialized pipelines. Policy rows can point to the same
candidate; do not treat them as independent repetitions. The dummy uses every
input column but ignores feature values; its reported k=561 is bookkeeping only.

`plot` produces feature-count curves, protocol comparisons, confusion matrices,
per-person plots, feature-selection frequencies, and feature-family summaries.
Families use the prefix before the first hyphen in the original feature name;
frequency is descriptive and does not establish a causal sensor contribution.
Audits include subject/activity counts, feature distributions, and highly correlated
pairs. Correlations are inspected, not used to remove features outside validation.
Fingerprints and environment metadata make runs traceable. Download provenance
includes a SHA-256 of the archive. Keep `download.json` with your data.

Only load joblib models you created or trust. Code and configuration are versioned;
data, virtual environments, and generated results are excluded from git.

## Sources

- Proposal supplied by the project team (three classifiers, feature sizes 20/50/100/200/561).
- [UCI HAR dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html)
