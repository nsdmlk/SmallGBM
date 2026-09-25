# SmallGBM

<p align="center">
  <b>Gradient boosting for small tabular data.</b><br>
  <sub>C backend · Outperforms XGBoost · Beats LightGBM · 2–3× faster training</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.5.0-blue" alt="version">
  <img src="https://img.shields.io/badge/python-3.8+-green" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="license">
  <img src="https://img.shields.io/badge/pip%20install-smallgbm-orange" alt="pip">
  <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21934674-blue" alt="DOI">
</p>

---

## What is SmallGBM?

SmallGBM is a gradient boosting library designed for **small datasets** (n < 1000). It combines robust leaf weight estimation with stochastic split selection to outperform XGBoost and LightGBM — with lower variance, no hyperparameter tuning, and a native C core.

**New in 1.5.0:** the decision tree is now implemented in C (histogram-based split search, 256 quantile bins) and called from Python via `ctypes`. Same algorithm, **2–3× faster training** and **~2× faster inference** compared to the pure-Python 1.4.x line.

---

## Benchmark

**22 datasets (15 synthetic + 7 real-world) · 5-fold cross-validation · mean ROC-AUC · same default hyperparameters for all models**

| Model              | AUC              | Fit (ms)       | Predict (ms)   |
| ------------------ | ---------------- | -------------- | -------------- |
| **SmallGBM** | **0.9101** | **13.4** | **0.29** |
| XGBoost            | 0.9036           | 36.5           | 0.46           |
| RandomForest       | 0.9007           | 28.8           | 1.57           |
| LightGBM           | 0.8958           | 34.3           | 0.57           |

> SmallGBM **outperforms XGBoost by +0.65%**, RandomForest by +0.94%, LightGBM by +1.43% — while training **2.7× faster** and predicting **1.6–5× faster**.

---

## Why Robust Leaf Weights?

Standard gradient boosting uses the **mean** of residuals per leaf. On small data, one outlier can destroy the estimate.

SmallGBM uses:

- **Median** for leaves with n ≤ 30
- **Inverse-distance weighted mean** for larger leaves
- **Signal-adaptive shrinkage** toward the parent node

This makes predictions robust to outliers and label noise — the main enemies of small-sample learning.

---

## Why Stochastic Split Selection?

Full enumeration of all possible split thresholds overfits on small data. SmallGBM uses **5 random thresholds per feature** (via the histogram) — less overfitting, faster training, and better generalization.

---

## Architecture

smallgbm/
├── smallgbm.py       # boosting logic (classifier + regressor)
├── tree.py           # ctypes wrapper around libtree
└── _c/
    ├── tree.c        # histogram-based decision tree in C
    └── tree.h


The C library is compiled automatically on `pip install`. On macOS it produces `libtree.dylib`, on Linux `libtree.so`, on Windows `tree.dll`. No manual compilation needed.

---

## Installation

```bash
pip install smallgbm
```

Requires a C compiler (`cc`, `clang`, or `gcc`) available on `PATH`. On macOS install Xcode Command Line Tools (`xcode-select --install`).

---

## Quickstart

```python
from smallgbm import SmallGBMClassifier

model = SmallGBMClassifier()
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)
```

---

## Parameters

| Parameter            | Default | Description               |
| -------------------- | ------- | ------------------------- |
| `n_estimators`     | 50      | Boosting rounds           |
| `max_depth`        | 3       | Max tree depth            |
| `min_samples_leaf` | 3       | Min samples per leaf      |
| `learning_rate`    | 0.1     | Shrinkage                 |
| `sigma_prior`      | 0.5     | Regularization strength   |
| `colsample_bytree` | 0.5     | Feature fraction per tree |
| `random_state`     | None    | Reproducibility           |
| `auto_scale`       | False   | RobustScaler internally   |

---

## Features

- **C core** — histogram-based tree with 256 quantile bins per feature
- **Robust leaf weights** — median + adaptive shrinkage
- **Stochastic split selection** — 5 random thresholds per feature
- **Column subsampling** — fights overfitting in high-dimensional small data
- **Uncertainty estimates** — `predict_with_uncertainty()`
- **scikit-learn compatible** — `fit`, `predict`, `predict_proba`
- **Auto-compiled on install** — no separate build step

---

## Reproducing the benchmark

```bash
# from the repository root
clang -O3 -march=native -flto -shared -fPIC smallgbm/_c/tree.c \
      -o smallgbm/_c/libtree.dylib -lm
python -m Tests.test
```

---

## Citation

```bibtex
@software{emelyanov2026smallgbm,
  author = {Emelyanov, Ilya},
  title = {SmallGBM: Gradient Boosting with Robust Leaf Regularization for Small-Sample Tabular Data},
  year = {2026},
  doi = {10.5281/zenodo.21934674},
  url = {https://github.com/nsdmlk/SmallGBM}
}
```

---

## License

MIT © [Emelyanov Ilya](https://github.com/nsdmlk), 2026
