# SmallGBM

<p align="center">
  <b>Gradient boosting for small tabular data.</b><br>
  <sub>Statistical parity with XGBoost · Outperforms LightGBM · Lowest variance</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.3.0-blue" alt="version">
  <img src="https://img.shields.io/badge/python-3.8+-green" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="license">
  <img src="https://img.shields.io/badge/pip%20install-smallgbm-orange" alt="pip">
  <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21934674-blue" alt="DOI">
</p>

---

## What is SmallGBM?

SmallGBM is a gradient boosting library designed for **small datasets** (n < 1000). It combines robust leaf weight estimation with controlled column subsampling to deliver accuracy on par with XGBoost — with lower variance and no hyperparameter tuning.

---

## Benchmark

**27 datasets (15 synthetic + 12 real-world) · 5-fold cross-validation · mean ROC-AUC**

| Model        | AUC    | Std     |
| ------------ | ------ | ------- |
| **SmallGBM** | **0.9157** | **±0.0723** |
| XGBoost      | 0.9156 | ±0.0792 |
| RandomForest | 0.9140 | ±0.0788 |
| LightGBM     | 0.9047 | ±0.0752 |

> SmallGBM achieves **statistical parity with XGBoost** (p > 0.05), outperforms RandomForest by +0.17%, LightGBM by +1.1%, and has the **lowest variance** among all models.

---

## Why Robust Leaf Weights?

Standard gradient boosting uses the **mean** of residuals per leaf. On small data, one outlier can destroy the estimate.

SmallGBM uses:

- **Median** for leaves with n ≤ 30
- **Inverse-distance weighted mean** for larger leaves
- **Signal-adaptive shrinkage** toward the parent node

This makes predictions robust to outliers and label noise — the main enemies of small-sample learning.

---

## Installation

```bash
pip install smallgbm
```

## Quickstart

```python
from smallgbm import SmallGBMClassifier

model = SmallGBMClassifier()
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)
```

---

## Parameters

| Parameter           | Default | Description                  |
| ------------------- | ------- | ---------------------------- |
| `n_estimators`      | 50      | Boosting rounds              |
| `max_depth`         | 3       | Max tree depth               |
| `min_samples_leaf`  | 3       | Min samples per leaf         |
| `learning_rate`     | 0.1     | Shrinkage                    |
| `sigma_prior`       | 0.5     | Regularization strength      |
| `colsample_bytree`  | 0.5     | Feature fraction per tree    |
| `random_state`      | None    | Reproducibility              |
| `auto_scale`        | False   | RobustScaler internally      |

---

## Features

- **Robust leaf weights** — median + adaptive shrinkage
- **Column subsampling** — fights overfitting in high-dimensional small data
- **Uncertainty estimates** — `predict_with_uncertainty()`
- **scikit-learn compatible** — `fit`, `predict`, `predict_proba`
- **Pure Python + NumPy** — no compilation, easy install

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

---

<p align="center">
  <sub>Built for researchers and engineers working with limited data.</sub>
</p>
