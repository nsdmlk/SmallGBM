# SmallGBM

<p align="center">
  <b>Gradient boosting that beats LightGBM and matches XGBoost on small data.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.3.0-blue" alt="version">
  <img src="https://img.shields.io/badge/python-3.8+-green" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="license">
  <img src="https://img.shields.io/badge/pip%20install-smallgbm-orange" alt="pip">
  <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21905013-blue" alt="DOI">
</p>

---

## What is SmallGBM?

SmallGBM is a gradient boosting library designed for **small datasets** (n < 1000). It combines robust leaf weight estimation with adaptive column sampling to deliver accuracy on par with XGBoost — while remaining simple, fast, and stable.

---

## Benchmark

**30 datasets, 5-fold cross-validation, mean ROC-AUC:**

| Model        | AUC    | Std    |
| ------------ | ------ | ------ |
| XGBoost      | 0.9089 | ±0.0755 |
| **SmallGBM** | **0.9085** | **±0.0697** |
| RandomForest | 0.9055 | ±0.0768 |
| LightGBM     | 0.8995 | ±0.0709 |

*SmallGBM is statistically indistinguishable from XGBoost, beats RandomForest by +0.3%, and LightGBM by +0.9%. SmallGBM also has the lowest variance across all models.*

---

## Why Robust Leaf Weights?

Standard gradient boosting uses the **mean** of residuals in each leaf. On small data, a single outlier can ruin the mean.

SmallGBM uses:
- **Median** for leaves with n ≤ 30
- **Weighted mean** with inverse-distance weights for larger leaves
- **Signal-adaptive shrinkage** toward the parent node value

This makes predictions robust to outliers and noise — the main enemies of small-sample learning.

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

## API

### SmallGBMClassifier

| Parameter            | Default | Description               |
| -------------------- | ------- | ------------------------- |
| `n_estimators`     | 50      | Number of boosting rounds |
| `max_depth`        | 3       | Maximum tree depth        |
| `min_samples_leaf` | 3       | Minimum samples per leaf  |
| `learning_rate`    | 0.1     | Shrinkage factor          |
| `sigma_prior`      | 0.5     | Regularization strength   |
| `colsample_bytree` | 0.5     | Fraction of features per tree |
| `random_state`     | None    | Random seed for reproducibility |
| `auto_scale`       | False   | Apply RobustScaler internally |

### SmallGBMRegressor

Same parameters, for regression tasks.

```python
from smallgbm import SmallGBMRegressor

model = SmallGBMRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

---

## Key Features

- **Robust leaf weights** — median + adaptive shrinkage
- **Column subsampling** — fights overfitting in high-dimensional small data
- **Uncertainty estimates** — `predict_with_uncertainty()` available
- **scikit-learn compatible** — `fit`, `predict`, `predict_proba`
- **Pure Python + NumPy** — no compilation, easy to install

---

## Citation

```bibtex
@software{emelyanov2026smallgbm,
  author = {Emelyanov, Ilya},
  title = {SmallGBM: Gradient Boosting with Robust Leaf Regularization for Small-Sample Tabular Data},
  year = {2026},
  doi = {10.5281/zenodo.21905013},
  url = {https://github.com/nsdmlk/SmallGBM}
}
```

---

## License

MIT © [Emelyanov Ilya](https://github.com/nsdmlk) 2026

---

<p align="center">
  <sub>Built for researchers and engineers working with limited data.</sub>
</p>
