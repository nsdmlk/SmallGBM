# SmallGBM

<p align="center">
  <b>Gradient boosting with Bayesian leaf regularization for small data.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.1.0-blue" alt="version">
  <img src="https://img.shields.io/badge/python-3.8+-green" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="license">
  <img src="https://img.shields.io/badge/pip%20install-smallgbm-orange" alt="pip">
  <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21905013-blue" alt="DOI">
</p>

---

## Why SmallGBM?

XGBoost and LightGBM are built for scale. They shine on thousands of rows. But when you only have **50, 100, or 500 samples**, their default hyperparameters fail — overfitting, instability, unpredictable results.

**SmallGBM** is designed from the ground up for datasets with fewer than 1000 samples, using Bayesian leaf weight regularization to prevent overfitting on small leaves.

| Feature                      | SmallGBM | XGBoost | LightGBM |
| ---------------------------- | -------- | ------- | -------- |
| Bayesian leaf weights        | ✅       | ❌      | ❌       |
| Uncertainty estimates        | ✅       | ❌      | ❌       |
| No bootstrap (uses all data) | ✅       | ❌      | ❌       |
| Stable under label noise     | ✅       | ❌      | ❌       |
| scikit-learn compatible      | ✅       | ✅      | ✅       |

---

## Research

SmallGBM has been evaluated on 16 datasets (synthetic and real-world) with sample sizes from 20 to 1000. Key findings:

- **Noise stability**: At 20% label noise, SmallGBM outperforms XGBoost and LightGBM
- **Prior insensitivity**: Performance is nearly invariant to `sigma_prior` across three orders of magnitude
- **Predictable learning curve**: Reliable performance begins at n ≈ 40

<p align="center">
  <img src="docs/noise_comparison.png" width="600" alt="Noise stability comparison">
</p>

*At 20% label noise, SmallGBM is the best performer. Bayesian regularization keeps it stable when others collapse.*

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
| `sigma_prior`      | 0.5     | Bayesian prior strength   |
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

## Citation

```bibtex
@software{emelyanov2026smallgbm,
  author = {Emelyanov, Ilya},
  title = {SmallGBM: Gradient Boosting with Bayesian Leaf Regularization for Small-Sample Tabular Data},
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
  <sub>Built with ❤️ for the small data community</sub>
</p>
