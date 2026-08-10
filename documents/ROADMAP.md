# ROADMAP: SmallGBM

Gradient boosting optimized for small datasets (n < 1000).
Goal: outperform XGBoost, LightGBM, and Random Forest on tabular data with limited samples.

---

## 1. Creation

Implement SmallGBM core from scratch.

- [ ] Derive gradient boosting from first principles (initial prediction, pseudo-residuals, tree fitting, update rule)
- [ ] Implement decision tree with strict constraints:
  - max_depth = 2–3
  - min_samples_leaf = 3–5
  - min_samples_split = 8–10
- [ ] Implement custom loss functions:
  - Binary cross-entropy for classification
  - MSE for regression
- [ ] Add aggressive regularization:
  - High learning rate shrinkage (eta)
  - Strong L2 regularization on leaf weights
  - No bootstrapping (use full data per iteration)
  - No column subsampling (use all features)
- [ ] Build iterative training loop
- [ ] Add early stopping with small patience (5–10 rounds)
- [ ] Write clean, readable Python code with NumPy only

---

## 2. Evaluation (Loop)

Run benchmarks against baselines, iterate until SmallGBM wins.

**Datasets:**

- [ ] Collect 20–30 small datasets from OpenML / UCI (n between 100 and 999)
- [ ] Mix of binary classification and regression tasks
- [ ] Preprocess consistently (no data leakage)

**Baselines:**

- [ ] XGBoost (default params)
- [ ] LightGBM (default params)
- [ ] Random Forest (default params)
- [ ] Logistic Regression / Ridge

**Metrics:**

- [ ] ROC-AUC (classification)
- [ ] RMSE / MAE (regression)
- [ ] Time to fit (CPU wall time)

**Protocol:**

- [ ] 5-fold cross-validation
- [ ] Mean + std across folds
- [ ] Hold-out test set for final comparison

**Iterate:**

- [ ] Tune regularization params until SmallGBM outperforms baselines on average
- [ ] Document every experiment with configs and results
- [ ] Stop when SmallGBM is statistically better (paired t-test or rank-sum)

---

## 3. API + Packaging

Make it pip-installable with scikit-learn-compatible API.

**API Design:**

- [ ] `SmallGBMClassifier` with `fit(X, y)`, `predict(X)`, `predict_proba(X)`
- [ ] `SmallGBMRegressor` with `fit(X, y)`, `predict(X)`
- [ ] Constructor params: `n_estimators`, `max_depth`, `min_samples_leaf`, `learning_rate`, `l2_reg`
- [ ] Sensible defaults tuned for n < 1000
- [ ] `get_params()` / `set_params()` for scikit-learn compatibility

**Packaging:**

- [ ] `setup.py` / `pyproject.toml`
- [ ] Upload to PyPI as `smallgbm`
- [ ] Test with `pip install smallgbm` in fresh virtual environment
- [ ] Add basic unit tests with pytest

---

## 4. Landing + Documentation

Static website with beautiful design and tutorial.

**Content:**

- [ ] Problem statement: why standard GBM fails on small data
- [ ] How SmallGBM works (key ideas, regularization)
- [ ] Benchmark results with clear charts (matplotlib/plotly)
- [ ] Interactive comparison table
- [ ] Installation: `pip install smallgbm`
- [ ] Quickstart tutorial with code snippets
- [ ] API reference
- [ ] Link to GitHub repo and paper

**Tech:**

- [ ] Static HTML/CSS or Jekyll/Hugo
- [ ] Host on GitHub Pages (free)
- [ ] Responsive design, clean typography

---

## 5. Publication

Write and submit the paper.

**Paper structure:**

- [ ] Abstract
- [ ] Introduction: small data problem in applied ML
- [ ] Related work: gradient boosting, regularization, small-sample methods
- [ ] Method: SmallGBM algorithm, regularization choices, default hyperparameters
- [ ] Experiments: datasets, baselines, metrics, results
- [ ] Discussion: limitations, future work
- [ ] Conclusion

**Target venues:**

- [ ] arXiv preprint (primary)
- [ ] Student track at small conferences (ML Reproducibility Challenge, Tiny Papers at ICLR, undergrad workshops)
- [ ] Journal of Open Source Software (JOSS) if tool is polished

**Extras:**

- [ ] Record 2-minute video explainer
- [ ] Share on Twitter/LinkedIn ML community
- [ ] Tag XGBoost/LightGBM authors for visibility

---

*No deadlines. Ship when each phase is solid.*
