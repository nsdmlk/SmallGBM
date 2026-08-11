---
title: 'SmallGBM: Gradient Boosting for Small Datasets'
tags:
  - Python
  - machine learning
  - gradient boosting
  - small data
  - bayesian methods
authors:
  - name: Emelyanov Ilya
    orcid: 0009-0002-6802-7544
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 11 August 2026
---
# Summary

SmallGBM is a Python library for gradient boosting on small datasets (n < 1000). Standard boosting libraries like XGBoost and LightGBM are optimized for large-scale data and often overfit when only a few hundred samples are available. SmallGBM addresses this with Bayesian leaf weight regularization, adaptive prior strength, and conservative default hyperparameters. The library provides a scikit-learn-compatible API and is available via `pip install smallgbm`.

# Statement of Need

Gradient boosting is one of the most powerful techniques for tabular data. However, state-of-the-art implementations (XGBoost, LightGBM, CatBoost) are designed for datasets with thousands to millions of rows. Their default hyperparameters are tuned for large-sample regimes. On small datasets — common in medical research, early-stage startups, and scientific experiments — these defaults lead to overfitting and unstable predictions.

SmallGBM fills this gap by providing a boosting implementation specifically designed for small-sample scenarios. It sacrifices some peak performance on large datasets in exchange for stability and reliability when data is scarce.

# Features

- **Bayesian leaf weights**: Leaf predictions are shrunk toward zero using a Bayesian prior, preventing overfitting on small leaves
- **Adaptive regularization**: Prior strength can automatically scale with sample size
- **No bootstrap**: Uses all training data for each tree, avoiding variance from subsampling
- **No column subsampling**: Uses all features at each split, appropriate for small feature sets
- **Dynamic tree depth**: Trees become shallower in later boosting rounds
- **scikit-learn compatible**: Implements `fit`, `predict`, and `predict_proba` interfaces

# Research

SmallGBM has been evaluated on synthetic and real-world datasets with sample sizes ranging from 15 to 500. Key findings from the characterization notebook include:

- **Noise stability**: SmallGBM degrades more gracefully than XGBoost and LightGBM under label noise, maintaining higher AUC at 10-25% noise levels
- **Prior insensitivity**: Performance is nearly invariant to `sigma_prior` across three orders of magnitude (0.01 to 100)
- **Predictable learning curve**: Reliable performance begins at approximately 40 training samples

---
