# SmallGBM

**Gradient boosting** that actually works on *small data*.

`XGBoost` and `LightGBM` are built for scale — they struggle when you only have a few hundred rows. **SmallGBM** is designed from the ground up for datasets with fewer than 1000 samples. Less overfitting, better metrics, simpler trees.

## Why?

Most real-world tabular problems start small. Medical studies, early-stage startups, niche surveys, rare events. Standard boosting libraries default to hyperparameters tuned for big data and fail badly on limited samples. SmallGBM fixes that.

## Installation

```bash
pip install smallgbm
```

## Quickstart

```python
from smallgbm import SmallGBMClassifier

model = SmallGBMClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

## License

MIT © Emelyanov Ilya 2026
