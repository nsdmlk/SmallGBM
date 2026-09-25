import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from smallgbm import SmallGBMClassifier


def make_huge(n, d, informative, noise, seed):
    X, y = make_classification(
        n_samples=n,
        n_features=d,
        n_informative=informative,
        n_redundant=0,
        n_repeated=0,
        flip_y=noise,
        class_sep=1.0,
        random_state=seed,
    )
    return X, y


def bench(model_name, model, X_train, y_train, X_test, y_test):
    t0 = time.perf_counter()
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"  [{model_name}] fit failed: {e}")
        return None, None, None
    t1 = time.perf_counter()

    try:
        y_pred = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"  [{model_name}] predict failed: {e}")
        return None, None, None
    t2 = time.perf_counter()

    try:
        auc = roc_auc_score(y_test, y_pred)
    except Exception:
        auc = float('nan')

    return auc, t1 - t0, t2 - t1


def make_models():
    return {
        'SmallGBM': lambda: SmallGBMClassifier(
            colsample_bytree=0.5, random_state=42,
            n_estimators=50, max_depth=3, min_samples_leaf=3,
        ),
        'XGBoost': lambda: XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            eval_metric='logloss', verbosity=0, colsample_bytree=0.5,
            tree_method='hist', n_jobs=-1,
        ),
        'LightGBM': lambda: LGBMClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            min_child_samples=3, verbose=-1, n_jobs=-1,
        ),
        'RandomForest': lambda: RandomForestClassifier(
            n_estimators=50, max_depth=3, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        ),
    }


def main():
    # три размера: 10k, 50k, 200k
    configs = [
        ("10k_x50",  10_000,  50, 25, 0.05, 1),
        ("50k_x100", 50_000, 100, 50, 0.05, 2),
        ("200k_x50",200_000,  50, 25, 0.05, 3),
    ]

    for name, n, d, informative, noise, seed in configs:
        print(f"\n{'='*70}")
        print(f"DATASET: {name}  n={n}, d={d}, informative={informative}, noise={noise}")
        print(f"{'='*70}")

        X, y = make_huge(n, d, informative, noise, seed)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        split_idx = list(cv.split(X, y))[0]  # один фолд, чтобы не ждать вечно
        train_idx, test_idx = split_idx

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"train: {X_train.shape}, test: {X_test.shape}\n")

        for model_name, factory in make_models().items():
            model = factory()
            auc, fit_t, pred_t = bench(model_name, model, X_train, y_train, X_test, y_test)
            if auc is not None:
                print(f"  {model_name:15s}: AUC={auc:.4f}  "
                      f"fit={fit_t:7.2f}s  predict={pred_t:6.3f}s")


if __name__ == "__main__":
    main()