import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import (make_classification, load_breast_cancer,
                              load_wine, load_iris, load_digits, fetch_openml)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

# Python-дерево
from smallgbm import SmallGBMClassifier


# ---------------- datasets ----------------

def build_datasets():
    datasets = []
    configs = [
        (50, 5, 0.05), (50, 10, 0.10), (80, 10, 0.10), (80, 15, 0.15),
        (100, 10, 0.05), (100, 20, 0.10), (150, 10, 0.10), (150, 20, 0.15),
        (200, 15, 0.05), (200, 25, 0.10), (300, 15, 0.10), (300, 20, 0.15),
        (500, 20, 0.10), (500, 30, 0.15), (800, 20, 0.05)
    ]
    for i, (n, d, noise) in enumerate(configs):
        X, y = make_classification(n_samples=n, n_features=d, n_informative=d//2,
                                   flip_y=noise, random_state=42+i)
        datasets.append((f'synth_{i}_{n}x{d}_noise{int(noise*100)}', X, y))

    bc = load_breast_cancer()
    wine = load_wine()
    iris = load_iris()
    digits = load_digits()

    datasets.append(('bc_full', bc.data, bc.target))
    datasets.append(('wine_0vsrest', wine.data, (wine.target == 0).astype(int)))
    datasets.append(('wine_1vsrest', wine.data, (wine.target == 1).astype(int)))
    datasets.append(('wine_2vsrest', wine.data, (wine.target == 2).astype(int)))
    datasets.append(('iris_0vsrest', iris.data, (iris.target == 0).astype(int)))
    datasets.append(('iris_1vsrest', iris.data, (iris.target == 1).astype(int)))

    mask = (digits.target == 3) | (digits.target == 8)
    datasets.append(('digits_3vs8', digits.data[mask],
                     (digits.target[mask] == 3).astype(int)))

    for name, loader in [('heart_disease', 'heart-disease'),
                         ('diabetes_pima', 'diabetes'),
                         ('banknote_auth', 'banknote-authentication'),
                         ('sonar', 'sonar'),
                         ('credit_approval', 'credit-approval')]:
        try:
            d = fetch_openml(loader, version=1, as_frame=False, parser='auto')
            X = d.data.astype(float)
            y = LabelEncoder().fit_transform(d.target.astype(str))
            datasets.append((name, X, y))
        except Exception:
            pass

    return datasets


# ---------------- models ----------------

def make_models():
    return {
        'SmallGBM': lambda: SmallGBMClassifier(colsample_bytree=0.5, random_state=42),
        'XGBoost':     lambda: XGBClassifier(n_estimators=50, max_depth=3,
                                             learning_rate=0.1, eval_metric='logloss',
                                             verbosity=0, colsample_bytree=0.5),
        'LightGBM':    lambda: LGBMClassifier(n_estimators=50, max_depth=3,
                                              learning_rate=0.1, min_child_samples=3,
                                              verbose=-1),
        'RandomForest':lambda: RandomForestClassifier(n_estimators=50, max_depth=3,
                                                      min_samples_leaf=3, random_state=42),
    }


# ---------------- bench ----------------

def main():
    datasets = build_datasets()
    print(f"Total datasets: {len(datasets)}")

    model_names = ['SmallGBM', 'XGBoost', 'LightGBM', 'RandomForest']

    aucs   = {name: [] for name in model_names}
    fit_ts = {name: [] for name in model_names}
    prd_ts = {name: [] for name in model_names}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for ds_name, X, y in datasets:
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for name, factory in make_models().items():
                model = factory()

                t0 = time.perf_counter()
                try:
                    model.fit(X_train, y_train)
                except Exception:
                    continue
                t1 = time.perf_counter()

                try:
                    y_pred = model.predict_proba(X_test)[:, 1]
                except Exception:
                    continue
                t2 = time.perf_counter()

                try:
                    aucs[name].append(roc_auc_score(y_test, y_pred))
                except Exception:
                    pass

                fit_ts[name].append(t1 - t0)
                prd_ts[name].append(t2 - t1)

    print(f"\n{'='*70}")
    print(f"FINAL BENCHMARK: {len(datasets)} datasets, 5-fold CV")
    print(f"{'='*70}")
    for name in model_names:
        a = np.array(aucs[name])
        f = np.array(fit_ts[name]) * 1000
        p = np.array(prd_ts[name]) * 1000
        if len(a):
            print(f"  {name:15s}: AUC={a.mean():.4f}±{a.std():.4f}  "
                  f"fit={f.mean():7.2f}ms  predict={p.mean():7.3f}ms  "
                  f"(n={len(a)})")


if __name__ == "__main__":
    main()