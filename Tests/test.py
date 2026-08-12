import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from smallgbm import SmallGBMClassifier

# Generate 20 diverse datasets
datasets = []

# 10 synthetic datasets with different properties
configs = [
    (30, 5, 2, 0.05),    # tiny, low noise
    (30, 20, 3, 0.15),   # tiny, high-dimensional, noisy
    (50, 10, 4, 0.10),   # very small
    (80, 15, 5, 0.15),   # small, moderate noise
    (100, 20, 6, 0.10),  # small
    (150, 10, 5, 0.05),  # medium-small, clean
    (200, 30, 8, 0.20),  # medium, noisy
    (300, 15, 6, 0.10),  # medium
    (500, 20, 8, 0.15),  # large-small
    (1000, 25, 10, 0.10) # upper bound
]

for i, (n, d, informative, noise) in enumerate(configs):
    X, y = make_classification(n_samples=n, n_features=d, n_informative=informative,
                               n_redundant=d//3, flip_y=noise, random_state=42+i)
    datasets.append((f'synth_{i}_{n}x{d}_noise{int(noise*100)}', X, y))

# 10 real-world datasets
real_datasets = [
    ('breast_cancer', load_breast_cancer().data, load_breast_cancer().target),
    ('wine_0vsrest', load_wine().data, (load_wine().target == 0).astype(int)),
]

# Wine variant: class 0 vs class 1+2
wine = load_wine()
datasets.append(('wine_1vsrest', wine.data, (wine.target == 1).astype(int)))

# Digits: 3 vs 8
digits = load_digits()
mask = (digits.target == 3) | (digits.target == 8)
datasets.append(('digits_3vs8', digits.data[mask], (digits.target[mask] == 3).astype(int)))

# Add breast cancer subsets with different train sizes
bc = load_breast_cancer()
for train_n in [30, 50, 80, 100]:
    X_train, X_test, y_train, y_test = train_test_split(
        bc.data, bc.target, train_size=train_n, stratify=bc.target, random_state=train_n
    )
    datasets.append((f'bc_train{train_n}', X_train, y_train))

results = {name: [] for name in ['SmallGBM', 'XGBoost', 'LightGBM']}

for ds_name, X, y in datasets:
    aucs = {name: [] for name in results}
    
    for seed in range(5):
        train_size = min(int(0.6 * len(y)), 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, stratify=y, random_state=seed
        )
        
        models = {
            'SmallGBM': SmallGBMClassifier(n_estimators=50, max_depth=3, min_samples_leaf=3,
                                            learning_rate=0.1, sigma_prior=0.5, random_state=seed),
            'XGBoost': XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                      eval_metric='logloss', verbosity=0),
            'LightGBM': LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                        min_child_samples=3, verbose=-1)
        }
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_test)[:, 1]
                aucs[name].append(roc_auc_score(y_test, y_pred))
            except:
                aucs[name].append(np.nan)
    
    print(f"\n{ds_name}:")
    for name in results:
        valid = [a for a in aucs[name] if not np.isnan(a)]
        if valid:
            mean_auc = np.mean(valid)
            results[name].append(mean_auc)
            print(f"  {name:15s}: {mean_auc:.4f}")
        else:
            results[name].append(np.nan)
            print(f"  {name:15s}: FAILED")

print(f"\n{'='*50}")
print("FINAL AVERAGE ACROSS ALL DATASETS:")
print(f"{'='*50}")
for name in results:
    valid = [a for a in results[name] if not np.isnan(a)]
    if valid:
        print(f"  {name:15s}: {np.mean(valid):.4f}  (on {len(valid)} datasets)")