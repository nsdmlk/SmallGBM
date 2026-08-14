import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer, load_wine, load_iris, load_digits, fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from smallgbm import SmallGBMClassifier

# 15 realistic synthetic datasets
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

# Real-world datasets
bc = load_breast_cancer()
wine = load_wine()
iris = load_iris()
digits = load_digits()

for n_train in [50, 80, 100, 150]:
    X_train, X_test, y_train, y_test = train_test_split(
        bc.data, bc.target, train_size=n_train, stratify=bc.target, random_state=n_train
    )
    datasets.append((f'bc_{n_train}', X_train, y_train))

datasets.append(('wine_0vsrest', wine.data, (wine.target == 0).astype(int)))
datasets.append(('wine_1vsrest', wine.data, (wine.target == 1).astype(int)))
datasets.append(('wine_2vsrest', wine.data, (wine.target == 2).astype(int)))
datasets.append(('iris_0vsrest', iris.data, (iris.target == 0).astype(int)))
datasets.append(('iris_1vsrest', iris.data, (iris.target == 1).astype(int)))

mask = (digits.target == 3) | (digits.target == 8)
datasets.append(('digits_3vs8', digits.data[mask], (digits.target[mask] == 3).astype(int)))

# Additional real-world datasets (medicine, finance, physics)
# Heart Disease (Cleveland)
try:
    heart = fetch_openml('heart-disease', version=1, as_frame=False, parser='auto')
    X_heart = heart.data.astype(float)
    y_heart = (heart.target.astype(int) > 0).astype(int)
    datasets.append(('heart_disease', X_heart, y_heart))
except:
    pass

# Diabetes (Pima Indians)
try:
    diabetes = fetch_openml('diabetes', version=1, as_frame=False, parser='auto')
    X_diab = diabetes.data.astype(float)
    y_diab = diabetes.target.astype(int)
    y_diab = LabelEncoder().fit_transform(y_diab)
    datasets.append(('diabetes_pima', X_diab, y_diab))
except:
    pass

# Banknote Authentication
try:
    banknote = fetch_openml('banknote-authentication', version=1, as_frame=False, parser='auto')
    X_bank = banknote.data.astype(float)
    y_bank = banknote.target.astype(int)
    y_bank = LabelEncoder().fit_transform(y_bank)
    datasets.append(('banknote_auth', X_bank, y_bank))
except:
    pass

# Sonar (physics)
try:
    sonar = fetch_openml('sonar', version=1, as_frame=False, parser='auto')
    X_sonar = sonar.data.astype(float)
    y_sonar = LabelEncoder().fit_transform(sonar.target.astype(str))
    datasets.append(('sonar', X_sonar, y_sonar))
except:
    pass

# Credit Approval
try:
    credit = fetch_openml('credit-approval', version=1, as_frame=False, parser='auto')
    X_credit = credit.data.astype(float)
    y_credit = LabelEncoder().fit_transform(credit.target.astype(str))
    datasets.append(('credit_approval', X_credit, y_credit))
except:
    pass

print(f"Total datasets: {len(datasets)}")

results = {name: [] for name in ['SmallGBM', 'XGBoost', 'LightGBM', 'RandomForest']}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for ds_name, X, y in datasets:
    aucs = {name: [] for name in results}
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        models = {
            'SmallGBM': SmallGBMClassifier(colsample_bytree=0.5, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                      eval_metric='logloss', verbosity=0, colsample_bytree=0.5),
            'LightGBM': LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                        min_child_samples=3, verbose=-1),
            'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=3,
                                                    min_samples_leaf=3, random_state=42)
        }
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_test)[:, 1]
                aucs[name].append(roc_auc_score(y_test, y_pred))
            except:
                aucs[name].append(np.nan)
    
    for name in results:
        valid = [a for a in aucs[name] if not np.isnan(a)]
        if valid:
            results[name].append(np.mean(valid))

print(f"\n{'='*60}")
print(f"FINAL BENCHMARK: {len(datasets)} datasets, 5-fold CV")
print(f"{'='*60}")
for name in results:
    valid = [a for a in results[name] if not np.isnan(a)]
    if valid:
        mean_auc = np.mean(valid)
        std_auc = np.std(valid)
        print(f"  {name:15s}: {mean_auc:.4f} ± {std_auc:.4f}  (on {len(valid)} datasets)")