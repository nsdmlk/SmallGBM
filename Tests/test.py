import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from smallgbm import SmallGBMClassifier

X, y = make_classification(n_samples=25, n_features=30, n_informative=2,
                           n_redundant=1, flip_y=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=15, stratify=y, random_state=42)

aucs = []
for seed in range(10):
    model = SmallGBMClassifier(random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    aucs.append(roc_auc_score(y_test, y_pred))

print(f"Corrected formula (extreme): AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")