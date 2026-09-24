import time
import numpy as np
from tree import RobustDecisionTree

M = 20

# --- читаем данные, сгенерированные в C ---
data = np.loadtxt("data.csv", delimiter=",")
X = data[:, :M]
y = data[:, M]

tree = RobustDecisionTree(max_depth=4, min_samples_leaf=20, sigma_prior=1.0,
                          n_splits=10, colsample_bytree=1.0)

t0 = time.perf_counter()
tree.fit(X, y)
t1 = time.perf_counter()
print(tree._best_split)
print(tree._build_tree)
print(tree._robust_weight)
preds = tree.predict(X)
t2 = time.perf_counter()

print(f"Py: fit = {t1 - t0:.3f} s, predict = {t2 - t1:.3f} s")
print(f"Py: first 5 preds = {preds[:5]}")