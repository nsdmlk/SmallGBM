import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from .tree import BayesianDecisionTree


class SmallGBMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=50, max_depth=3, min_samples_leaf=3,
             learning_rate=0.1, sigma_prior=0.5, adaptive_prior=False,
             dynamic_depth=False, weighted_residuals=False, soft_bootstrap=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.sigma_prior = sigma_prior
        self.adaptive_prior = adaptive_prior
        self.dynamic_depth = dynamic_depth
        self.weighted_residuals = weighted_residuals
        self.soft_bootstrap = soft_bootstrap

    def _log_odds(self, y):
        pos = np.sum(y == 1)
        neg = np.sum(y == 0)
        return np.log((pos + 1e-10) / (neg + 1e-10))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples = len(y)

        init = self._log_odds(y)
        self.base_score_ = init
        self._trees = []

        current_pred = np.full(y.shape, init)

        for i in range(self.n_estimators):
            proba = self._sigmoid(current_pred)
            residuals = y - proba

            # 2. Weighted residuals: weight = proba * (1 - proba)
            if self.weighted_residuals:
                confidence = proba * (1 - proba)
                confidence = np.clip(confidence, 1e-10, None)
                sample_weights = confidence / confidence.sum() * n_samples
            else:
                sample_weights = np.ones(n_samples)

            # 4. Soft bootstrap: weighted sampling without replacement bias
            if self.soft_bootstrap:
                # Use all data but weight the residuals by sample_weights
                weighted_residuals = residuals * sample_weights
            else:
                weighted_residuals = residuals

            # 3. Dynamic depth: deeper early, shallower later
            if self.dynamic_depth:
                progress = i / self.n_estimators
                current_max_depth = max(1, int(self.max_depth * (1 - progress * 0.5)))
            else:
                current_max_depth = self.max_depth

            # 1. Adaptive sigma_prior: 1 / sqrt(n) if not specified
            if self.adaptive_prior and self.sigma_prior is None:
                sigma_prior = 1.0 / np.sqrt(n_samples)
            elif self.sigma_prior is not None:
                sigma_prior = self.sigma_prior
            else:
                sigma_prior = 1.0

            tree = BayesianDecisionTree(
                max_depth=current_max_depth,
                min_samples_leaf=self.min_samples_leaf,
                sigma_prior=sigma_prior,
                n_splits=20
            )
            tree.fit(X, weighted_residuals)

            update = tree.predict(X)
            current_pred += self.learning_rate * update
            self._trees.append(tree)

        return self

    def predict_proba(self, X):
        X = np.array(X)
        current_pred = np.full(X.shape[0], self.base_score_)
        for tree in self._trees:
            current_pred += self.learning_rate * tree.predict(X)
        proba_pos = self._sigmoid(current_pred)
        return np.column_stack([1 - proba_pos, proba_pos])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)
    
class SmallGBMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=50, max_depth=3, min_samples_leaf=3,
                 learning_rate=0.1, sigma_prior=0.5, adaptive_prior=False,
                 dynamic_depth=False, weighted_residuals=False, soft_bootstrap=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.sigma_prior = sigma_prior
        self.adaptive_prior = adaptive_prior
        self.dynamic_depth = dynamic_depth
        self.weighted_residuals = weighted_residuals
        self.soft_bootstrap = soft_bootstrap

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).astype(float)
        n_samples = len(y)

        # Initial prediction: mean of y
        init = np.mean(y)
        self.base_score_ = init
        self._trees = []

        current_pred = np.full(y.shape, init)

        for i in range(self.n_estimators):
            residuals = y - current_pred  # MSE residuals

            if self.weighted_residuals:
                confidence = np.abs(residuals)
                confidence = np.clip(confidence, 1e-10, None)
                sample_weights = confidence / confidence.sum() * n_samples
            else:
                sample_weights = np.ones(n_samples)

            if self.soft_bootstrap:
                weighted_residuals = residuals * sample_weights
            else:
                weighted_residuals = residuals

            if self.dynamic_depth:
                progress = i / self.n_estimators
                current_max_depth = max(1, int(self.max_depth * (1 - progress * 0.5)))
            else:
                current_max_depth = self.max_depth

            if self.adaptive_prior and self.sigma_prior is None:
                sigma_prior = 1.0 / np.sqrt(n_samples)
            elif self.sigma_prior is not None:
                sigma_prior = self.sigma_prior
            else:
                sigma_prior = 1.0

            tree = BayesianDecisionTree(
                max_depth=current_max_depth,
                min_samples_leaf=self.min_samples_leaf,
                sigma_prior=sigma_prior)
            tree.fit(X, weighted_residuals)

            update = tree.predict(X)
            current_pred += self.learning_rate * update
            self._trees.append(tree)

        return self

    def predict(self, X):
        X = np.array(X)
        current_pred = np.full(X.shape[0], self.base_score_)
        for tree in self._trees:
            current_pred += self.learning_rate * tree.predict(X)
        return current_pred