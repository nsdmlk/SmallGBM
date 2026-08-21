import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import RobustScaler
from .tree import RobustDecisionTree


class SmallGBMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=50, max_depth=3, min_samples_leaf=3,
                 learning_rate=0.1, sigma_prior=0.5, adaptive_prior=False,
                 dynamic_depth=False, weighted_residuals=False, soft_bootstrap=False,
                 n_bags=1, bagging_fraction=0.8, sub_bagging=False,
                 random_state=None, auto_scale=False, colsample_bytree=0.5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.sigma_prior = sigma_prior
        self.adaptive_prior = adaptive_prior
        self.dynamic_depth = dynamic_depth
        self.weighted_residuals = weighted_residuals
        self.soft_bootstrap = soft_bootstrap
        self.n_bags = n_bags
        self.bagging_fraction = bagging_fraction
        self.sub_bagging = sub_bagging
        self.random_state = random_state
        self.auto_scale = auto_scale
        self.colsample_bytree = colsample_bytree

    def _log_odds(self, y):
        pos = np.sum(y == 1)
        neg = np.sum(y == 0)
        return np.log((pos + 1e-10) / (neg + 1e-10))

    def _sigmoid(self, x):
        x = np.clip(x, -30, 30)
        return 1.0 / (1.0 + np.exp(-x))

    def _fit_single_boost(self, X, y, seed, colsample):
        if seed is not None:
            np.random.seed(seed)
        
        n_samples = len(y)
        
        if self.sub_bagging and n_samples > 20:
            n_bag = int(n_samples * self.bagging_fraction)
            indices = np.random.choice(n_samples, n_bag, replace=False)
            X_train = X[indices]
            y_train = y[indices]
        else:
            X_train = X
            y_train = y
        
        init = self._log_odds(y_train)
        trees = []
        current_pred = np.full(y_train.shape, init, dtype=np.float64)
        
        for i in range(self.n_estimators):
            proba = self._sigmoid(current_pred)
            residuals = y_train - proba
            
            if self.dynamic_depth:
                progress = i / self.n_estimators
                current_max_depth = max(1, int(self.max_depth * (1 - progress * 0.5)))
            else:
                current_max_depth = self.max_depth
            
            sigma_prior = self.sigma_prior
            if self.adaptive_prior and sigma_prior is None:
                sigma_prior = 1.0 / np.sqrt(len(y_train))
            
            tree = RobustDecisionTree(
                max_depth=current_max_depth,
                min_samples_leaf=self.min_samples_leaf,
                sigma_prior=sigma_prior,
                colsample_bytree=colsample
            )
            tree.fit(X_train, residuals)
            
            update = tree.predict(X_train)
            current_pred += self.learning_rate * update
            trees.append(tree)
        
        return init, trees

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        
        colsample = self.colsample_bytree
        if colsample == 'auto':
            n_samples = len(y)
            colsample = min(1.0, 0.3 + 0.7 * (n_samples / 1000))
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        if self.auto_scale:
            self._scaler = RobustScaler()
            X = self._scaler.fit_transform(X)
        
        self._models = []
        for m in range(self.n_bags):
            seed = self.random_state + m if self.random_state is not None else m
            init, trees = self._fit_single_boost(X, y, seed, colsample)
            self._models.append((init, trees))
        
        return self

    def _predict_proba_single(self, X, init, trees):
        current_pred = np.full(X.shape[0], init, dtype=np.float64)
        for tree in trees:
            current_pred += self.learning_rate * tree.predict(X)
        return self._sigmoid(current_pred)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if hasattr(self, '_scaler'):
            X = self._scaler.transform(X)
        
        if self.n_bags > 1:
            all_probas = [self._predict_proba_single(X, init, trees) for init, trees in self._models]
            proba_pos = np.mean(all_probas, axis=0)
        else:
            init, trees = self._models[0]
            proba_pos = self._predict_proba_single(X, init, trees)
        
        return np.column_stack([1 - proba_pos, proba_pos])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)


class SmallGBMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=50, max_depth=3, min_samples_leaf=3,
                 learning_rate=0.1, sigma_prior=0.5, adaptive_prior=False,
                 dynamic_depth=False, weighted_residuals=False, soft_bootstrap=False,
                 n_bags=1, bagging_fraction=0.8, sub_bagging=False,
                 random_state=None, auto_scale=False, colsample_bytree=0.5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.sigma_prior = sigma_prior
        self.adaptive_prior = adaptive_prior
        self.dynamic_depth = dynamic_depth
        self.weighted_residuals = weighted_residuals
        self.soft_bootstrap = soft_bootstrap
        self.n_bags = n_bags
        self.bagging_fraction = bagging_fraction
        self.sub_bagging = sub_bagging
        self.random_state = random_state
        self.auto_scale = auto_scale
        self.colsample_bytree = colsample_bytree

    def _fit_single_boost(self, X, y, seed, colsample):
        if seed is not None:
            np.random.seed(seed)
        
        n_samples = len(y)
        
        if self.sub_bagging and n_samples > 20:
            n_bag = int(n_samples * self.bagging_fraction)
            indices = np.random.choice(n_samples, n_bag, replace=False)
            X_train = X[indices]
            y_train = y[indices]
        else:
            X_train = X
            y_train = y
        
        init = np.mean(y_train)
        trees = []
        current_pred = np.full(y_train.shape, init, dtype=np.float64)
        
        for i in range(self.n_estimators):
            residuals = y_train - current_pred
            
            if self.dynamic_depth:
                progress = i / self.n_estimators
                current_max_depth = max(1, int(self.max_depth * (1 - progress * 0.5)))
            else:
                current_max_depth = self.max_depth
            
            sigma_prior = self.sigma_prior
            if self.adaptive_prior and sigma_prior is None:
                sigma_prior = 1.0 / np.sqrt(len(y_train))
            
            tree = RobustDecisionTree(
                max_depth=current_max_depth,
                min_samples_leaf=self.min_samples_leaf,
                sigma_prior=sigma_prior,
                colsample_bytree=colsample
            )
            tree.fit(X_train, residuals)
            
            update = tree.predict(X_train)
            current_pred += self.learning_rate * update
            trees.append(tree)
        
        return init, trees

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        colsample = self.colsample_bytree
        if colsample == 'auto':
            n_samples = len(y)
            colsample = min(1.0, 0.3 + 0.7 * (n_samples / 1000))
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        if self.auto_scale:
            self._scaler = RobustScaler()
            X = self._scaler.fit_transform(X)
        
        self._models = []
        for m in range(self.n_bags):
            seed = self.random_state + m if self.random_state is not None else m
            init, trees = self._fit_single_boost(X, y, seed, colsample)
            self._models.append((init, trees))
        
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if hasattr(self, '_scaler'):
            X = self._scaler.transform(X)
        
        all_preds = []
        for init, trees in self._models:
            current_pred = np.full(X.shape[0], init, dtype=np.float64)
            for tree in trees:
                current_pred += self.learning_rate * tree.predict(X)
            all_preds.append(current_pred)
        
        return np.mean(all_preds, axis=0) if self.n_bags > 1 else all_preds[0]