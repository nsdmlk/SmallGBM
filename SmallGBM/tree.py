import numpy as np

class RobustDecisionTree:
    """Decision tree with robust (median/IQM) leaf weight estimation."""
    
    def __init__(self, max_depth=2, min_samples_leaf=5, sigma_prior=1.0, n_splits=10,
                 colsample_bytree=1.0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sigma_prior = sigma_prior
        self.sigma_prior_sq = sigma_prior ** 2
        self.n_splits = n_splits
        self.colsample_bytree = colsample_bytree
        self.tree_ = None
        self.feature_importances_ = None
    
    def _robust_weight(self, residuals, parent_mu=0.0):
        n = len(residuals)
        
        if n == 0:
            return parent_mu, 0.0
        
        threshold = 30
        if n <= threshold:
            center = np.median(residuals)
            mad = np.median(np.abs(residuals - center))
            sigma_noise = 1.4826 * mad if mad > 0 else self.sigma_prior
        else:
            med = np.median(residuals)
            distances = np.abs(residuals - med)
            weights = 1.0 / (distances + self.sigma_prior)
            weights /= weights.sum()
            center = np.sum(weights * residuals)
            sigma_noise = np.sqrt(np.sum(weights * (residuals - center)**2))
        
        k = 1.5
        ratio = sigma_noise ** 2 / (n * self.sigma_prior_sq)
        shrinkage = 1.0 / (1.0 + ratio ** k)
        mu = shrinkage * center + (1 - shrinkage) * parent_mu
        std = sigma_noise / np.sqrt(n)
        
        return mu, std
    
    def _best_split(self, X, residuals):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        n = len(residuals)
        
        if self.colsample_bytree < 1.0 and n_features > 1:
            n_cols = max(1, int(n_features * self.colsample_bytree))
            feature_indices = np.random.choice(n_features, n_cols, replace=False)
        else:
            feature_indices = range(n_features)
        
        for feature in feature_indices:
            values = X[:, feature]
            sort_idx = np.argsort(values)
            sorted_values = values[sort_idx]
            sorted_residuals = residuals[sort_idx]
            
            cumsum = np.cumsum(sorted_residuals)
            total_sum = cumsum[-1]
            
            for pos in range(self.min_samples_leaf, n - self.min_samples_leaf + 1):
                left_n = pos
                right_n = n - pos
                
                left_sum = cumsum[pos - 1]
                right_sum = total_sum - left_sum
                
                lambda_reg = self.sigma_prior_sq
                left_score = left_sum ** 2 / (left_n + lambda_reg)
                right_score = right_sum ** 2 / (right_n + lambda_reg)
                parent_score = total_sum ** 2 / (n + lambda_reg)
                
                gain = left_score + right_score - parent_score
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (sorted_values[pos - 1] + sorted_values[pos]) / 2
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, residuals, depth=0, parent_mu=0.0):
        n = len(residuals)
        
        if (depth >= self.max_depth or 
            n < self.min_samples_leaf * 2 or 
            len(np.unique(residuals)) == 1):
            mu, std = self._robust_weight(residuals, parent_mu)
            return {'type': 'leaf', 'weight': mu, 'uncertainty': std, 'n_samples': n}
        
        feature, threshold, gain = self._best_split(X, residuals)
        
        if feature is None or gain <= 0:
            mu, std = self._robust_weight(residuals, parent_mu)
            return {'type': 'leaf', 'weight': mu, 'uncertainty': std, 'n_samples': n}
        
        node_mu, _ = self._robust_weight(residuals, parent_mu)
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        return {
            'type': 'node',
            'feature': feature,
            'threshold': threshold,
            'weight': node_mu,
            'n_samples': n,
            'left': self._build_tree(X[left_mask], residuals[left_mask], depth + 1, node_mu),
            'right': self._build_tree(X[right_mask], residuals[right_mask], depth + 1, node_mu)
        }
    
    def predict_with_uncertainty(self, X):
        preds, uncertainties = [], []
        for x in np.array(X):
            node = self.tree_
            while node['type'] != 'leaf':
                node = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
            preds.append(node['weight'])
            uncertainties.append(node['uncertainty'])
        return np.array(preds), np.array(uncertainties)
    
    def fit(self, X, residuals):
        self.tree_ = self._build_tree(np.asarray(X), np.asarray(residuals))
        return self
    
    def _predict_one(self, x, node):
        while node['type'] != 'leaf':
            node = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
        return node['weight']
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in np.asarray(X)])