import numpy as np

class BayesianDecisionTree:
    """Optimized decision tree with Bayesian leaf weight estimation."""
    
    def __init__(self, max_depth=2, min_samples_leaf=5, sigma_prior=1.0, n_splits=10):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sigma_prior = sigma_prior
        self.n_splits = n_splits
        self.tree_ = None
    
    def _variance(self, y):
        if len(y) <= 1:
            return 0.0
        return np.var(y)
    
    def _bayesian_weight(self, residuals):
        n = len(residuals)
        sum_r = np.sum(residuals)
        sigma_noise = self._variance(residuals)
        if sigma_noise == 0:
            return sum_r / n if n > 0 else 0.0
        shrinkage = sigma_noise / (self.sigma_prior ** 2)
        return sum_r / (n + shrinkage)
    
    def _best_split(self, X, residuals):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        n = len(residuals)
        
        for feature in range(n_features):
            values = X[:, feature]
            sort_idx = np.argsort(values)
            sorted_values = values[sort_idx]
            sorted_residuals = residuals[sort_idx]
            
            # Prefix sums for O(1) variance computation
            cumsum = np.cumsum(sorted_residuals)
            cumsum2 = np.cumsum(sorted_residuals ** 2)
            total_sum = cumsum[-1]
            total_sum2 = cumsum2[-1]
            parent_var = (total_sum2 - total_sum**2 / n) / n if n > 1 else 0
            
            # Full search over all valid split positions
            for pos in range(self.min_samples_leaf, n - self.min_samples_leaf + 1):
                left_n = pos
                right_n = n - pos
                
                left_sum = cumsum[pos - 1]
                left_sum2 = cumsum2[pos - 1]
                left_var = (left_sum2 - left_sum**2 / left_n) / left_n if left_n > 1 else 0
                
                right_sum = total_sum - left_sum
                right_sum2 = total_sum2 - left_sum2
                right_var = (right_sum2 - right_sum**2 / right_n) / right_n if right_n > 1 else 0
                
                gain = parent_var - (left_n / n * left_var + right_n / n * right_var)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (sorted_values[pos - 1] + sorted_values[pos]) / 2
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, residuals, depth=0):
        n = len(residuals)
        
        if (depth >= self.max_depth or 
            n < self.min_samples_leaf * 2 or 
            len(np.unique(residuals)) == 1):
            return {
                'type': 'leaf',
                'weight': self._bayesian_weight(residuals),
                'n_samples': n
            }
        
        feature, threshold, gain = self._best_split(X, residuals)
        
        if feature is None or gain <= 0:
            return {
                'type': 'leaf',
                'weight': self._bayesian_weight(residuals),
                'n_samples': n
            }
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        return {
            'type': 'node',
            'feature': feature,
            'threshold': threshold,
            'n_samples': n,
            'left': self._build_tree(X[left_mask], residuals[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], residuals[right_mask], depth + 1)
        }
    
    def fit(self, X, residuals):
        self.tree_ = self._build_tree(np.array(X), np.array(residuals))
        return self
    
    def _predict_one(self, x, node):
        if node['type'] == 'leaf':
            return node['weight']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in np.array(X)])