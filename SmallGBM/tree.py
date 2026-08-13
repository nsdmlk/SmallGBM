import numpy as np

class BayesianDecisionTree:
    """Optimized decision tree with Bayesian leaf weight estimation."""
    
    def __init__(self, max_depth=2, min_samples_leaf=5, sigma_prior=1.0, n_splits=10, global_sigma_noise_sq=None):
        self.global_sigma_noise_sq = global_sigma_noise_sq
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sigma_prior = sigma_prior
        self.n_splits = n_splits
        self.tree_ = None
    
    def _bayesian_weight(self, residuals, parent_mu=0.0):
        n = len(residuals)
        sum_r = np.sum(residuals)
        
        # Empirical Bayes for a: scales inversely with n
        a = 0.5 / (n + 9)**0.5
        b = self.sigma_prior ** 2 * (n + 5)
        
        a_post = a + n / 3.33
        # Center by parent_mu
        b_post = b + np.sum((residuals - parent_mu) ** 2) / 2
        
        if a_post > 1:
            sigma_noise_sq = b_post / (a_post - 1)
        else:
            sigma_noise_sq = self.sigma_prior ** 2
        
        if sigma_noise_sq == 0:
            return sum_r / n if n > 0 else 0.0, 0.0
        
        # Posterior mean: pull toward parent_mu
        precision_data = n / sigma_noise_sq
        precision_prior = 1.0 / (self.sigma_prior ** 2)
        mu = (precision_data * (sum_r / n) + precision_prior * parent_mu) / (precision_data + precision_prior)
        
        std = 1.0 / np.sqrt(precision_data + precision_prior)
        return mu, std
    
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
            
            cumsum = np.cumsum(sorted_residuals)
            total_sum = cumsum[-1]
            
            for pos in range(self.min_samples_leaf, n - self.min_samples_leaf + 1):
                left_n = pos
                right_n = n - pos
                
                left_sum = cumsum[pos - 1]
                right_sum = total_sum - left_sum
                
                lambda_reg = self.sigma_prior ** 2
                left_score = (left_sum ** 2) / (left_n + lambda_reg)
                right_score = (right_sum ** 2) / (right_n + lambda_reg)
                parent_score = (total_sum ** 2) / (n + lambda_reg)
                
                gain = left_score + right_score - parent_score
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (sorted_values[pos - 1] + sorted_values[pos]) / 2
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, residuals, depth=0, parent_mu=0.0):
        n = len(residuals)
        
        # Leaf node
        if (depth >= self.max_depth or 
            n < self.min_samples_leaf * 2 or 
            len(np.unique(residuals)) == 1):
            mu, std = self._bayesian_weight(residuals, parent_mu)
            return {
                'type': 'leaf',
                'weight': mu,
                'uncertainty': std,
                'n_samples': n
            }
        
        feature, threshold, gain = self._best_split(X, residuals)
        
        # No valid split -> leaf
        if feature is None or gain <= 0:
            mu, std = self._bayesian_weight(residuals, parent_mu)
            return {
                'type': 'leaf',
                'weight': mu,
                'uncertainty': std,
                'n_samples': n
            }
        
        # Compute this node's weight to pass as parent_mu
        node_mu, _ = self._bayesian_weight(residuals, parent_mu)
        
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
        preds = []
        uncertainties = []
        for x in np.array(X):
            node = self.tree_
            while node['type'] != 'leaf':
                if x[node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            preds.append(node['weight'])
            uncertainties.append(node['uncertainty'])
        return np.array(preds), np.array(uncertainties)
    
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