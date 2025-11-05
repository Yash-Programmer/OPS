"""
Base Shapley estimator with exact and Monte Carlo methods.
Implements exact enumeration for n≤10 and naive MC sampling baseline.
"""

import numpy as np
from itertools import permutations, combinations
from typing import List, Callable, Optional


class ShapleyEstimator:
    """
    Base class for Shapley value estimation.
    
    Provides:
    - exact_shapley(): Enumerate all n! permutations (for n≤10)
    - mc_shapley(): Naive Monte Carlo baseline
    """
    
    def __init__(self, model: Callable, X: np.ndarray, baseline: Optional[np.ndarray] = None):
        """
        Args:
            model: Trained model with predict() method or callable
            X: Data points to explain (n_samples, n_features)
            baseline: Reference point for marginal contributions
        """
        self.model = model
        self.X = X if X.ndim == 2 else X.reshape(1, -1)
        self.n_features = self.X.shape[1]
        
        if baseline is None:
            self.baseline = np.zeros(self.n_features)
        else:
            self.baseline = baseline
            
    def _marginal_contribution(self, feature_idx: int, S: List[int]) -> float:
        """
        Compute marginal contribution v(S ∪ {i}) - v(S).
        
        Args:
            feature_idx: Index of feature i
            S: Subset of features (not including i, can be list or set)
            
        Returns:
            marginal: v(S ∪ {i}) - v(S)
        """
        # Convert S to list if it's a set (for indexing)
        S_list = list(S) if isinstance(S, set) else S
        
        # Create input with S features from X, rest from baseline
        x_S = self.baseline.copy()
        x_S[S_list] = self.X[0, S_list]  # Use first data point
        
        x_S_union_i = x_S.copy()
        x_S_union_i[feature_idx] = self.X[0, feature_idx]
        
        # Compute model predictions
        if hasattr(self.model, 'predict'):
            # For classification, use predict_proba if available
            if hasattr(self.model, 'predict_proba'):
                v_S = self.model.predict_proba(x_S.reshape(1, -1))[0]
                v_S_union_i = self.model.predict_proba(x_S_union_i.reshape(1, -1))[0]
                # Use probability of positive class (binary) or first class
                v_S = v_S[1] if len(v_S) == 2 else v_S[0]
                v_S_union_i = v_S_union_i[1] if len(v_S_union_i) == 2 else v_S_union_i[0]
            else:
                # Regression or models without predict_proba
                v_S_pred = self.model.predict(x_S.reshape(1, -1))
                v_S_union_i_pred = self.model.predict(x_S_union_i.reshape(1, -1))
                # Handle both scalar and array outputs
                v_S = float(v_S_pred[0]) if hasattr(v_S_pred, '__len__') else float(v_S_pred)
                v_S_union_i = float(v_S_union_i_pred[0]) if hasattr(v_S_union_i_pred, '__len__') else float(v_S_union_i_pred)
        else:
            # Model is callable function
            v_S = self.model(x_S.reshape(1, -1))[0]
            v_S_union_i = self.model(x_S_union_i.reshape(1, -1))[0]
            
        return float(v_S_union_i - v_S)
    
    def exact_shapley(self, feature_idx: int) -> float:
        """
        Compute exact Shapley value by enumerating all n! permutations.
        Only use for n ≤ 10 due to computational cost.
        
        Args:
            feature_idx: Index of feature to compute Shapley value for
            
        Returns:
            phi: Exact Shapley value
        """
        n = self.n_features
        
        if n > 10:
            raise ValueError(f"Exact Shapley computation infeasible for n={n}>10")
            
        # Enumerate all permutations
        feature_indices = list(range(n))
        all_perms = list(permutations(feature_indices))
        
        phi_sum = 0.0
        
        for perm in all_perms:
            # Find position k where feature_idx appears
            k = perm.index(feature_idx)
            
            # S = features appearing before k in permutation
            S = list(perm[:k])
            
            # Add marginal contribution
            marginal = self._marginal_contribution(feature_idx, S)
            phi_sum += marginal
            
        # Average over all n! permutations
        phi = phi_sum / len(all_perms)
        
        return phi
    
    def mc_shapley(self, feature_idx: int, n_samples: int = 1000, seed: Optional[int] = None) -> float:
        """
        Estimate Shapley value using naive Monte Carlo sampling.
        This is the baseline method for comparison.
        
        Args:
            feature_idx: Index of feature to compute Shapley value for
            n_samples: Number of random permutations to sample
            seed: Random seed for reproducibility
            
        Returns:
            phi: MC estimate of Shapley value
        """
        n = self.n_features
        
        if seed is not None:
            np.random.seed(seed)
            
        phi_sum = 0.0
        
        for _ in range(n_samples):
            # Sample random permutation
            perm = np.random.permutation(n)
            
            # Find position k where feature_idx appears
            k = np.where(perm == feature_idx)[0][0]
            
            # S = features appearing before k
            S = perm[:k].tolist()
            
            # Add marginal contribution
            marginal = self._marginal_contribution(feature_idx, S)
            phi_sum += marginal
            
        # Average over samples
        phi = phi_sum / n_samples
        
        return phi
