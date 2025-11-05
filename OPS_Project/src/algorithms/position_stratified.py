"""
Position-Stratified Shapley estimator (Algorithm 1).
Implements stratification over feature ranks with provable variance reduction.
"""

import numpy as np
from typing import List, Set, Optional, Tuple, Dict
from .shapley_base import ShapleyEstimator


class PositionStratifiedShapley(ShapleyEstimator):
    """
    Position-Stratified Shapley estimator (Algorithm 1 from paper).
    
    Stratifies permutation space by rank k ∈ {0, ..., n-1} where feature i appears.
    Eliminates between-stratum variance, keeping only within-stratum variance.
    
    Theory:
        - Lemma 1 (Rank-Conditional Decomposition)
        - Theorem 1 (Variance Decomposition)
    """
    
    def _sample_k_subset(self, feature_idx: int, k: int) -> List[int]:
        """
        Sample subset S of size k from N\{i} uniformly.
        
        Args:
            feature_idx: Feature i to exclude
            k: Subset size
            
        Returns:
            S: Random subset of size k (not including i)
        """
        # N \ {i}: all features except i
        N_minus_i = [j for j in range(self.n_features) if j != feature_idx]
        
        # Sample k features uniformly
        S = np.random.choice(N_minus_i, size=k, replace=False).tolist()
        
        return S
    
    def compute(self, feature_idx: int, L_budget: int, seed: Optional[int] = None) -> float:
        """
        Compute Shapley value using position stratification.
        
        Args:
            feature_idx: Index of feature to compute Shapley value for
            L_budget: Total sampling budget
            seed: Random seed
            
        Returns:
            phi: Position-stratified estimate
        """
        n = self.n_features
        
        if seed is not None:
            np.random.seed(seed)
            
        # Allocate budget uniformly across strata
        L_k = L_budget // n  # Samples per stratum
        
        # Estimate Shapley value for each stratum k
        phi_k = np.zeros(n)
        
        for k in range(n):
            marginal_sum = 0.0
            
            for _ in range(L_k):
                # Sample S ~ P_k (subset with |S| = k, i not in S)
                S = self._sample_k_subset(feature_idx, k)
                
                # Compute marginal contribution
                marginal = self._marginal_contribution(feature_idx, S)
                marginal_sum += marginal
                
            # Average for stratum k
            phi_k[k] = marginal_sum / L_k if L_k > 0 else 0.0
            
        # Average across strata (uniform weights)
        phi = np.mean(phi_k)
        
        return phi
    
    def compute_with_variance(self, feature_idx: int, L_budget: int, seed: Optional[int] = None) -> Tuple[float, float, Dict]:
        """
        Compute Shapley value with variance decomposition (Theorem 1).
        
        Returns:
            phi: Position-stratified estimate
            variance: Theoretical variance from Theorem 1
            stats: Dictionary with per-stratum statistics
        """
        n = self.n_features
        
        if seed is not None:
            np.random.seed(seed)
            
        # Allocate budget uniformly
        L_k = L_budget // n
        
        # Track samples and variance per stratum
        phi_k = np.zeros(n)
        sigma_k_sq = np.zeros(n)
        
        for k in range(n):
            marginals = []
            
            for _ in range(L_k):
                S = self._sample_k_subset(feature_idx, k)
                marginal = self._marginal_contribution(feature_idx, S)
                marginals.append(marginal)
                
            # Stratum mean and variance
            phi_k[k] = np.mean(marginals)
            sigma_k_sq[k] = np.var(marginals)
            
        # Overall estimate
        phi = np.mean(phi_k)
        
        # Variance decomposition (Theorem 1):
        # Var[φ_PS] = (1/n²) * Σ_k (σ²_k / L_k)
        variance = (1 / n**2) * np.sum(sigma_k_sq / L_k)
        
        stats = {
            'phi_k': phi_k,
            'sigma_k_sq': sigma_k_sq,
            'L_k': L_k
        }
        
        return phi, variance, stats
