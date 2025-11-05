"""
Algorithm 3: OPS with Antithetic Coupling
Implements orthogonal permutation sampling with negative correlation
"""

import numpy as np
from typing import Tuple, Dict, List
from .shapley_base import ShapleyEstimator


class OPSAntitheticShapley(ShapleyEstimator):
    """
    OPS with antithetic coupling estimator.
    
    Based on Theorem 3: Uses orthogonal Latin squares to create
    negatively correlated permutation pairs, reducing variance.
    """
    
    def _generate_orthogonal_permutations(self, n: int, n_pairs: int, seed: int = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate pairs of orthogonal permutations.
        
        For true orthogonality, we'd need orthogonal Latin squares.
        Here we use a simpler antithetic approach: create pairs where
        the second permutation is designed to be negatively correlated.
        
        Args:
            n: Number of features
            n_pairs: Number of permutation pairs to generate
            seed: Random seed
            
        Returns:
            List of (perm1, perm2) tuples
        """
        if seed is not None:
            np.random.seed(seed)
        
        pairs = []
        
        for _ in range(n_pairs):
            # Generate first permutation randomly
            perm1 = np.random.permutation(n)
            
            # Generate antithetic permutation
            # Strategy: reverse the permutation for maximum negative correlation
            perm2 = perm1[::-1]
            
            pairs.append((perm1.copy(), perm2.copy()))
        
        return pairs
    
    def compute_with_variance(self, feature_idx: int, budget: int, seed: int = None) -> Tuple[float, float, Dict]:
        """
        Compute Shapley value using antithetic OPS.
        
        Args:
            feature_idx: Index of feature to explain
            budget: Total sampling budget (should be even for pairing)
            seed: Random seed
            
        Returns:
            (estimate, variance, stats_dict)
        """
        n = self.X.shape[1]
        
        # Ensure even budget for pairing
        n_pairs = budget // 2
        actual_budget = n_pairs * 2
        
        # Generate orthogonal permutation pairs
        perm_pairs = self._generate_orthogonal_permutations(n, n_pairs, seed=seed)
        
        # Collect estimates from each permutation
        all_estimates = []
        pair_estimates = []
        
        for perm1, perm2 in perm_pairs:
            # Process first permutation
            est1 = self._estimate_from_permutation(feature_idx, perm1)
            all_estimates.append(est1)
            
            # Process second permutation (antithetic)
            est2 = self._estimate_from_permutation(feature_idx, perm2)
            all_estimates.append(est2)
            
            # Store pair average (antithetic variance reduction)
            pair_estimates.append((est1 + est2) / 2)
        
        # Final estimate
        phi = np.mean(all_estimates)
        
        # Variance estimate using antithetic pairs
        # Var(pair_avg) = Var(X1 + X2) / 4 = (Var(X1) + Var(X2) + 2*Cov(X1,X2)) / 4
        # For antithetic variables, Cov(X1, X2) < 0
        pair_variance = np.var(pair_estimates, ddof=1) if len(pair_estimates) > 1 else 0.0
        
        # Empirical variance of all samples
        empirical_variance = np.var(all_estimates, ddof=1) if len(all_estimates) > 1 else 0.0
        
        stats = {
            'n_pairs': n_pairs,
            'actual_budget': actual_budget,
            'pair_variance': pair_variance,
            'empirical_variance': empirical_variance,
            'variance_reduction': empirical_variance / pair_variance if pair_variance > 0 else 1.0
        }
        
        return phi, pair_variance, stats
    
    def _estimate_from_permutation(self, feature_idx: int, permutation: np.ndarray) -> float:
        """
        Estimate Shapley value from a single permutation.
        
        Args:
            feature_idx: Feature index
            permutation: Permutation of feature indices
            
        Returns:
            Shapley value estimate from this permutation
        """
        # Find position of feature in permutation
        pos = np.where(permutation == feature_idx)[0][0]
        
        # S = features before feature_idx in the permutation
        S = set(permutation[:pos].tolist())
        
        # Compute marginal contribution
        return self._marginal_contribution(feature_idx, S)
    
    def compute(self, feature_idx: int, budget: int, seed: int = None) -> float:
        """Simple interface returning just the estimate."""
        phi, _, _ = self.compute_with_variance(feature_idx, budget, seed=seed)
        return phi
