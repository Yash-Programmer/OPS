"""
Algorithm 2: Neyman Allocation for Shapley Value Estimation
Implements optimal stratum allocation based on variance estimates
"""

import numpy as np
from typing import Tuple, Dict
from .position_stratified import PositionStratifiedShapley


class NeymanAllocationShapley(PositionStratifiedShapley):
    """
    Neyman allocation estimator that allocates budget proportional to stratum standard deviations.
    
    Based on Theorem 2: Optimal allocation minimizes variance when n_k ∝ σ_k
    """
    
    def compute_with_neyman(self, feature_idx: int, total_budget: int, 
                           pilot_budget: int = 100, seed: int = None) -> Tuple[float, float, Dict]:
        """
        Compute Shapley value using Neyman allocation.
        
        Args:
            feature_idx: Index of feature to explain
            total_budget: Total sampling budget
            pilot_budget: Budget for pilot phase to estimate variances
            seed: Random seed
            
        Returns:
            (estimate, variance, stats_dict)
        """
        if seed is not None:
            np.random.seed(seed)
        
        n = self.X.shape[1]
        
        # Phase 1: Pilot study with uniform allocation
        pilot_per_stratum = max(1, pilot_budget // n)
        pilot_estimates = []
        pilot_variances = []
        
        for k in range(n):
            samples = []
            for _ in range(pilot_per_stratum):
                S = self._sample_k_subset(feature_idx, k)
                marginal = self._marginal_contribution(feature_idx, S)
                samples.append(marginal)
            
            pilot_estimates.append(np.mean(samples))
            pilot_variances.append(np.var(samples, ddof=1) if len(samples) > 1 else 0.0)
        
        # Phase 2: Neyman allocation based on pilot variances
        pilot_std = np.sqrt(np.array(pilot_variances))
        
        # Avoid division by zero
        pilot_std = np.maximum(pilot_std, 1e-10)
        
        # Compute Neyman allocation: n_k ∝ σ_k
        total_std = np.sum(pilot_std)
        if total_std > 0:
            allocation = (pilot_std / total_std * (total_budget - pilot_budget)).astype(int)
        else:
            # Fallback to uniform allocation
            allocation = np.full(n, (total_budget - pilot_budget) // n, dtype=int)
        
        # Ensure minimum 1 sample per stratum
        allocation = np.maximum(allocation, 1)
        
        # Adjust to match budget exactly
        while np.sum(allocation) < total_budget - pilot_budget:
            # Add to stratum with highest variance
            allocation[np.argmax(pilot_variances)] += 1
        while np.sum(allocation) > total_budget - pilot_budget:
            # Remove from stratum with lowest variance (but keep >= 1)
            removable = np.where(allocation > 1)[0]
            if len(removable) > 0:
                idx = removable[np.argmin(pilot_variances[removable])]
                allocation[idx] -= 1
            else:
                break
        
        # Phase 3: Main sampling with Neyman allocation
        stratum_estimates = []
        stratum_counts = []
        
        for k in range(n):
            n_k = allocation[k]
            samples = []
            
            for _ in range(n_k):
                S = self._sample_k_subset(feature_idx, k)
                marginal = self._marginal_contribution(feature_idx, S)
                samples.append(marginal)
            
            stratum_estimates.append(np.mean(samples) if len(samples) > 0 else 0.0)
            stratum_counts.append(n_k)
        
        # Combine with pilot estimates (use weighted average)
        combined_estimates = []
        for k in range(n):
            pilot_weight = pilot_per_stratum
            main_weight = stratum_counts[k]
            total_weight = pilot_weight + main_weight
            
            combined = (pilot_estimates[k] * pilot_weight + stratum_estimates[k] * main_weight) / total_weight
            combined_estimates.append(combined)
        
        # Compute final estimate
        phi = sum(combined_estimates) / n
        
        # Estimate variance using Neyman formula
        variance_est = sum(
            (pilot_std[k] ** 2) / (pilot_per_stratum + stratum_counts[k])
            for k in range(n)
        ) / (n ** 2)
        
        stats = {
            'pilot_budget': pilot_budget,
            'pilot_variances': pilot_variances,
            'allocation': allocation.tolist(),
            'stratum_estimates': combined_estimates,
            'total_samples': pilot_budget + np.sum(allocation)
        }
        
        return phi, variance_est, stats
    
    def compute(self, feature_idx: int, budget: int, seed: int = None) -> float:
        """Simple interface returning just the estimate."""
        phi, _, _ = self.compute_with_neyman(feature_idx, budget, seed=seed)
        return phi
