"""
Test all 5 algorithms to ensure they work correctly before running full experiments
"""

import numpy as np
import pickle
from pathlib import Path

from src.algorithms import (
    ShapleyEstimator,
    PositionStratifiedShapley,
    NeymanAllocationShapley,
    OPSAntitheticShapley,
    OPSControlVariatesShapley
)

print("="*80)
print("TESTING ALL ALGORITHMS")
print("="*80)

# Load a simple model for testing
model_path = Path('data') / 'models' / 'iris_random_forest.pkl'
with open(model_path, 'rb') as f:
    data = pickle.load(f)

model = data['model']
X_test = data['X_test']
baseline = np.zeros(X_test.shape[1])

print(f"\nTest model: iris/random_forest")
print(f"Features: {X_test.shape[1]}, Test samples: {len(X_test)}")

feature_idx = 0
budget = 100
seed = 42

print(f"\nTesting with feature_idx={feature_idx}, budget={budget}, seed={seed}")
print("-"*80)

# Test 1: Monte Carlo
print("\n1. Monte Carlo (MC):")
try:
    estimator = ShapleyEstimator(model, X_test[[0]], baseline)
    phi = estimator.mc_shapley(feature_idx, n_samples=budget, seed=seed)
    print(f"   ✅ MC estimate: {phi:.6f}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Position Stratified
print("\n2. Position Stratified (PS):")
try:
    estimator = PositionStratifiedShapley(model, X_test[[0]], baseline)
    phi, var, stats = estimator.compute_with_variance(feature_idx, budget, seed=seed)
    print(f"   ✅ PS estimate: {phi:.6f}, variance: {var:.8f}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Neyman Allocation
print("\n3. Neyman Allocation:")
try:
    estimator = NeymanAllocationShapley(model, X_test[[0]], baseline)
    phi, var, stats = estimator.compute_with_neyman(feature_idx, budget, pilot_budget=20, seed=seed)
    print(f"   ✅ Neyman estimate: {phi:.6f}, variance: {var:.8f}")
    print(f"   Allocation: {stats['allocation']}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: OPS Antithetic
print("\n4. OPS Antithetic:")
try:
    estimator = OPSAntitheticShapley(model, X_test[[0]], baseline)
    phi, var, stats = estimator.compute_with_variance(feature_idx, budget, seed=seed)
    print(f"   ✅ OPS estimate: {phi:.6f}, variance: {var:.8f}")
    print(f"   Pairs: {stats['n_pairs']}, VR: {stats['variance_reduction']:.2f}x")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: OPS Control Variates
print("\n5. OPS Control Variates (OPS-CV):")
try:
    estimator = OPSControlVariatesShapley(model, X_test[[0]], baseline)
    phi, var, stats = estimator.compute_with_cv(feature_idx, budget, seed=seed)
    print(f"   ✅ OPS-CV estimate: {phi:.6f}, variance: {var:.8f}")
    print(f"   Main budget: {stats['main_budget']}, CV budget: {stats['cv_budget']}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("ALGORITHM TESTING COMPLETE")
print("="*80)
