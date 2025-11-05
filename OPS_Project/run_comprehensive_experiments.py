"""
Phase 7: Comprehensive Experimental Evaluation
Tests all 5 algorithms (MC, PS, Neyman, OPS, OPS-CV) across all datasets
"""

import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path

from src.algorithms import (
    ShapleyEstimator,
    PositionStratifiedShapley,
    NeymanAllocationShapley,
    OPSAntitheticShapley,
    OPSControlVariatesShapley
)

# Configuration
BUDGETS = [100, 500, 1000, 2500, 5000]
N_TRIALS = 30
ALGORITHMS = ['mc', 'ps', 'neyman', 'ops', 'ops_cv']

# Dataset/model pairs to evaluate
EVAL_CONFIGS = [
    ('iris', 'random_forest'),
    ('iris', 'neural_net'),
    ('california_housing', 'xgboost'),
    ('adult_income', 'random_forest'),
    ('mnist_pca', 'svm'),
    ('synthetic_svm', 'neural_net'),
]

def load_model_data(dataset_name, model_name):
    """Load trained model and test data."""
    model_path = Path('data') / 'models' / f'{dataset_name}_{model_name}.pkl'
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data

def run_experiment(model, X_test, feature_idx, algorithm, budget, seed):
    """
    Run single experiment trial.
    
    Returns:
        dict with estimate, time, and variance metrics
    """
    baseline = np.zeros(X_test.shape[1])
    
    if algorithm == 'mc':
        estimator = ShapleyEstimator(model, X_test[[0]], baseline)
        t0 = time.time()
        phi = estimator.mc_shapley(feature_idx, n_samples=budget, seed=seed)
        elapsed = time.time() - t0
        
        return {
            'phi': phi,
            'time': elapsed,
            'variance_est': None
        }
        
    elif algorithm == 'ps':
        estimator = PositionStratifiedShapley(model, X_test[[0]], baseline)
        t0 = time.time()
        phi, var_theoretical, stats = estimator.compute_with_variance(feature_idx, budget, seed=seed)
        elapsed = time.time() - t0
        
        return {
            'phi': phi,
            'time': elapsed,
            'variance_est': var_theoretical
        }
    
    elif algorithm == 'neyman':
        estimator = NeymanAllocationShapley(model, X_test[[0]], baseline)
        t0 = time.time()
        phi, var_est, stats = estimator.compute_with_neyman(feature_idx, budget, seed=seed)
        elapsed = time.time() - t0
        
        return {
            'phi': phi,
            'time': elapsed,
            'variance_est': var_est
        }
    
    elif algorithm == 'ops':
        estimator = OPSAntitheticShapley(model, X_test[[0]], baseline)
        t0 = time.time()
        phi, var_est, stats = estimator.compute_with_variance(feature_idx, budget, seed=seed)
        elapsed = time.time() - t0
        
        return {
            'phi': phi,
            'time': elapsed,
            'variance_est': var_est
        }
    
    elif algorithm == 'ops_cv':
        estimator = OPSControlVariatesShapley(model, X_test[[0]], baseline)
        t0 = time.time()
        phi, var_est, stats = estimator.compute_with_cv(feature_idx, budget, seed=seed)
        elapsed = time.time() - t0
        
        return {
            'phi': phi,
            'time': elapsed,
            'variance_est': var_est
        }
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

def evaluate_configuration(dataset_name, model_name, feature_idx=0):
    """
    Evaluate all algorithms on one dataset/model pair.
    """
    config_id = f"{dataset_name}/{model_name}"
    print(f"\n{'='*80}")
    print(f"Configuration: {config_id}")
    print(f"{'='*80}")
    
    # Load model
    model_data = load_model_data(dataset_name, model_name)
    model = model_data['model']
    X_test = model_data['X_test']
    
    n_features = X_test.shape[1]
    print(f"  Features: {n_features}, Test samples: {len(X_test)}")
    
    results = []
    total_experiments = len(BUDGETS) * len(ALGORITHMS) * N_TRIALS
    completed = 0
    
    for budget in BUDGETS:
        print(f"\n  Budget: {budget}")
        
        for algorithm in ALGORITHMS:
            estimates = []
            times = []
            
            print(f"    {algorithm.upper():8s}: ", end='', flush=True)
            
            try:
                for trial in range(N_TRIALS):
                    seed = 42 + trial
                    result = run_experiment(model, X_test, feature_idx, algorithm, budget, seed)
                    
                    estimates.append(result['phi'])
                    times.append(result['time'])
                    
                    # Progress indicator
                    if (trial + 1) % 10 == 0:
                        print(f"{trial+1}", end=' ', flush=True)
                    
                    completed += 1
                
                # Compute statistics
                mean_estimate = np.mean(estimates)
                empirical_variance = np.var(estimates, ddof=1) if len(estimates) > 1 else 0.0
                empirical_std = np.std(estimates, ddof=1) if len(estimates) > 1 else 0.0
                mean_time = np.mean(times) * 1000  # Convert to ms
                
                results.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'algorithm': algorithm,
                    'budget': budget,
                    'n_trials': N_TRIALS,
                    'mean_estimate': mean_estimate,
                    'empirical_variance': empirical_variance,
                    'empirical_std': empirical_std,
                    'mean_time': mean_time,
                    'feature_idx': feature_idx
                })
                
                print(f"✅ Mean: {mean_estimate:.6f}, Var: {empirical_variance:.8f}, Time: {mean_time:.2f}ms")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                completed += N_TRIALS  # Skip trials
                continue
    
    progress_pct = (completed / total_experiments) * 100
    print(f"\n  Progress: {completed}/{total_experiments} ({progress_pct:.1f}%)")
    
    return results

def main():
    """Run comprehensive evaluation."""
    print("="*80)
    print("PHASE 7: COMPREHENSIVE EXPERIMENTAL EVALUATION")
    print("="*80)
    
    total_experiments = len(EVAL_CONFIGS) * len(BUDGETS) * len(ALGORITHMS) * N_TRIALS
    print(f"\nConfiguration:")
    print(f"  Budgets: {BUDGETS}")
    print(f"  Trials per config: {N_TRIALS}")
    print(f"  Algorithms: {ALGORITHMS}")
    print(f"  Evaluation pairs: {len(EVAL_CONFIGS)}")
    print(f"  Total experiments: {total_experiments}")
    
    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80)
    
    all_results = []
    
    for idx, (dataset_name, model_name) in enumerate(EVAL_CONFIGS, 1):
        print(f"\n[{idx}/{len(EVAL_CONFIGS)}] Starting: {dataset_name}/{model_name}")
        
        config_results = evaluate_configuration(dataset_name, model_name)
        all_results.extend(config_results)
        
        # Save incremental results
        df = pd.DataFrame(all_results)
        output_path = Path('results') / 'experiments' / 'phase7_comprehensive_results.csv'
        df.to_csv(output_path, index=False)
        print(f"  💾 Saved {len(all_results)} results to {output_path}")
    
    print("\n" + "="*80)
    print("PHASE 7 COMPLETE")
    print("="*80)
    print(f"\nTotal experiments: {len(all_results)}")
    print(f"Results saved to: results/experiments/phase7_comprehensive_results.csv")

if __name__ == '__main__':
    main()
