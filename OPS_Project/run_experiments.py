"""
Phase 4: Experimental Evaluation Framework
Run comprehensive experiments: 6 datasets × 6 models × 5 algorithms × 5 budgets
"""

import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from algorithms.shapley_base import ShapleyEstimator
from algorithms.position_stratified import PositionStratifiedShapley
from baselines.shap_baselines import BaselineMethods

# Directories
models_dir = project_root / 'data' / 'models'
results_dir = project_root / 'results' / 'experiments'
results_dir.mkdir(exist_ok=True, parents=True)

# Experimental configuration
BUDGETS = [100, 500, 1000, 2500, 5000]
N_TRIALS = 30  # Trials per configuration for variance estimation
ALGORITHMS = ['mc', 'ps']  # Start with MC and PS for quick validation

# Dataset/model pairs to evaluate
EVAL_CONFIGS = [
    ('iris', 'random_forest'),
    ('iris', 'neural_net'),
    ('california_housing', 'xgboost'),
    ('adult_income', 'random_forest'),
    ('mnist_pca', 'svm'),
    ('synthetic_svm', 'neural_net')
]

print("=" * 80)
print("PHASE 4: EXPERIMENTAL EVALUATION")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Budgets: {BUDGETS}")
print(f"  Trials per config: {N_TRIALS}")
print(f"  Algorithms: {ALGORITHMS}")
print(f"  Evaluation pairs: {len(EVAL_CONFIGS)}")
print(f"  Total experiments: {len(EVAL_CONFIGS) * len(ALGORITHMS) * len(BUDGETS) * N_TRIALS}")

def load_model_data(dataset_name, model_name):
    """Load trained model and data splits."""
    model_path = models_dir / f'{dataset_name}_{model_name}.pkl'
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def run_experiment(model, X_test, feature_idx, algorithm, budget, seed):
    """
    Run single experiment trial.
    
    Returns:
        dict with estimate, time, and any algorithm-specific metrics
    """
    baseline = np.zeros(X_test.shape[1])
    
    try:
        if algorithm == 'mc':
            estimator = ShapleyEstimator(model, X_test[[0]], baseline)
            t0 = time.time()
            phi = estimator.mc_shapley(feature_idx, n_samples=budget, seed=seed)
            elapsed = time.time() - t0
            
            return {
                'phi': float(phi),
                'time': elapsed,
                'variance_est': None,
                'success': True
            }
            
        elif algorithm == 'ps':
            estimator = PositionStratifiedShapley(model, X_test[[0]], baseline)
            t0 = time.time()
            phi, var_theoretical, stats = estimator.compute_with_variance(feature_idx, budget, seed=seed)
            elapsed = time.time() - t0
            
            return {
                'phi': float(phi),
                'time': elapsed,
                'variance_est': float(var_theoretical),
                'success': True
            }
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
    except Exception as e:
        return {
            'phi': np.nan,
            'time': 0.0,
            'variance_est': None,
            'success': False,
            'error': str(e)
        }

def evaluate_configuration(dataset_name, model_name, config_num, total_configs, feature_idx=0):
    """
    Evaluate all algorithms on one dataset/model pair.
    """
    print(f"\n{'='*80}")
    print(f"[{config_num}/{total_configs}] Evaluating: {dataset_name} / {model_name}")
    print(f"{'='*80}")
    
    # Load model
    model_data = load_model_data(dataset_name, model_name)
    model = model_data['model']
    X_test = model_data['X_test']
    
    # Use first test sample, first feature
    print(f"  Test sample shape: {X_test.shape}")
    print(f"  Feature to explain: {feature_idx}")
    
    results = []
    
    for budget in BUDGETS:
        print(f"\n  Budget: {budget}")
        
        for algorithm in ALGORITHMS:
            print(f"    {algorithm.upper()}:", end=' ', flush=True)
            
            trial_results = []
            trial_times = []
            errors = []
            
            for trial in range(N_TRIALS):
                if (trial + 1) % 10 == 0:
                    print(f".{trial+1}", end='', flush=True)
                elif (trial + 1) % 5 == 0:
                    print(".", end='', flush=True)
                
                seed = 42 + trial
                result = run_experiment(model, X_test, feature_idx, algorithm, budget, seed)
                
                if result['success']:
                    trial_results.append(result['phi'])
                    trial_times.append(result['time'])
                else:
                    errors.append(result.get('error', 'Unknown'))
            
            if len(trial_results) == 0:
                print(f" ❌ All trials failed: {errors[0] if errors else 'Unknown error'}")
                continue
            
            # Compute statistics
            estimates = np.array(trial_results)
            mean_estimate = np.mean(estimates)
            empirical_variance = np.var(estimates)
            empirical_std = np.std(estimates)
            mean_time = np.mean(trial_times)
            
            status = f"✅ ({len(trial_results)}/{N_TRIALS})" if len(errors) == 0 else f"⚠️  ({len(trial_results)}/{N_TRIALS})"
            print(f" {status} Mean: {mean_estimate:.6f}, Var: {empirical_variance:.8f}, Time: {mean_time*1000:.2f}ms")
            
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
    
    print(f"\n✅ [{config_num}/{total_configs}] COMPLETED: {dataset_name} / {model_name}")
    return results

# Run all experiments
print("\n" + "=" * 80)
print("RUNNING EXPERIMENTS")
print("=" * 80)

all_results = []
results_path = results_dir / 'phase4_initial_results.csv'

for idx, (dataset_name, model_name) in enumerate(EVAL_CONFIGS, 1):
    try:
        config_results = evaluate_configuration(dataset_name, model_name, idx, len(EVAL_CONFIGS))
        all_results.extend(config_results)
        
        # Save incremental results after each configuration
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_path, index=False)
        print(f"  💾 Saved {len(all_results)} results to {results_path.name}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Saving partial results...")
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(results_path, index=False)
            print(f"  💾 Saved {len(all_results)} partial results")
        raise
    except Exception as e:
        print(f"  ❌ Error in {dataset_name}/{model_name}: {e}")
        continue

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

# Variance Reduction Analysis
print("\nVariance Reduction Factors (MC baseline):")
print("-" * 80)

for budget in BUDGETS:
    budget_results = results_df[results_df['budget'] == budget]
    
    mc_var = budget_results[budget_results['algorithm'] == 'mc']['empirical_variance'].mean()
    ps_var = budget_results[budget_results['algorithm'] == 'ps']['empirical_variance'].mean()
    
    vrf_ps = mc_var / ps_var if ps_var > 0 else float('inf')
    
    print(f"  Budget {budget:5d}: PS VRF = {vrf_ps:6.2f}×")

print(f"\n✅ Phase 4 Initial Experiments Complete")
print(f"   Results saved to: {results_path}")
print(f"   Total experiments run: {len(all_results)}")
print(f"   Next: Add remaining algorithms (Neyman, OPS, OPS-CV)")
