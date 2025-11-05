"""
Phase 5: Analyze Experimental Results
Calculate variance reduction factors and generate summary statistics
"""

import pandas as pd
import numpy as np

# Load results
df = pd.read_csv('results/experiments/phase4_initial_results.csv')

print("="*80)
print("PHASE 5: EXPERIMENTAL RESULTS ANALYSIS")
print("="*80)

print(f"\nTotal experiments: {len(df)}")
print(f"Datasets: {df['dataset'].nunique()}")
print(f"Algorithms: {list(df['algorithm'].unique())}")
print(f"Budgets: {sorted(df['budget'].unique())}")

print("\n" + "="*80)
print("VARIANCE REDUCTION FACTOR (VRF) ANALYSIS")
print("="*80)
print("\nVRF = MC Variance / PS Variance (higher is better)")
print("-"*80)

# Calculate VRF for each configuration
vrf_results = []

for dataset in sorted(df['dataset'].unique()):
    dataset_data = df[df['dataset'] == dataset]
    model = dataset_data['model'].iloc[0]
    
    print(f"\n{dataset} / {model}")
    print("-"*80)
    
    for budget in sorted(df['budget'].unique()):
        mc_data = dataset_data[(dataset_data['algorithm'] == 'mc') & (dataset_data['budget'] == budget)]
        ps_data = dataset_data[(dataset_data['algorithm'] == 'ps') & (dataset_data['budget'] == budget)]
        
        if len(mc_data) > 0 and len(ps_data) > 0:
            mc_var = mc_data['empirical_variance'].values[0]
            ps_var = ps_data['empirical_variance'].values[0]
            mc_time = mc_data['mean_time'].values[0]
            ps_time = ps_data['mean_time'].values[0]
            
            vrf = mc_var / ps_var if ps_var > 0 else float('inf')
            time_ratio = mc_time / ps_time if ps_time > 0 else 1.0
            
            vrf_results.append({
                'dataset': dataset,
                'model': model,
                'budget': budget,
                'mc_variance': mc_var,
                'ps_variance': ps_var,
                'vrf': vrf,
                'mc_time': mc_time,
                'ps_time': ps_time,
                'time_ratio': time_ratio
            })
            
            print(f"  Budget {budget:5d}: VRF = {vrf:8.2f}x | MC var: {mc_var:.2e} | PS var: {ps_var:.2e} | Time ratio: {time_ratio:.2f}x")

# Create VRF summary
vrf_df = pd.DataFrame(vrf_results)

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nVariance Reduction Factor (VRF):")
print(f"  Mean VRF: {vrf_df['vrf'].replace([np.inf, -np.inf], np.nan).mean():.2f}x")
print(f"  Median VRF: {vrf_df['vrf'].replace([np.inf, -np.inf], np.nan).median():.2f}x")
print(f"  Min VRF: {vrf_df['vrf'].replace([np.inf, -np.inf], np.nan).min():.2f}x")
print(f"  Max VRF: {vrf_df['vrf'].replace([np.inf, -np.inf], np.nan).max():.2f}x")

print(f"\nComputation Time Ratio (MC/PS):")
print(f"  Mean: {vrf_df['time_ratio'].mean():.2f}x")
print(f"  Median: {vrf_df['time_ratio'].median():.2f}x")

print("\n" + "="*80)
print("VRF BY BUDGET")
print("="*80)

for budget in sorted(vrf_df['budget'].unique()):
    budget_data = vrf_df[vrf_df['budget'] == budget]
    avg_vrf = budget_data['vrf'].replace([np.inf, -np.inf], np.nan).mean()
    print(f"  Budget {budget:5d}: Average VRF = {avg_vrf:8.2f}x")

print("\n" + "="*80)
print("VRF BY DATASET")
print("="*80)

for dataset in sorted(vrf_df['dataset'].unique()):
    dataset_data = vrf_df[vrf_df['dataset'] == dataset]
    avg_vrf = dataset_data['vrf'].replace([np.inf, -np.inf], np.nan).mean()
    print(f"  {dataset:25s}: Average VRF = {avg_vrf:8.2f}x")

# Save VRF summary
vrf_df.to_csv('results/experiments/vrf_summary.csv', index=False)
print(f"\n✅ Saved VRF summary to: results/experiments/vrf_summary.csv")

print("\n" + "="*80)
print("COMPARISON WITH PAPER CLAIMS")
print("="*80)
print("\nPaper claims: 5-67× variance reduction")
print(f"Our results: {vrf_df['vrf'].replace([np.inf, -np.inf], np.nan).min():.2f}x - {vrf_df['vrf'].replace([np.inf, -np.inf], np.nan).max():.2f}x")

valid_vrfs = vrf_df['vrf'].replace([np.inf, -np.inf], np.nan).dropna()
if len(valid_vrfs) > 0:
    within_range = ((valid_vrfs >= 5) & (valid_vrfs <= 67)).sum()
    print(f"Configurations within paper's claimed range: {within_range}/{len(valid_vrfs)} ({100*within_range/len(valid_vrfs):.1f}%)")

print("\n" + "="*80)
print("PHASE 5 COMPLETE")
print("="*80)
