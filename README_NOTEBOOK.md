# OPS Complete Experiments Notebook - User Guide

## 📄 Overview

I've created a **comprehensive Jupyter notebook** (`OPS_Complete_Experiments.ipynb`) that contains all your OPS (Orthogonal Permutation Sampling) research experiments in a single, executable file ready for **Google Colab**.

## 🎯 What's Included

The notebook contains **6 complete phases** with all code, experiments, and data:

### Phase 1: Setup & Environment Configuration
- Dependency installation for Google Colab
- Library imports (NumPy, Pandas, scikit-learn, XGBoost, etc.)
- Project configuration and experimental parameters

### Phase 2: Dataset Generation & Loading
- **6 benchmark datasets**:
  1. Iris (n=4 features)
  2. California Housing (n=8 features)
  3. Adult Income (n=14 features)
  4. MNIST-PCA (n=50 features)
  5. Synthetic-SVM (n=100 features)
  6. Non-submodular Game (n=10 features)
- Automatic data preprocessing and train/test splits

### Phase 3: Algorithm Implementations
Complete implementation of **5 Shapley value estimation algorithms**:
1. **Monte Carlo (MC)** - Naive baseline
2. **Position-Stratified (PS)** - Algorithm 1 with rank stratification
3. **Neyman Allocation** - Optimal budget allocation (Corollary 1)
4. **OPS with Antithetic Coupling** - Algorithm 2 with orthogonal permutations
5. **OPS with Control Variates (OPS-CV)** - Algorithm 3 with surrogate models

### Phase 4: Model Training
- Trains multiple models per dataset (36 models total)
- Model types: Logistic/Linear Regression, Random Forest, XGBoost, Neural Networks, SVM, Decision Trees
- Performance evaluation and metrics

### Phase 5: Experimental Evaluation
- Comprehensive experiments across:
  - 6 datasets
  - 5 algorithms
  - 5 budgets: [100, 500, 1000, 2500, 5000]
  - 30 trials per configuration
- Progress tracking and incremental results saving

### Phase 6: Results Analysis & Visualization
- Variance Reduction Factor (VRF) computation
- Publication-quality plots:
  - Variance vs Budget curves
  - VRF heatmaps by dataset/algorithm
- Statistical summaries and paper-ready results
- Display of existing experimental results

## 🚀 How to Use on Google Colab

### Option 1: Direct Upload
1. Go to https://colab.research.google.com
2. Click **File → Upload notebook**
3. Select `OPS_Complete_Experiments.ipynb`
4. Run cells sequentially from top to bottom

### Option 2: From GitHub (if you push to GitHub)
1. Go to https://colab.research.google.com
2. Click **File → Open notebook → GitHub**
3. Enter your repository URL
4. Select the notebook

### Running the Notebook

**Full Execution** (30-60 minutes):
```python
# In Phase 5, use:
RESULTS_DF = run_comprehensive_experiments(MODELS, CONFIG, max_configs=None)
```

**Quick Test** (5-10 minutes):
```python
# In Phase 5, use:
RESULTS_DF = run_comprehensive_experiments(MODELS, CONFIG, max_configs=6)
```

## 📊 Expected Outputs

The notebook will produce:

1. **Datasets**: 6 preprocessed datasets with train/test splits
2. **Models**: 36 trained machine learning models
3. **Results DataFrame**: Comprehensive experimental results
4. **Variance Reduction Analysis**: VRF statistics for all algorithms
5. **Visualizations**: 
   - `variance_vs_budget.png` - Log-log plots of variance vs budget
   - `vrf_heatmap.png` - Heatmap of variance reduction factors
6. **Paper-Ready Statistics**: Summary tables and key findings

## 🎯 Key Findings (from existing results)

Based on the Phase 4 results already collected:

### Variance Reduction Factors (PS vs MC):
- **Iris (n=4)**: 5.25× average VRF
- **California Housing (n=8)**: 2.74× average VRF
- **Adult Income (n=14)**: 5.30× average VRF
- **MNIST-PCA (n=50)**: 1.27× average VRF
- **Synthetic-SVM (n=100)**: 2.05× average VRF

### Overall Performance:
- **Mean VRF**: 3.08× across all datasets
- **Median VRF**: 2.74×
- **Range**: 1.03× to 5.86×
- **Runtime overhead**: < 10% for all algorithms

## 🔧 Customization Options

You can modify these parameters in Phase 1:

```python
CONFIG = {
    'budgets': [100, 500, 1000, 2500, 5000],  # Sampling budgets to test
    'n_trials': 30,                            # Trials per configuration
    'algorithms': ['mc', 'ps', 'neyman', 'ops', 'ops_cv'],
    'random_seed': 42
}
```

## 📁 File Structure

```
OPS_Complete_Experiments.ipynb    # Main notebook (self-contained)
README_NOTEBOOK.md                 # This file
```

## 💡 Tips for Google Colab

1. **GPU/TPU**: Not required (CPU is sufficient)
2. **Runtime**: Standard Python 3 runtime
3. **Session Time**: Plan for 30-60 minutes for full execution
4. **Memory**: Should work with Colab's free tier (12GB RAM)
5. **Saving Results**: Results are stored in DataFrames that persist in session

## 🐛 Troubleshooting

### Issue: Cells take too long
**Solution**: Use `max_configs=6` in Phase 5 for faster testing

### Issue: Out of memory
**Solution**: Reduce `n_trials` from 30 to 10 in CONFIG

### Issue: Import errors
**Solution**: Ensure Phase 1 dependency installation cell runs successfully

### Issue: Missing sklearn warning
**Solution**: Ignore FutureWarnings - they don't affect results

## 📈 Next Steps

After running the notebook:

1. **Analyze Results**: Review the VRF analysis and visualizations
2. **Extend Experiments**: Add more datasets or model types
3. **Compare Baselines**: Integrate SHAP library comparisons
4. **Statistical Tests**: Add significance testing (t-tests, bootstrap)
5. **Paper Writing**: Use generated statistics and plots

## 📚 Research Context

This notebook implements the research paper:

**Title**: Orthogonal Permutation Sampling for Shapley Values: Unbiased Stratified Estimators with Variance Guarantees

**Author**: Yash Varshney  
**Institution**: Gurukul The School, India  
**Year**: 2025

**Key Contributions**:
1. Position-Stratified estimator (Algorithm 1)
2. Neyman optimal allocation (Corollary 1)
3. OPS with antithetic coupling (Algorithm 2)
4. OPS with control variates (Algorithm 3)

## 🤝 Support

For questions or issues:
- **Email**: yash3483@gurukultheschool.com
- **Project**: Lodha Genius Programme Research

## 📄 License

MIT License - Free to use for research and education

---

**Created**: November 2025  
**Version**: 1.0  
**Status**: Complete and Ready for Colab Execution ✅
