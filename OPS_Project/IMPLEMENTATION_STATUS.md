# Orthogonal Permutation Sampling (OPS) for Shapley Values - Implementation

Research implementation of the paper "Orthogonal Permutation Sampling for Shapley Values" achieving 5-67× variance reduction over Monte Carlo methods.

## 📊 Project Status

**Implementation Progress: 30% Complete (3/10 Phases)**

- ✅ **Phase 1**: Environment & Dataset Preparation
- ✅ **Phase 2**: Core Algorithm Implementation  
- ✅ **Phase 3**: Model Training (36 models trained)
- 🔄 **Phase 4**: Experimental Evaluation (in progress)
- ⏳ **Phases 5-10**: Pending

## 🎯 Key Achievements

### Phase 1: Datasets
- 6 benchmark datasets generated (150-20,640 samples, 4-100 features)
- Iris (n=4), California Housing (n=8), Adult Income (n=14), MNIST-PCA (n=50), Synthetic-SVM (n=100), Non-submodular (n=10)

### Phase 2: Algorithms Implemented
1. **Monte Carlo Baseline** - Naive permutation sampling
2. **Position-Stratified (PS)** - Algorithm 1 with rank stratification
3. **Neyman Allocation** - Optimal budget allocation (Corollary 1)
4. **Orthogonal Permutation Sampling (OPS)** - Antithetic coupling (Algorithm 2)
5. **OPS with Control Variates (OPS-CV)** - Linearized surrogate (Algorithm 3)

### Phase 3: Models Trained
- **36 models** across 6 datasets × 6 model types
- Model types: Logistic/Linear Regression, Random Forest, XGBoost, Neural Network, SVM, Decision Tree
- **Classification performance**: 83.1% avg accuracy, 83.0% avg F1
- **Regression performance**: 47.5% avg R²

## 📁 Project Structure

```
OPS_Project/
├── src/
│   ├── algorithms/          # Core OPS algorithms
│   │   ├── shapley_base.py
│   │   ├── position_stratified.py
│   │   └── __init__.py
│   └── baselines/           # Comparison methods
│       ├── shap_baselines.py
│       └── __init__.py
├── data/
│   ├── processed/           # 6 benchmark datasets (.pkl)
│   └── models/              # 36 trained models (.pkl)
├── results/
│   ├── experiments/         # Experimental results
│   └── model_training_summary.csv
├── notebooks/
│   ├── OPS_Implementation_Phase1.ipynb
│   ├── OPS_Implementation_Phase2.ipynb
│   └── OPS_Implementation_Phase3.ipynb
├── generate_datasets.py     # Dataset generation script
├── train_models.py          # Model training script
├── test_algorithms.py       # Algorithm validation
└── run_experiments.py       # Experimental evaluation
```

## 🚀 Quick Start

### 1. Generate Datasets
```bash
python generate_datasets.py
```

### 2. Train Models
```bash
python train_models.py
```

### 3. Test Algorithms
```bash
python test_algorithms.py
```

### 4. Run Experiments
```bash
python run_experiments.py
```

## 📈 Model Performance Summary

### Classification Tasks (24 models)
| Dataset | Best Model | Accuracy | F1 Score |
|---------|------------|----------|----------|
| Iris | Neural Net | 1.000 | 1.000 |
| Adult Income | Neural Net | 0.964 | 0.964 |
| MNIST-PCA | SVM | 0.828 | 0.828 |
| Synthetic-SVM | SVM | 0.940 | 0.940 |

**Average**: 83.1% accuracy, 83.0% F1 score

### Regression Tasks (12 models)
| Dataset | Best Model | MSE | R² |
|---------|------------|-----|-----|
| California Housing | XGBoost | 0.223 | 0.830 |
| Non-submodular | Neural Net | 0.718 | 0.787 |

**Average**: R² = 0.475

## 🔬 Algorithms Overview

### 1. Monte Carlo (MC) Baseline
- Naive uniform permutation sampling
- Reference baseline for variance reduction

### 2. Position-Stratified Shapley (PS)
- Stratifies by feature rank k ∈ {0, ..., n-1}
- **Theorem 1**: Eliminates between-stratum variance
- Uniform budget allocation: L_k = L/n

### 3. Neyman Allocation
- Optimal allocation proportional to stratum std dev
- **Corollary 1**: L_k* = L · σ_k / Σ(σ_j)
- Two-phase: pilot + allocation

### 4. Orthogonal Permutation Sampling (OPS)
- Antithetic pairs: π and π^⊥
- **Theorem 3**: ≥2× variance reduction when negatively correlated
- Construction: π^⊥(j) = n - 1 - π(n - 1 - j)

### 5. OPS with Control Variates (OPS-CV)
- Uses linearized surrogate model
- **Theorem 4**: Var[φ_CV] = Var[φ_OPS](1 - ρ²)
- Additional reduction proportional to correlation ρ

## 📊 Expected Results (from paper)

- **Variance Reduction**: 5-67× over Monte Carlo
- **MSE Reduction**: 2-5× lower than KernelSHAP
- **Computation**: Comparable cost to naive MC
- **Datasets**: Effective across n=4 to n=100 features

## 🔧 Technical Details

### Dependencies
- Python 3.10+
- NumPy, pandas, scikit-learn
- XGBoost, SHAP
- matplotlib, seaborn (for visualization)

### Experimental Configuration
- **Budgets**: L ∈ {100, 500, 1000, 2500, 5000}
- **Trials**: 30-50 per configuration
- **Metrics**: Variance, MSE, computation time
- **Baselines**: KernelSHAP, TreeExplainer

## 📝 Implementation Notes

### Phase 1 Completed
- All datasets generated with proper preprocessing
- Train/test splits with stratification
- Dataset statistics validated

### Phase 2 Completed
- All 5 algorithms implemented and tested
- Modular class hierarchy (inheritance from ShapleyEstimator)
- Exact Shapley computation for n≤10 validation
- Test script confirms correctness on linear model

### Phase 3 Completed
- 36 models trained successfully
- Models saved with train/test splits
- Performance metrics recorded
- Ready for Shapley value experiments

### Phase 4 In Progress
- Experimental framework created
- Running initial MC vs PS comparison
- 1,800 total experiments planned
- Results saved incrementally

## 🎯 Next Steps

### Phase 4: Complete Experimental Evaluation
- [ ] Finish MC vs PS experiments (6 configs)
- [ ] Add Neyman, OPS, OPS-CV to comparison
- [ ] Expand to all 36 model/dataset pairs
- [ ] Compute variance reduction factors

### Phase 5: Visualization & Analysis
- [ ] Variance vs budget plots
- [ ] VRF heatmaps by dataset/model
- [ ] MSE comparison with baselines
- [ ] Computation time analysis

### Phase 6-10: Advanced Analysis
- [ ] Ablation studies
- [ ] Non-submodular game analysis
- [ ] High-dimensional experiments
- [ ] Optimization & parallelization
- [ ] Final validation & presentation

## 📚 References

**Paper**: "Orthogonal Permutation Sampling for Shapley Values" (Yash Varshney, 2025)

**Key Contributions**:
1. Position stratification with variance decomposition
2. Neyman allocation for optimal sampling
3. Orthogonal permutation coupling (antithetic variance reduction)
4. Control variate acceleration

## 🤝 Contact

Research implementation by: Yash Varshney  
Date: November 2025

---

**Status Last Updated**: November 5, 2025
