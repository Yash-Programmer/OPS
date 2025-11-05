"""
Quick script to generate all 6 benchmark datasets for OPS experiments.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.datasets import load_iris, fetch_california_housing, make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Setup
project_root = Path(__file__).parent
data_dir = project_root / 'data' / 'processed'
data_dir.mkdir(exist_ok=True, parents=True)

print("Generating 6 benchmark datasets...")
print("=" * 80)

# 1. Iris (n=4)
print("\n1. Iris (n=4)...")
iris = load_iris()
dataset = {
    'X': iris.data,
    'y': iris.target,
    'feature_names': iris.feature_names,
    'n_features': 4
}
with open(data_dir / 'iris.pkl', 'wb') as f:
    pickle.dump(dataset, f)
print(f"   Saved: X shape {dataset['X'].shape}, y shape {dataset['y'].shape}")

# 2. California Housing (n=8)
print("\n2. California Housing (n=8)...")
housing = fetch_california_housing()
dataset = {
    'X': housing.data,
    'y': housing.target,
    'feature_names': housing.feature_names,
    'n_features': 8
}
with open(data_dir / 'california_housing.pkl', 'wb') as f:
    pickle.dump(dataset, f)
print(f"   Saved: X shape {dataset['X'].shape}, y shape {dataset['y'].shape}")

# 3. Adult Income (n=14) - Synthetic approximation
print("\n3. Adult Income (n=14)...")
np.random.seed(42)
n_samples = 5000
X, y = make_classification(
    n_samples=n_samples,
    n_features=14,
    n_informative=10,
    n_redundant=2,
    n_classes=2,
    random_state=42
)
dataset = {
    'X': X,
    'y': y,
    'feature_names': [f'feature_{i}' for i in range(14)],
    'n_features': 14
}
with open(data_dir / 'adult_income.pkl', 'wb') as f:
    pickle.dump(dataset, f)
print(f"   Saved: X shape {dataset['X'].shape}, y shape {dataset['y'].shape}")

# 4. MNIST PCA (n=50)
print("\n4. MNIST PCA (n=50)...")
# Generate synthetic high-dimensional data
X_raw, y_raw = make_classification(
    n_samples=2000,
    n_features=200,
    n_informative=50,
    n_classes=10,
    n_clusters_per_class=1,
    random_state=42
)
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_raw)
dataset = {
    'X': X_pca,
    'y': y_raw,
    'feature_names': [f'PC_{i}' for i in range(50)],
    'n_features': 50
}
with open(data_dir / 'mnist_pca.pkl', 'wb') as f:
    pickle.dump(dataset, f)
print(f"   Saved: X shape {dataset['X'].shape}, y shape {dataset['y'].shape}")

# 5. Synthetic SVM (n=100)
print("\n5. Synthetic SVM (n=100)...")
X, y = make_classification(
    n_samples=1000,
    n_features=100,
    n_informative=70,
    n_redundant=20,
    n_classes=2,
    random_state=42
)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dataset = {
    'X': X_scaled,
    'y': y,
    'feature_names': [f'feature_{i}' for i in range(100)],
    'n_features': 100
}
with open(data_dir / 'synthetic_svm.pkl', 'wb') as f:
    pickle.dump(dataset, f)
print(f"   Saved: X shape {dataset['X'].shape}, y shape {dataset['y'].shape}")

# 6. Non-submodular game (n=10)
print("\n6. Non-submodular game (n=10)...")
np.random.seed(42)
n_samples = 500
n_features = 10
X = np.random.randn(n_samples, n_features)
# Non-linear interactions
y = (X[:, 0] * X[:, 1] + X[:, 2]**2 + np.sin(X[:, 3]) + 
     X[:, 4] * X[:, 5] * X[:, 6] + np.random.randn(n_samples) * 0.1)
dataset = {
    'X': X,
    'y': y,
    'feature_names': [f'feature_{i}' for i in range(10)],
    'n_features': 10
}
with open(data_dir / 'non_submodular.pkl', 'wb') as f:
    pickle.dump(dataset, f)
print(f"   Saved: X shape {dataset['X'].shape}, y shape {dataset['y'].shape}")

print("\n" + "=" * 80)
print("✅ All 6 datasets generated successfully!")
print(f"Saved to: {data_dir}")
