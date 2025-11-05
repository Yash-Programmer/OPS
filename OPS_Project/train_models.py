"""
Train all 6 ML models on all 6 datasets for Phase 3.
Saves trained models and performance metrics.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
import time
warnings.filterwarnings('ignore')

# ML models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import xgboost as xgb

# Setup
project_root = Path(__file__).parent
data_dir = project_root / 'data' / 'processed'
models_dir = project_root / 'data' / 'models'
results_dir = project_root / 'results'
models_dir.mkdir(exist_ok=True, parents=True)
results_dir.mkdir(exist_ok=True, parents=True)

def get_models(task_type='classification'):
    """Get 6 ML models for classification or regression."""
    if task_type == 'classification':
        return {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42, max_depth=6, eval_metric='logloss', use_label_encoder=False),
            'neural_net': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'decision_tree': DecisionTreeClassifier(max_depth=10, random_state=42)
        }
    else:
        return {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=6),
            'neural_net': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'svm': SVR(kernel='rbf'),
            'decision_tree': DecisionTreeRegressor(max_depth=10, random_state=42)
        }

def train_and_evaluate(X, y, model, task_type='classification'):
    """Train model and evaluate performance."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=y if task_type == 'classification' else None
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if task_type == 'classification':
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
    else:
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    
    return model, metrics, X_train, X_test, y_train, y_test

# Dataset configuration
dataset_config = {
    'iris': 'classification',
    'california_housing': 'regression',
    'adult_income': 'classification',
    'mnist_pca': 'classification',
    'synthetic_svm': 'classification',
    'non_submodular': 'regression'
}

dataset_names = list(dataset_config.keys())

# Load datasets
print("Loading datasets...")
datasets = {}
for name in dataset_names:
    with open(data_dir / f'{name}.pkl', 'rb') as f:
        datasets[name] = pickle.load(f)
    print(f"  ✅ {name}: X{datasets[name]['X'].shape}, y{datasets[name]['y'].shape}")

# Train all models
print("\n" + "=" * 80)
print("TRAINING MODELS ON ALL DATASETS")
print("=" * 80)

trained_models = {}
results_summary = []
total_trained = 0

for dataset_name in dataset_names:
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    X = datasets[dataset_name]['X']
    y = datasets[dataset_name]['y']
    task_type = dataset_config[dataset_name]
    
    print(f"Task: {task_type}, Shape: X{X.shape}, y{y.shape}")
    
    models = get_models(task_type)
    trained_models[dataset_name] = {}
    
    for model_name, model in models.items():
        print(f"  Training {model_name}...", end=' ', flush=True)
        
        try:
            t0 = time.time()
            trained_model, metrics, X_train, X_test, y_train, y_test = train_and_evaluate(
                X, y, model, task_type
            )
            train_time = time.time() - t0
            
            # Store
            trained_models[dataset_name][model_name] = {
                'model': trained_model,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'metrics': metrics,
                'task_type': task_type,
                'train_time': train_time
            }
            
            # Save
            model_path = models_dir / f'{dataset_name}_{model_name}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(trained_models[dataset_name][model_name], f)
            
            # Print
            if task_type == 'classification':
                print(f"Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, Time: {train_time:.2f}s ✅")
            else:
                print(f"MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.4f}, Time: {train_time:.2f}s ✅")
            
            results_summary.append({
                'dataset': dataset_name,
                'model': model_name,
                'task': task_type,
                'train_time': train_time,
                **metrics
            })
            total_trained += 1
            
        except Exception as e:
            print(f"FAILED: {e} ❌")

# Save summary
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results_summary)

print("\nCLASSIFICATION TASKS:")
print("-" * 80)
clf_results = results_df[results_df['task'] == 'classification']
if len(clf_results) > 0:
    print(clf_results[['dataset', 'model', 'accuracy', 'f1_score', 'train_time']].to_string(index=False))
    print(f"\nAverage Accuracy: {clf_results['accuracy'].mean():.4f}")
    print(f"Average F1 Score: {clf_results['f1_score'].mean():.4f}")

print("\n\nREGRESSION TASKS:")
print("-" * 80)
reg_results = results_df[results_df['task'] == 'regression']
if len(reg_results) > 0:
    print(reg_results[['dataset', 'model', 'mse', 'r2', 'train_time']].to_string(index=False))
    print(f"\nAverage R²: {reg_results['r2'].mean():.4f}")

# Save
summary_path = results_dir / 'model_training_summary.csv'
results_df.to_csv(summary_path, index=False)

print("\n" + "=" * 80)
print("✅ PHASE 3 COMPLETE: MODEL TRAINING")
print("=" * 80)
print(f"Total models trained: {total_trained}/{len(dataset_names) * 6}")
print(f"Models saved to: {models_dir}")
print(f"Summary saved to: {summary_path}")
print("\n🎯 Ready for Phase 4: Experimental Evaluation")
