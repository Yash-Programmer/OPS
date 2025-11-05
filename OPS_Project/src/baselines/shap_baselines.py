"""
Baseline Shapley estimation methods for comparison.
Implements KernelSHAP and TreeExplainer.
"""

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class BaselineMethods:
    """
    Wrapper for baseline Shapley estimation methods.
    """
    
    @staticmethod
    def kernelshap(model, X_background, X_explain, n_samples=1000):
        """
        Estimate Shapley values using KernelSHAP.
        
        Args:
            model: Trained model
            X_background: Background dataset for sampling
            X_explain: Data points to explain (n_samples, n_features)
            n_samples: Number of samples for kernel approximation
            
        Returns:
            shap_values: SHAP values (n_samples, n_features)
        """
        # Sample background if too large
        if X_background.shape[0] > 100:
            idx = np.random.choice(X_background.shape[0], 100, replace=False)
            X_background = X_background[idx]
        
        # Create explainer
        if hasattr(model, 'predict_proba'):
            def predict_fn(X):
                proba = model.predict_proba(X)
                # Return probabilities for positive class (binary) or all classes
                if proba.shape[1] == 2:
                    return proba[:, 1]
                return proba
            explainer = shap.KernelExplainer(predict_fn, X_background)
        else:
            explainer = shap.KernelExplainer(model.predict, X_background)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_explain, nsamples=n_samples)
        
        # Handle multi-output case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Use first class
            
        return shap_values
    
    @staticmethod
    def tree_explainer(model, X_explain):
        """
        Estimate Shapley values using TreeExplainer (exact for tree models).
        
        Args:
            model: Trained tree-based model
            X_explain: Data points to explain
            
        Returns:
            shap_values: Exact SHAP values (n_samples, n_features)
        """
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
        
        # Handle multi-output case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        return shap_values
    
    @staticmethod
    def is_tree_based(model):
        """Check if model supports TreeExplainer."""
        tree_types = (
            RandomForestClassifier, RandomForestRegressor,
            DecisionTreeClassifier, DecisionTreeRegressor
        )
        
        if HAS_XGB:
            tree_types = tree_types + (xgb.XGBClassifier, xgb.XGBRegressor)
            
        return isinstance(model, tree_types)
    
    @staticmethod
    def get_best_baseline(model, X_background, X_explain, n_samples=1000):
        """
        Automatically select best baseline method.
        Uses TreeExplainer for tree models, KernelSHAP otherwise.
        """
        if BaselineMethods.is_tree_based(model):
            return BaselineMethods.tree_explainer(model, X_explain)
        else:
            return BaselineMethods.kernelshap(model, X_background, X_explain, n_samples)
