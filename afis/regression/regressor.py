"""
A-FIS Regression Utilities

This module provides a clean API for A-FIS regression tasks:
- AFISRegressor: Train, predict, save/load models
- evaluate_kfold: K-fold cross-validation
- generate_rule_base: Wang-Mendel rule generation wrapper
"""

import copy
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial import distance

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

# Import A-FIS components from core module
from ..core.A_FIS import A_FIS, format_FN_N_Dim
from ..core.afis_utils import centroid
from ..core import wangmendel


# =============================================================================
# AFISRegressor Class
# =============================================================================

class AFISRegressor:
    """
    A-FIS Regressor with distance-weighted local inference.
    
    This class encapsulates a complete trained A-FIS regression model,
    including the rule base, point-rule associations, optimized parameters,
    and the optimal k value.
    
    Example
    -------
    >>> model = AFISRegressor(config)
    >>> model.fit(X_train, y_train, rule_base)
    >>> predictions = model.predict(X_test)
    >>> model.save('model.pkl')
    >>> 
    >>> # Later...
    >>> model = AFISRegressor.load('model.pkl')
    >>> new_predictions = model.predict(X_new)
    """
    
    def __init__(self, config=None):
        """
        Initialize the regressor with configuration.
        
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary with keys:
            - agg_method: Aggregation method (default: 'avg')
            - t_norm_type: T-norm type (default: 'product')
            - imp_params: Implication parameters (default: ['luka', 1])
            - disc: Discretization (default: 100)
            - k_max: Max neighbors for k-search (default: 10)
            - p_norm: Minkowski distance norm (default: 1)
            - param_range: Implication param search range (default: (0.1, 50.0))
            - param_step: Implication param search step (default: 0.5)
        """
        default_config = {
            'agg_method': 'avg',
            't_norm_type': 'product',
            'imp_params': ['luka', 1],
            'disc': 100,
            'k_max': 10,
            'p_norm': 1,
            'param_range': (0.1, 50.0),
            'param_step': 0.5,
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Model state (populated by fit())
        self.rule_base = None
        self.df_train = None
        self.optimal_k = None
        self.scaler = None
        self.is_fitted = False
        
    def fit(self, X_train, y_train, rule_base, X_val=None, y_val=None,
            optimize_imp_params=True, optimize_k=False, show_progress=True):
        """
        Train the model.
        
        Steps:
        1. Build point-rule associations
        2. Optimize implication parameters (if parametric and optimize_imp_params=True)
        3. Find optimal k (if X_val provided and optimize_k=True)
        
        Parameters
        ----------
        X_train : np.ndarray
            Training inputs, shape (n_samples, n_features).
        y_train : np.ndarray
            Training targets, shape (n_samples,).
        rule_base : FuzzyRuleBase
            Pre-generated fuzzy rule base.
        X_val : np.ndarray, optional
            Validation inputs for k-optimization.
        y_val : np.ndarray, optional
            Validation targets for k-optimization.
        optimize_imp_params : bool
            Whether to optimize implication parameters (default: True).
        optimize_k : bool
            Whether to search for optimal k (default: False).
        show_progress : bool
            Whether to show progress bars (default: True).
            
        Returns
        -------
        self : AFISRegressor
            The fitted model.
        """
        self.rule_base = rule_base
        
        # Step 1: Build point-rule associations
        if show_progress:
            print("[1/3] Building point-rule associations...")
        
        self.df_train = _build_point_rule_associations(
            rule_base, X_train, y_train,
            agg_method=self.config['agg_method'],
            disc=self.config['disc'],
            t_norm_type=self.config['t_norm_type'],
            imp_params=self.config['imp_params'],
            show_progress=show_progress
        )
        
        # Step 2: Optimize implication parameters (if parametric)
        if optimize_imp_params:
            if show_progress:
                print("[2/3] Optimizing implication parameters...")
            
            self.df_train = _optimize_imp_params(
                self.df_train, rule_base,
                agg_method=self.config['agg_method'],
                t_norm_type=self.config['t_norm_type'],
                imp_params=self.config['imp_params'],
                disc=self.config['disc'],
                param_range=self.config['param_range'],
                param_step=self.config['param_step'],
                show_progress=show_progress
            )
        else:
            if show_progress:
                print("[2/3] Skipping implication parameter optimization")
            self.df_train['best_param_value'] = self.config['imp_params'][1]
        
        # Step 3: Find optimal k
        k_fixed = self.config.get('k_fixed')
        
        if k_fixed is not None:
            # Use fixed k value (no optimization)
            if show_progress:
                print(f"[3/3] Using fixed k = {k_fixed}")
            self.optimal_k = k_fixed
        elif optimize_k and X_val is not None and y_val is not None:
            if show_progress:
                print("[3/3] Finding optimal k...")
            
            self.optimal_k, _, _ = _find_optimal_k(
                X_val, y_val, self.df_train, rule_base,
                k_max=self.config['k_max'],
                p_norm=self.config['p_norm'],
                agg_method=self.config['agg_method'],
                t_norm_type=self.config['t_norm_type'],
                imp_params=self.config['imp_params'],
                disc=self.config['disc'],
                show_progress=show_progress
            )
            if show_progress:
                print(f"    Optimal k: {self.optimal_k}")
        else:
            if show_progress:
                print("[3/3] Using default k (no validation data)")
            self.optimal_k = self.config['k_max']
        
        # Fit scaler for prediction
        X_train_stacked = np.vstack(self.df_train['X_train'].values)
        self.scaler = MinMaxScaler()
        self.scaler.fit(X_train_stacked)
        
        self.is_fitted = True
        return self
    
    def predict(self, X, show_progress=True):
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data, shape (n_samples, n_features).
        show_progress : bool
            Whether to show progress bar (default: True).
            
        Returns
        -------
        predictions : np.ndarray
            Predicted values, shape (n_samples,).
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        predictions = _predict_with_model(
            X, self.df_train, self.rule_base, self.scaler,
            k_neighbors=self.optimal_k,
            p_norm=self.config['p_norm'],
            agg_method=self.config['agg_method'],
            t_norm_type=self.config['t_norm_type'],
            imp_params=self.config['imp_params'],
            disc=self.config['disc'],
            show_progress=show_progress
        )
        
        return np.array(predictions)
    
    def save(self, path):
        """
        Save the model to disk.
        
        Parameters
        ----------
        path : str
            File path (typically .pkl extension).
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        model_state = {
            'config': self.config,
            'rule_base': self.rule_base,
            'df_train': self.df_train,
            'optimal_k': self.optimal_k,
            'scaler': self.scaler,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
        
        print(f"Model saved to: {path}")
    
    @classmethod
    def load(cls, path):
        """
        Load a model from disk.
        
        Parameters
        ----------
        path : str
            File path to saved model.
            
        Returns
        -------
        model : AFISRegressor
            Loaded model ready for prediction.
        """
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        model = cls(config=model_state['config'])
        model.rule_base = model_state['rule_base']
        model.df_train = model_state['df_train']
        model.optimal_k = model_state['optimal_k']
        model.scaler = model_state['scaler']
        model.is_fitted = True
        
        print(f"Model loaded from: {path}")
        return model
    
    def get_params(self):
        """
        Get model parameters summary.
        
        Returns
        -------
        params : dict
            Dictionary with model parameters.
        """
        params = {
            'config': self.config,
            'is_fitted': self.is_fitted,
        }
        
        if self.is_fitted:
            params.update({
                'n_rules': len(self.rule_base.ruleBase),
                'n_train_points': len(self.df_train),
                'optimal_k': self.optimal_k,
            })
            
            if 'best_param_value' in self.df_train.columns:
                params['imp_param_stats'] = {
                    'mean': self.df_train['best_param_value'].mean(),
                    'std': self.df_train['best_param_value'].std(),
                    'min': self.df_train['best_param_value'].min(),
                    'max': self.df_train['best_param_value'].max(),
                }
        
        return params


# =============================================================================
# High-Level Functions
# =============================================================================

def generate_rule_base(X, y, n_fuzzy_partitions, n_rules, 
                       filter_method='balanced', verbose=False, margin=0.05):
    """
    Generate a fuzzy rule base using Wang-Mendel algorithm.
    
    Parameters
    ----------
    X : np.ndarray
        Input data, shape (n_samples, n_features).
    y : np.ndarray
        Target values, shape (n_samples,).
    n_fuzzy_partitions : int
        Number of fuzzy partitions per dimension.
    n_rules : int
        Target number of rules.
    filter_method : str, optional
        'balanced' - Keep N rules per consequent for balanced coverage (default)
        'top_n' - Keep top N rules by strength (traditional Wang-Mendel)
    verbose : bool, optional
        If True, print rule statistics after each step. Default False.
    margin : float, optional
        Percentage margin to extend fuzzy regions beyond data min/max.
        Default 0.05 (5%). Set to 0 for exact data boundaries.
        
    Returns
    -------
    rule_base : FuzzyRuleBase
        Generated fuzzy rule base.
    """
    return wangmendel.generate_rule_base(
        X, y, n_fuzzy_partitions, n_rules, 
        filter_method=filter_method, verbose=verbose, margin=margin
    )


def compute_correlation_weights(X, y):
    """
    Compute correlation-based weights for weighted aggregation.
    
    Computes the absolute correlation between each feature and the target,
    then normalizes so weights sum to 1.
    
    Parameters
    ----------
    X : np.ndarray
        Input data, shape (n_samples, n_features).
    y : np.ndarray
        Target values, shape (n_samples,).
        
    Returns
    -------
    weights : list
        Normalized correlation weights for each feature.
    """
    n_features = X.shape[1]
    correlations = []
    
    for i in range(n_features):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append(abs(corr) if not np.isnan(corr) else 0)
    
    correlations = np.array(correlations)
    
    # Normalize to sum to 1
    if correlations.sum() > 0:
        weights = correlations / correlations.sum()
    else:
        weights = np.ones(n_features) / n_features
    
    return weights.tolist()


def evaluate_kfold(X, y, rule_base_generator, config, n_splits=5, 
                   n_repetitions=1, random_state=42, show_progress=True):
    """
    Evaluate A-FIS regression using (repeated) K-fold cross-validation.
    
    If n_repetitions > 1, performs multiple rounds of K-fold with different
    random seeds and returns mean ± std statistics across all evaluations.
    
    If config['agg_method'] is ['weighted_avg', ...], correlation weights
    are recomputed for each fold using that fold's training data.
    
    Parameters
    ----------
    X : np.ndarray
        Input data.
    y : np.ndarray
        Target values.
    rule_base_generator : callable
        Function that takes (X_train, y_train) and returns a FuzzyRuleBase.
        Example: lambda X, y: generate_rule_base(X, y, n_parts, n_rules)
    config : dict
        Configuration dictionary for AFISRegressor.
    n_splits : int
        Number of folds per repetition (default: 5).
    n_repetitions : int
        Number of times to repeat K-fold (default: 1).
        If > 1, std is computed across all (n_repetitions × n_splits) evaluations.
    random_state : int
        Base random seed (default: 42). Each repetition uses random_state + rep.
    show_progress : bool
        Whether to show progress (default: True).
        
    Returns
    -------
    results : dict
        Dictionary with:
        - 'mean_rmse': Mean RMSE across all folds/repetitions
        - 'std_rmse': Std of RMSE (meaningful when n_repetitions > 1)
        - 'all_results': List of per-fold results (with repetition info)
        - 'best_model': AFISRegressor from best fold
        - 'n_repetitions': Number of repetitions performed
        - 'n_splits': Number of folds per repetition
    """
    # Check if using weighted_avg (needs per-fold weight computation)
    uses_weighted_avg = (
        isinstance(config.get('agg_method'), list) and 
        len(config.get('agg_method', [])) >= 1 and
        config['agg_method'][0] == 'weighted_avg'
    )
    
    all_results = []
    best_rmse = float('inf')
    best_model = None
    
    total_evals = n_repetitions * n_splits
    eval_count = 0
    
    for rep in range(n_repetitions):
        rep_seed = random_state + rep  # Different seed per repetition
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=rep_seed)
        
        if show_progress and n_repetitions > 1:
            print(f"\n{'#'*60}")
            print(f"Repetition {rep + 1}/{n_repetitions}")
            print(f"{'#'*60}")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            eval_count += 1
            
            if show_progress:
                if n_repetitions > 1:
                    print(f"\n  Fold {fold + 1}/{n_splits} (eval {eval_count}/{total_evals})")
                else:
                    print(f"\n{'='*60}")
                    print(f"Fold {fold + 1}/{n_splits}")
                    print(f"{'='*60}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create fold-specific config
            fold_config = copy.deepcopy(config)
            
            # Recompute correlation weights for this fold if using weighted_avg
            if uses_weighted_avg:
                fold_weights = compute_correlation_weights(X_train, y_train)
                fold_config['agg_method'] = ['weighted_avg', fold_weights]
                if show_progress:
                    print(f"  Correlation weights: {[f'{w:.3f}' for w in fold_weights]}")
            
            # Generate rule base for this fold
            rule_base = rule_base_generator(X_train, y_train)
            if show_progress:
                print(f"  Rules: {len(rule_base.ruleBase)}")
            
            # Train model with fold-specific config
            model = AFISRegressor(fold_config)
            model.fit(X_train, y_train, rule_base, X_val, y_val, 
                     show_progress=show_progress and n_repetitions == 1)
            
            # Evaluate on validation
            val_predictions = model.predict(X_val, show_progress=False)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
            
            if show_progress:
                print(f"  Validation RMSE: {val_rmse:.4f}")
            
            all_results.append({
                'repetition': rep + 1,
                'fold': fold + 1,
                'val_rmse': val_rmse,
                'optimal_k': model.optimal_k,
                'n_rules': len(rule_base.ruleBase),
            })
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_model = model
    
    # Compute statistics
    rmse_values = [r['val_rmse'] for r in all_results]
    mean_rmse = np.mean(rmse_values)
    std_rmse = np.std(rmse_values)
    
    results = {
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'all_results': all_results,
        'best_model': best_model,
        'n_repetitions': n_repetitions,
        'n_splits': n_splits,
    }
    
    # Summary
    if show_progress:
        print(f"\n{'='*60}")
        if n_repetitions > 1:
            print(f"Repeated K-Fold Summary ({n_repetitions}×{n_splits} = {total_evals} evaluations)")
        else:
            print("K-Fold Summary")
        print(f"{'='*60}")
        print(f"Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
        
        if n_repetitions > 1:
            # Per-repetition summary
            for rep in range(n_repetitions):
                rep_rmses = [r['val_rmse'] for r in all_results if r['repetition'] == rep + 1]
                print(f"  Rep {rep + 1}: {np.mean(rep_rmses):.4f} ± {np.std(rep_rmses):.4f}")
        
        best_idx = rmse_values.index(min(rmse_values))
        best_result = all_results[best_idx]
        if n_repetitions > 1:
            print(f"Best: Rep {best_result['repetition']}, Fold {best_result['fold']} "
                  f"(RMSE={best_result['val_rmse']:.4f})")
        else:
            print(f"Best fold: {best_result['fold']} (RMSE={best_result['val_rmse']:.4f})")
    
    return results


def benchmark_method(X, y, rule_base_generator, config, n_splits=5, 
                     n_repetitions=10, test_size=0.2, random_state=42, 
                     show_progress=True):
    """
    Benchmark A-FIS regression method with varying train/test splits.
    
    This function is designed for METHOD EVALUATION, not model deployment.
    Each repetition uses a different train/test split, providing robust
    statistics (mean ± std) of TEST performance across different data scenarios.
    
    Parameters
    ----------
    X : np.ndarray
        Full input data.
    y : np.ndarray
        Full target values.
    rule_base_generator : callable
        Function that takes (X_train, y_train) and returns a FuzzyRuleBase.
    config : dict
        Configuration dictionary for AFISRegressor.
    n_splits : int
        Number of K-fold splits within each repetition (default: 5).
    n_repetitions : int
        Number of repetitions with different train/test splits (default: 10).
    test_size : float
        Fraction of data for test set (default: 0.2).
    random_state : int
        Base random seed (default: 42).
    show_progress : bool
        Whether to show progress (default: True).
        
    Returns
    -------
    results : dict
        Dictionary with:
        - 'mean_test_rmse': Mean TEST RMSE across repetitions
        - 'std_test_rmse': Std of TEST RMSE across repetitions
        - 'repetition_results': List of per-repetition results
        - 'all_test_rmses': List of all test RMSEs
    """
    # Check if using weighted_avg
    uses_weighted_avg = (
        isinstance(config.get('agg_method'), list) and 
        len(config.get('agg_method', [])) >= 1 and
        config['agg_method'][0] == 'weighted_avg'
    )
    
    repetition_results = []
    all_test_rmses = []
    
    for rep in range(n_repetitions):
        rep_seed = random_state + rep
        
        if show_progress:
            print(f"\n{'#'*60}")
            print(f"Repetition {rep + 1}/{n_repetitions} (seed={rep_seed})")
            print(f"{'#'*60}")
        
        # Fresh train/test split for this repetition
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=rep_seed
        )
        
        if show_progress:
            print(f"Data split: trainval={len(X_trainval)}, test={len(X_test)}")
        
        # Run K-fold on trainval (with suppressed progress for cleaner output)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=rep_seed)
        
        best_val_rmse = float('inf')
        best_fold_model = None
        fold_val_rmses = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
            X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
            y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
            
            # Fold-specific config
            fold_config = copy.deepcopy(config)
            if uses_weighted_avg:
                fold_weights = compute_correlation_weights(X_train, y_train)
                fold_config['agg_method'] = ['weighted_avg', fold_weights]
            
            # Generate rule base and train
            rule_base = rule_base_generator(X_train, y_train)
            model = AFISRegressor(fold_config)
            model.fit(X_train, y_train, rule_base, X_val, y_val, show_progress=False)
            
            # Validate
            val_pred = model.predict(X_val, show_progress=False)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            fold_val_rmses.append(val_rmse)
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_fold_model = model
        
        if show_progress:
            print(f"K-fold validation: {np.mean(fold_val_rmses):.4f} ± {np.std(fold_val_rmses):.4f}")
        
        # Test best model from this repetition on THIS repetition's test set
        test_predictions = best_fold_model.predict(X_test, show_progress=False)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        all_test_rmses.append(test_rmse)
        
        if show_progress:
            print(f"TEST RMSE: {test_rmse:.4f}")
        
        repetition_results.append({
            'repetition': rep + 1,
            'seed': rep_seed,
            'val_rmse_mean': np.mean(fold_val_rmses),
            'val_rmse_std': np.std(fold_val_rmses),
            'test_rmse': test_rmse,
            'optimal_k': best_fold_model.optimal_k,
        })
    
    # Aggregate statistics
    mean_test_rmse = np.mean(all_test_rmses)
    std_test_rmse = np.std(all_test_rmses)
    
    results = {
        'mean_test_rmse': mean_test_rmse,
        'std_test_rmse': std_test_rmse,
        'all_test_rmses': all_test_rmses,
        'repetition_results': repetition_results,
        'n_repetitions': n_repetitions,
        'n_splits': n_splits,
    }
    
    # Final summary
    if show_progress:
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY ({n_repetitions} repetitions × {n_splits}-fold)")
        print(f"{'='*60}")
        print(f"TEST RMSE: {mean_test_rmse:.4f} ± {std_test_rmse:.4f}")
        print(f"Range: [{min(all_test_rmses):.4f}, {max(all_test_rmses):.4f}]")
        print(f"{'='*60}")
    
    return results


def compute_metrics(y_true, y_pred):
    """
    Compute regression metrics.
    
    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
        
    Returns
    -------
    metrics : dict
        Dictionary with RMSE, MAE, R².
    """
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }


def print_metrics(y_true, y_pred):
    """Print regression metrics."""
    metrics = compute_metrics(y_true, y_pred)
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")


# =============================================================================
# Internal Helper Functions
# =============================================================================

def _get_consequent_centroids(rule_base, disc=100):
    """Compute centroid of each unique consequent."""
    consequents = list(set(rule.consequent for rule in rule_base.ruleBase))
    Y = np.linspace(rule_base.outputRange[0], rule_base.outputRange[1], disc)
    
    centroid_list = []
    for cons in consequents:
        membership_values = np.array([cons.set.pertinence(y) for y in Y])
        cons_centroid = centroid(Y, membership_values)
        centroid_list.append((cons, cons_centroid))
    
    return sorted(centroid_list, key=lambda x: x[1])


def _group_rules_by_consequent(rule_base):
    """Group rules by consequent name."""
    groups = {}
    for rule in rule_base.ruleBase:
        name = rule.consequent.name
        if name not in groups:
            groups[name] = []
        groups[name].append(rule)
    return groups


def _find_bracketing_consequents(y, centroid_list):
    """Find consequent indices that bracket value y."""
    centroids = [c[1] for c in centroid_list]
    
    if y <= centroids[0]:
        return (0, 0)
    elif y >= centroids[-1]:
        n = len(centroids) - 1
        return (n, n)
    else:
        for i, c in enumerate(centroids):
            if y < c:
                return (i - 1, i)
    return (0, 0)


def _build_point_rule_associations(rule_base, X_train, y_train, 
                                   agg_method='avg', disc=100,
                                   t_norm_type='product', imp_params=['luka', 1],
                                   show_progress=True):
    """Build point-rule associations for training data."""
    centroid_list = _get_consequent_centroids(rule_base, disc)
    rule_groups = _group_rules_by_consequent(rule_base)
    
    associations = []
    
    iterator = range(len(X_train))
    if show_progress:
        iterator = tqdm(iterator, desc="Building associations")
    
    for i in iterator:
        x = X_train[i]
        y = y_train[i]
        
        left_idx, right_idx = _find_bracketing_consequents(y, centroid_list)
        left_cons = centroid_list[left_idx][0]
        right_cons = centroid_list[right_idx][0]
        
        left_rules = rule_groups.get(left_cons.name, [])
        right_rules = rule_groups.get(right_cons.name, [])
        
        input_formatted = format_FN_N_Dim(x)
        
        # Find best rule on each side
        best_left_rule = _find_best_rule(
            input_formatted, left_rules, rule_base,
            agg_method, disc, t_norm_type, imp_params
        )
        best_right_rule = _find_best_rule(
            input_formatted, right_rules, rule_base,
            agg_method, disc, t_norm_type, imp_params
        )
        
        associated_rules = []
        if best_left_rule is not None:
            associated_rules.append(best_left_rule)
        if best_right_rule is not None and best_right_rule != best_left_rule:
            associated_rules.append(best_right_rule)
        
        associations.append({
            'X_train': x,
            'y_train': y,
            'ruleBase': associated_rules
        })
    
    return pd.DataFrame(associations)


def _find_best_rule(input_formatted, rules, rule_base, agg_method, disc, t_norm_type, imp_params):
    """Find rule with highest activation."""
    best_rule = None
    best_activation = -1
    
    for rule in rules:
        temp_rb = copy.copy(rule_base)
        temp_rb.ruleBase = [rule]
        _, _, max_vals = A_FIS(
            input_formatted, temp_rb, agg_method, disc,
            t_norm_type, imp_params
        )
        activation = next(iter(max_vals.values()))[0] if max_vals else 0
        if activation > best_activation:
            best_activation = activation
            best_rule = rule
    
    return best_rule


def _optimize_imp_params(df_train, rule_base, agg_method, t_norm_type, imp_params,
                         disc=100, param_range=(0.1, 50.0), param_step=0.5,
                         show_progress=True):
    """Optimize implication parameter for each training point."""
    parametric_imps = ['hamacher', 'dombi']
    
    if imp_params[0].lower() not in parametric_imps:
        if show_progress:
            print(f"    Skipping ('{imp_params[0]}' is not parametric)")
        df_train['best_param_value'] = imp_params[1]
        return df_train
    
    param_list = np.arange(param_range[0], param_range[1], param_step)
    best_params = []
    
    iterator = df_train.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(df_train), desc="Optimizing params")
    
    for idx, row in iterator:
        x_train = row['X_train']
        y_train = row['y_train']
        local_rules = row['ruleBase']
        
        local_rb = copy.copy(rule_base)
        local_rb.ruleBase = local_rules
        
        # Grid search for best parameter
        errors = []
        for param in param_list:
            y_pred = _afis_predict_single(
                x_train, local_rb, agg_method, t_norm_type,
                imp_params, param, disc
            )
            errors.append(abs(y_train - y_pred))
        
        best_params.append(param_list[np.argmin(errors)])
    
    df_train['best_param_value'] = best_params
    
    if show_progress:
        print(f"    Param stats: mean={np.mean(best_params):.2f}, "
              f"std={np.std(best_params):.2f}")
    
    return df_train


def _afis_predict_single(x_input, rule_base, agg_method, t_norm_type, imp_params, param_value, disc):
    """Run A-FIS with specific implication parameter."""
    imp_params_copy = copy.copy(imp_params)
    imp_params_copy[1] = param_value
    
    t_norm_copy = copy.copy(t_norm_type) if isinstance(t_norm_type, list) else t_norm_type
    if isinstance(t_norm_copy, list):
        t_norm_copy[1] = param_value
    
    input_formatted = format_FN_N_Dim(x_input)
    y_output, U, _ = A_FIS(
        input_formatted, rule_base, agg_method, disc,
        t_norm_copy, imp_params_copy
    )
    
    return centroid(U, y_output)


def _find_optimal_k(X_val, y_val, df_train, rule_base, k_max, p_norm,
                    agg_method, t_norm_type, imp_params, disc=100,
                    show_progress=True):
    """Find optimal k using validation data."""
    X_train_stacked = np.vstack(df_train['X_train'].values)
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train_stacked)
    
    has_optimized_params = 'best_param_value' in df_train.columns
    
    all_predictions = []
    
    iterator = enumerate(X_val)
    if show_progress:
        iterator = tqdm(iterator, total=len(X_val), desc="Optimizing k")
    
    for j, x_val in iterator:
        x_normalized = scaler.transform(x_val.reshape(1, -1)).flatten()
        distances = [distance.minkowski(x, x_normalized, p_norm) for x in X_train_normalized]
        
        df_distances = pd.DataFrame({
            'Index': df_train.index,
            'Distance': distances
        }).sort_values(by='Distance')
        
        sorted_indices = df_distances['Index'].tolist()
        sorted_distances = df_distances['Distance'].tolist()
        
        input_formatted = format_FN_N_Dim(x_val)
        predictions_for_k = []
        neighbor_outputs = []
        
        for k in range(min(k_max, len(sorted_indices))):
            idx = sorted_indices[k]
            row = df_train.loc[idx]
            
            local_rules = row['ruleBase']
            if len(local_rules) == 0:
                continue
            
            local_rb = copy.copy(rule_base)
            local_rb.ruleBase = local_rules
            
            imp_params_k = copy.copy(imp_params)
            if has_optimized_params and imp_params[0].lower() in ['hamacher', 'dombi']:
                imp_params_k[1] = row['best_param_value']
            
            output, U, _ = A_FIS(
                input_formatted, local_rb, agg_method, disc,
                t_norm_type, imp_params_k
            )
            y_pred = centroid(U, output)
            neighbor_outputs.append(y_pred)
            
            k_distances = sorted_distances[:len(neighbor_outputs)]
            weights = _compute_distance_weights(k_distances)
            weighted_pred = np.average(neighbor_outputs, weights=weights)
            predictions_for_k.append(weighted_pred)
        
        while len(predictions_for_k) < k_max:
            predictions_for_k.append(predictions_for_k[-1] if predictions_for_k else 0)
        
        all_predictions.append(predictions_for_k)
    
    all_predictions = np.array(all_predictions)
    
    rmse_by_k = []
    for k in range(k_max):
        preds_k = all_predictions[:, k]
        rmse_k = np.sqrt(mean_squared_error(y_val, preds_k))
        rmse_by_k.append(rmse_k)
    
    best_k = np.argmin(rmse_by_k) + 1
    best_rmse = rmse_by_k[best_k - 1]
    
    return best_k, best_rmse, rmse_by_k


def _compute_distance_weights(distances):
    """Compute inverse-distance weights."""
    distances = np.array(distances)
    
    if np.sum(distances) == 0:
        return np.ones(len(distances)) / len(distances)
    
    total_dist = np.sum(distances)
    weights = (total_dist - distances)
    
    if np.sum(weights) == 0:
        return np.ones(len(distances)) / len(distances)
    
    return weights / np.sum(weights)


def _predict_with_model(X_test, df_train, rule_base, scaler, k_neighbors, p_norm,
                        agg_method, t_norm_type, imp_params, disc=100,
                        show_progress=True):
    """Make predictions using trained model components."""
    X_train_normalized = scaler.transform(np.vstack(df_train['X_train'].values))
    
    has_optimized_params = 'best_param_value' in df_train.columns
    use_optimized = has_optimized_params and imp_params[0].lower() in ['hamacher', 'dombi']
    
    predictions = []
    
    iterator = enumerate(X_test)
    if show_progress:
        iterator = tqdm(iterator, total=len(X_test), desc="Predicting")
    
    for j, x_test in iterator:
        x_normalized = scaler.transform(x_test.reshape(1, -1)).flatten()
        
        distances = [distance.minkowski(x, x_normalized, p_norm) for x in X_train_normalized]
        df_distances = pd.DataFrame({
            'Index': df_train.index,
            'Distance': distances
        }).sort_values(by='Distance')
        
        k_indices = df_distances['Index'].tolist()[:k_neighbors]
        k_distances = df_distances['Distance'].tolist()[:k_neighbors]
        
        input_formatted = format_FN_N_Dim(x_test)
        neighbor_predictions = []
        
        for idx in k_indices:
            row = df_train.loc[idx]
            local_rules = row['ruleBase']
            
            if len(local_rules) == 0:
                continue
            
            local_rb = copy.copy(rule_base)
            local_rb.ruleBase = local_rules
            
            imp_params_local = copy.copy(imp_params)
            if use_optimized:
                imp_params_local[1] = row['best_param_value']
            
            output, U, _ = A_FIS(
                input_formatted, local_rb, agg_method, disc,
                t_norm_type, imp_params_local
            )
            y_pred = centroid(U, output)
            neighbor_predictions.append(y_pred)
        
        if not neighbor_predictions:
            predictions.append(0)
        else:
            weights = _compute_distance_weights(k_distances[:len(neighbor_predictions)])
            predictions.append(float(np.average(neighbor_predictions, weights=weights)))
    
    return predictions

