"""
A-FIS Regression Module

Contains utilities for A-FIS regression tasks:
- AFISRegressor: Main regressor class with NW local inference (train, predict, save/load)
- evaluate_kfold: K-fold cross-validation (for model selection)
- benchmark_method: Repeated K-fold with varying test sets (for method evaluation)
- generate_rule_base: Wang-Mendel rule generation wrapper
- Metrics and visualization utilities
"""

from .regressor import (
    AFISRegressor,
    generate_rule_base,
    compute_correlation_weights,
    evaluate_kfold,
    benchmark_method,
    compute_metrics,
    print_metrics,
)
from .utils import (
    plot_correlation_matrix,
    print_correlation_summary,
    plot_predictions_vs_actual,
    plot_residuals,
)

__all__ = [
    # Main class
    'AFISRegressor',
    # Functions
    'generate_rule_base',
    'compute_correlation_weights',
    'evaluate_kfold',
    'benchmark_method',
    'compute_metrics',
    'print_metrics',
    # Visualization
    'plot_correlation_matrix',
    'print_correlation_summary',
    'plot_predictions_vs_actual',
    'plot_residuals',
]

