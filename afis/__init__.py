"""
AFIS - A-Fuzzy Inference System Library

A comprehensive library for A-FIS (A-subsethood based Fuzzy Inference System).

Modules:
- core: Core A-FIS algorithm, fuzzy structures, and rule generation
- visualization: Plotting and diagnostic tools
- regression: Regression utilities and model classes

Usage:
    from afis.core import A_FIS, FuzzyRuleBase, wangmendel
    from afis.visualization import plot_results, plot_supremum
    from afis.regression import AFISRegressor, evaluate_kfold

You can also import directly:
    from afis import A_FIS
"""

__version__ = "1.0.0"

# Re-export core components for backwards compatibility
from .core import (
    # A_FIS main functions
    A_FIS,
    format_FN_N_Dim,
    cuenta_antecedentes,
    sort_antecedents_spatially,
    D_LR,
    fuzzy_imp,
    # Fuzzy structures
    FuzzyRuleBase,
    FuzzyRule,
    FuzzySet,
    Triangular,
    Trapezoidal,
    InferiorBorder,
    SuperiorBorder,
    Gaussian,
    centroid,
    # Supremum calculations
    nu_A_vee_B,
    nu_A_vee_B_auto,
    nu_numerical,
    nu_A_vee_B_numerical,
    # Submodules
    wangmendel,
)

__all__ = [
    # Core A_FIS
    'A_FIS',
    'format_FN_N_Dim',
    'cuenta_antecedentes',
    'sort_antecedents_spatially',
    'D_LR',
    'fuzzy_imp',
    # Fuzzy structures
    'FuzzyRuleBase',
    'FuzzyRule',
    'FuzzySet',
    'Triangular',
    'Trapezoidal',
    'InferiorBorder',
    'SuperiorBorder',
    'Gaussian',
    'centroid',
    # Supremum
    'nu_A_vee_B',
    'nu_A_vee_B_auto',
    'nu_numerical',
    'nu_A_vee_B_numerical',
    # Submodules
    'wangmendel',
]
