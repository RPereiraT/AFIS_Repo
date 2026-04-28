"""
A-FIS Core Module

Contains the core A-FIS algorithm and supporting components:
- A_FIS: Main inference algorithm
- A_vee_B: Supremum calculations (analytical and numerical)
- utils: Fuzzy structures (Triangular, Trapezoidal, Gaussian, etc.)
- wangmendel: Rule generation using Wang-Mendel algorithm
"""

from .A_FIS import (
    A_FIS, 
    format_FN_N_Dim, 
    cuenta_antecedentes, 
    sort_antecedents_spatially, 
    D_LR, 
    fuzzy_imp
)
from .afis_utils import (
    FuzzyRuleBase, 
    FuzzyRule, 
    FuzzySet,
    Triangular, 
    Trapezoidal, 
    InferiorBorder, 
    SuperiorBorder,
    Gaussian,
    centroid
)
from .A_vee_B import (
    nu_A_vee_B,
    nu_A_vee_B_auto,
    nu_numerical,
    nu_A_vee_B_numerical
)
from . import wangmendel

__all__ = [
    # A_FIS
    'A_FIS',
    'format_FN_N_Dim',
    'cuenta_antecedentes',
    'sort_antecedents_spatially',
    'D_LR',
    'fuzzy_imp',
    # utils
    'FuzzyRuleBase',
    'FuzzyRule',
    'FuzzySet',
    'Triangular',
    'Trapezoidal',
    'InferiorBorder',
    'SuperiorBorder',
    'Gaussian',
    'centroid',
    # A_vee_B
    'nu_A_vee_B',
    'nu_A_vee_B_auto',
    'nu_numerical',
    'nu_A_vee_B_numerical',
    # wangmendel
    'wangmendel',
]

