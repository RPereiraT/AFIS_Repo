"""
A-FIS Visualization Module

Contains plotting and diagnostic tools for A-FIS:
- plot_results: Plot antecedents/consequents with inputs/outputs
- plot_supremum: Visualize supremum of fuzzy sets
- show_svi_table: Display SVI and A-subsethood measures
- And more...
"""

from .plotting import (
    # Setup
    create_rule_base,
    run_afis,
    # Plotting
    plot_results,
    plot_antecedents_stacked,
    plot_antecedents_3d,
    plot_supremum,
    # Diagnostics
    show_svi_table,
    show_detailed_diagnostic,
    show_detailed_diagnostic_nd,
    # Experiments
    test_swapped_rule_base,
    test_multiple_random_shuffles,
    # Activation curves
    compute_activation_curves,
    plot_activation_curves,
)

__all__ = [
    'create_rule_base',
    'run_afis',
    'plot_results',
    'plot_antecedents_stacked',
    'plot_antecedents_3d',
    'plot_supremum',
    'show_svi_table',
    'show_detailed_diagnostic',
    'show_detailed_diagnostic_nd',
    'test_swapped_rule_base',
    'test_multiple_random_shuffles',
    'compute_activation_curves',
    'plot_activation_curves',
]

